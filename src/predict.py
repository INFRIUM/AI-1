"""
학습된 Isolation Forest 로 Window 별 Anomaly Score 계산 및 Normal / Anomaly 판정.

   9. Anomaly Score   : 정상 기준 대비 z-score → IF 점수 → 0~1 변환 (train.to_unit_score)
  10. 판정            : (1) Window 1개 기준 raw_prediction
                            IF 점수 > 임계값  또는  (보조 규칙 사용 시) 최대|z| > z 임계값
                        (2) 연속 판정 prediction (최종, 사용 시)
                            최근 persist_n 개 Window 중 persist_k 개 이상이 (1)에서 Anomaly
                            → 과거 Window 만 보므로 실시간 데이터에도 그대로 적용 가능
      모니터링        : 센서별 모델로 <센서>_score / <센서>_state, 그리고
                        anomaly_sensors (이상 센서 목록), main_sensor·main_location (가장 크게 벗어난 센서와 위치),
                        symptom (무엇이 평소와 다른가. 예: "변형률 평균 증가 (+18.1σ)")
                        ※ symptom 은 '증상'이며 D1~D3 원인 유형 판정이 아니다

입력: data/processed/<source>/<bridge>/features.parquet
      models/<source>/<bridge>/isolation_forest.joblib
출력: results/<source>/<bridge>/predictions.csv   ← 최종 결과 (AI② 위험도 모델·Backend 전달용)
      한 행 = 실험 1개 × Window 1개

정답(damage_type, severity_level)은 읽지 않는다. 실제 운영처럼 센서 데이터만으로 판정한다.

실행: python src/predict.py                         ← 데이터셋 전체 예측 → predictions.csv
      python src/predict.py --file <센서파일 경로> [--position C]
                                                    ← 14. 새로 들어온 센서 파일 1개 판정 (하중 위치 코드 지정)
                                                        → results/<source>/<bridge>/new/<파일명>_predictions.csv
                                                        → results/<source>/<bridge>/new/<파일명>_ai2_messages.jsonl
                                                          (구간마다 raw + AI① 결과 메시지. config.AI2_API url 이 있으면 AI② 로 POST)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ai2_message import build_messages, load_confidence, save_jsonl, send
from config import (MODEL_SUFFIX, ACTIVE_BRIDGE, AI2_API, AXIS_KO, BRIDGES, DECISION, EXPERIMENT, FEATURE_KO, FILTER, IFOREST,
                    MODELS_DIR, OUTPUT_FEATURES, PREDICTION_LABELS, PROCESSED_DIR, RESULTS_DIR, SENSOR_TYPE_KO, SPAN_MM, WINDOW)
from preprocess import PUBLIC, load_experiment, process_experiment, sensor_short_name, window_setting
from preprocess import META_COLS as FEATURE_META_COLS
from train import AXIS_SUFFIXES, feature_name, raw_iforest_score, to_unit_score, to_zscore

HOLD, NOT_SCORED = "판정 보류", "판정 불가"   # 공개 데이터: 열화본 잡음 수준 Window / 센서 없음·계산 불가 Window

FEAT_PATH = PROCESSED_DIR / ACTIVE_BRIDGE / "features.parquet"
MODEL_PATH = MODELS_DIR / ACTIVE_BRIDGE / f"isolation_forest{MODEL_SUFFIX}.joblib"
OUT_PATH = RESULTS_DIR / ACTIVE_BRIDGE / f"predictions{MODEL_SUFFIX}.csv"
NORMAL, ANOMALY = PREDICTION_LABELS
TOP_K = 2  # 이상에 가장 크게 영향을 준 특징 몇 개를 기록할지


def top_features(z, cols, k=TOP_K):
    """정상 기준에서 가장 많이 벗어난 특징 k개와 그 z-score (+: 정상보다 큼, -: 정상보다 작음).
    특징 이름 앞부분이 센서 약칭이므로 이상 위치의 힌트가 된다 (예: strain02_mean → L/2 변형률)."""
    order = np.argsort(-np.abs(z), axis=1)[:, :k]
    return np.asarray(cols)[order], np.take_along_axis(z, order, axis=1)


def persistence(ratio_by_exp):
    """최근 n 개 Window 중 k 번째로 큰 decision_ratio. 1 을 넘으면 'k 개 이상이 이상'과 같은 뜻.
    실험(측정 세션) 시작 직후 Window 가 k 개 모이기 전에는 판정하지 않는다 (0)."""
    n, k = DECISION["persist_n"], DECISION["persist_k"]
    return ratio_by_exp.rolling(n, min_periods=k).apply(lambda a: np.sort(a)[-k], raw=True).fillna(0.0)


def output_feature_columns(sensors, feats):
    """최종 CSV 에 넣을 특징 열 (config.OUTPUT_FEATURES). 가속도는 축마다, 센서 간 비교(cross_)는 모두."""
    cols = []
    for s in sensors:
        prefix = sensor_short_name(s["sensor_id"])
        for name in OUTPUT_FEATURES.get(s["sensor_type"], ()):
            if s["sensor_type"] == "acceleration":
                cols += [f"{prefix}_{name}_{axis.lower()}" for axis in s["axes"]]
            else:
                cols.append(f"{prefix}_{name}")
    cols += [c for c in feats.columns if c.startswith("cross_")]
    return [c for c in cols if c in feats.columns]


def score_and_ratio(z, entry):
    """IF 점수(0~1)와 decision_ratio. ratio = 각 임계값 대비 비율 중 큰 값 → 1 을 넘으면 Window 1개 기준 Anomaly."""
    score = to_unit_score(raw_iforest_score(entry["model"], z), entry["calib"])
    max_abs_z = np.abs(z).max(axis=1)
    ratio = score / entry["threshold"]
    if IFOREST["use_z_rule"]:
        ratio = np.maximum(ratio, max_abs_z / entry["z_threshold"])
    return score, max_abs_z, ratio


def final_ratio(ratio, feats):
    """연속 판정 적용 (실험마다 시간 순서대로). 사용하지 않으면 Window 1개 기준 ratio 그대로."""
    if not DECISION["use_persistence"]:
        return pd.Series(ratio, index=feats.index)
    order = feats.sort_values(["experiment_id", "window_start_us"]).index
    r = pd.Series(ratio, index=feats.index).loc[order]
    return r.groupby(feats.loc[order, "experiment_id"]).transform(persistence).reindex(feats.index)


def symptom_text(sensor_type, col, z):
    """'가속도 수직(Z) 감쇠비 증가 (+6.2σ)' 처럼 무엇이 평소와 다른지 알려주는 문구."""
    direction = "증가" if z > 0 else "감소"
    if col.startswith("cross_"):
        name = col.removeprefix("cross_")
        return f"{SENSOR_TYPE_KO['cross']} {FEATURE_KO.get(name, name)} {direction} ({z:+.1f}σ)"
    feat = feature_name(col)
    axis = col.rsplit("_", 1)[-1]
    axis_text = f" {AXIS_KO[axis.upper()]}" if axis in AXIS_SUFFIXES and col.count("_") >= 2 else ""
    return f"{SENSOR_TYPE_KO.get(sensor_type, sensor_type)}{axis_text} {FEATURE_KO.get(feat, feat)} {direction} ({z:+.1f}σ)"


def predict(feats, bundle):
    """HW 데이터: 모든 Window 판정. 공개 데이터: 판정 가능한 Window 만 판정하고 나머지는 판정 보류/불가로 표시."""
    if not PUBLIC:
        return predict_core(feats, bundle)
    cols = bundle["feature_cols"]
    complete = feats[cols].notna().all(axis=1)
    held = feats["hold_main"].astype(bool)
    usable = complete & ~held
    if bundle.get("model_type") == "zscore":
        core = predict_zscore(feats[usable], bundle)
    else:
        core = predict_core(feats[usable], bundle)
    out = core.reindex(feats.index)
    out.loc[~usable, "prediction"] = np.where(held[~usable], HOLD, NOT_SCORED)
    if "raw_prediction" in out:
        out.loc[~usable, "raw_prediction"] = out.loc[~usable, "prediction"]
    return out


def predict_zscore(feats, bundle):
    """z-score 전용 (Old ADA): 정상 기준에서 가장 크게 벗어난 정도와 특징만. 판정은 하지 않는다 (서술 보고)."""
    cols = bundle["feature_cols"]
    z = to_zscore(feats, cols, bundle["baseline"])
    names, values = top_features(z, cols)
    out = pd.DataFrame({"max_abs_z": np.abs(z).max(axis=1).round(2), "prediction": "판정 없음 (z-score 전용)"},
                       index=feats.index)
    for i in range(TOP_K):
        out[f"top{i + 1}_feature"] = names[:, i]
        out[f"top{i + 1}_z"] = values[:, i].round(2)
    return out


def predict_core(feats, bundle):
    cols = bundle["feature_cols"]
    z = to_zscore(feats, cols, bundle["baseline"])

    # ── 교량 전체: 교량이 평소와 다른가 ──
    score, max_abs_z, ratio = score_and_ratio(z, bundle)
    names, values = top_features(z, cols)
    out = pd.DataFrame({
        "anomaly_score": score.round(4),
        "threshold": round(bundle["threshold"], 4),
        "max_abs_z": max_abs_z.round(2),
        "z_threshold": round(bundle["z_threshold"], 2),
        "decision_ratio": ratio.round(3),
        "raw_prediction": np.where(ratio > 1, ANOMALY, NORMAL),
    }, index=feats.index)
    for i in range(TOP_K):
        out[f"top{i + 1}_feature"] = names[:, i]
        out[f"top{i + 1}_z"] = values[:, i].round(2)
    persist = final_ratio(ratio, feats)
    if DECISION["use_persistence"]:
        out["persist_ratio"] = persist.round(3)
    out["prediction"] = np.where(persist > 1, ANOMALY, NORMAL)

    # ── 센서별: 어느 센서가 평소와 다른가 ──
    sensor_ratio = {}
    for sid, m in bundle["sensor_models"].items():
        zs = z[:, m["col_idx"]]
        s_score, _, s_ratio = score_and_ratio(zs, m)
        s_final = final_ratio(s_ratio, feats)
        out[f"{m['prefix']}_score"] = s_score.round(4)
        out[f"{m['prefix']}_state"] = np.where(s_final > 1, ANOMALY, NORMAL)
        sensor_ratio[sid] = s_final.to_numpy()

    # ── 모니터링 요약: 어디의, 어느 센서가, 무엇이 다른가 ──
    sids = list(sensor_ratio)
    if not sids:   # v2: 센서별로 나뉘지 않는 모양 Feature 라 센서별 모델이 없음
        out["anomaly_sensors"] = out["main_sensor"] = out["main_location"] = out["symptom"] = ""
        return out
    ratios = np.column_stack([sensor_ratio[s] for s in sids])
    anomalous = ratios > 1
    out["anomaly_sensors"] = [";".join(np.asarray(sids)[row]) for row in anomalous]
    main_sensor, main_location, symptom = [], [], []
    for r in range(len(out)):
        if not anomalous[r].any() and out["prediction"].iat[r] == NORMAL:
            main_sensor.append(""), main_location.append(""), symptom.append("")
            continue
        sid = sids[int(np.argmax(ratios[r]))]  # 가장 크게 벗어난 센서
        m = bundle["sensor_models"][sid]
        zs = z[r, m["col_idx"]]
        j = int(np.argmax(np.abs(zs)))
        main_sensor.append(sid)
        main_location.append(f"{m['sensor_location']} (x/L={m['x_ratio']:.2f})" if PUBLIC
                             else f"{m['sensor_location']} (x={m['x_ratio'] * SPAN_MM:.0f}mm)")
        symptom.append(symptom_text(m["sensor_type"], cols[m["col_idx"][j]], zs[j]))
    out["main_sensor"] = main_sensor
    out["main_location"] = main_location
    out["symptom"] = symptom
    return out


def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    if bundle["window"] != window_setting():
        raise SystemExit(f"모델 학습 때 Window 설정 {bundle['window']} 과 현재 설정 {window_setting()} 이 다릅니다.\n"
                         f"→ preprocess.py 와 train.py 를 다시 실행하세요.")
    return bundle


def build_output(feats, result, sensors):
    """최종 결과 표: 구간 정보 + 주요 특징 + 점수·판정·모니터링 열."""
    return pd.concat([
        feats[["experiment_id", "bridge_type"]].assign(model_material=BRIDGES[ACTIVE_BRIDGE]["model_material"]),
        feats[["window_id"]],
        pd.DataFrame({
            "window_start": pd.to_datetime(feats["window_start_us"], unit="us"),
            "window_end": pd.to_datetime(feats["window_end_us"], unit="us"),
        }, index=feats.index),
        feats[["window_start_us", "window_end_us", "load_level_kg", "load_position", "condition", "temperature_c"]],
        # 공개 데이터: 모든 특징·AI② 지표(1-TAC, AR)·판정 보류 열을 그대로 전달 (합의서 v3 8절)
        feats[[c for c in feats.columns if c not in FEATURE_META_COLS + ["split"]]
              + [c for c in ("tac_group", "hold_main") if c in feats]
              if PUBLIC else output_feature_columns(sensors, feats)],
        result,
    ], axis=1).sort_values(["experiment_id", "window_start_us"], ignore_index=True)


def monitoring_summary(final):
    """실험별 요약: 교량·센서 이상 비율 + 가장 많이 나온 (주 이상 센서, 위치, 증상)."""
    summary = final.groupby("experiment_id").agg(
        교량이상=("prediction", lambda s: f"{(s == ANOMALY).mean():.0%}"),
        센서이상=("anomaly_sensors", lambda s: f"{(s != '').mean():.0%}"),
        조건=("condition", "first"),
    )
    flagged = final[final["main_sensor"] != ""].assign(
        symptom_kind=lambda d: d["symptom"].str.replace(r" \(.*\)", "", regex=True))
    top = (flagged.groupby(["experiment_id", "main_sensor", "main_location", "symptom_kind"]).size()
           .rename("n").reset_index().sort_values("n", ascending=False)
           .drop_duplicates("experiment_id").set_index("experiment_id")
           .rename(columns={"main_sensor": "주이상센서", "main_location": "위치", "symptom_kind": "증상"})
           .drop(columns="n"))
    return summary.join(top).fillna("")


def method_text():
    method = "IF 점수"
    if IFOREST["use_z_rule"]:
        method += " + z 보조 규칙"
    if DECISION["use_persistence"]:
        method += f" + 연속 판정({DECISION['persist_k']}/{DECISION['persist_n']})"
    return method


def predict_dataset():
    """features.parquet 전체 (학습에 쓴 데이터 포함 모든 실험) 예측 → predictions.csv"""
    if PUBLIC and PUBLIC["ai1_model"] == "none":
        # 사전 등록서 v4 (LUMO, Vänersborg): AI① 모델 판정 없이 특징·지표·보류만 전달
        feats = pd.read_parquet(FEAT_PATH).sort_values(["experiment_id", "window_start_us"], ignore_index=True)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
        print(f"[9~10] 판정 없음 (특징만): Window {len(feats):,}개 → {OUT_PATH}")
        return
    bundle = load_bundle()
    feats = pd.read_parquet(FEAT_PATH)
    sensors = BRIDGES[ACTIVE_BRIDGE]["sensors"]
    final = build_output(feats, predict(feats, bundle), sensors)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print(f"[9~10] 예측 완료: Window {len(final):,}개 → {OUT_PATH}")
    print(f"  열 {final.shape[1]}개: " + ", ".join(final.columns))
    if PUBLIC:
        # 공개 데이터: 손상 구분 결과는 evaluate.py 에서 정해 둔 규칙대로 본다. 여기서는 판정 개수만
        print(f"  판정 방식: {'z-score 전용 (판정 없음)' if bundle.get('model_type') == 'zscore' else method_text()}")
        print("  판정별 Window 수:", final["prediction"].value_counts().to_dict())
        return
    print(f"  판정 방식: {method_text()}")
    print("\n[실험별 모니터링 요약] (정답은 보지 않은 상태)")
    print("  교량이상 = 교량 전체 판정 Anomaly 비율 / 센서이상 = 센서 하나라도 Anomaly 인 Window 비율")
    print("  주 이상 센서·위치·증상 = 이상 Window 에서 가장 많이 나온 (센서, 증상) 조합")
    print(monitoring_summary(final).to_string())


def judge_raw(raw, experiment_id, load_position="", bundle=None):
    """raw (RAW_COLUMNS 형식 DataFrame) 1개 측정 세션을 판정 → (구간당 1행 결과, 제외한 구간 수). 구간이 없으면 결과는 None.
    파일 판정(predict_new_file)과 수신 API(ai1_server.py)가 같이 쓴다."""
    bundle = bundle or load_bundle()
    sensors = BRIDGES[ACTIVE_BRIDGE]["sensors"]
    max_rate = max(s["sample_rate_hz"] for s in sensors)
    exp = {
        "experiment_id": experiment_id,
        "bridge_type": BRIDGES[ACTIVE_BRIDGE]["bridge_type"],
        "raw": raw,
        "start_us": int(raw["timestamp_us"].min()),
        "end_us": int(raw["timestamp_us"].max() + 1e6 / max_rate),
        "load_position": load_position,
    }
    rows, dropped, _ = process_experiment(exp, sensors, FILTER["enabled"])
    if not rows:
        return None, dropped
    feats = pd.DataFrame(rows)
    return build_output(feats, predict(feats, bundle), sensors), dropped


def predict_new_file(file_path, load_position=""):
    """14. 새로 들어온 센서 파일 1개 (parquet 또는 csv, 데이터 사양 RAW_COLUMNS 형식) 를 바로 판정.

    센서 목록은 config.BRIDGES 에서 읽으므로 실제 HW 파일에도 그대로 쓸 수 있다.
    정답 열(damage_type, severity_level)이 들어 있어도 읽자마자 버린다.
    하중 위치(load_position)는 센서로 측정되지 않으므로 실험 조건으로 함께 알려 줘야 한다 (하중 0 kg 이면 생략 가능).
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise SystemExit(f"파일이 없습니다: {path}")
    if load_position and load_position not in EXPERIMENT["load_positions"]:
        raise SystemExit(f"하중 위치 '{load_position}' 는 config 의 load_positions {list(EXPERIMENT['load_positions'])} 에 없습니다.")
    raw = load_experiment(path)
    experiment_id = str(raw["experiment_id"].iat[0]) if "experiment_id" in raw else path.stem
    final, dropped = judge_raw(raw, experiment_id, load_position)
    if final is None:
        raise SystemExit("사용할 수 있는 Window 가 없습니다 (결측이 너무 많거나 데이터가 2초보다 짧음).")

    out = RESULTS_DIR / ACTIVE_BRIDGE / "new" / f"{path.stem}_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print(f"[14] 새 데이터 판정: {path.name}  (교량 {ACTIVE_BRIDGE}, Window {len(final)}개, 결측 제외 {dropped}개)")
    print(f"  판정 방식: {method_text()}")
    print(monitoring_summary(final).to_string())
    anomalies = final[final["prediction"] == ANOMALY]
    if len(anomalies):
        print("\n[Anomaly 로 판정된 Window (처음 5개)]")
        print(anomalies[["window_id", "window_start", "anomaly_score", "anomaly_sensors", "main_location", "symptom"]]
              .head().to_string(index=False))
    print(f"\n저장: {out}")

    # AI② 전달: 구간마다 raw + AI① 결과 메시지 → 파일 저장, API 주소가 있으면 전송
    messages = build_messages(final, raw, ACTIVE_BRIDGE, load_confidence(ACTIVE_BRIDGE))
    msg_path = out.with_name(f"{path.stem}_ai2_messages.jsonl")
    save_jsonl(messages, msg_path)
    print(f"AI② 메시지 {len(messages)}건 저장: {msg_path}")
    if AI2_API["url"]:
        ok, failed = send(messages)
        print(f"AI② API 전송: 성공 {ok}건 / 실패 {len(failed)}건 → {AI2_API['url']}")
        for wid, err in failed[:5]:
            print(f"  실패 {wid}: {err}")
    else:
        print("AI② API 주소 없음 (config.AI2_API['url'] 또는 환경변수 AI2_API_URL) → 전송 생략")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        position = sys.argv[sys.argv.index("--position") + 1] if "--position" in sys.argv else ""
        predict_new_file(sys.argv[2], position)
    else:
        predict_dataset()
