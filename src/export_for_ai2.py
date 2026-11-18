"""
AI② (위험도 분석) 전달용 폴더 만들기.

전달 방식: 구간(Window)마다 메시지 1건 (raw + AI① 결과) 을 API (HTTP POST, JSON) 로 AI② 에 보낸다.
          메시지 생성·전송은 ai2_message.py, 실시간 판정은 predict.py --file. 이 스크립트는 양식 확인용 폴더를 만든다.

출력: results/<source>/for_AI2/
  README.md                       전달 구조 · 메시지 양식 · 규칙 · AI① 성능
  examples/<bridge>_normal_message.json, <bridge>_anomaly_message.json   ★ 메시지 1건 예시 (정상 / 이상)
  examples/<bridge>_stream_<실험>.jsonl                                  이상 실험 1개 전체 메시지 (전송 순서)
  <bridge>_model_info.json        모델 성능(정확도 등) · 신뢰도 계산 방법 · 필드/특징 설명
  <bridge>_anomaly_list.json      이상 구간 목록 (확인용, raw·특징값 제외)
  sensor_info.json                센서 규격표
  experiment_labels.json          실험별 정답(D·L)·하중 조건·AI② 평가 사용 가능 여부

predict.py · evaluate.py 를 실행한 뒤에 실행한다. (run_pipeline.py 마지막 단계에서 자동 실행)
실행: python src/export_for_ai2.py

공개 데이터 (AI①·AI② 합의서 v3 8절): python src/export_for_ai2.py public
  → results/public/for_AI2/ 에 파일 10개
     <dataset>_anomaly_results_raw.csv / _deg.csv   (ponte_moesa, hbta, old_ada) 원본·HW 규격 열화본 기반 이상탐지 결과
     experiment_labels.csv, column_guide.csv, sensor_info.csv, data_provenance.json
"""
import json
import re
from datetime import date, datetime

import pandas as pd

from ai2_message import anomaly_probability, build_messages, clean, fit_confidence, save_jsonl
from config import (AI2_API, AXIS_KO, BRIDGES, DATA_SOURCE, DECISION, EXPERIMENT, FEATURE_KO, GROUND_TRUTH,
                    MODELS_DIR, PREDICTION_LABELS, RAW_DIR, RESULTS_DIR, SENSOR_TYPE_KO, SPAN_MM, WINDOW)
from preprocess import load_experiment, sensor_short_name
from train import AXIS_SUFFIXES, feature_name

OUT_DIR = RESULTS_DIR / "for_AI2"
NORMAL, ANOMALY = PREDICTION_LABELS
D0 = GROUND_TRUTH["normal_code"]

# 열 사용 구분
KEY, INPUT, COND, REF, SHOW, BAN = "연결 키", "AI② 입력 권장", "실험 조건 (입력 가능)", "참고", "표시용", "입력 금지"


def feature_unit(sensor_unit, feature):
    if feature.startswith("band_"):
        return "비율 (0~1)"
    return {"dom_freq": "Hz", "crest": "무차원", "damping": "무차원 (감쇠비)"}.get(feature, sensor_unit)


def window_text():
    if WINDOW["mode"] == "tap":
        return f"타격 1번 = 구간 1개 ({WINDOW['length_s']:.0f}초, 타격 {WINDOW['tap_pre_s']}초 전부터)"
    return f"{WINDOW['length_s']:.0f}초 구간, {WINDOW['overlap']:.0%} 겹침"


# ---------------------------------------------------------------------------
# sensor_info.json
# ---------------------------------------------------------------------------
def sensor_info():
    rows = []
    for bridge in BRIDGES.values():
        for s in bridge["sensors"]:
            role = "구조 응답 (AI① 모델 입력)" if s["sensor_type"] == "acceleration" else "실험 조건 (AI① 모델 입력 아님, 정상 기준 구분·보정에 사용)"
            rows.append({
                "bridge_type": bridge["bridge_type"],
                "sensor_id": s["sensor_id"],
                "결과파일_약칭": sensor_short_name(s["sensor_id"]),
                "센서종류": SENSOR_TYPE_KO[s["sensor_type"]],
                "sensor_type": s["sensor_type"],
                "모델": s.get("device", ""),
                "설치위치": s["sensor_location"],
                "x_mm (길이방향, 좌측 받침=0)": round(s["x_ratio"] * SPAN_MM),
                "y_mm (폭방향, 중심선=0)": s.get("y_mm", 0),
                "측정축": ",".join(s["axes"]) if s["axes"] != ["NA"] else "-",
                "측정속도_Hz": s["sample_rate_hz"],
                "단위": s["unit"],
                "역할": role,
                "모델 재료": bridge["model_material"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 열·특징 설명 (meta.feature_guide 에 사용)
# ---------------------------------------------------------------------------
FIXED_COLUMNS = {
    "experiment_id": ("실험 ID. ⚠️ 정답(D·L)이 문자열에 들어 있음 (예: BOX_D1_L2_018)", "", KEY + " / " + BAN),
    "bridge_type": ("교량 구조형식 (BOX / TRUSS)", "", REF),
    "model_material": ("축소모형 재료 (ALUMINUM)", "", REF),
    "window_id": (f"구간 번호 (W001~). {window_text()}", "", KEY),
    "window_start": ("구간 시작 시각", "", KEY),
    "window_end": ("구간 끝 시각", "", REF),
    "window_start_us": ("구간 시작 시각 (1970-01-01 기준 마이크로초)", "µs", KEY),
    "window_end_us": ("구간 끝 시각 (마이크로초)", "µs", REF),
    "load_level_kg": ("로드셀 값에 가장 가까운 하중 단계", "kg", COND),
    "load_position": ("하중 위치 코드 (" + ", ".join(f"{k}={v['name']}" for k, v in EXPERIMENT["load_positions"].items())
                      + "). 하중 0 kg 은 빈칸", "", COND),
    "condition": ("하중 조건 = 하중 크기 × 위치 (예: 1kg_E). AI① 은 같은 조건의 정상 상태와 비교", "", COND),
    "temperature_c": ("주변 공기 온도 (구간 중앙 시각 값)", "°C", COND),
    "anomaly_score": ("교량 전체 Isolation Forest 점수. 0~1, 클수록 이상 (정상 ≈ 0.42~0.45)", "0~1", REF),
    "threshold": ("anomaly_score 판정 기준 (교량마다 다름)", "0~1", REF),
    "max_abs_z": ("같은 하중 조건의 정상 상태에서 가장 크게 벗어난 정도 (표준편차 몇 배)", "σ", INPUT),
    "z_threshold": ("max_abs_z 판정 기준", "σ", REF),
    "decision_ratio": ("구간 1개 기준 이상 정도 = max(점수/기준, |z|/z기준). 1 을 넘으면 이상. 교량이 달라도 비교 가능", "배", INPUT),
    "raw_prediction": ("구간 1개 기준 판정 (Normal / Anomaly)", "", REF),
    "persist_ratio": (f"최근 {DECISION['persist_n']}개 구간 기준 이상 정도. 1 을 넘으면 이상", "배", REF),
    "prediction": (f"★ 최종 판정 (Normal / Anomaly). 최근 {DECISION['persist_n']}개 구간 중 {DECISION['persist_k']}개 이상이 이상이면 Anomaly", "", INPUT),
    "top1_feature": ("정상에서 가장 많이 벗어난 특징 이름 (앞부분 = 센서 약칭, 끝 = 축)", "", REF),
    "top1_z": ("top1_feature 가 벗어난 정도 (+: 정상보다 큼, -: 작음)", "σ", INPUT),
    "top2_feature": ("두 번째로 많이 벗어난 특징 이름", "", REF),
    "top2_z": ("top2_feature 가 벗어난 정도", "σ", REF),
    "anomaly_sensors": ("Anomaly 로 판정된 센서 ID 목록 (; 구분, 없으면 빈칸)", "", INPUT),
    "main_sensor": ("가장 크게 벗어난 센서 ID", "", SHOW),
    "main_location": ("main_sensor 의 설치 위치와 x 좌표 (센서가 교량당 2개라 위치는 참고 수준)", "", SHOW),
    "symptom": ("무엇이 평소와 다른지 (예: 가속도 수직(Z) 감쇠비 증가 (+6.2σ)). ⚠️ 증상이지 원인(D1~D3) 판정이 아님", "", SHOW),
}


def column_guide(columns, sensors):
    by_prefix = {sensor_short_name(s["sensor_id"]): s for s in sensors}
    rows = []
    for col in columns:
        if col in FIXED_COLUMNS:
            desc, unit, use = FIXED_COLUMNS[col]
            rows.append({"열 이름": col, "설명": desc, "단위": unit, "관련 센서 ID": "", "AI② 사용 구분": use})
            continue
        if col.startswith("cross_"):
            name = col.removeprefix("cross_")
            rows.append({"열 이름": col, "설명": f"[센서 간 비교] {FEATURE_KO.get(name, name)}. 두 신호 RMS 의 비",
                         "단위": "비", "관련 센서 ID": "", "AI② 사용 구분": INPUT})
            continue
        prefix, rest = col.split("_", 1)
        s = by_prefix[prefix]
        kind = SENSOR_TYPE_KO[s["sensor_type"]]
        where = f"{s['sensor_location']} (x={s['x_ratio'] * SPAN_MM:.0f}mm)"
        if rest == "score":
            rows.append({"열 이름": col, "설명": f"[센서별 점수] {kind} 센서 {where}의 Isolation Forest 점수. 0~1, 클수록 이상",
                         "단위": "0~1", "관련 센서 ID": s["sensor_id"], "AI② 사용 구분": INPUT})
        elif rest == "state":
            rows.append({"열 이름": col, "설명": f"[센서별 판정] {kind} 센서 {where}의 Normal / Anomaly (연속 판정 적용)",
                         "단위": "", "관련 센서 ID": s["sensor_id"], "AI② 사용 구분": INPUT})
        else:
            feat = feature_name(col)
            axis = col.rsplit("_", 1)[-1]
            axis_text = f" {AXIS_KO[axis.upper()]}" if s["sensor_type"] == "acceleration" and axis in AXIS_SUFFIXES else ""
            note = " (평균을 뺀 흔들림 성분)" if s["sensor_type"] == "acceleration" and feat in ("rms", "max", "min", "p2p") else ""
            use = INPUT if s["sensor_type"] == "acceleration" else COND
            rows.append({"열 이름": col, "설명": f"[특징] {where} {kind}{axis_text}의 {FEATURE_KO.get(feat, feat)}{note}, 구간 값",
                         "단위": feature_unit(s["unit"], feat), "관련 센서 ID": s["sensor_id"], "AI② 사용 구분": use})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# experiment_labels.json
# ---------------------------------------------------------------------------
def experiment_labels(key):
    exps = pd.read_csv(RAW_DIR / key / "experiments.csv", keep_default_na=False, na_values=[""])
    split = pd.read_csv(MODELS_DIR / key / "split.csv", usecols=["experiment_id", "split"])
    df = exps.merge(split.rename(columns={"split": "ai1_split"}), on="experiment_id", how="left")
    df["AI②_평가에_사용"] = df["ai1_split"].map(lambda s: "가능" if s == "test" else "금지 (AI① 학습·기준 설정에 사용된 실험)")
    return df[["experiment_id", "bridge_type", "damage_type", "severity_level", "damage_meaning", "damage_name",
               "condition", "applied_load_kg", "load_position", "ai1_split", "AI②_평가에_사용"]]


# ---------------------------------------------------------------------------
# <bridge>_model_info.json / <bridge>_anomaly_list.json / examples/
# ---------------------------------------------------------------------------
def model_performance(key, test):
    """AI① test 실험 기준 성능 (evaluate.py 결과) — 최종 판정(연속 판정) 행"""
    final = pd.read_csv(RESULTS_DIR / key / "evaluation_summary.csv").iloc[0]
    by_state = pd.read_csv(RESULTS_DIR / key / "evaluation_by_state.csv")
    recall_col = by_state.columns[2]
    return {
        "accuracy": round(final["Accuracy"], 4),
        "precision": round(final["Precision"], 4),
        "recall": round(final["Recall"], 4),
        "f1": round(final["F1"], 4),
        "false_alarm_rate_d0": round(final["오탐률(D0)"], 4),
        "auc": round(final["AUC(점수)"], 4),
        "confusion_matrix": {k: int(final[k]) for k in ("TN", "FP", "FN", "TP")},
        "recall_by_state": {f"{r['damage_type']}_{r['severity_level']}": round(r[recall_col], 4)
                            for _, r in by_state.iterrows() if r["damage_type"] != D0},
        "evaluated_on": f"AI① test 실험 {test['experiment_id'].nunique()}개 · 구간 {len(test):,}개 ({DATA_SOURCE} 데이터)",
    }


FIELD_GUIDE = {
    "raw": "이전 구간 끝 ~ 이 구간 끝의 센서 원본값. 메시지를 순서대로 이으면 raw 전체. 센서별 t0_us(첫 샘플 시각) + sample_rate_hz 로 각 샘플 시각 계산",
    "raw.sensors.<ID>.axes": "축별 값 배열 (3축 가속도 X/Y/Z, 1축 센서 NA). 결측은 null",
    "raw.sensors.<ID>.quality": "OK 가 아닌 샘플 위치 (축별 → 플래그별 index 목록). SAT 포화, DROP 누락, CAL 보정 중",
    "ai1.prediction": "★ 최종 판정 (Normal / Anomaly)",
    "ai1.warming_up": "true = 측정 시작 직후라 연속 판정에 필요한 구간이 아직 안 모임 → prediction 은 Normal 로 나가지만 판정 보류, confidence 는 null",
    "ai1.confidence": "★ 이 판정이 맞을 확률 (0~1). Anomaly → P(이상), Normal → 1 - P(이상). warming_up 이면 null",
    "ai1.anomaly_probability": "실제 이상일 확률 (0~1)",
    "ai1.decision_ratio": "구간 1개 기준 이상 정도. 1 초과 = 이상. 교량이 달라도 비교 가능",
    "ai1.persist_ratio": f"최근 {DECISION['persist_n']}개 구간 기준 이상 정도. 1 초과 → 최종 Anomaly (실험 시작 직후는 0)",
    "ai1.raw_prediction": "구간 1개 기준 판정 (연속 판정 전)",
    "ai1.anomaly_score / threshold": "교량 전체 Isolation Forest 점수 (0~1, 클수록 이상) / 판정 기준",
    "ai1.max_abs_z / z_threshold": "같은 하중 조건의 정상 대비 최대 벗어남 (σ) / 판정 기준",
    "ai1.condition": "실험 조건: 하중 크기×위치, 하중 kg, 위치 코드, 온도 °C",
    "ai1.anomaly_sensors": "Anomaly 로 판정된 센서 ID 목록",
    "ai1.main_sensor / main_location": "가장 크게 벗어난 센서와 위치 (센서 2개라 위치는 참고 수준)",
    "ai1.symptom": "무엇이 평소와 다른지. ⚠️ 증상이지 원인(D1~D3) 판정이 아님",
    "ai1.top_features": "정상에서 가장 많이 벗어난 특징 2개와 z (σ, +: 정상보다 큼)",
    "ai1.sensors": "센서별 Isolation Forest 점수와 판정 (sensor = 약칭, sensor_short_names 참고)",
    "ai1.features": "구간 특징값 (설명은 feature_guide). XGBoost 입력으로 바로 사용 가능",
}


def model_info(key, pred, calib, test):
    bridge = BRIDGES[key]
    guide = column_guide(pred.columns, bridge["sensors"])
    guide = guide[guide["설명"].str.startswith(("[특징]", "[센서 간 비교]"))]
    return {
        "schema_version": AI2_API["schema_version"],
        "bridge_type": bridge["bridge_type"],
        "model_material": bridge["model_material"],
        "data_source": DATA_SOURCE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ai1_method": "Isolation Forest + 같은 하중 조건 정상 대비 z-score",
        "window": window_text(),
        "decision_rule": f"최근 {DECISION['persist_n']}개 구간 중 {DECISION['persist_k']}개 이상이 이상이면 Anomaly",
        "n_windows_total": len(pred),
        "n_anomalies": int((pred["prediction"] == ANOMALY).sum()),
        "model_performance": model_performance(key, test),
        "confidence_method": f"로지스틱 회귀 P(실제 이상 | {calib['ratio_column']}), 정상·이상 같은 무게, AI① test 실험 {calib['n_windows']:,}개 구간으로 보정",
        "field_guide": FIELD_GUIDE,
        "feature_guide": {r["열 이름"]: f"{r['설명']} [{r['단위']}]" for _, r in guide.iterrows()},
        "sensor_short_names": {sensor_short_name(s["sensor_id"]): s["sensor_id"] for s in bridge["sensors"]},
    }


def anomaly_list(pred, calib):
    """확인용: 이상 구간만, raw·특징값 없이 핵심만"""
    a = pred[pred["prediction"] == ANOMALY]
    return [{
        "experiment_id": r["experiment_id"], "window_id": r["window_id"], "window_start": r["window_start"],
        "confidence": round(anomaly_probability(r[calib["ratio_column"]], calib), 4),
        "decision_ratio": clean(r["decision_ratio"]),
        "anomaly_sensors": [] if pd.isna(r["anomaly_sensors"]) else r["anomaly_sensors"].split(";"),
        "main_sensor": clean(r["main_sensor"]), "main_location": clean(r["main_location"]), "symptom": clean(r["symptom"]),
    } for _, r in a.iterrows()]


def example_experiments(test):
    """예시용 test 실험: 정상(D0) 1개, 이상 1개 (최종 Anomaly 구간이 가장 많은 실험)"""
    normal = test.loc[test["damage_type"] == D0, "experiment_id"].iat[0]
    anomalous = test[test["prediction"] == ANOMALY].groupby("experiment_id").size().idxmax()
    return normal, anomalous


def write_examples(key, pred, calib, test):
    """examples/: 정상·이상 메시지 1건씩 (보기 쉽게 들여쓰기) + 이상 실험 1개 전체 스트림 (jsonl, 실제 전송 순서)"""
    name = BRIDGES[key]["bridge_type"].lower()
    out = OUT_DIR / "examples"
    normal_exp, anomaly_exp = example_experiments(test)
    picked = {}
    for exp_id in (normal_exp, anomaly_exp):
        file = pd.read_csv(RAW_DIR / key / "experiments.csv").set_index("experiment_id").at[exp_id, "file"]
        msgs = build_messages(pred[pred["experiment_id"] == exp_id], load_experiment(RAW_DIR / key / file), key, calib)
        picked[exp_id] = msgs
    normal_msg = picked[normal_exp][len(picked[normal_exp]) // 2]
    anomaly_msg = max((m for m in picked[anomaly_exp] if m["ai1"]["prediction"] == ANOMALY),
                      key=lambda m: m["ai1"]["confidence"])
    write_json(normal_msg, out / f"{name}_normal_message.json")
    write_json(anomaly_msg, out / f"{name}_anomaly_message.json")
    save_jsonl(picked[anomaly_exp], out / f"{name}_stream_{anomaly_exp}.jsonl")
    return {"normal": f"{name}_normal_message.json", "anomaly": f"{name}_anomaly_message.json",
            "stream": f"{name}_stream_{anomaly_exp}.jsonl", "stream_n": len(picked[anomaly_exp])}


def write_json(obj, path):
    """들여쓰기 JSON. 숫자·문자열만 든 배열(raw 값 등)은 한 줄로 접어 읽기 쉽게."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    text = re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------
def performance_table(metas):
    lines = ["| 교량 | Accuracy | Precision | Recall | F1 | 정상(D0) 오탐률 | AUC |", "|---|---|---|---|---|---|---|"]
    for m in metas:
        p = m["model_performance"]
        lines.append(f"| {m['bridge_type']} | {p['accuracy']:.1%} | {p['precision']:.1%} | {p['recall']:.1%} | "
                     f"{p['f1']:.1%} | {p['false_alarm_rate_d0']:.1%} | {p['auc']:.3f} |")
    return "\n".join(lines)


def detection_table(metas, lists):
    lines = ["| 교량 | 전체 구간 | 이상 구간 | 이상 실험 수 | 신뢰도 ≥0.9 | 0.7~0.9 | <0.7 |", "|---|---|---|---|---|---|---|"]
    for m, a in zip(metas, lists):
        c = pd.Series([r["confidence"] for r in a], dtype=float)
        n_exp = len({r["experiment_id"] for r in a})
        lines.append(f"| {m['bridge_type']} | {m['n_windows_total']:,} | {m['n_anomalies']:,} | {n_exp} | "
                     f"{(c >= 0.9).sum():,} | {((c >= 0.7) & (c < 0.9)).sum():,} | {(c < 0.7).sum():,} |")
    return "\n".join(lines)


def ground_truth_table():
    lines = ["| 코드 | 뜻 (두 교량 공통) | " + " | ".join(BRIDGES[k]["bridge_type"] for k in BRIDGES) + " |",
             "|---|---|" + "---|" * len(BRIDGES),
             "| D0 | 정상 | " + " | ".join("정상" for _ in BRIDGES) + " |"]
    for code in GROUND_TRUTH["damage_codes"]:
        names = [BRIDGES[k].get("damage_types", {}).get(code, {}).get("name", "") for k in BRIDGES]
        lines.append(f"| {code} | {GROUND_TRUTH['damage_meaning'][code]} | " + " | ".join(names) + " |")
    return "\n".join(lines)


def readme(metas, lists, examples):
    names = [m["bridge_type"].lower() for m in metas]
    example_rows = "\n".join(
        f"| `examples/{e['normal']}` · `examples/{e['anomaly']}` | {m['bridge_type']} 메시지 1건 예시 (정상 / 이상). **양식은 이것만 보면 됨** |\n"
        f"| `examples/{e['stream']}` | {m['bridge_type']} 이상 실험 1개를 실제 전송 순서대로 ({e['stream_n']}건, 한 줄 = 메시지 1건) |"
        for m, e in zip(metas, examples))
    info_rows = "\n".join(
        f"| `{n}_model_info.json` | {m['bridge_type']} 모델 성능(정확도 등)·신뢰도 계산 방법·필드 설명·특징 설명 |\n"
        f"| `{n}_anomaly_list.json` | {m['bridge_type']} 이상 구간 목록 ({m['n_anomalies']:,}건, raw·특징값 없이 핵심만. 확인용) |"
        for n, m in zip(names, metas))
    return f"""# AI① → AI② 전달 (API · JSON)

작성: AI① 이상징후 탐지 담당 / 갱신일: {date.today()} / 이 폴더는 `src/export_for_ai2.py` 가 자동으로 만듭니다.

## 1. 전달 구조

```
HW(ESP32) ─raw→ AI① 이상탐지 ─[구간마다 메시지 1건: raw + AI① 결과]→ AI② (XGBoost 분류·위험도) → BE(Django) → FE
```

- **방식**: HTTP `POST` (JSON), 구간(Window) 1개마다 메시지 1건. **정상 구간도 보냅니다.**
- **주소**: AI② 가 정해 주면 `src/config.py` 의 `AI2_API["url"]` (또는 환경변수 `AI2_API_URL`) 에 넣습니다.
- **스키마 버전**: `schema_version = "{AI2_API['schema_version']}"`. 필드가 바뀌면 버전을 올립니다.
- AI② 는 받은 메시지에 `ai2` 블록(구조 상태 분류·위험도)을 붙여 BE 로 넘기면 됩니다.

## 2. 메시지 1건의 모양

```
{{
  "schema_version": "{AI2_API['schema_version']}",
  "message_type": "ai1_window",
  "bridge_type": "BOX",
  "experiment_id": "...", "window_id": "W013",
  "window_start": "...", "window_start_us": ..., "window_end_us": ...,

  "raw": {{                                   ← HW 원본값 (정답 열 제거)
    "start_us": ..., "end_us": ...,
    "sensors": {{
      "BOX_ACC_01": {{ "sample_rate_hz": 400, "unit": "m/s^2", "t0_us": ..., "n": ...,
                      "axes": {{ "X": [...], "Y": [...], "Z": [...] }}, "quality": {{...}} }},
      "LOAD_01":    {{ ..., "axes": {{ "NA": [...] }} }},
      ...
    }}
  }},

  "ai1": {{                                   ← AI① 결과
    "prediction": "Normal | Anomaly",
    "warming_up": false,                     ← true 면 측정 시작 직후라 판정 보류 (confidence = null)
    "confidence": 0~1,                       ← 판정이 맞을 확률
    "anomaly_probability": 0~1,
    "decision_ratio", "persist_ratio", "raw_prediction", "anomaly_score", "threshold", "max_abs_z", "z_threshold",
    "condition": {{ "condition", "load_level_kg", "load_position", "temperature_c" }},
    "anomaly_sensors": [...], "main_sensor", "main_location", "symptom",
    "top_features": [...], "sensors": [...],
    "features": {{ "acc01_rms_x": ..., ... }}  ← XGBoost 입력으로 바로 사용 가능
  }}
}}
```

- **raw 범위**: 이전 구간 끝 ~ 이 구간 끝. 메시지를 순서대로 이으면 raw 전체가 빠짐·겹침 없이 복원됩니다.
  (구간 = 타격 1번 2초. raw 는 그 사이 시간까지 포함)
- 각 샘플 시각 = `t0_us + index × 1,000,000 / sample_rate_hz`. 간격이 고르지 않으면 `timestamp_us` 배열이 추가됩니다.
- 각 필드 뜻은 `<교량>_model_info.json` 의 `field_guide`, 특징값 뜻·단위는 `feature_guide`.

## 3. 파일 안내

| 파일 | 내용 |
|---|---|
| `API_SPEC.md` | ★ **AI② API 명세서** (호출 방향·엔드포인트·전송 주기·필드 정의·22개 진동 Feature·FastAPI 예시) |
{example_rows}
{info_rows}
| `sensor_info.json` | 센서 규격표 (센서 ID ↔ 약칭, 종류, 모델, 위치 mm, 측정 축·속도, 단위, 역할) |
| `experiment_labels.json` | 실험별 정답(D·L)·하중 조건·**AI② 평가에 써도 되는지** (XGBoost 학습 라벨) |

## 4. 한눈에 보기

### 탐지 요약
{detection_table(metas, lists)}

### AI① 모델 성능 (AI① test 실험 기준)
{performance_table(metas)}

## 5. 꼭 지킬 규칙 (데이터 누수 방지)

1. 정답과는 **`experiment_id` + `window_id`** 로 연결합니다.
2. **`experiment_labels.json` 에서 `AI②_평가에_사용 = 금지` 인 실험은 XGBoost 평가에 쓰지 마세요.** (AI① 학습에 쓴 실험)
3. 학습/평가 분할은 **실험 단위**로 합니다. 구간을 무작위로 섞으면 누수가 생깁니다.
4. `experiment_id` 문자열에는 정답이 들어 있으니 (예: `BOX_D1_L2_018`) **모델 입력으로 쓰지 마세요.**
   실제 운영에서는 실험 ID 대신 장치 ID·시각을 키로 쓸 예정입니다.
5. `confidence` 는 AI① test 실험으로 보정한 값이라 같은 실험으로 평가하면 약간 낙관적일 수 있습니다.

## 6. 이상상태 코드 (HW 합의 2026-10-12, 두 교량 공통 의미)

{ground_truth_table()}

L1~L3 = 같은 유형 안의 상대 심각도 (위험도 단계 아님). 두 교량 모두 **알루미늄 축소모형**입니다.

## 7. 같이 정해야 할 것

1. AI② API 주소·인증 방식, 응답 코드 (성공 2xx 외 재전송 규칙)
2. 실제 운영 시 키: `experiment_id` 대신 `device_id` + 시각
3. `features` 를 계속 넣을지 (XGBoost 가 raw 에서 직접 특징을 만든다면 뺄 수 있음)
"""


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    keys = [k for k in BRIDGES if (RESULTS_DIR / k / "predictions.csv").exists()]
    if not keys:
        raise SystemExit("predictions.csv 가 없습니다. predict.py 와 evaluate.py 를 먼저 실행하세요.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in [*OUT_DIR.glob("*.csv"), *OUT_DIR.glob("*.json"), *OUT_DIR.glob("examples/*")]:  # 이전 버전 파일 정리
        old.unlink()

    metas, lists, examples, labels = [], [], [], []
    for key in keys:
        pred = pd.read_csv(RESULTS_DIR / key / "predictions.csv")
        calib, test = fit_confidence(key, pred)
        name = BRIDGES[key]["bridge_type"].lower()
        meta = model_info(key, pred, calib, test)
        lst = anomaly_list(pred, calib)
        write_json(meta, OUT_DIR / f"{name}_model_info.json")
        write_json(lst, OUT_DIR / f"{name}_anomaly_list.json")
        examples.append(write_examples(key, pred, calib, test))
        metas.append(meta)
        lists.append(lst)
        labels.append(experiment_labels(key))

    write_json(sensor_info().to_dict(orient="records"), OUT_DIR / "sensor_info.json")
    lab = pd.concat(labels).astype(object)
    write_json(lab.where(lab.notna(), None).to_dict(orient="records"), OUT_DIR / "experiment_labels.json")
    (OUT_DIR / "README.md").write_text(readme(metas, lists, examples), encoding="utf-8")

    print(f"AI② 전달용 폴더 → {OUT_DIR}")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(OUT_DIR)):<48} {f.stat().st_size / 1024:>8,.0f} KB")


# ---------------------------------------------------------------------------
# 공개 데이터 전달 파일 (합의서 v3 8절)
# ---------------------------------------------------------------------------
PUBLIC_DATASETS = ("ponte_moesa", "hbta", "old_ada")
PUBLIC_OUT = RESULTS_DIR.parent / "public" / "for_AI2"
MODEL_VERSION = "AI1-public-1.1"   # 1.1: AI① v2 결과 열 추가 (v1 열 유지)
V2_RESULT_COLS = ["anomaly_score", "threshold", "max_abs_z", "z_threshold", "decision_ratio", "raw_prediction",
                  "persist_ratio", "prediction", "top1_feature", "top1_z", "top2_feature", "top2_z"]

STAGE_TEXT = {
    "ponte_moesa": {
        "DS2": "I·II·III 철근 노출 (기준 상태)", "DS3": "II·III 철근 절단 + IV 천공",
        "DS5": "II 긴장재 절반 절단", "DS6": "II 긴장재 전체 절단 (참고, 등급 판정에 쓰지 않음)",
        "DS7": "DS6 + IV 천공 (누적)",
    },
    "hbta": {
        "UDS": "손상 없음", "DS1": "세로보-가로보 연결부 1곳", "DS2": "세로보-가로보 연결부 여러 곳",
        "DS3": "세로보 가로재 1곳", "DS4": "세로보 가로재 여러 곳", "DS5": "수평 가새 연결부 1곳",
        "DS6": "수평 가새 연결부 1곳", "DS7": "수평 가새 연결부 여러 곳", "DS8": "가로보-주부재 연결부 1곳",
    },
    "old_ada": {
        "INT": "정상", "DMG1": "T1(경간 중앙 수직재) 절반 절단", "DMG2": "T1 완전 절단",
        "RCV": "T1 보수 후 (회복 불완전, 오경보 계산에서 제외)", "DMG3": "T2(5/8 지점 수직재) 완전 절단 + T1 보수 후 잔여 손상 (다중 손상)",
    },
}
HW_POSITION = {"acc01": "HW L/4 가속도계에 대응", "acc02": "HW L/2 가속도계에 대응", "acc03": "세션 차이 확인용 (HW 에 없음)"}
LABEL_COLS = ["damage_stage", "stage_description", "label_status", "ai1_split", "cv_group", "subset", "source_record",
              "condition", "setup", "temperature_c", "train_passing", "wind_range", "speed_kmh"]


def public_paths(name, source):
    return (RESULTS_DIR.parent / source / name / "predictions.csv",
            RESULTS_DIR.parents[1] / "data" / "raw" / source / name)


def public_experiments(name):
    """실험 기록 (원본·열화본 동일). 파일 경로·내부 ID 는 빼고 정답·분할·조건만."""
    exps = pd.read_csv(public_paths(name, "public_raw")[1] / "experiments.csv", keep_default_na=False)
    exps = exps.rename(columns={"split": "ai1_split"})
    exps["stage_description"] = exps["damage_stage"].map(STAGE_TEXT[name]).fillna("")
    exps.insert(0, "dataset", name)
    return exps


def public_results(name, source, exps):
    """<dataset>_anomaly_results_<raw|deg>.csv: AI① 이상탐지 결과 + 특징 + AI② 지표 + 판정 보류 + 라벨·조건"""
    pred = pd.read_csv(public_paths(name, source)[0], keep_default_na=False, na_values=[""])
    tag = "raw" if source == "public_raw" else "deg"
    for c in [c for c in pred.columns if c.count("_") >= 2 and c.split("_")[1] == "rms" and c.startswith("acc")]:
        p, _, axis = c.split("_")
        pred[f"{tag}_rms_{p}_{axis}"] = pred[c]          # 판정 보류 판단 값 (2~25Hz RMS)
    pred = pred.copy()   # 열을 많이 붙인 뒤 표 정리 (pandas 성능 경고 방지)
    v2 = public_paths(name, source)[0].with_name("predictions_v2.csv")
    if v2.exists():   # AI① v2 결과 (참고): 같은 키로 붙이고 열 이름 앞에 v2_
        r2 = pd.read_csv(v2, keep_default_na=False, na_values=[""])
        r2 = r2[["experiment_id", "window_start_us", *[c for c in V2_RESULT_COLS if c in r2]]]
        pred = pred.merge(r2.rename(columns={c: f"v2_{c}" for c in V2_RESULT_COLS}), on=["experiment_id", "window_start_us"], how="left")
    keep = ["experiment_id", *[c for c in LABEL_COLS if c in exps.columns]]
    out = pred.drop(columns=[c for c in ("condition", "temperature_c", "load_level_kg", "load_position") if c in pred])
    out = out.merge(exps[keep], on="experiment_id", how="left")
    out.insert(0, "dataset", name)
    out["model_version"] = MODEL_VERSION
    front = ["dataset", "experiment_id", "window_id", "window_start_us", "window_end_us", *[c for c in LABEL_COLS if c in out]]
    return out[front + [c for c in out.columns if c not in front]]


def public_column_guide(frames):
    """모든 결과 파일의 열 설명. 1-TAC·AR 은 AI② 지표이며 AI① 이상탐지 모델 입력이 아님을 표시 (AI② 요청)"""
    fixed = {
        "dataset": ("데이터셋 (ponte_moesa / hbta / old_ada)", "", "연결 키"),
        "experiment_id": ("실험 ID (중립 ID, 정답 없음). 정상 측정은 시간 블록마다 따로", "", "연결 키"),
        "window_id": ("실험 내 Window 순번", "", "참고"), "window_start_us": ("Window 시작 시각 µs. 키 = experiment_id + window_start_us", "µs", "연결 키"),
        "window_end_us": ("Window 끝 시각 µs", "µs", "참고"),
        "window_start": ("Window 시작 시각 (window_start_us 를 시간대 변환 없이 표시)", "", "참고"),
        "window_end": ("Window 끝 시각", "", "참고"),
        "damage_stage": ("손상 단계 라벨 (정답)", "", "라벨, 모델 입력 금지"),
        "stage_description": ("손상 단계 설명", "", "참고"),
        "label_status": ("확실 / 참고 / 속도 영향 확인용 (참고, Old ADA 30·50km/h 정상 주행) / 참고 (회복 불완전). 주 평가는 '확실'만", "", "라벨"),
        "ai1_split": ("AI① 분할: train / val / ai2_calib(AI② T1 보정) / test / session(세션 차이 확인용)", "", "분할"),
        "cv_group": ("손상 상태 교차 검증 묶음 A / B (Ponte Moesa)", "", "분할"),
        "subset": ("보고용 구분: 작업전 / 작업중 (DS2 test), 저녁 / 낮 (세션)", "", "분할"),
        "source_record": ("원본 측정 이름. 정답이 들어 있음", "", "모델 입력 금지"),
        "condition": ("가진 조건 (예: NM_P2_Z, AMB_S5, VBA_40)", "", "조건"), "setup": ("Ponte Moesa 센서 구성 번호", "", "조건"),
        "temperature_c": ("측정 온도 (HBTA 측정 단위 값)", "°C", "조건"), "train_passing": ("HBTA 측정 중 열차 통과 기록", "", "조건"),
        "wind_range": ("HBTA 바람 등급", "", "조건"), "speed_kmh": ("Old ADA 차량 속도 (광센서 기준)", "km/h", "조건"),
        "model_material": ("교량 재료", "", "참고"), "bridge_type": ("교량 형식 BOX / TRUSS", "", "참고"),
        "model_version": ("AI① 모델·전처리 버전", "", "참고"),
        "tac_group": ("HBTA 1-TAC 묶음 (기본 Window 3개 = 30.72초가 같은 값)", "", "참고"),
        "hold_main": ("AI① 판정 보류 (열화본에서 주 지표 쌍이 잡음 수준)", "", "판정 보류"),
        "anomaly_score": ("AI① Isolation Forest 점수 0~1, 클수록 이상. ai1_split = train 행은 사용 금지", "0~1", "AI① 결과"),
        "threshold": ("anomaly_score 판정 기준 (AI① val)", "0~1", "AI① 결과"),
        "max_abs_z": ("정상 기준(AI① train) 대비 가장 크게 벗어난 특징의 |z|", "σ", "AI① 결과"),
        "z_threshold": ("max_abs_z 판정 기준", "σ", "AI① 결과"),
        "decision_ratio": ("Window 1개 기준 이상 정도 max(점수/기준, |z|/z기준). 1 초과 = 이상", "배", "AI① 결과"),
        "raw_prediction": ("Window 1개 기준 판정", "", "AI① 결과"), "persist_ratio": ("최근 3개 중 2개 기준 이상 정도", "배", "AI① 결과"),
        "prediction": ("AI① 최종 판정: Normal / Anomaly / 판정 보류 / 판정 불가 / 판정 없음 (z-score 전용, Old ADA)", "", "AI① 결과"),
        "anomaly_sensors": ("Anomaly 로 판정된 센서", "", "AI① 결과"), "main_sensor": ("가장 크게 벗어난 센서", "", "AI① 결과"),
        "main_location": ("main_sensor 위치", "", "AI① 결과"), "symptom": ("무엇이 평소와 다른지 (증상, 원인 판정 아님)", "", "AI① 결과"),
    }
    unit = {"rms": "m/s^2", "std": "m/s^2", "max": "m/s^2", "min": "m/s^2", "p2p": "m/s^2", "mean": "m/s^2",
            "dom_freq": "Hz", "crest": "무차원", "damping": "무차원 (감쇠비)"}
    rows = []
    for col in dict.fromkeys(c for f in frames.values() for c in f.columns):
        present = ",".join(n for n, f in frames.items() if col in f.columns)
        if col in fixed:
            desc, u, use = fixed[col]
        elif col.startswith("top") and ("_feature" in col or col.endswith("_z")):
            desc, u, use = "정상에서 가장 많이 벗어난 특징 이름 / z", "σ", "AI① 결과"
        elif col.startswith("v2_") and col.removeprefix("v2_") in V2_RESULT_COLS:
            base = fixed.get(col.removeprefix("v2_"), ("", "", ""))
            desc, u, use = f"[AI① v2, 참고] {base[0]}", base[1], "AI① v2 결과 (결과 열람 후 변경한 방법, 참고)"
        elif re.match(r"v2_(f|shape|e)\d", col):
            k = re.findall(r"\d", col)[1] if col.startswith("v2_e") else col[-1]
            kind = ("모드 {k} 피크 진동수 (모드 대역 ±15%, 포물선 보간)" if col.startswith("v2_f") else
                    "모드 {k} 형상 비 √(acc01 대역 에너지 / acc02 대역 에너지)" if col.startswith("v2_shape") else
                    "모드 {k} 대역 에너지 / 2–25Hz 전체 에너지 (" + col.rsplit("_", 1)[1] + ")").format(k=k)
            desc = f"[AI① v2 Feature] {kind}. 모드 대역은 원본 AI① train 평균 스펙트럼으로 결정. AI② 지표(1-TAC, AR)와 두 센서 응답 비·스펙트럼 정보를 일부 공유"
            u = "Hz" if col.startswith("v2_f") else "비" if col.startswith("v2_shape") else "비율 (0~1)"
            use = "Feature (AI① v2 모델 입력은 v2_selection 의 선택 Feature 만)"
        elif col.startswith("tac_"):
            desc, u, use = f"1-TAC (쌍 {col[4:-2]}, 축 {col[-1].upper()}). 기준 전달함수 = AI① train Window. AI① v2 Feature 와 두 센서 응답 비·스펙트럼 정보를 일부 공유 (독립 근거 아님)", "0~1", "AI② 등급 지표 (AI① 모델 입력 아님)"
        elif re.match(r"acc\d+_ar\d+_[xyz]$", col):
            desc, u, use = "AR(5) 계수 (평균 제거, 절편 없음, 최소제곱). AI① v2 Feature 와 두 센서 응답 비·스펙트럼 정보를 일부 공유 (독립 근거 아님)", "무차원", "AI② 등급 지표 (AI① 모델 입력 아님)"
        elif col.startswith(("hold_flag_", "raw_hold_flag_")):
            desc = f"센서 쌍 {col.split('flag_')[1]} 판정 보류 (주 지표 축). " + ("원본 센서 잡음 기준" if col.startswith("raw") else "LIS2DW12 잡음 기준")
            u, use = "", "판정 보류"
        elif col.startswith("hold_"):
            desc, u, use = "채널별 판정 보류 (2~25Hz RMS < 잡음 RMS × √2)", "", "판정 보류"
        elif col.startswith(("raw_rms_", "deg_rms_")):
            desc, u, use = "판정 보류 판단 값: 채널별 2~25Hz RMS", "m/s^2", "판정 보류"
        elif col.endswith(("_score", "_state")):
            desc, u, use = "센서별 Isolation Forest 점수 / 판정", "", "AI① 결과"
        elif col.startswith("cross_"):
            desc, u, use = f"센서 간 비교 {FEATURE_KO.get(col[6:], col[6:])} (RMS 비)", "비", "Feature"
        elif re.match(r"acc\d+_", col):
            p, feat = col.split("_", 1)[0], feature_name(col)
            desc = f"{p} ({HW_POSITION.get(p, '')}) {AXIS_KO.get(col[-1].upper(), '')} {FEATURE_KO.get(feat, feat)}, 2~25Hz 필터 후"
            u = "비율 (0~1)" if feat.startswith("band_") else unit.get(feat, "")
            use = "Feature (AI① 모델은 주 지표 축만 사용)"
        else:
            desc, u, use = "", "", "참고"
        rows.append({"열 이름": col, "설명": desc, "단위": u, "AI② 사용 구분": use, "있는 데이터셋": present})
    return pd.DataFrame(rows)


def public_sensor_info():
    rows = []
    for name in PUBLIC_DATASETS:
        s = pd.read_csv(public_paths(name, "public_raw")[1] / "sensors.csv", keep_default_na=False)
        s.insert(0, "dataset", name)
        s.insert(2, "결과파일_약칭", s["sensor_id"].map(sensor_short_name))
        s["HW 대응"] = s["결과파일_약칭"].map(HW_POSITION)
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def public_provenance(frames):
    from config import HBTA, LIS2DW12, MOESA, OLD_ADA, ORIGINAL_DIR
    files = {"ponte_moesa": sorted({*MOESA["zips"].values(), "moesa/Acceleration_Data.zip"}),
             "hbta": [HBTA["file"]], "old_ada": [OLD_ADA["zip"]]}
    counts = {}
    for key, f in frames.items():
        g = f.groupby(["ai1_split", "damage_stage", "label_status"]).agg(
            windows=("window_id", "size"), judged=("prediction", lambda s: int(s.isin([NORMAL, ANOMALY]).sum())))
        counts[key] = [{"ai1_split": a, "damage_stage": b, "label_status": c, **{k: int(v) for k, v in r.items()}}
                       for (a, b, c), r in g.iterrows()]
    moesa_raw = frames["ponte_moesa_raw"]
    pre = moesa_raw[moesa_raw["subset"] == "작업전"]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "model_version": MODEL_VERSION,
        "agreement": "AI①·AI② 공개 데이터 인터페이스 합의서 v3 (2026-11-12 확정, INFRIUM/AI-1#3)",
        "datasets": {
            "ponte_moesa": {"doi": "10.3929/ethz-b-000716850", "license": "CC BY-NC 4.0 (받은 Lisence.txt 는 MIT, 더 엄격한 쪽 적용)",
                            "paper": "Garcia-Ramonda et al. (2025) postprint", "unit": "m/s^2 (센서 ±4g, 4 초과 값이 많아 확정)",
                            "label_source": "공식 목록 Ambient.txt + 논문 표 1. 라벨 규칙 ① 목록·그림 6 일치만 사용 ② 불일치 제외 (AMB_2~6, AMB_19~24, AMB_32~33, AMB_46) ③ AMB_36 은 15:20~15:22 만 주 평가, 전체는 참고",
                            "excluded": "DS4 (문헌 불일치), FVA, 센서 6번, AMB_48 (트럭 충격)",
                            "sensors": "acc01 = 4번 (S5, x=53.75m), acc02 = 2번 (x=45.5m), acc03 = 3번 (x=78.5m, 세션 차이용)",
                            "time": "원본 파일 시각 그대로 (현지 시각, 시간대 표기 없음)",
                            "limits": ["DS2 기준 상태는 단일 세션 (11/27 오후, 굴착기 작업 중)", "열화본에서 조용한 구간은 대부분 판정 보류",
                                       "저진폭 기준 0.0785 m/s^2 는 거의 걸러지지 않음", "세션 차이는 원본으로만 확인 (열화본 미검증)",
                                       "DS7 묶음 A/B 는 같은 측정 (독립 검증 아님)", "DS5 는 보류 후 30개 미만 → 경고 등급 잠정"],
                            "work_effect_check": {"rule": "원본 T1 = AI② 보정 블록, 4번·2번 z 원본 보류 후 n. 오경보 > 0.05 × n 이면 한계 기록, n < 10 이면 참고",
                                                  "exception": "'원본은 열화본과 같은 Window 만' 규칙의 예외 (진동원 확인 목적)",
                                                  "pre_work_windows_total": int(len(pre)),
                                                  "pre_work_windows_after_raw_hold": int((pre["raw_hold_flag_4_2"].astype(str) == "False").sum())}},
            "hbta": {"doi": "10.5281/zenodo.14632942", "license": "CC BY 4.0", "paper": "Svendsen et al. (2022)",
                     "records": HBTA["records"], "sensors": "acc01 = AG03 (x=-7m), acc02 = AG05 (x=0)", "unit": "m/s^2",
                     "time": "측정 날짜 00:00 UTC + 경과 시간 (실제 측정 시각 아님)",
                     "notes": ["손상은 측정마다 보수 (누적 아님)", "UDS_02 블록 전체 train_passing (열차 구간 검출 방법 없음)",
                               "온도 보정 끔", "열화본 판정은 z 축만 (y 축은 보류되어 원인 분석용)", "1-TAC 는 기본 Window 3개 (30.72초) 묶음",
                               "논문의 10.24초 × 80 시험은 측정 1개를 자른 것으로 추정 (독립 시험 아님)"]},
            "old_ada": {"doi": "10.17632/sc8whx4pvm.2", "license": "CC BY 4.0",
                        "papers": ["Kim et al. (2021) J. Bridge Eng. (본문 미확인)", "Quqa & Landi (2023) MSSP 186, 109907 (센서 정보)"],
                        "sensors": "acc01 = A1 (진동 2열, L/4), acc02 = A3 (진동 4열, L/2). 열 대응은 AI① 이 모드 형상으로 확인 (추정)",
                        "unit": {"value": "m/s^2 (간접 근거 2가지로 추정, 문서 확정 아님)",
                                 "evidence": ["잡음 바닥: 70–95Hz 약 6×10⁻⁵/√Hz. m/s^2 면 약 6 µg/√Hz, gal 이면 약 0.06 µg/√Hz 로 일반 계측용 가속도계로는 매우 어려움",
                                              "상시 진동 수준: 2–25Hz RMS 0.001–0.022. m/s^2 면 0.1–2.2 mg 로 교량 상시 진동 범위"],
                                 "not_usable": ["설명 페이지에 단위 표기 없음", "원시 데이터 평균 0 (영점 제거) → 중력 성분 확인 불가"],
                                 "limit": "단위가 gal 이면 열화본 Window 는 전부 판정 보류에 해당함"},
                        "time": "측정 시각 정보 없음 → 주행마다 1시간 간격 가상 시각",
                        "notes": ["Window = 진출 광센서 +1~31초 자유 감쇠, 주행당 1개", "AI① 은 Isolation Forest 없이 z-score 만 (정상 학습 주행 4개)",
                                  "통과 판정 없이 서술 보고", "DMG1 12회 사용 (논문은 10회, 차이 이유 불명)",
                                  "사전 기록: DMG1 은 2~25Hz 동적 Feature 로 탐지되지 않을 수 있음 (Quqa & Landi 2023)",
                                  "OLDADA_0059 (INT 50km/h #5, 49.2km/h) 판정 불가: acc01 감쇠비 계산 실패로 모델 입력이 빠짐. Window 시작 직후 짧은 고주파 충격이 있어 acc01 포락선이 0.2초 만에 최댓값의 20% 아래로 떨어짐 (감쇠비 계산에는 최소 3주기, 약 1초 필요). 2–25Hz RMS 는 다른 주행보다 작음 (acc01 0.090 / acc02 0.095 m/s^2, 다른 주행 중앙값 0.215 / 0.221), 8–15Hz·15–25Hz 에너지 비는 다른 주행의 31–41배. 충격의 원인 기록 없음",
                                  "30·50km/h INT 주행은 '속도 영향 확인용 (참고)'. 40km/h test 3회를 주로 서술하고 따로 보고",
                                  "AI① v1 |z| 중앙값: 40km/h test 2.8, 30km/h 5.1, 50km/h 약 100. 정상 기준 표본이 train 4회뿐이라 표준편차가 작아 |z| 가 과장될 수 있음"],
                        "limits": ["진폭 Feature 는 가진 조건(차량 속도)에 민감함: 손상이 없어도 속도가 바뀌면 크게 벗어남"]},
        },
        "original_file_md5": {n: {f: file_md5(ORIGINAL_DIR / f) for f in fs} for n, fs in files.items()},
        "degradation": {"seed_rule": "채널 난수 = numpy default_rng([20261109, 데이터셋 번호 (HBTA 1, Ponte Moesa 2, Old ADA 3), 측정 번호, 채널 번호]). "
                                     "HBTA 측정 번호: AI1-public-1.1 의 10개는 config 순서대로 1–10, 확인용 새 측정은 사전 등록서 목록 순서대로 11부터. "
                                     "채널 번호: 센서·축 순서 (acc01 Y = 1, acc01 Z = 2, acc02 Y = 3, acc02 Z = 4). 기존 측정의 열화 결과는 바뀌지 않음",
                        "sensor": "LIS2DW12 (±2g, 14bit, 고성능 + 저잡음, 90 µg/√Hz)", "order": "잡음 → 양자화 → 범위 제한(SAT) → 2~25Hz 필터 → Feature",
                        "z_axis_gravity": "수직축은 1g 를 더한 상태로 양자화·범위 제한 후 다시 뺌", "seed": LIS2DW12["seed"],
                        "noise_band_rms_mg": {"200Hz": 0.411, "100Hz": 0.416},
                        "hold_threshold_m_s2": {"deg_200Hz": 0.0057, "deg_100Hz": 0.0058, "raw_moesa_MR3000C": 0.00063},
                        "hold_threshold_exact_m_s2": {"deg_200Hz": 0.0057041894, "deg_100Hz": 0.0057711219, "raw_moesa_MR3000C": 0.0006318644,
                                                      "rule": "2–25Hz RMS < 기준 이면 보류. 기준 = 잡음 RMS × √2, 잡음 RMS = 잡음 밀도 × √(필터 등가 대역폭) (LIS2DW12 는 14bit 양자화 잡음 포함)"}},
        "preprocess": {"filter": "4차 Butterworth 대역 통과 2~25Hz, 양방향, 실험 앞뒤 10초 여유", "windows": {"ponte_moesa": "60초 / 30초 이동", "hbta": "10.24초, 겹침 없음", "old_ada": "30초 (자유 감쇠)"},
                       "tac": "Welch (Hann, 50% 겹침), T = G_ba / G_bb, 기준 = AI① train Window 평균 스펙트럼의 비",
                       "ar": "AR(5), 채널별, 평균 제거, 절편 없음, numpy 최소제곱", "channel_hold": "주 지표 채널 보류 → Window 판정 보류, 보조 채널 → 그 채널 Feature 빈 값 (열화본)"},
        "windows_by_split": counts,
        "ai1_v2": v2_provenance(),
        "information_overlap": "AI① v2 Feature (모드 형상 비, 모드별 진동수) 와 AI② 지표 (1-TAC, AR) 는 두 센서 응답 비와 스펙트럼 정보를 일부 공유함. 같은 결론을 서로 독립인 두 근거로 쓰지 않음",
        "fva_label_mismatch": "FVA_2·3, FVA_17~22, FVB_1·2, FVB_5~8 라벨이 같은 시각 상시 진동 라벨과 다름, FVB_12 날짜 오기, FVA_36 구성 오기 (FVA 는 사용 안 함)",
    }


def v2_provenance():
    """AI① v2 (모양 Feature) 기록: 선택 규칙·결과·모드 대역. 참고 방법임을 명시"""
    root = RESULTS_DIR.parents[1]
    out = {"status": "결과 열람 후 변경한 방법. 참고 결과로만 사용, 한 번만 실행 (AI② 확인 2026-11-18)",
           "rule": {"stability": "|중앙값1 − 중앙값2| ÷ √(s1² + s2²) < 1.0, s = MAD × 1.4826, 열화본 정상 Window (판정 보류 제외)",
                    "noise": "열화본 정상 s ÷ 원본 정상 s < 2.0 (같은 Window)", "no_pass": "통과 Feature 없으면 v2 미실행, v1 만 보고",
                    "modes": "원본 AI① train 평균 스펙트럼(두 센서 z 합)에서 2–25Hz 돌출도 상위 3개, 대역 ±15% 겹치면 큰 쪽. 열화본도 같은 대역"},
           "limits": ["Ponte Moesa: 안정성은 같은 측정(AMB_18) train vs val → 세션 간 안정성 검증되지 않음",
                      "Old ADA: 안정성 미적용 (정상 4회), 잡음 견딤도 4개 Window 로 계산",
                      "모드 진동수는 손상에 둔감할 수 있음 (Garcia-Ramonda et al. 2025), 단독 사용 안 함",
                      "Ponte Moesa: 열화본 DS3 비율 상승은 AMB_28의 약 15Hz 피크(v2_f3)와 원본·열화본 판정 기준 차이(z 기준 6.44 / 4.84)에서 나옴. 15Hz 피크는 DS5·DS7에서 보이지 않아 측정 시점의 진동원일 가능성(원인 기록 없음). 탐지 개선 근거로 쓰지 않음 (AI② 확인 INFRIUM/AI-1#4, AI① 재확인: AMB_28 v2_f3 중앙값 15.02Hz, 14.5Hz 초과 87–88%, 다른 측정 13.2–13.6Hz, 열화본 DS3 Anomaly 72개 중 52개의 top1 이 v2_f3)",
                      "Ponte Moesa 원인 확인 (같은 Window 기준): 분모 차이 아님 (판정 보류 안 된 같은 Window 에서도 원본 4%, 열화본 80%). DS3 은 조용한 측정 아님 (원본 2–25Hz RMS 중앙값 acc01 0.0076 / acc02 0.0141 m/s^2, AI① train·val 0.0086 / 0.0106). 원본→열화본 변화가 가장 큰 Feature 는 v2_shape1 (정상 s 의 1.3–2.1배), v2_shape3, v2_e1_acc01 (0.6–1.0배). v2_shape1 값 이동은 DS3 상승의 원인은 아니지만, 값의 퍼짐만 보는 잡음 견딤 규칙의 한계로 기록",
                      "Old ADA: 속도 영향이 손상 영향보다 큼 (정상 50km/h |z| 58.4 > DMG1 25.1, DMG2 24.2). 정상 40km/h AI② 기준 3회도 6.5 로 정상 주행 사이 변동이 큼 (정상 기준 4회, Feature 13개 중 최댓값). |z| 는 순서로만 해석",
                      "Old ADA DMG1: 사전 기록('탐지되지 않을 수 있음')과 달리 v2 |z| 25.1. 위 두 이유로 'v2 가 DMG1 을 탐지함'의 근거로 쓰지 않음. RCV (17.0) 도 해석하지 않음"],
           "hw_rule_lessons": ["잡음 견딤은 진동이 작은 측정까지 포함해 확인 (Ponte Moesa)", "가진 조건(타격 세기·위치)이 바뀌어도 정상 판정이 유지되는지 따로 확인 (Old ADA)"],
           "datasets": {}}
    for name in PUBLIC_DATASETS:
        sel_path = root / "results" / "public" / "v2_selection" / f"{name}_v2_selection.csv"
        modes_path = root / "data" / "processed" / "public_raw" / name / "v2_modes.json"
        if not sel_path.exists():
            continue
        sel = pd.read_csv(sel_path)
        out["datasets"][name] = {
            "modes_hz": json.loads(modes_path.read_text(encoding="utf-8"))["modes_hz"] if modes_path.exists() else None,
            "selected": sel.loc[sel["selected"], "feature"].tolist(),
            "rejected": {r.feature: {"stability": None if pd.isna(r.stability) else round(float(r.stability), 3),
                                     "noise_ratio": round(float(r.noise_ratio), 3)} for r in sel[~sel["selected"]].itertuples()},
            "windows": {k: (str(sel[k].iat[0]) if k == "used_splits" else int(sel[k].iat[0]))
                        for k in ("windows_train", "windows_val", "excluded_held_deg", "raw_windows_used", "used_splits")},
        }
    return out


def export_public():
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    frames, labels = {}, []
    for name in PUBLIC_DATASETS:
        exps = public_experiments(name)
        labels.append(exps.drop(columns=[c for c in ("file", "record_id", "damage_type", "severity_level", "load_position", "bridge_type") if c in exps]))
        for source in ("public_raw", "public_deg"):
            out = public_results(name, source, exps)
            key = f"{name}_{'raw' if source == 'public_raw' else 'deg'}"
            frames[key] = out
            out.to_csv(PUBLIC_OUT / f"{name}_anomaly_results_{key.rsplit('_', 1)[1]}.csv", index=False, encoding="utf-8-sig")
    pd.concat(labels, ignore_index=True).to_csv(PUBLIC_OUT / "experiment_labels.csv", index=False, encoding="utf-8-sig")
    public_column_guide(frames).to_csv(PUBLIC_OUT / "column_guide.csv", index=False, encoding="utf-8-sig")
    public_sensor_info().to_csv(PUBLIC_OUT / "sensor_info.csv", index=False, encoding="utf-8-sig")
    prov = public_provenance(frames)
    (PUBLIC_OUT / "data_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"공개 데이터 전달 파일 → {PUBLIC_OUT}")
    for f in sorted(PUBLIC_OUT.iterdir()):
        print(f"  {f.name:<44} {f.stat().st_size / 1024:>10,.0f} KB")
    for key in ("ponte_moesa_raw", "ponte_moesa_deg", "hbta_raw", "hbta_deg", "old_ada_raw", "old_ada_deg"):
        a, b = frames[key.rsplit("_", 1)[0] + "_raw"], frames[key]
        same = a[["experiment_id", "window_start_us"]].equals(b[["experiment_id", "window_start_us"]])
        print(f"  {key}: Window {len(b):,}개, 원본과 같은 키·순서: {same}")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["public"]:
        export_public()
    else:
        main()
