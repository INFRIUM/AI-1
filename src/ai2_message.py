"""
AI② 전달 메시지 만들기 / API 전송.

메시지 1건 = 구간(Window) 1개 = raw 센서값 + AI① 이상탐지 결과. 정상(Normal) 구간도 보낸다.
  {
    "schema_version", "message_type": "ai1_window", "bridge_type", "experiment_id", "window_id",
    "window_start_us", "window_end_us",
    "raw": { "start_us", "end_us", "sensors": { <센서 ID>: { "sample_rate_hz", "unit", "t0_us", "n",
                                                            "axes": { "X": [...], ... }, "quality": {...} } } },
    "ai1": { "prediction", "confidence", "anomaly_probability", "decision_ratio", ..., "features": {...} }
  }

raw 범위: 이전 구간 끝 ~ 이 구간 끝 (첫 구간은 실험 시작부터, 마지막 구간은 실험 끝까지)
  → 메시지를 순서대로 이어 붙이면 실험 raw 전체가 빠짐·겹침 없이 복원된다.
  구간(타격) 자체의 시간은 window_start_us ~ window_end_us.
raw 의 정답 열(damage_type, severity_level)은 load_experiment 에서 이미 제거된다.

confidence: 판정이 맞을 확률. Anomaly → P(이상), Normal → 1 - P(이상).
  P(이상) 은 persist_ratio 로지스틱 회귀 (AI① test 실험으로 보정, models/<source>/<bridge>/confidence.json)
  정상·이상 구간 수가 달라도 한쪽으로 치우치지 않도록 두 클래스를 같은 무게로 보정 (class_weight="balanced")

사용: predict.py --file (새 파일 판정 → 메시지 저장·전송), export_for_ai2.py (예시 메시지)
"""
import json
import math
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import AI2_API, BRIDGES, DATA_SOURCE, GROUND_TRUTH, MODELS_DIR, PREDICTION_LABELS
from preprocess import sensor_short_name

NORMAL, ANOMALY = PREDICTION_LABELS
D0 = GROUND_TRUTH["normal_code"]


def warming_up(df):
    """실험(세션) 시작 직후 연속 판정에 필요한 구간이 아직 안 모인 구간 (predict.persistence 가 0 을 넣음)"""
    return df["persist_ratio"].eq(0) if "persist_ratio" in df else pd.Series(False, index=df.index)


def ratio_column(pred):
    return "persist_ratio" if "persist_ratio" in pred else "decision_ratio"


# ---------------------------------------------------------------------------
# 신뢰도 (confidence)
# ---------------------------------------------------------------------------
def fit_confidence(key, pred):
    """AI① test 실험으로 P(실제 이상 | 이상 정도) 로지스틱 보정 → confidence.json 저장. (보정 정보, test 구간 표) 반환"""
    col = ratio_column(pred)
    split = pd.read_csv(MODELS_DIR / key / "split.csv", usecols=["experiment_id", "split", "damage_type"])
    test = pred.merge(split, on="experiment_id")
    test = test[(test["split"] == "test") & ~warming_up(test)]
    lr = LogisticRegression(class_weight="balanced").fit(test[[col]].to_numpy(), (test["damage_type"] != D0).astype(int))
    calib = {"ratio_column": col, "coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]),
             "n_windows": len(test), "data_source": DATA_SOURCE}
    (MODELS_DIR / key / "confidence.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib, test


def load_confidence(key):
    path = MODELS_DIR / key / "confidence.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def anomaly_probability(ratio, calib):
    if calib is None or ratio is None or math.isnan(ratio):
        return None
    return 1 / (1 + math.exp(-(calib["coef"] * ratio + calib["intercept"])))


# ---------------------------------------------------------------------------
# ai1 블록
# ---------------------------------------------------------------------------
def clean(v, digits=6):
    """NaN → None, numpy 숫자 → 파이썬 숫자 (JSON 저장용)"""
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), digits)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def feature_columns(pred, sensors):
    prefixes = tuple(f"{sensor_short_name(s['sensor_id'])}_" for s in sensors) + ("cross_",)
    return [c for c in pred.columns if c.startswith(prefixes) and not c.endswith(("_score", "_state"))]


def sensor_prefixes(pred):
    return [c.removesuffix("_score") for c in pred.columns if c.endswith("_score") and c != "anomaly_score"]


def ai1_block(row, prefixes, feature_cols, calib):
    warming = "persist_ratio" in row and row["persist_ratio"] == 0
    p = None if warming else anomaly_probability(row.get(calib["ratio_column"], np.nan) if calib else None, calib)
    is_anomaly = row["prediction"] == ANOMALY
    return {
        "prediction": row["prediction"],
        "warming_up": bool(warming),
        "confidence": None if p is None else round(p if is_anomaly else 1 - p, 4),
        "anomaly_probability": None if p is None else round(p, 4),
        "decision_ratio": clean(row["decision_ratio"]),
        "persist_ratio": clean(row.get("persist_ratio", np.nan)),
        "raw_prediction": row["raw_prediction"],
        "anomaly_score": clean(row["anomaly_score"]),
        "threshold": clean(row["threshold"]),
        "max_abs_z": clean(row["max_abs_z"]),
        "z_threshold": clean(row["z_threshold"]),
        "condition": {
            "condition": row["condition"],
            "load_level_kg": clean(row["load_level_kg"]),
            "load_position": clean(row["load_position"]) or None,  # 하중 위치 없음 → null (빈 문자열 대신)
            "temperature_c": clean(row["temperature_c"]),
        },
        "anomaly_sensors": [] if pd.isna(row["anomaly_sensors"]) or row["anomaly_sensors"] == "" else row["anomaly_sensors"].split(";"),
        "main_sensor": clean(row["main_sensor"]),
        "main_location": clean(row["main_location"]),
        "symptom": clean(row["symptom"]),
        "top_features": [{"feature": row[f"top{i}_feature"], "z": clean(row[f"top{i}_z"])}
                         for i in (1, 2) if f"top{i}_feature" in row and not pd.isna(row[f"top{i}_feature"])],
        "sensors": [{"sensor": p, "score": clean(row[f"{p}_score"]), "state": row[f"{p}_state"]} for p in prefixes],
        "features": {c: clean(row[c]) for c in feature_cols},
    }


# ---------------------------------------------------------------------------
# raw 블록
# ---------------------------------------------------------------------------
def raw_block(raw, start_us, end_us):
    """[start_us, end_us) 구간의 raw. 센서마다 시작 시각 + 측정 속도 + 축별 값 배열 (시각 간격이 고르지 않으면 timestamp_us 도 포함)."""
    part = raw[(raw["timestamp_us"] >= start_us) & (raw["timestamp_us"] < end_us)]
    sensors = {}
    for sid, g in part.groupby("sensor_id", sort=False):
        axes, quality, ts = {}, {}, None
        for axis, a in g.sort_values("timestamp_us").groupby("axis", sort=False):
            axes[axis] = [clean(v, 5) for v in a["value"].to_numpy()]
            bad = a["quality_flag"].to_numpy() != "OK"
            if bad.any():
                quality[axis] = {flag: np.flatnonzero(a["quality_flag"].to_numpy() == flag).tolist()
                                 for flag in sorted(set(a["quality_flag"][bad]))}
            ts = a["timestamp_us"].to_numpy()
        entry = {"sample_rate_hz": clean(g["sample_rate_hz"].iat[0]), "unit": g["unit"].iat[0],
                 "t0_us": int(ts[0]), "n": len(ts), "axes": axes}
        if len(ts) > 1 and len(np.unique(np.diff(ts))) > 1:
            entry["timestamp_us"] = ts.tolist()
        if quality:
            entry["quality"] = quality  # 값이 OK 가 아닌 샘플의 위치 (축별, 플래그별 index)
        sensors[sid] = entry
    return {"start_us": int(start_us), "end_us": int(end_us), "sensors": sensors}


# ---------------------------------------------------------------------------
# 메시지
# ---------------------------------------------------------------------------
def build_messages(final, raw, key, calib=None, only=None, to_raw_end=True):
    """실험 1개의 판정 결과(final, 구간당 1행) + raw → 구간마다 메시지 1건 (시간 순).
    only: 이 window_start_us 집합에 든 구간만 메시지로 만든다 (수신 API 에서 새로 확정된 구간만 보낼 때).
    to_raw_end: 마지막 구간의 raw 를 데이터 끝까지 붙일지 (세션이 끝났을 때만 True)."""
    sensors = BRIDGES[key]["sensors"]
    prefixes, feature_cols = sensor_prefixes(final), feature_columns(final, sensors)
    final = final.sort_values("window_start_us").reset_index(drop=True)
    raw_start, raw_end = int(raw["timestamp_us"].min()), int(raw["timestamp_us"].max()) + 1
    messages = []
    for i, row in final.iterrows():
        if only is not None and int(row["window_start_us"]) not in only:
            continue
        start = raw_start if i == 0 else int(final.at[i - 1, "window_end_us"])
        end = raw_end if to_raw_end and i == len(final) - 1 else int(row["window_end_us"])
        messages.append({
            "schema_version": AI2_API["schema_version"],
            "message_type": "ai1_window",
            "bridge_type": BRIDGES[key]["bridge_type"],
            "experiment_id": row["experiment_id"],
            "window_id": row["window_id"],
            "window_start": clean(row["window_start"]),
            "window_start_us": int(row["window_start_us"]),
            "window_end_us": int(row["window_end_us"]),
            "raw": raw_block(raw, start, end),
            "ai1": ai1_block(row, prefixes, feature_cols, calib),
        })
    return messages


def save_jsonl(messages, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def post_one(message):
    """메시지 1건 POST. 네트워크 오류·5xx 면 max_retries 번까지 재전송 (간격 1·2·4초…), 4xx 는 바로 실패.
    반환: None (성공) 또는 실패 사유 문자열"""
    data = json.dumps(message, ensure_ascii=False).encode("utf-8")
    error = None
    for attempt in range(AI2_API["max_retries"] + 1):
        if attempt:
            time.sleep(AI2_API["retry_backoff_s"] * 2 ** (attempt - 1))
        req = urllib.request.Request(AI2_API["url"], data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=AI2_API["timeout_s"]):
                return None
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}"
            if e.code < 500:  # 형식 오류 등: 다시 보내도 같은 결과
                return error
        except (urllib.error.URLError, OSError) as e:  # 연결 실패·시간 초과
            error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
    return f"{error} (재전송 {AI2_API['max_retries']}회 후 실패)"


def send(messages):
    """AI2_API url 로 메시지를 1건씩 순서대로 POST. url 이 비어 있으면 보내지 않는다. (성공 건수, 실패 목록) 반환"""
    if not AI2_API["url"]:
        return 0, []
    ok, failed = 0, []
    for m in messages:
        error = post_one(m)
        if error is None:
            ok += 1
        else:  # 실패는 기록만 하고 다음 메시지 계속
            failed.append((m["window_id"], error))
    return ok, failed
