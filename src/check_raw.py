"""
raw 데이터 구조 확인 (실제 센서 데이터).

확인 항목
[1] 센서 등록 정보   [2] 실험 구성 (D0~D3 × 심각도 × split, 하중 조건)
[3] 파일 열 구성이 데이터 사양(RAW_COLUMNS)과 같은지
[4] 센서·축별 측정 속도 / quality_flag 개수 / 값 범위 (OK 데이터 기준)
[5] 실험별 주요 값: 가속도 우세 진동수(수직 Z·폭방향 Y), 하중, 온도 → 조건·이상상태에 따라 기대한 방향으로 변하는지

실행: python src/check_raw.py        (트러스: $env:IOT_BRIDGE="truss"; python src/check_raw.py)
"""
import numpy as np
import pandas as pd

from config import ACTIVE_BRIDGE, RAW_COLUMNS, RAW_DIR

BRIDGE_DIR = RAW_DIR / ACTIVE_BRIDGE


def dominant_freq(values, fs, min_hz=2.0):
    """평균(중력 등)을 뺀 뒤 가장 강한 진동수 [Hz]."""
    x = values - values.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    spec[freqs < min_hz] = 0
    return freqs[np.argmax(spec)]


def channel_summary(df):
    rows = []
    for (sid, axis), g in df.groupby(["sensor_id", "axis"], sort=False):
        dt = np.diff(g["timestamp_us"].to_numpy()) / 1e6
        ok = g.loc[g["quality_flag"] == "OK", "value"]
        flags = g["quality_flag"].value_counts()
        rows.append({
            "sensor_id": sid, "axis": axis, "rows": len(g),
            "fs_est_hz": round(1 / np.median(dt), 2),
            "fs_spec_hz": g["sample_rate_hz"].iloc[0],
            **{f: int(flags.get(f, 0)) for f in ("OK", "SAT", "DROP", "CAL")},
            "mean(OK)": ok.mean(), "std(OK)": ok.std(), "unit": g["unit"].iloc[0],
        })
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    sensors = pd.read_csv(BRIDGE_DIR / "sensors.csv", keep_default_na=False)
    exps = pd.read_csv(BRIDGE_DIR / "experiments.csv", keep_default_na=False, na_values=[""])

    print("[1] 센서 등록 정보 (sensors.csv)")
    print(sensors.drop(columns=["device"]).to_string(index=False))

    print("\n[2] 실험 구성: (damage_type, severity_level) × split  /  하중 조건별 실험 수")
    print(pd.crosstab([exps["damage_type"], exps["severity_level"]], exps["split"]))
    print(exps["condition"].value_counts().sort_index().to_string())

    first = exps.iloc[0]
    df = pd.read_parquet(BRIDGE_DIR / first["file"])
    print(f"\n[3] 센서 데이터 파일 예시: {first['file']}  ({len(df):,} rows)")
    print(f"열 구성이 데이터 사양과 일치: {list(df.columns) == RAW_COLUMNS}")
    print(df.head(3).to_string(index=False))

    print(f"\n[4] 센서·축별 요약 ({first['experiment_id']}, 조건 {first['condition']})")
    print(channel_summary(df).to_string(index=False))

    acc = sensors[sensors["sensor_type"] == "acceleration"]
    side = acc.sort_values("y_mm").iloc[-1]   # 측면(비틀림 확인용) 가속도계
    temp_ids = sensors.loc[sensors["sensor_type"] == "temperature", "sensor_id"].tolist()
    load_ids = sensors.loc[sensors["sensor_type"] == "load", "sensor_id"].tolist()
    fs = side["sample_rate_hz"]

    print(f"\n[5] 실험별 주요 값 (OK 데이터만): {side['sensor_id']} 수직(Z)·폭방향(Y) 우세 진동수 / 하중 / 온도")
    rows = []
    for exp in exps.itertuples():
        d = pd.read_parquet(BRIDGE_DIR / exp.file, filters=[("quality_flag", "==", "OK")])
        acc_d = d[d["sensor_id"] == side["sensor_id"]]
        row = {
            "experiment_id": exp.experiment_id, "condition": exp.condition,
            "Z_peak_hz": round(dominant_freq(acc_d.loc[acc_d["axis"] == "Z", "value"].to_numpy(), fs), 1),
            "Y_peak_hz": round(dominant_freq(acc_d.loc[acc_d["axis"] == "Y", "value"].to_numpy(), fs), 1),
            "Y_rms": acc_d.loc[acc_d["axis"] == "Y", "value"].std(),
            "load_N": d.loc[d["sensor_id"].isin(load_ids), "value"].mean(),
            "temp_C": d.loc[d["sensor_id"].isin(temp_ids), "value"].mean(),
        }
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
