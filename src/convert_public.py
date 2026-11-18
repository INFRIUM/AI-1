"""
공개 교량 데이터 → AI① 데이터 사양(RAW_COLUMNS) 변환 + HW 가속도계(LIS2DW12) 수준 열화본 생성.

AI①·AI② 합의서 v3 (2026-11-12) 기준. 설정값은 config.py 의 [A-7] 에만 둔다.

출력 (데이터셋마다 두 벌, 같은 실험·같은 시각)
  data/raw/public_raw/<dataset>/   원본 신호
  data/raw/public_deg/<dataset>/   LIS2DW12 열화본 (잡음 → 14bit 양자화 → ±2g 범위 제한)
    <record>.parquet   측정 1개의 센서 데이터 (RAW_COLUMNS, 정답 열은 비워 둠)
    experiments.csv    실험(블록) 목록: 시각 범위, 분할, 정답, 조건
    sensors.csv        센서 등록 정보 (원본 센서와의 대응 포함)

실험(experiment) = 분할 단위. 정상 측정은 시간 블록으로 나눠 블록마다 실험 1개,
손상 측정은 측정 전체가 실험 1개. experiment_id 는 정답이 드러나지 않는 중립 ID.

열화
  - 잡음: 백색 잡음, 표준편차 = 잡음 밀도 × √(fs/2)
  - 수직(Z)축은 실제 센서처럼 중력 1g 를 더한 상태에서 양자화·범위 제한 후 다시 뺀다
  - 난수: 시드 20261109 + (데이터셋, 측정, 채널) 번호로 채널마다 독립. 같은 입력이면 항상 같은 결과

실행: python src/convert_public.py                ← 원본이 있는 데이터셋 모두
      python src/convert_public.py hbta           ← 지정한 데이터셋만 (hbta / ponte_moesa / old_ada)
"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import GRAVITY, HBTA, LIS2DW12, MOESA, OLD_ADA, ORIGINAL_DIR, RAW_COLUMNS, ROOT

DATASETS = {"hbta": "hbta", "ponte_moesa": "moesa", "old_ada": "old_ada"}   # 데이터셋 → data/original 아래 폴더
DATASET_INDEX = {"hbta": 1, "ponte_moesa": 2, "old_ada": 3}               # 열화 난수 시드에 쓰는 번호 (바꾸지 않음)


# ---------------------------------------------------------------------------
# LIS2DW12 열화
# ---------------------------------------------------------------------------
def lis2dw12_noise_std(fs):
    """한 샘플의 잡음 표준편차 [m/s^2] = 잡음 밀도 × √(나이퀴스트 대역)."""
    return LIS2DW12["noise_ug_per_rthz"] * 1e-6 * GRAVITY * np.sqrt(fs / 2)


def degrade(x, fs, axis, rng):
    """원본 가속도 [m/s^2] → LIS2DW12 출력. 반환: (열화 신호, 범위 초과 여부)"""
    offset = GRAVITY if axis == "Z" else 0.0          # 수직축은 중력이 더해진 상태로 측정됨
    full = LIS2DW12["range_g"] * GRAVITY
    lsb = 2 * full / 2 ** LIS2DW12["bits"]
    y = x + offset + rng.normal(0.0, lis2dw12_noise_std(fs), len(x))
    y = np.round(y / lsb) * lsb
    sat = np.abs(y) > full
    y = np.clip(y, -full, full)
    return y - offset, sat


def channel_rng(dataset, record_no, channel_no):
    return np.random.default_rng([LIS2DW12["seed"], DATASET_INDEX[dataset], record_no, channel_no])


# ---------------------------------------------------------------------------
# 공통: 측정 1개를 RAW_COLUMNS 표로
# ---------------------------------------------------------------------------
def to_frame(signals, t_us, fs, sat=None):
    """signals: {(sensor_id, axis): 값 배열} → RAW_COLUMNS 표 (정답 열은 비워 둠, experiment_id 는 측정 ID)"""
    parts = []
    for (sid, axis), v in signals.items():
        flag = np.where(np.isnan(v), "DROP", "OK").astype(object)
        if sat is not None:
            flag[sat[(sid, axis)]] = "SAT"
        parts.append(pd.DataFrame({
            "sensor_id": sid, "timestamp_us": t_us, "axis": axis, "value": v,
            "unit": "m/s^2", "sample_rate_hz": fs, "quality_flag": flag,
        }))
    df = pd.concat(parts, ignore_index=True)
    for col in ("experiment_id", "bridge_type", "damage_type", "severity_level"):
        df[col] = ""
    return df[RAW_COLUMNS]


def write_dataset(name, records, experiments, sensors):
    """records: [(record_id, {(sensor_id, axis): 원본 배열}, t_us, fs)] → public_raw / public_deg 두 벌 저장"""
    out = {src: ROOT / "data" / "raw" / src / name for src in ("public_raw", "public_deg")}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    sat_total = 0
    for r_no, (rid, signals, t_us, fs) in enumerate(records, start=1):
        to_frame(signals, t_us, fs).to_parquet(out["public_raw"] / f"{rid}.parquet", index=False)
        deg, sat = {}, {}
        for c_no, (key, v) in enumerate(signals.items(), start=1):
            deg[key], sat[key] = degrade(v, fs, key[1], channel_rng(name, r_no, c_no))
            sat_total += int(sat[key].sum())
        to_frame(deg, t_us, fs, sat).to_parquet(out["public_deg"] / f"{rid}.parquet", index=False)
    for d in out.values():
        experiments.to_csv(d / "experiments.csv", index=False, encoding="utf-8-sig")
        sensors.to_csv(d / "sensors.csv", index=False, encoding="utf-8-sig")
    print(f"[{name}] 측정 {len(records)}개, 실험(블록) {len(experiments)}개 → {out['public_raw']} / {out['public_deg']}")
    print(f"  열화본 범위 초과(SAT) 샘플: {sat_total:,}")
    return out


def date_to_us(date_str):
    """'2020-09-27' → 그날 00:00 UTC 의 Unix µs. 공개 데이터에 측정 시작 시각이 없을 때 시간축 기준으로 쓴다."""
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1e6)


def neutral_ids(prefix, n, salt):
    """정답이 드러나지 않도록 순서를 섞은 ID 번호 [1..n]"""
    order = np.random.default_rng([LIS2DW12["seed"], salt]).permutation(n) + 1
    return [f"{prefix}_{k:02d}" for k in order]


# ---------------------------------------------------------------------------
# HBTA
# ---------------------------------------------------------------------------
def hbta_blocks(state, n_windows):
    """측정 하나를 실험 블록으로 나눈다. 반환: [(split, 시작 Window 번호, Window 수)]"""
    if state != "UDS":
        return [("test", 0, n_windows)]
    s = HBTA["uds_split_windows"]
    blocks, k = [], 0
    for split, n in (("train", s["train"]), ("val", s["val"]), ("ai2_calib", s["ai2_calib"])):
        blocks.append((split, k, n))
        k += n
    for n in s["test"]:
        blocks.append(("test", k, n))
        k += n
    if k > n_windows:
        raise ValueError(f"HBTA 분할 Window 합 {k} 가 안정 구간 Window 수 {n_windows} 보다 큼")
    return blocks


def convert_hbta():
    import h5py

    fs = HBTA["sample_rate_hz"]
    win = HBTA["window_s"]
    t0_s, t1_s = HBTA["steady_s"]
    n_windows = int((t1_s - t0_s) // win)
    rec_ids = neutral_ids("HBTA_R", len(HBTA["records"]), salt=101)

    records, rows = [], []
    with h5py.File(ORIGINAL_DIR / HBTA["file"], "r") as f:
        for rid, name in zip(rec_ids, HBTA["records"]):
            g = f[name]
            a = g.attrs
            state = str(a["damage_state"])
            signals = {(sid, axis): g["acceleration"][src][axis.lower()][()].astype(float)
                       for sid, src in HBTA["sensors"].items() for axis in HBTA["axes"]}
            n = len(next(iter(signals.values())))
            base_us = date_to_us(str(a["date"]))
            t_us = base_us + np.round(np.arange(n) * 1e6 / fs).astype(np.int64)
            records.append((rid, signals, t_us, fs))
            for split, k, nw in hbta_blocks(state, n_windows):
                start_s = t0_s + k * win
                rows.append({
                    "record_id": rid, "source_record": name, "file": f"{rid}.parquet",
                    "start_us": int(base_us + round(start_s * 1e6)),
                    "end_us": int(base_us + round((start_s + nw * win) * 1e6)),
                    "n_windows": nw, "split": split, "cv_group": "",
                    "bridge_type": "TRUSS",
                    "damage_type": "D0" if state == "UDS" else "DMG", "severity_level": "",
                    "damage_stage": state, "label_status": "확실",
                    "condition": "NM_P2_Z", "load_position": "",
                    "temperature_c": int(a["temperature"]),
                    "train_passing": str(a["train_passing"]), "wind_range": str(a["wind_range"]),
                    "date": str(a["date"]),
                })

    exps = pd.DataFrame(rows).sort_values(["record_id", "start_us"], ignore_index=True)
    exps.insert(0, "experiment_id", [f"HBTA_{i:04d}" for i in range(1, len(exps) + 1)])
    sensors = pd.DataFrame([
        {"sensor_id": sid, "sensor_type": "acceleration", "device": "Dytran (HBTA 원본) / LIS2DW12 (열화본)",
         "source_sensor": src, "sensor_location": loc, "x_m": x, "x_ratio": xr, "y_m": 2.25,
         "sample_rate_hz": fs, "unit": "m/s^2", "axes": ",".join(HBTA["axes"])}
        for (sid, src), loc, x, xr in zip(HBTA["sensors"].items(), ("L/4 부근", "L/2"), (-7.0, 0.0), (0.30, 0.50))
    ])
    write_dataset("hbta", records, exps, sensors)
    print(exps.groupby(["damage_stage", "split"]).agg(실험=("experiment_id", "size"), Window=("n_windows", "sum"))
          .to_string())


# ---------------------------------------------------------------------------
# Ponte Moesa
# ---------------------------------------------------------------------------
MOESA_CHANNELS = [f"{a}{i}" for i in range(1, 8) for a in "XYZ"]   # .mat 안 21개 채널 순서 (X_1, Y_1, Z_1, ..., Z_7)


def moesa_day(day):
    """Nov<day>_Sall.mat (MATLAB 7.3 timetable) → (시각 µs 배열, {채널: 값 배열}). 필요하면 ZIP 에서 꺼낸다."""
    import zipfile

    import h5py

    out = MOESA["extract_dir"]
    path = out / f"Nov{day}_Sall.mat"
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ORIGINAL_DIR / MOESA["zips"][day]) as z:
            z.extract(f"Nov{day}_Sall.mat", out)
    f = h5py.File(path, "r")
    refs = f["#refs#"]
    big = [k for k in refs if isinstance(refs[k], h5py.Dataset) and refs[k].size > 10000]
    time_keys = [k for k in big if refs[k].dtype == np.float64]
    data_keys = sorted((k for k in big if refs[k].dtype == np.float32), key=lambda k: (k.upper(), k))
    if len(time_keys) != 1 or len(data_keys) != len(MOESA_CHANNELS):
        raise ValueError(f"{path.name}: 시간 열 {len(time_keys)}개, 채널 {len(data_keys)}개 (기대: 1, 21)")
    t_us = np.round(refs[time_keys[0]][0, :] * 1000).astype(np.int64)   # 원본은 ms
    return f, t_us, dict(zip(MOESA_CHANNELS, (refs[k] for k in data_keys)))


def moesa_time_us(text):
    """'27 12:45:00' → 2019-11-27 12:45:00 (파일의 시각 기준) µs"""
    day, hms = text.split()
    h, m, s = (int(x) for x in hms.split(":"))
    return int(datetime(2019, 11, int(day), h, m, s, tzinfo=timezone.utc).timestamp() * 1e6)


def convert_moesa():
    fs = MOESA["sample_rate_hz"]
    blocks = pd.DataFrame(MOESA["blocks"], columns=["amb", "start", "end", "damage_stage", "split", "group", "label_status"])
    blocks["cv_group"] = blocks["group"].where(blocks["group"].isin(["A", "B"]), "")   # 교차 검증 묶음
    blocks["subset"] = blocks["group"].where(~blocks["group"].isin(["A", "B"]), "")    # 작업전/작업중, 저녁/낮
    blocks["start_us"] = blocks["start"].map(moesa_time_us)
    blocks["end_us"] = blocks["end"].map(moesa_time_us)
    blocks["day"] = blocks["start"].str[:2].astype(int)
    blocks["setup"] = blocks["amb"].map(MOESA["setup"]).fillna(5).astype(int)
    rec_ids = dict(zip(blocks["amb"].unique(), neutral_ids("MOESA_R", blocks["amb"].nunique(), salt=201)))

    records = []
    for day, g in blocks.groupby("day"):
        f, t_us, ch = moesa_day(day)
        for amb, b in g.groupby("amb", sort=False):
            i0, i1 = np.searchsorted(t_us, [b["start_us"].min(), b["end_us"].max()])
            setup = int(b["setup"].iat[0])
            signals = {}
            for sid, num in MOESA["sensors"].items():
                if num == 4 and setup != 5:     # 이동 센서 4번은 S5 에서만 주 평가 위치
                    continue
                for axis in "XYZ":
                    signals[(sid, axis)] = ch[f"{axis}{num}"][0, i0:i1].astype(float)
            records.append((rec_ids[amb], signals, t_us[i0:i1], fs))
        f.close()

    rows = []
    for b in blocks.itertuples():
        n_win = int(((b.end_us - b.start_us) / 1e6 - MOESA["window_s"]) // (MOESA["window_s"] * (1 - MOESA["overlap"])) + 1)
        rows.append({
            "record_id": rec_ids[b.amb], "source_record": b.amb, "file": f"{rec_ids[b.amb]}.parquet",
            "start_us": b.start_us, "end_us": b.end_us, "n_windows": max(n_win, 0),
            "split": b.split, "cv_group": b.cv_group, "subset": b.subset, "bridge_type": "BOX",
            "damage_type": "D0" if b.damage_stage == "DS2" else "DMG", "severity_level": "",
            "damage_stage": b.damage_stage, "label_status": b.label_status,
            "condition": f"AMB_S{b.setup}", "setup": b.setup, "load_position": "",
            "temperature_c": "", "train_passing": "", "wind_range": "", "date": f"2019-11-{b.day}",
        })
    exps = pd.DataFrame(rows).sort_values(["record_id", "start_us"], ignore_index=True)
    exps.insert(0, "experiment_id", [f"MOESA_{i:04d}" for i in range(1, len(exps) + 1)])
    sensors = pd.DataFrame([
        {"sensor_id": "MOESA_ACC_01", "sensor_type": "acceleration", "device": "Syscom MR3000C / LIS2DW12 (열화본)",
         "source_sensor": "4 (이동, S5)", "sensor_location": "2경간 3/4L (L/4 대칭 위치)", "x_m": 53.75, "x_ratio": 0.75, "y_m": 3.0},
        {"sensor_id": "MOESA_ACC_02", "sensor_type": "acceleration", "device": "Syscom MR3000C / LIS2DW12 (열화본)",
         "source_sensor": "2 (고정)", "sensor_location": "2경간 L/2", "x_m": 45.5, "x_ratio": 0.50, "y_m": -3.0},
        {"sensor_id": "MOESA_ACC_03", "sensor_type": "acceleration", "device": "Syscom MR3000C / LIS2DW12 (열화본)",
         "source_sensor": "3 (고정)", "sensor_location": "3경간 약 0.57L (세션 차이용)", "x_m": 78.5, "x_ratio": 0.57, "y_m": 3.0},
    ]).assign(sample_rate_hz=fs, unit="m/s^2", axes="X,Y,Z")
    write_dataset("ponte_moesa", records, exps, sensors)
    print(exps.groupby(["damage_stage", "split", "label_status"]).agg(실험=("experiment_id", "size"), Window=("n_windows", "sum"))
          .to_string())


# ---------------------------------------------------------------------------
# Old ADA
# ---------------------------------------------------------------------------
OLD_ADA_STAGE = {"INT": "INT", "DMG1": "DMG1", "DMG2": "DMG2", "RCV": "RCV", "DMG3": "DMG3"}


def convert_old_ada():
    import io
    import zipfile

    import scipy.io as sio

    fs = OLD_ADA["sample_rate_hz"]
    with zipfile.ZipFile(ORIGINAL_DIR / OLD_ADA["zip"]) as z:
        name = next(n for n in z.namelist() if n.endswith(".mat"))
        vba = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)["VBA"]

    runs = []
    for r in range(1, vba.shape[0]):
        case, speed = str(vba[r, 0]).split()                # 예: "INT 40km"
        for c in range(1, vba.shape[1]):
            x = vba[r, c]
            if x.size:
                runs.append((case, int(speed.removesuffix("km")), c, x))
    rec_ids = neutral_ids("OLDADA_R", len(runs), salt=301)

    records, rows = [], []
    a0, a1 = OLD_ADA["free_decay_s"]
    for rid, (case, speed, run_no, x) in zip(rec_ids, runs):
        t = x[:, 0]
        t_exit = float(t[np.argmax(x[:, OLD_ADA["optical_exit_col"]] > 0.5)])
        base_us = date_to_us("2012-01-01") + int(rid.split("_")[-1]) * 3_600_000_000   # 측정 시각 정보 없음 → 주행마다 1시간 간격 가상 시각
        t_us = base_us + np.round(t * 1e6).astype(np.int64)
        signals = {(sid, "Z"): x[:, 1 + col].astype(float) for sid, col in OLD_ADA["sensors"].items()}
        records.append((rid, signals, t_us, fs))
        if case == "INT" and speed == 40:
            split = next(s for s, nos in OLD_ADA["int40_split"].items() if run_no in nos)
        else:
            split = "test"
        status = ("속도 영향 확인용 (참고)" if case == "INT" and speed != 40 else
                  "참고 (회복 불완전)" if case == "RCV" else "확실")
        rows.append({
            "record_id": rid, "source_record": f"VBA {case} {speed}km #{run_no}", "file": f"{rid}.parquet",
            "start_us": int(base_us + round((t_exit + a0) * 1e6)), "end_us": int(base_us + round((t_exit + a1) * 1e6)),
            "n_windows": 1, "split": split, "cv_group": "", "bridge_type": "TRUSS",
            "damage_type": "D0" if case == "INT" else "DMG", "severity_level": "",
            "damage_stage": OLD_ADA_STAGE[case], "label_status": status,
            "condition": f"VBA_{speed}", "speed_kmh": round(59.2 / t_exit * 3.6, 1), "load_position": "",
            "temperature_c": "", "train_passing": "", "wind_range": "", "date": "",
        })
    exps = pd.DataFrame(rows).sort_values("record_id", ignore_index=True)
    exps.insert(0, "experiment_id", [f"OLDADA_{i:04d}" for i in range(1, len(exps) + 1)])
    sensors = pd.DataFrame([
        {"sensor_id": "OLDADA_ACC_01", "source_sensor": "A1 (진동 2열)", "sensor_location": "L/4", "x_m": 14.8, "x_ratio": 0.25},
        {"sensor_id": "OLDADA_ACC_02", "source_sensor": "A3 (진동 4열)", "sensor_location": "L/2", "x_m": 29.6, "x_ratio": 0.50},
    ]).assign(sensor_type="acceleration", device="Tokyo Sokki ARS-A / LIS2DW12 (열화본)", y_m="",
              sample_rate_hz=fs, unit="m/s^2 (추정)", axes="Z")
    write_dataset("old_ada", records, exps, sensors)
    print(exps.groupby(["condition", "damage_stage", "split"]).agg(실험=("experiment_id", "size")).to_string())


CONVERTERS = {"hbta": convert_hbta, "ponte_moesa": convert_moesa, "old_ada": convert_old_ada}


def main():
    names = sys.argv[1:] or [d for d, folder in DATASETS.items() if (ORIGINAL_DIR / folder).exists()]
    for name in names:
        if name not in CONVERTERS:
            raise SystemExit(f"'{name}' 변환기가 없습니다. 가능한 값: {list(CONVERTERS)}")
        CONVERTERS[name]()


if __name__ == "__main__":
    main()
