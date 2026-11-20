"""
Raw 센서 데이터 → Window 단위 특징 추출 (3축 가속도 + 하중 + 온도).

처리 순서 (실험 1개)
  3. 데이터 로딩    load_experiment       : 파일 읽기, 정답 열(damage_type, severity_level) 즉시 제거
  4. 전처리         apply_quality_policy  : quality_flag 처리 (SAT·DROP·CAL → 결측)
                    to_uniform_grid       : 센서·축(채널)마다 자기 측정 속도 그대로 시간 격자에 배치
                    fill_short_gaps       : 짧은 결측만 선형 보간 (긴 결측이 있는 Window 는 제외)
                    apply_filter          : (선택) 노이즈 필터. config.FILTER["enabled"]
  5. Window 생성    tap 모드   : 수직 가속도에서 타격 순간을 찾아 '타격 1번 = 구간 1개' (감쇠비 계산 가능)
                    fixed 모드 : 실험 시작부터 일정 길이·겹침으로 자름
  6. 특징 추출      가속도 (센서·축마다) : mean, rms, std, max, min, p2p, crest, dom_freq, band energy, (Z축) damping
                    센서 간 비교         : config 의 cross_features (예: L/4÷L/2 수직 진폭 비, 측면 센서 비틀림 지표)
                    하중·온도            : 실험 조건 (모델 입력 아님). 조건 열 condition = 하중 크기 × 하중 위치

센서마다 측정 속도가 다르므로 하나의 속도로 강제 변환하지 않는다.
각 채널은 자기 속도를 유지한 채 timestamp_us 기준으로 같은 시간 구간(Window)에 맞춘다.

입력: data/raw/<source>/<bridge>/        (실험별 parquet 또는 csv + experiments.csv + sensors.csv)
출력: data/processed/<source>/<bridge>/features.parquet
      한 행 = 실험 1개 × Window 1개
      특징 열 이름: 가속도 <센서 약칭>_<특징>_<축>  (예: acc02_dom_freq_z)
                    하중·온도 <센서 약칭>_<특징> (예: load01_mean, temp01_mean), 센서 간 비교 cross_<이름>

데이터 누수 방지
- 센서 파일의 damage_type, severity_level 열은 읽자마자 버린다.
- experiments.csv 에서는 실험 ID·파일명·시작·끝 시각·하중 위치(실험 조건)만 읽는다.
- experiment_id 문자열(정답 포함)은 행을 구분하는 용도로만 쓰고 특징으로 쓰지 않는다.

공개 데이터 교량 (config BRIDGES 에 "public" 항목이 있는 교량, 합의서 v3)
  - 조건(condition)·온도는 센서가 아니라 experiments.csv 의 실험 기록에서 읽는다
  - Window·대역 필터(2~25Hz)·Band 구간·감쇠비 사용 여부는 교량별 설정
  - 블록 경계의 필터 왜곡을 막으려고 실험 앞뒤 10초를 더 읽어 필터한 뒤 잘라 쓴다
  - 실험에 없는 센서(예: S5 가 아닌 측정의 이동 센서)는 Window 를 버리지 않고 그 센서 특징만 빈 값
  - 판정 보류: 채널별 2~25Hz RMS 가 센서 잡음 RMS 의 √2배 미만이면 보류
      public_deg: LIS2DW12 잡음 기준 (hold_flag_<쌍>), public_raw: 원본 센서 잡음 기준 (raw_hold_flag_<쌍>, 사양이 있을 때만)
      열화본에서 주 지표 축이 아닌 채널이 보류되면 그 채널 특징은 빈 값
  - AI② 위험 등급 지표 (AI① 모델 입력 아님): tac_<쌍>_<축> (1-TAC), <센서>_ar<k>_<축> (AR 계수)
      1-TAC 기준 전달함수는 AI① train 실험의 Window 로만 계산

실행: python src/preprocess.py                              ← 특징 추출 (config 설정 사용)
      python src/preprocess.py --compare BOX_D0_NORMAL_002   ← 원본 vs 필터 결과 비교
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from config import (ACTIVE_BRIDGE, BRIDGES, DATA_SOURCE, EXPERIMENT, FEATURES, FILTER, GRAVITY, LABEL_COLUMNS,
                    LIS2DW12, PREPROCESS, PROCESSED_DIR, QUALITY, RAW_DIR, RESULTS_DIR, WINDOW)

IN_DIR = RAW_DIR / ACTIVE_BRIDGE
OUT_DIR = PROCESSED_DIR / ACTIVE_BRIDGE
PUBLIC = BRIDGES[ACTIVE_BRIDGE].get("public")   # 공개 데이터 교량 설정 (HW 교량은 None)
PAD_S = 10.0 if PUBLIC else 0.0                 # 필터 경계 왜곡 방지용 앞뒤 여유 [초]


# ---------------------------------------------------------------------------
# 3. 데이터 로딩
# ---------------------------------------------------------------------------
def load_experiment(file_name):
    """실험 1개의 센서 데이터를 읽는다 (parquet / csv). 정답 열은 바로 제거.
    file_name 이 파일명이면 data/raw/<source>/<bridge>/ 에서, 전체 경로면 그 위치에서 읽는다."""
    path = Path(file_name)
    if not path.is_absolute():
        path = IN_DIR / file_name
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        # CSV: axis 의 "NA"(1축 센서)를 결측으로 읽지 않도록, 빈 칸만 결측(NaN)으로 처리
        df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"axis": str, "quality_flag": str})
    return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])


def sensor_short_name(sensor_id):
    """BOX_ACC_01 → acc01, LOAD_01 → load01 (교량 접두어 제거). 특징 열 이름에 사용."""
    parts = sensor_id.split("_")
    if len(parts) >= 3:   # 교량·데이터셋 접두어 제거 (BOX_ACC_01, HBTA_ACC_01 → acc01)
        parts = parts[1:]
    return "".join(parts).lower()


def channels(sensors):
    """처리할 신호 목록: [(sensor_id, sensor_type, axis, sample_rate_hz)]. 3축 센서는 축마다 1개."""
    out = []
    for s in sensors:
        for axis in s["axes"]:
            out.append((s["sensor_id"], s["sensor_type"], axis, s["sample_rate_hz"]))
    return out


# ---------------------------------------------------------------------------
# 4. 전처리
# ---------------------------------------------------------------------------
def apply_quality_policy(df):
    """quality_flag 처리. 'drop' 인 플래그의 값은 결측(NaN)으로 바꾼다 (행은 남겨 결측 비율 계산에 사용)."""
    df = df.copy()
    counts = df["quality_flag"].value_counts().to_dict()
    for flag in QUALITY["flags"]:
        if flag in QUALITY["use"]:
            continue
        if PREPROCESS["quality_policy"].get(flag, "drop") == "drop":
            df.loc[df["quality_flag"] == flag, "value"] = np.nan
    return df, counts


def to_uniform_grid(g, sample_rate, start_us, duration_s):
    """한 채널의 데이터를 자기 측정 속도의 시간 격자에 배치. 빠진 칸은 NaN."""
    n = int(round(duration_s * sample_rate))
    grid = np.full(n, np.nan)
    idx = np.round((g["timestamp_us"].to_numpy() - start_us) * sample_rate / 1e6).astype(int)
    ok = (idx >= 0) & (idx < n)
    grid[idx[ok]] = g["value"].to_numpy(dtype=float)[ok]
    return grid


def fill_short_gaps(v, sample_rate):
    """길이가 max_gap_s 이하인 내부 결측만 선형 보간. 긴 결측과 앞뒤 끝의 결측은 NaN 으로 남긴다."""
    max_gap = int(PREPROCESS["max_gap_s"] * sample_rate)
    s = pd.Series(v)
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("size")
    filled = s.interpolate(limit_area="inside")
    return np.where(isna & (run_len > max_gap), np.nan, filled.to_numpy())


def filter_spec(sensor_type):
    """(필터 종류, 차단 주파수). 공개 데이터 교량의 가속도는 교량 설정 대역(2~25Hz) 대역 통과."""
    if PUBLIC and sensor_type == "acceleration":
        return "bandpass", PUBLIC["filter_band_hz"]
    return FILTER.get(sensor_type)


def apply_filter(v, sensor_type, sample_rate):
    """Butterworth 필터 (양방향). 결측은 잠시 보간해 필터를 적용한 뒤 다시 NaN 으로 되돌린다."""
    spec = filter_spec(sensor_type)
    if spec is None:
        return v
    kind, cutoff = spec
    nyq = sample_rate / 2
    cutoff = (cutoff[0], min(cutoff[1], 0.95 * nyq)) if kind == "bandpass" else min(cutoff, 0.95 * nyq)
    sos = signal.butter(FILTER["order"], cutoff, btype=kind, fs=sample_rate, output="sos")
    nan = np.isnan(v)
    if nan.all():
        return v
    out = signal.sosfiltfilt(sos, pd.Series(v).interpolate(limit_direction="both").to_numpy())
    out[nan] = np.nan
    return out


# ---------------------------------------------------------------------------
# 5. Window 생성
# ---------------------------------------------------------------------------
def window_setting():
    return PUBLIC["window"] if PUBLIC else WINDOW


def fixed_windows(duration_s):
    """[(시작초, 끝초), ...]. 실험 시간 안에 완전히 들어가는 Window 만."""
    w = window_setting()
    length = w["length_s"]
    step = length * (1 - w["overlap"])
    starts = np.arange(0, duration_s - length + 1e-6, step)
    return [(s, s + length) for s in starts]


def detect_taps(acc_z_signals, fs):
    """수직 가속도에서 타격 순간 [초] 를 찾는다.
    평균을 뺀 |가속도| 를 센서끼리 더한 뒤, 평소 흔들림(중앙값)의 tap_threshold 배를 넘는 봉우리를 타격으로 본다."""
    r = np.zeros(len(acc_z_signals[0]))
    for z in acc_z_signals:
        r += np.nan_to_num(np.abs(z - np.nanmedian(z)))
    level = np.median(r) + 1e-12
    peaks, _ = signal.find_peaks(r, height=WINDOW["tap_threshold"] * level, distance=int(WINDOW["tap_min_gap_s"] * fs))
    return peaks / fs


def tap_windows(taps_s, duration_s):
    starts = [t - WINDOW["tap_pre_s"] for t in taps_s]
    return [(s, s + WINDOW["length_s"]) for s in starts if s >= 0 and s + WINDOW["length_s"] <= duration_s]


# ---------------------------------------------------------------------------
# 6. 특징 추출
# ---------------------------------------------------------------------------
def spectrum(x, fs):
    """detrend + Hann 창 → FFT 크기 스펙트럼."""
    x = signal.detrend(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / fs), spec


def dominant_freq(freqs, spec):
    """가장 강하게 흔들린 진동수 [Hz]. 주파수 칸 사이 값은 포물선 보간으로 보정."""
    spec = np.where(freqs < FEATURES["min_freq_hz"], 0, spec)
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return freqs[k] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0])
    return freqs[k]


def damping_ratio(x, fs, f_dom):
    """타격 직후 자유진동이 줄어드는 속도로 감쇠비를 추정 (구할 수 없으면 NaN).
    주 진동수 근처만 남긴 신호의 포락선(진폭 윤곽)이 exp(-ζ·2πf·t) 로 줄어든다고 보고 log 기울기로 ζ 를 구한다."""
    if not FEATURES["min_freq_hz"] < f_dom < 0.4 * fs:
        return np.nan
    sos = signal.butter(2, (0.7 * f_dom, min(1.3 * f_dom, 0.95 * fs / 2)), btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x - x.mean())))
    start = int(np.argmax(env[: len(env) // 3]))           # 구간 앞부분의 최대 진폭 = 타격 직후
    end = start + int(np.argmax(env[start:] < 0.2 * env[start])) if (env[start:] < 0.2 * env[start]).any() else len(env)
    if end - start < max(10, int(FEATURES["damping_min_cycles"] * fs / f_dom)):
        return np.nan
    t = np.arange(end - start) / fs
    slope = np.polyfit(t, np.log(env[start:end] + 1e-12), 1)[0]
    zeta = -slope / (2 * np.pi * f_dom)
    return float(zeta) if 0 < zeta < 0.5 else np.nan


def acceleration_features(v, fs, with_damping):
    """가속도 한 축. mean 은 원본(중력·영점 포함), 나머지 크기 특징은 평균을 뺀 '흔들림 성분'으로 계산."""
    x = v - v.mean()
    rms = np.sqrt(np.mean(x**2))
    freqs, spec = spectrum(v, fs)
    energy = spec**2
    total = energy[freqs >= FEATURES["min_freq_hz"]].sum()
    f_dom = dominant_freq(freqs, spec)
    feats = {
        "mean": v.mean(), "rms": rms, "std": v.std(), "max": x.max(), "min": x.min(),
        "p2p": x.max() - x.min(), "crest": np.abs(x).max() / rms if rms > 0 else 0.0, "dom_freq": f_dom,
    }
    for lo, hi in (PUBLIC["acc_bands_hz"] if PUBLIC else FEATURES["acc_bands_hz"]):
        band = (freqs >= lo) & (freqs < min(hi, fs / 2 + 1e-9))
        feats[f"band_{lo}_{hi}"] = energy[band].sum() / total if total > 0 else 0.0
    if with_damping:
        feats["damping"] = damping_ratio(v, fs, f_dom)
    return feats


# ---------------------------------------------------------------------------
# 6-1. 공개 데이터: AI② 위험 등급 지표 (1-TAC, AR) — AI① 이상탐지 모델 입력 아님
# ---------------------------------------------------------------------------
def transmissibility_spectra(a, b, fs):
    """1-TAC 용 교차·자기 스펙트럼 (Welch, Hann, 50% 겹침). 대역 안 값만 반환: (G_ba, G_bb)
    전달함수 T = G_ba / G_bb (b 기준 a 의 응답). G_ba = E[conj(X_b) X_a]"""
    tac = PUBLIC["tac"]
    nper = min(tac["nperseg"], len(a))
    f, g_ba = signal.csd(b, a, fs, nperseg=nper)
    _, g_bb = signal.welch(b, fs, nperseg=nper)
    lo, hi = tac["band_hz"]
    m = (f >= lo) & (f <= hi)
    return g_ba[m], g_bb[m]


def one_minus_tac(t_d, t_r):
    """1 − |T_dᴴ T_r|² / ((T_dᴴ T_d)(T_rᴴ T_r)). 0 = 기준과 같은 모양, 1 에 가까울수록 다름"""
    num = np.abs(np.vdot(t_d, t_r)) ** 2
    den = np.real(np.vdot(t_d, t_d)) * np.real(np.vdot(t_r, t_r))
    return float(1 - num / den) if den > 0 else np.nan


def ar_coefficients(x, order):
    """AR(order) 계수. 평균 제거, 절편 없음, 최소제곱 (statsmodels AutoReg(trend='n') 와 같은 OLS)"""
    x = x - x.mean()
    lagged = np.column_stack([x[order - k: len(x) - k] for k in range(1, order + 1)])
    coef, *_ = np.linalg.lstsq(lagged, x[order:], rcond=None)
    return coef


def noise_band_rms(density_ug_per_rthz, fs):
    """센서 잡음(백색, 밀도 [µg/√Hz])이 교량 대역 필터(양방향)를 지난 뒤의 RMS [m/s^2]"""
    lo, hi = PUBLIC["filter_band_hz"]
    sos = signal.butter(FILTER["order"], (lo, min(hi, 0.95 * fs / 2)), btype="bandpass", fs=fs, output="sos")
    f, h = signal.sosfreqz(sos, worN=200000, fs=fs)
    enbw = np.trapezoid(np.abs(h) ** 4, f)   # 양방향 필터 → 전력 이득 |H|^4
    lsb = 2 * LIS2DW12["range_g"] * GRAVITY / 2 ** LIS2DW12["bits"]
    quant = lsb / np.sqrt(12) * np.sqrt(enbw / (fs / 2)) if density_ug_per_rthz == LIS2DW12["noise_ug_per_rthz"] else 0.0
    return float(np.hypot(density_ug_per_rthz * 1e-6 * GRAVITY * np.sqrt(enbw), quant))


def hold_threshold(fs):
    """판정 보류 기준 [m/s^2] = 잡음 RMS × √2. public_deg 는 LIS2DW12, public_raw 는 원본 센서 사양 (없으면 None)"""
    if DATA_SOURCE == "public_deg":
        return np.sqrt(2) * noise_band_rms(LIS2DW12["noise_ug_per_rthz"], fs)
    if PUBLIC.get("raw_noise_ug_per_rthz"):
        return np.sqrt(2) * noise_band_rms(PUBLIC["raw_noise_ug_per_rthz"], fs)
    return None


def condition_features(v):
    """하중·온도: 구조 이상이 아니라 실험 조건. 참고 변수 및 정상 기준 구분에 사용."""
    return {"mean": v.mean(), "max": v.max(), "min": v.min(), "std": v.std()}


def nearest_load_level(load_n):
    """로드셀 측정값(N)에 가장 가까운 하중 단계 [kg] (config 의 loads_kg 중 하나)."""
    levels = np.asarray(EXPERIMENT["loads_kg"], dtype=float)
    return float(levels[np.argmin(np.abs(levels * GRAVITY - load_n))])


def condition_name(load_kg, position):
    """정상 기준을 나누는 조건 이름. 하중 0 kg 은 위치 구분 없음. 예: '0kg', '1kg_C', '2kg_E'"""
    if load_kg == 0 or not position:
        return f"{load_kg:g}kg"
    return f"{load_kg:g}kg_{position}"


# ---------------------------------------------------------------------------
# 실험 1개 처리
# ---------------------------------------------------------------------------
def prepare_signals(exp, sensors, use_filter):
    """채널(센서·축)마다 (원본 결측 여부, 전처리된 신호) 를 만든다."""
    # exp["raw"]: 메모리에 있는 raw (API 로 받은 데이터). 없으면 exp["file"] 에서 읽는다
    df = exp["raw"].drop(columns=[c for c in LABEL_COLUMNS if c in exp["raw"].columns]) if "raw" in exp else load_experiment(exp["file"])
    df = df.sort_values(["sensor_id", "axis", "timestamp_us"])
    df, flag_counts = apply_quality_policy(df)
    duration_s = (exp["end_us"] - exp["start_us"]) / 1e6
    # 공개 데이터: 앞뒤 PAD_S 초를 더 읽어 필터한 뒤 잘라 쓴다 (블록 경계의 필터 왜곡 방지)
    grid_start_us = exp["start_us"] - int(PAD_S * 1e6)
    signals = {}
    for sid, stype, axis, sr in channels(sensors):
        g = df[(df["sensor_id"] == sid) & (df["axis"].astype(str) == axis)]
        if PUBLIC and g.empty:   # 이 실험에 없는 센서 → 특징만 빈 값
            continue
        raw = to_uniform_grid(g, sr, grid_start_us, duration_s + 2 * PAD_S)
        v = fill_short_gaps(raw, sr)
        if use_filter:
            v = apply_filter(v, stype, sr)
        cut = slice(int(round(PAD_S * sr)), int(round((PAD_S + duration_s) * sr)))
        signals[(sid, axis)] = {"raw_missing": np.isnan(raw[cut]), "value": v[cut], "type": stype, "fs": sr}
    return signals, duration_s, flag_counts


def process_experiment(exp, sensors, use_filter):
    """반환: (Window 별 특징 행 목록, 제외한 Window 수, quality_flag 개수)"""
    signals, duration_s, flag_counts = prepare_signals(exp, sensors, use_filter)
    acc_z = [(k, v) for k, v in signals.items() if v["type"] == "acceleration" and k[1] == "Z"]
    if not PUBLIC and WINDOW["mode"] == "tap":
        windows = tap_windows(detect_taps([v["value"] for _, v in acc_z], acc_z[0][1]["fs"]), duration_s)
    else:
        windows = fixed_windows(duration_s)
    with_damping = PUBLIC["use_damping"] if PUBLIC else (FEATURES["use_damping"] and WINDOW["mode"] == "tap")
    cross = BRIDGES[ACTIVE_BRIDGE].get("cross_features", [])

    rows, dropped = [], 0
    for w, (ws, we) in enumerate(windows):
        row = {
            "experiment_id": exp["experiment_id"],
            "bridge_type": exp["bridge_type"],
            "window_id": f"W{w + 1:03d}",
            "window_start_us": int(exp["start_us"] + ws * 1e6),
            "window_end_us": int(exp["start_us"] + we * 1e6),
        }
        ok, seg = True, {}
        for (sid, axis), sig in signals.items():
            fs = sig["fs"]
            prefix = sensor_short_name(sid)
            if sig["type"] == "temperature":
                # 온도는 10초에 1번 측정 → 구간 중앙 시각의 값을 앞뒤 측정값으로 보간
                t = np.arange(len(sig["value"])) / fs
                valid = ~np.isnan(sig["value"])
                if not valid.any():
                    ok = False
                    break
                row[f"{prefix}_mean"] = float(np.interp((ws + we) / 2, t[valid], sig["value"][valid]))
                continue
            i0, i1 = int(round(ws * fs)), int(round(we * fs))
            v = sig["value"][i0:i1]
            if sig["raw_missing"][i0:i1].mean() > WINDOW["max_missing_ratio"] or np.isnan(v).any():
                ok = False
                break
            seg[(sid, axis)] = v
            if sig["type"] == "acceleration":
                feats = acceleration_features(v, fs, with_damping and axis == FEATURES["damping_axis"])
                row.update({f"{prefix}_{k}_{axis.lower()}": val for k, val in feats.items()})
            else:
                row.update({f"{prefix}_{k}": val for k, val in condition_features(v).items()})
        if not ok:
            dropped += 1
            continue
        for name, num, den in cross:
            if num not in seg or den not in seg:   # 공개 데이터: 이 실험에 없는 센서
                row[f"cross_{name}"] = np.nan
                continue
            a, b = seg[num] - seg[num].mean(), seg[den] - seg[den].mean()
            row[f"cross_{name}"] = np.sqrt(np.mean(a**2)) / (np.sqrt(np.mean(b**2)) + 1e-12)
        if PUBLIC:
            # 공개 데이터는 감쇠비 등을 못 구해도 Window 를 남기고 빈 값으로 둔다. 조건·온도는 실험 기록에서
            row.update(public_indicators(seg, fs_of(signals)))
            row["_w"] = w
            row["load_level_kg"] = np.nan
            row["load_position"] = ""
            row["condition"] = exp.get("condition", "")
            row["temperature_c"] = pd.to_numeric(exp.get("temperature_c"), errors="coerce")
            rows.append(row)
            continue
        feat_vals = [v for k, v in row.items() if k not in ("experiment_id", "bridge_type", "window_id")]
        if any(isinstance(v, float) and np.isnan(v) for v in feat_vals):
            dropped += 1  # 감쇠비 등을 계산하지 못한 Window
            continue
        load_col = [k for k in row if k.startswith("load") and k.endswith("_mean")]
        row["load_level_kg"] = nearest_load_level(row[load_col[0]]) if load_col else 0.0
        row["load_position"] = exp.get("load_position", "") or ""
        row["condition"] = condition_name(row["load_level_kg"], row["load_position"])
        temp_col = [k for k in row if k.startswith("temp") and k.endswith("_mean")]
        row["temperature_c"] = row[temp_col[0]] if temp_col else np.nan
        rows.append(row)
    if PUBLIC and PUBLIC["tac"]["windows"] > 1:
        add_grouped_tac(rows, signals, windows)
    return rows, dropped, flag_counts


def fs_of(signals):
    return next(iter(signals.values()))["fs"] if signals else None


def pair_channels(pair, axis):
    a, b = PUBLIC["pairs"][pair]
    return (a, axis), (b, axis)


def public_indicators(seg, fs):
    """Window 1개의 AI② 지표: AR 계수, (Window 1개 단위일 때) 1-TAC 용 스펙트럼 (_spec_<쌍>_<축>, 나중에 1-TAC 로 바뀜)"""
    out = {}
    ar = PUBLIC.get("ar")
    if ar:
        for sid, axis in ar["channels"]:
            key = (sid, axis)
            coef = ar_coefficients(seg[key], ar["order"]) if key in seg else [np.nan] * ar["order"]
            out.update({f"{sensor_short_name(sid)}_ar{k}_{axis.lower()}": float(c) for k, c in enumerate(coef, start=1)})
    a, b = pair_channels(PUBLIC["main_pair"], PUBLIC["main_axis"])
    if a in seg and b in seg:   # AI① v2 모드 Feature 용 자기 스펙트럼 (나중에 모드 대역으로 요약)
        nper = min(PUBLIC["tac"]["nperseg"], len(seg[a]))
        f, p_aa = signal.welch(seg[a], fs, nperseg=nper)
        _, p_bb = signal.welch(seg[b], fs, nperseg=nper)
        out["_psd"] = (f, p_aa, p_bb)
    if PUBLIC["tac"]["windows"] == 1:
        for pair, axis in PUBLIC["tac"]["pairs"]:
            a, b = pair_channels(pair, axis)
            out[f"_spec_{pair}_{axis.lower()}"] = transmissibility_spectra(seg[a], seg[b], fs) if a in seg and b in seg else None
    return out


def add_grouped_tac(rows, signals, windows):
    """1-TAC 를 기본 Window k 개를 이어 붙인 구간으로 계산 (HBTA: 10.24초 × 3). 같은 스펙트럼을 묶음의 Window 에 함께 둔다."""
    k = PUBLIC["tac"]["windows"]
    by_w = {r["_w"]: r for r in rows}
    for g0 in range(0, len(windows) - k + 1, k):
        members = [w for w in range(g0, g0 + k) if w in by_w]
        ws, we = windows[g0][0], windows[g0 + k - 1][1]
        for pair, axis in PUBLIC["tac"]["pairs"]:
            a, b = pair_channels(pair, axis)
            spec = None
            if len(members) == k and a in signals and b in signals:
                fs = signals[a]["fs"]
                sa, sb = (signals[c]["value"][int(round(ws * fs)):int(round(we * fs))] for c in (a, b))
                if not (np.isnan(sa).any() or np.isnan(sb).any()):
                    spec = transmissibility_spectra(sa, sb, fs)
            for w in members:
                by_w[w][f"_spec_{pair}_{axis.lower()}"] = spec
                by_w[w]["tac_group"] = f"T{g0 // k + 1:03d}"


def read_inputs():
    sensors = BRIDGES[ACTIVE_BRIDGE]["sensors"]
    # 정답 열을 실수로 쓰지 않도록 필요한 열만 읽는다 (load_position 은 정답이 아니라 실험 조건)
    exps = pd.read_csv(IN_DIR / "experiments.csv", keep_default_na=False, na_values=[""])
    cols = ["experiment_id", "bridge_type", "file", "start_us", "end_us", "load_position"]
    if PUBLIC:   # 공개 데이터: 실험 기록의 조건·온도, 1-TAC 기준을 정할 분할(split)도 읽는다 (정답 아님)
        cols += ["condition", "temperature_c", "split"]
    exps = exps[[c for c in cols if c in exps]]
    if "load_position" in exps:
        exps["load_position"] = exps["load_position"].fillna("")
    return sensors, exps


V2_MODES = 3            # 모드 수: AI① train 평균 스펙트럼에서 돌출도가 큰 피크 3개
V2_BAND = 0.15          # 모드 대역 = 모드 진동수 ± 15%


def find_modes(f, p):
    """정상(AI① train) 평균 스펙트럼에서 모드 진동수. 2~25Hz 안 돌출도 상위 V2_MODES 개, 대역이 겹치면 큰 쪽만."""
    lo, hi = PUBLIC["filter_band_hz"]
    m = (f >= lo) & (f <= hi)
    pk, prop = signal.find_peaks(p[m], prominence=0)
    chosen = []
    for i in pk[np.argsort(prop["prominences"])[::-1]]:
        fm = float(f[m][i])
        if all(abs(fm - c) > V2_BAND * (fm + c) for c in chosen):
            chosen.append(fm)
        if len(chosen) == V2_MODES:
            break
    return sorted(chosen)


def add_v2_mode_features(feats, train):
    """AI① v2 Feature (진동 크기 대신 모양). 모드 대역은 AI① train Window 만으로 정한다 (손상 라벨 안 봄).
    v2_f<m>: 모드 대역 피크 진동수 [Hz] (포물선 보간), v2_shape<m>: √(acc01 대역 에너지 / acc02 대역 에너지),
    v2_e<m>_acc01·acc02: 모드 대역 에너지 / 2~25Hz 전체 에너지"""
    if "_psd" not in feats:
        return feats
    psd = [s if isinstance(s, tuple) else None for s in feats["_psd"]]
    ref = [s for s, t in zip(psd, train) if t and s is not None]
    if not ref:
        return feats
    f = ref[0][0]
    # 모드 대역은 원본(public_raw)의 AI① train 에서 정하고 저장. 열화본은 같은 대역을 쓴다 (두 버전을 같은 대역으로 비교)
    import json
    modes_file = PROCESSED_DIR.parent / "public_raw" / ACTIVE_BRIDGE / "v2_modes.json"
    if DATA_SOURCE == "public_deg":
        if not modes_file.exists():
            raise SystemExit(f"{modes_file} 가 없습니다. 원본(IOT_SOURCE=public_raw) 전처리를 먼저 실행하세요.")
        modes = json.loads(modes_file.read_text(encoding="utf-8"))["modes_hz"]
    else:
        modes = find_modes(f, np.mean([s[1] + s[2] for s in ref], axis=0))
        modes_file.parent.mkdir(parents=True, exist_ok=True)
        modes_file.write_text(json.dumps({"modes_hz": modes, "source": "public_raw AI① train 평균 스펙트럼",
                                          "band": f"±{V2_BAND:.0%}"}, ensure_ascii=False), encoding="utf-8")
    lo, hi = PUBLIC["filter_band_hz"]
    full = (f >= lo) & (f <= hi)
    cols = {}
    for k, fm in enumerate(modes, start=1):
        band = (f >= fm * (1 - V2_BAND)) & (f <= fm * (1 + V2_BAND))
        fb = f[band]
        out = {f"v2_f{k}": [], f"v2_shape{k}": [], f"v2_e{k}_acc01": [], f"v2_e{k}_acc02": []}
        for s in psd:
            if s is None:
                for c in out:
                    out[c].append(np.nan)
                continue
            _, paa, pbb = s
            tot = paa[band] + pbb[band]
            j = int(np.argmax(tot))
            peak = fb[j]
            if 0 < j < len(tot) - 1:
                y0, y1, y2 = tot[j - 1], tot[j], tot[j + 1]
                den = y0 - 2 * y1 + y2
                peak = fb[j] + (0.5 * (y0 - y2) / den * (fb[1] - fb[0]) if den != 0 else 0.0)
            out[f"v2_f{k}"].append(float(peak))
            out[f"v2_shape{k}"].append(float(np.sqrt(paa[band].sum() / pbb[band].sum())) if pbb[band].sum() > 0 else np.nan)
            out[f"v2_e{k}_acc01"].append(float(paa[band].sum() / paa[full].sum()) if paa[full].sum() > 0 else np.nan)
            out[f"v2_e{k}_acc02"].append(float(pbb[band].sum() / pbb[full].sum()) if pbb[full].sum() > 0 else np.nan)
        cols.update(out)
    print("AI① v2 모드 진동수 (원본 AI① train 평균 스펙트럼): " + ", ".join(f"{fm:.2f}Hz" for fm in modes))
    return pd.concat([feats, pd.DataFrame(cols, index=feats.index)], axis=1)


def finish_public(feats, exps, sensors):
    """공개 데이터 후처리: 1-TAC (기준 = AI① train Window), 판정 보류 표시, 열화본 보조 채널 빈 값 처리"""
    split = feats["experiment_id"].map(exps.set_index("experiment_id")["split"])
    train = (split == "train").to_numpy()
    for col in [c for c in feats.columns if c.startswith("_spec_")]:
        specs = [s if isinstance(s, tuple) else None for s in feats[col]]   # 묶음에 안 든 Window 는 빈 값
        ref = [s for s, t in zip(specs, train) if t and s is not None]
        out = f"tac_{col.removeprefix('_spec_')}"
        if not ref:
            feats[out] = np.nan
            continue
        t_r = np.mean([s[0] for s in ref], axis=0) / np.mean([s[1] for s in ref], axis=0)
        feats[out] = [one_minus_tac(s[0] / s[1], t_r) if s is not None else np.nan for s in specs]
    feats = add_v2_mode_features(feats, train)
    feats = feats.drop(columns=[c for c in feats.columns if c.startswith("_spec_") or c in ("_w", "_psd")])

    fs = sensors[0]["sample_rate_hz"]
    thr = hold_threshold(fs)
    deg = DATA_SOURCE == "public_deg"
    main_axis = PUBLIC["main_axis"].lower()
    main_sensors = {sensor_short_name(s) for s in PUBLIC["pairs"][PUBLIC["main_pair"]]}
    if thr is not None:
        for s in sensors:
            p = sensor_short_name(s["sensor_id"])
            for axis in s["axes"]:
                rms = feats.get(f"{p}_rms_{axis.lower()}")
                if rms is None:
                    continue
                held = (rms < thr).fillna(False)
                feats[f"hold_{p}_{axis.lower()}"] = held.astype("boolean").mask(rms.isna())   # 센서 없음 = 빈 값
                if deg and not (p in main_sensors and axis.lower() == main_axis):
                    # 열화본: 주 지표 축이 아닌 채널이 보류되면 그 채널 특징과 그 채널을 쓰는 지표를 빈 값으로
                    cols = [c for c in feats.columns if c.startswith(f"{p}_") and c.endswith(f"_{axis.lower()}")]
                    cols += [f"cross_{name}" for name, num, den in BRIDGES[ACTIVE_BRIDGE].get("cross_features", [])
                             if (s["sensor_id"], axis) in (num, den) and f"cross_{name}" in feats]
                    cols += [f"tac_{pair}_{ax.lower()}" for pair, ax in PUBLIC["tac"]["pairs"]
                             if ax == axis and s["sensor_id"] in PUBLIC["pairs"][pair] and f"tac_{pair}_{ax.lower()}" in feats]
                    feats.loc[held, cols] = np.nan
        flag = "hold_flag_" if deg else "raw_hold_flag_"
        for pair, (a, b) in PUBLIC["pairs"].items():
            ca, cb = (f"hold_{sensor_short_name(x)}_{main_axis}" for x in (a, b))
            if ca in feats and cb in feats:
                # 두 센서 중 하나라도 보류면 보류. 한쪽 센서가 없는 실험은 빈 값
                feats[f"{flag}{pair}"] = (feats[ca] | feats[cb]).mask(feats[ca].isna() | feats[cb].isna())
    # AI① 판정 보류: 열화본에서 주 지표 쌍이 보류된(또는 센서가 없는) Window. 원본은 보류하지 않음
    main_flag = f"hold_flag_{PUBLIC['main_pair']}"
    feats["hold_main"] = feats[main_flag].fillna(True).astype(bool) if deg and main_flag in feats else False
    print(f"판정 보류 기준: {thr:.6f} m/s^2 ({'LIS2DW12' if deg else '원본 센서'} 잡음 RMS × √2)" if thr else
          "판정 보류 기준: 없음 (원본, 센서 잡음 사양 없음)")
    return feats


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
META_COLS = ["experiment_id", "bridge_type", "window_id", "window_start_us", "window_start", "window_end_us",
             "load_level_kg", "load_position", "condition", "temperature_c", "tac_group", "hold_main"]


def run_all():
    sensors, exps = read_inputs()
    all_rows, total_dropped, total_flags = [], 0, {}
    use_filter = bool(PUBLIC) or FILTER["enabled"]   # 공개 데이터는 교량 대역 필터를 항상 적용 (합의서 v3)
    for exp in exps.to_dict("records"):
        rows, dropped, flags = process_experiment(exp, sensors, use_filter)
        all_rows += rows
        total_dropped += dropped
        for k, v in flags.items():
            total_flags[k] = total_flags.get(k, 0) + v
    feats = pd.DataFrame(all_rows)
    if PUBLIC:
        feats = finish_public(feats, exps, sensors)
    feats.insert(4, "window_start", pd.to_datetime(feats["window_start_us"], unit="us"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "features.parquet"
    feats.to_parquet(path, index=False)

    w = window_setting()
    mode = (f"타격 기준 (타격 1번 = {w['length_s']}초 구간 1개)" if w["mode"] == "tap"
            else f"고정 {w['length_s']}초, 겹침 {w['overlap']:.0%}")
    print(f"[{ACTIVE_BRIDGE} / {DATA_SOURCE}] Window: {mode} / 필터 {'ON' if use_filter else 'OFF'}")
    print(f"실험 {len(exps)}개 → Window {len(feats) + total_dropped}개 중 {len(feats)}개 사용, {total_dropped}개 제외 (결측·감쇠비 계산 실패)")
    print(f"실험당 Window 수: 평균 {feats.groupby('experiment_id').size().mean():.1f}개")
    print("quality_flag 전체 개수:", ", ".join(f"{k}={v:,}" for k, v in sorted(total_flags.items())))
    feat_cols = [c for c in feats.columns if c not in META_COLS]
    print(f"특징 {len(feat_cols)}개 → {path}")
    groups = {}
    for c in feat_cols:
        prefix, rest = c.split("_", 1)
        groups.setdefault(prefix, []).append(rest)
    for prefix, names in groups.items():
        print(f"  {prefix:<7} ({len(names):>2}) {', '.join(names[:14])}{' ...' if len(names) > 14 else ''}")
    print("조건별 Window 수:", feats["condition"].value_counts().sort_index().to_dict())


def compare_filter(experiment_id):
    """한 실험에서 필터 OFF(원본) / ON 의 특징을 비교해 출력·저장."""
    sensors, exps = read_inputs()
    exp = exps.set_index("experiment_id").loc[experiment_id].to_dict() | {"experiment_id": experiment_id}
    raw, flt = (pd.DataFrame(process_experiment(exp, sensors, use_filter=f)[0]) for f in (False, True))
    cols = [c for c in raw.columns if any(c.split("_", 1)[-1].startswith(k) for k in ("rms", "dom_freq", "damping", "mean"))]
    cols = [c for c in cols if c in flt.columns and c not in META_COLS]
    table = pd.DataFrame({"원본 평균": raw[cols].mean(), "필터 평균": flt[cols].mean()})
    table["변화율"] = (table["필터 평균"] - table["원본 평균"]) / table["원본 평균"].abs().replace(0, np.nan)
    out = RESULTS_DIR / ACTIVE_BRIDGE / f"filter_compare_{experiment_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, encoding="utf-8-sig")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"[{experiment_id}] 원본 vs 필터 (Window 평균)")
    print(table.to_string())
    print(f"\n저장: {out}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        compare_filter(sys.argv[2])
    else:
        run_all()
