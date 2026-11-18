"""
정상(D0) 데이터로 교량별 Isolation Forest 학습.

  7. 데이터 분할   split_experiments : experiment_id 단위로 train / val / test 결정
  8. 학습          fit_baseline      : train D0 로 '하중 조건(크기 × 위치)별 정상 기준' (특징별 평균·표준편차)
                                       + 온도 보정 (온도 1°C 당 특징 변화량)
                   to_zscore         : 모든 특징을 정상 기준 대비 z-score 로 변환
                                       → 하중·온도 때문에 달라진 값은 0 근처, 구조 이상으로 달라진 값만 커짐
                   IsolationForest   : train z-score 로 학습
                   임계값            : val D0 의 점수 분포로 판정 기준 결정 (IF 점수, 보조 규칙 |z|)

입력: data/processed/<source>/<bridge>/features.parquet
      data/raw/<source>/<bridge>/experiments.csv  (분할용 split 열, 안전 확인용 damage_type)
출력: models/<source>/<bridge>/isolation_forest.joblib   (모델 + 정상 기준 + 임계값 + 사용 특징 목록)
      models/<source>/<bridge>/split.csv                  (실험별 분할 결과. AI② 와 공유용)

데이터 누수 방지
- 정상 기준·모델은 train 만, 임계값은 val 만 사용. test 는 여기서 쓰지 않는다.
- damage_type 은 'train/val 이 전부 D0 인지' 확인하는 데만 쓰고 모델 입력에는 넣지 않는다.

실행: python src/train.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (ACTIVE_BRIDGE, BASELINE, BRIDGES, FEATURES, GROUND_TRUTH, IFOREST, MODEL_INPUT, MODELS_DIR,
                    PREPROCESS, PROCESSED_DIR, RAW_DIR, SPLIT, WINDOW, AI1_MODEL, MODEL_SUFFIX)
from preprocess import META_COLS as FEATURE_META_COLS
from preprocess import PUBLIC, sensor_short_name, window_setting

FEAT_PATH = PROCESSED_DIR / ACTIVE_BRIDGE / "features.parquet"
MODEL_DIR = MODELS_DIR / ACTIVE_BRIDGE
MODEL_PATH = MODEL_DIR / f"isolation_forest{MODEL_SUFFIX}.joblib"

# 특징이 아닌 정보 열 (조건 열 condition·temperature_c 포함)
META_COLS = FEATURE_META_COLS + ["split"]
AXIS_SUFFIXES = ("x", "y", "z")


def feature_name(col):
    """열 이름에서 센서 약칭과 축을 뗀 특징 이름. acc01_dom_freq_z → dom_freq, load01_mean → mean"""
    rest = col.split("_", 1)[1]
    head, _, tail = rest.rpartition("_")
    return head if tail in AXIS_SUFFIXES and head else rest


# ---------------------------------------------------------------------------
# 7. 데이터 분할 (experiment_id 단위)
# ---------------------------------------------------------------------------
def split_experiments(exps):
    """experiment_id → split. 같은 실험의 모든 Window 는 항상 같은 split 에 들어간다."""
    normal = exps["damage_type"] == GROUND_TRUTH["normal_code"]
    if SPLIT["method"] == "column":
        split = exps.set_index("experiment_id")["split"]
    else:
        rng = np.random.default_rng(SPLIT["random_state"])
        ids = rng.permutation(exps.loc[normal, "experiment_id"].to_numpy())
        n_train = int(round(len(ids) * SPLIT["ratios"]["train"]))
        n_val = int(round(len(ids) * SPLIT["ratios"]["val"]))
        split = pd.Series("test", index=exps["experiment_id"])
        split[ids[:n_train]] = "train"
        split[ids[n_train:n_train + n_val]] = "val"

    # 안전장치: train/val 에 이상(D1~D3) 실험이 섞이면 학습을 멈춘다
    bad = exps.loc[split.reindex(exps["experiment_id"]).isin(["train", "val"]).to_numpy() & ~normal.to_numpy(),
                   "experiment_id"]
    if len(bad):
        raise SystemExit(f"train/val 에 정상(D0)이 아닌 실험이 있습니다: {list(bad)}")
    return split


# ---------------------------------------------------------------------------
# 8. 학습 — predict.py 에서도 같은 함수를 쓴다
# ---------------------------------------------------------------------------
def model_feature_columns(feats, sensors):
    """모델 입력 특징 열. 하중·온도 센서 특징, 제외 특징 이름(acc mean 등)은 뺀다. 센서 간 비교(cross_)는 포함.
    공개 데이터: 주 지표 쌍 센서의 주 지표 축(z) 특징 + 수직 진폭 비만 쓴다.
    1-TAC(tac_)·AR(_ar<k>_)·판정 보류(hold_) 열은 AI② 지표·표시라 넣지 않는다."""
    if PUBLIC and AI1_MODEL == "v2":
        # v2: 정상 데이터만으로 고른 모양 Feature (select_v2 결과, AI② 확인 2026-11-18)
        from config import ROOT
        sel = pd.read_csv(ROOT / "results" / "public" / "v2_selection" / f"{ACTIVE_BRIDGE}_v2_selection.csv")
        return sel.loc[sel["selected"], "feature"].tolist()
    if PUBLIC:
        prefixes = {sensor_short_name(s) for s in PUBLIC["model_sensors"]}
        axis = PUBLIC["main_axis"].lower()
        cols = [c for c in feats.columns
                if c.split("_", 1)[0] in prefixes and c.endswith(f"_{axis}") and c.count("_") >= 2
                and feature_name(c) not in MODEL_INPUT["exclude_feature_names"]
                and not feature_name(c).startswith("ar") and c not in META_COLS]
        return cols + [c for c in feats.columns if c == "cross_amp_ratio_L4_L2_z"]
    excluded = {sensor_short_name(s["sensor_id"]) for s in sensors
                if s["sensor_type"] in MODEL_INPUT["exclude_sensor_types"]}
    cols = []
    for c in feats.columns:
        if c in META_COLS or c in MODEL_INPUT["exclude_features"] or c.split("_", 1)[0] in excluded:
            continue
        if not c.startswith("cross_") and feature_name(c) in MODEL_INPUT["exclude_feature_names"]:
            continue
        cols.append(c)
    return cols


def fit_baseline(train_df, cols):
    """조건(하중 크기 × 위치)별 정상 기준 + 온도 보정.
    평균      : 조건마다
    온도 보정 : 조건 평균을 뺀 나머지가 온도 1°C 에 얼마나 변하는지(기울기)를 구해 빼 준다
    표준편차  : config 방식 (pooled = 조건을 합쳐 계산, per_condition = 조건마다)"""
    train_df = train_df.assign(condition=baseline_condition(train_df))
    mean = train_df.groupby("condition")[cols].mean()
    resid = train_df[cols] - mean.reindex(train_df["condition"]).to_numpy()
    temps = train_df["temperature_c"]
    # 온도 기록이 없으면(공개 데이터 일부) 기준 온도를 0 으로 두고 보정하지 않는다 (빈 값이 z-score 로 퍼지지 않게)
    temp_ref, slope = (float(temps.mean()) if temps.notna().any() else 0.0), pd.Series(0.0, index=cols)
    # 공개 데이터는 온도 보정을 하지 않는다 (HBTA 온도는 측정 단위 값이라 측정 차이와 구분 불가, 합의서 v3)
    if (BASELINE["temperature_correction"] and not PUBLIC and temps.notna().all()
            and temps.max() - temps.min() >= BASELINE["temp_min_range_c"]):
        dt = (temps - temp_ref).to_numpy()
        raw = dt @ resid.to_numpy() / np.sum(dt**2)
        # 기울기의 추정 오차 → 뚜렷한(|기울기/오차| ≥ temp_min_t) 특징에만 보정
        after = resid.to_numpy() - np.outer(dt, raw)
        se = np.sqrt(after.var(axis=0, ddof=2) / np.sum(dt**2))
        significant = np.abs(raw) >= BASELINE["temp_min_t"] * se
        slope = pd.Series(np.where(significant, raw, 0.0), index=cols)
        resid = resid - np.outer(dt, slope.to_numpy())
    if BASELINE["std"] == "pooled":
        std = pd.DataFrame([resid.std(ddof=1)] * len(mean), index=mean.index)
    else:
        std = resid.groupby(train_df["condition"].to_numpy()).std(ddof=1)
    return {"mean": mean, "std": std.clip(lower=BASELINE["std_floor"]), "temp_ref": temp_ref, "temp_slope": slope}


def baseline_condition(df):
    """정상 기준을 나누는 조건. 공개 데이터는 하중 조건이 없어 하나로 묶는다 (가진 조건은 데이터셋마다 하나)."""
    return pd.Series("ALL", index=df.index) if PUBLIC else df["condition"]


def to_zscore(df, cols, baseline):
    """각 Window 를 '같은 조건(하중 크기 × 위치)의 정상 기준'에서 벗어난 정도로 변환 (온도 보정 포함)."""
    conds = baseline_condition(df)
    unknown = sorted(set(conds) - set(baseline["mean"].index))
    if unknown:
        raise ValueError(f"정상 기준이 없는 하중 조건: {unknown}\n"
                         f"→ train D0 데이터에 해당 조건(하중 크기 × 위치)이 있어야 합니다.")
    mean = baseline["mean"].reindex(conds)[cols].to_numpy()
    std = baseline["std"].reindex(conds)[cols].to_numpy()
    dt = (df["temperature_c"].fillna(baseline["temp_ref"]) - baseline["temp_ref"]).fillna(0.0).to_numpy()
    temp_effect = np.outer(dt, baseline["temp_slope"][cols].to_numpy())
    return (df[cols].to_numpy() - mean - temp_effect) / std


def sensor_feature_groups(cols, sensors):
    """센서별 모델 입력 특징 위치. {sensor_id: (센서 약칭, [cols 안의 열 번호])}. 센서 간 비교(cross_)는 제외."""
    groups = {}
    for s in sensors:
        prefix = sensor_short_name(s["sensor_id"])
        idx = [i for i, c in enumerate(cols) if c.split("_", 1)[0] == prefix]
        if idx:
            groups[s["sensor_id"]] = (prefix, idx)
    return groups


def fit_iforest(z_train, z_val):
    """Isolation Forest 1개 학습 + val 로 판정 기준 결정. 교량 전체 모델과 센서별 모델이 같은 방식을 쓴다."""
    model = IsolationForest(
        n_estimators=IFOREST["n_estimators"],
        max_samples=IFOREST["max_samples"],
        contamination=IFOREST["contamination"],
        random_state=IFOREST["random_state"],
    ).fit(z_train)
    raw_val = raw_iforest_score(model, z_val)
    calib = {"val_p01": float(np.quantile(raw_val, 0.01)), "val_p99": float(np.quantile(raw_val, 0.99))}
    s_val = to_unit_score(raw_val, calib)
    return {
        "model": model,
        "calib": calib,
        "threshold": float(np.quantile(s_val, IFOREST["threshold_quantile"])),
        "z_threshold": float(np.quantile(np.abs(z_val).max(axis=1), IFOREST["z_threshold_quantile"])),
    }


def raw_iforest_score(model, z):
    """Isolation Forest 원 논문 점수 (0~1, 클수록 이상). sklearn score_samples 의 부호를 바꾼 값."""
    return -model.score_samples(z)


def to_unit_score(raw, calib):
    """0~1 Anomaly Score 로 변환. 방식은 config IFOREST["score_method"] (변환 방식을 바꾸려면 이 함수만 수정)."""
    if IFOREST["score_method"] == "iforest_paper":
        return raw
    if IFOREST["score_method"] == "minmax_val":
        lo, hi = calib["val_p01"], calib["val_p99"]
        return np.clip((raw - lo) / (hi - lo), 0, 1)
    raise ValueError(f"알 수 없는 score_method: {IFOREST['score_method']}")


def save_zscore_only(train_df, cols, exps, split):
    """정상 학습 데이터가 너무 적은 공개 데이터(Old ADA): Isolation Forest 없이 정상 기준(평균·표준편차)만 저장.
    predict.py 는 정상 대비 z-score 만 계산하고 Normal/Anomaly 판정은 하지 않는다 (서술 보고용)."""
    baseline = fit_baseline(train_df, cols)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bridge": ACTIVE_BRIDGE, "window": dict(window_setting()), "model_type": "zscore",
                 "feature_cols": cols, "baseline": baseline, "n_train": len(train_df)}, MODEL_PATH)
    (exps.assign(split=exps["experiment_id"].map(split))[["experiment_id", "split", "damage_type", "severity_level"]]
     .to_csv(MODEL_DIR / "split.csv", index=False, encoding="utf-8-sig"))
    print(f"[z-score 전용] 정상 학습 Window {len(train_df)}개로 정상 기준만 계산 (Isolation Forest·판정 없음)")
    print(f"  모델 입력 특징 {len(cols)}개: {', '.join(cols)}")
    print(f"저장: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    feats = pd.read_parquet(FEAT_PATH)
    exps = pd.read_csv(RAW_DIR / ACTIVE_BRIDGE / "experiments.csv",
                       usecols=["experiment_id", "split", "damage_type", "severity_level"])
    sensors = BRIDGES[ACTIVE_BRIDGE]["sensors"]

    split = split_experiments(exps)
    feats["split"] = feats["experiment_id"].map(split)
    cols = model_feature_columns(feats, sensors)
    usable = pd.Series(True, index=feats.index)
    if PUBLIC:
        # 공개 데이터: 판정 보류(열화본에서 주 지표 쌍이 잡음 수준) Window 와 센서가 없는 Window 는 학습·기준에서 뺀다
        usable = feats[cols].notna().all(axis=1) & ~feats["hold_main"].astype(bool)
        print(f"[공개 데이터] 학습·판정에 쓸 수 있는 Window: {int(usable.sum()):,} / {len(feats):,} "
              f"(판정 보류 {int(feats['hold_main'].astype(bool).sum()):,}, 센서 없음·계산 불가 {int((~feats[cols].notna().all(axis=1)).sum()):,})")
    nan_cols = feats.loc[usable, cols].columns[feats.loc[usable, cols].isna().any()].tolist()
    if nan_cols:
        raise SystemExit(f"특징에 결측(NaN)이 있습니다: {nan_cols}\n→ preprocess.py 의 해당 특징 계산을 확인하세요.")
    train_df = feats[(feats["split"] == "train") & usable]
    val_df = feats[(feats["split"] == "val") & usable]
    if PUBLIC and PUBLIC["ai1_model"] == "zscore":
        save_zscore_only(train_df, cols, exps, split)
        return

    baseline = fit_baseline(train_df, cols)
    z_train = to_zscore(train_df, cols, baseline)
    z_val = to_zscore(val_df, cols, baseline)

    # 교량 전체 모델: 모든 센서 특징을 함께 봄 → "교량이 평소와 다른가"
    bridge_model = fit_iforest(z_train, z_val)
    model, calib = bridge_model["model"], bridge_model["calib"]
    threshold, z_threshold = bridge_model["threshold"], bridge_model["z_threshold"]
    s_train = to_unit_score(raw_iforest_score(model, z_train), calib)
    s_val = to_unit_score(raw_iforest_score(model, z_val), calib)

    # 센서별 모델: 그 센서의 특징만 봄 → "어느 센서가 평소와 다른가" (위치 모니터링용)
    sensor_info = pd.DataFrame(sensors).set_index("sensor_id")
    sensor_models = {}
    for sid, (prefix, idx) in sensor_feature_groups(cols, sensors).items():
        sensor_models[sid] = fit_iforest(z_train[:, idx], z_val[:, idx]) | {
            "prefix": prefix,
            "col_idx": idx,
            "sensor_type": sensor_info.loc[sid, "sensor_type"],
            "sensor_location": sensor_info.loc[sid, "sensor_location"],
            "x_ratio": float(sensor_info.loc[sid, "x_ratio"]),
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "bridge": ACTIVE_BRIDGE,
        "window": dict(window_setting()),
        "features": {"use_damping": FEATURES["use_damping"]},
        "preprocess": dict(PREPROCESS),
        "iforest_params": dict(IFOREST),
        "feature_cols": cols,
        "baseline": baseline,
        "model": model,
        "calib": calib,
        "threshold": threshold,
        "z_threshold": z_threshold,
        "sensor_models": sensor_models,
    }, MODEL_PATH)
    split_table = (exps.assign(split=exps["experiment_id"].map(split))
                   [["experiment_id", "split", "damage_type", "severity_level"]])
    split_table.to_csv(MODEL_DIR / "split.csv", index=False, encoding="utf-8-sig")

    print("[7] 실험 단위 분할 (experiment 수 / Window 수)")
    counts = feats.groupby("split").agg(experiments=("experiment_id", "nunique"), windows=("window_id", "size"))
    print(counts.to_string())
    overlap = set(train_df["experiment_id"]) & set(feats.loc[feats["split"] != "train", "experiment_id"])
    print(f"train 과 다른 split 에 동시에 들어간 실험: {len(overlap)}개 (0 이어야 정상)")

    print(f"\n[8] 학습 완료: 모델 입력 특징 {len(cols)}개 (하중·온도·가속도 평균 제외 → 조건으로 사용)")
    print(f"  조건별 정상 기준: {list(baseline['mean'].index)} / 표준편차 방식: {BASELINE['std']}")
    if BASELINE["temperature_correction"]:
        t = train_df["temperature_c"]
        corrected = baseline["temp_slope"][baseline["temp_slope"] != 0]
        print(f"  온도 보정: train 온도 {t.min():.1f}~{t.max():.1f}°C (기준 {baseline['temp_ref']:.1f}°C), "
              f"보정 적용 특징 {len(corrected)}개"
              + (": " + ", ".join(f"{c} ({v:+.3g}/°C)" for c, v in corrected.abs().sort_values(ascending=False)
                                  .head(4).items()) if len(corrected) else ""))
    print(f"  점수 변환: {IFOREST['score_method']}")
    print(f"  train 점수 중앙값 {np.median(s_train):.3f} / val 점수 중앙값 {np.median(s_val):.3f}")
    print(f"  판정 기준: IF 점수 > {threshold:.3f}" + (f"  또는  최대|z| > {z_threshold:.2f}" if IFOREST["use_z_rule"] else ""))
    print(f"  val 오탐률 (IF 기준): {np.mean(s_val > threshold):.1%}  ← 임계값을 val 로 정했으므로 약 1% 가 정상")
    print(f"\n  센서별 모델 {len(sensor_models)}개 (어느 센서가 이상인지 모니터링용)")
    for sid, m in sensor_models.items():
        print(f"    {sid:<16} {m['sensor_location']:<20} 특징 {len(m['col_idx']):>2}개  "
              f"판정 기준: 점수 > {m['threshold']:.3f} 또는 최대|z| > {m['z_threshold']:.2f}")
    print(f"\n저장: {MODEL_PATH}\n      {MODEL_DIR / 'split.csv'}")


# ---------------------------------------------------------------------------
# AI① v2 Feature 선택 (공개 데이터, AI②와 합의한 규칙. 손상 라벨을 보지 않고 정상 데이터만 사용)
# ---------------------------------------------------------------------------
V2_STABILITY_MAX = 1.0   # 정상 구간 간 안정성: |중앙값₁ − 중앙값₂| ÷ √(s₁² + s₂²) < 1.0  (s = MAD × 1.4826)
V2_NOISE_MAX = 2.0       # 잡음 견딤: 열화본 정상 s ÷ 원본 정상 s < 2.0


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def v2_candidates(feats):
    """v2 후보: 모드 Feature(v2_*), 수직 진폭 비, (자유 감쇠 데이터면) 감쇠비"""
    cols = [c for c in feats.columns if c.startswith("v2_")] + ["cross_amp_ratio_L4_L2_z"]
    if PUBLIC["use_damping"]:
        cols += [c for c in feats.columns if c.startswith("acc") and feature_name(c) == "damping" and c.endswith("_z")]
    return cols


def select_v2():
    """정상(AI① train·val) Window 만으로 v2 Feature 를 고르고 표로 저장. 모델은 실행하지 않는다."""
    from config import ROOT

    exps = pd.read_csv(RAW_DIR / ACTIVE_BRIDGE / "experiments.csv", keep_default_na=False)
    feats = {src: pd.read_parquet(ROOT / "data" / "processed" / src / ACTIVE_BRIDGE / "features.parquet")
             for src in ("public_raw", "public_deg")}
    info = exps.set_index("experiment_id")
    normal = {}
    for src, f in feats.items():
        split = f["experiment_id"].map(info["split"])
        ok = split.isin(["train", "val"]) & ~f["hold_main"].astype(bool)
        normal[src] = f[ok].assign(_split=split[ok], _record=f.loc[ok, "experiment_id"].map(info["source_record"]))
    deg = normal["public_deg"]
    # 잡음 견딤은 같은 Window 끼리 비교: 원본도 열화본에서 판정 보류되지 않은 정상 Window 만
    key = ["experiment_id", "window_start_us"]
    raw = normal["public_raw"].merge(deg[key], on=key)

    # 안정성을 볼 두 정상 구간 (데이터셋별, AI② 회신 2.3)
    if ACTIVE_BRIDGE == "hbta":
        groups, how = deg["_record"], "UDS_01 vs UDS_02 (각 측정의 train·val 블록)"
    elif ACTIVE_BRIDGE == "ponte_moesa":
        groups, how = deg["_split"], "train vs val (같은 측정 AMB_18 의 연속 구간 → 세션 간 안정성은 검증되지 않음)"
    else:
        groups, how = None, "적용 안 함 (정상 학습 주행 4회라 두 구간으로 나누면 표준편차를 2개 값으로 계산하게 됨)"

    held = {src: int((f["experiment_id"].map(info["split"]).isin(["train", "val"]) & f["hold_main"].astype(bool)).sum())
            for src, f in feats.items()}
    counts = {"windows_train": int((deg["_split"] == "train").sum()), "windows_val": int((deg["_split"] == "val").sum()),
              "excluded_held_deg": held["public_deg"], "raw_windows_used": len(raw),
              "used_splits": "train,val (ai2_calib·test 미사용)"}
    rows = []
    for c in v2_candidates(deg):
        row = {"feature": c, "n_normal_deg": int(deg[c].notna().sum())}
        if groups is not None:
            g1, g2 = (deg.loc[groups == k, c] for k in sorted(groups.unique()))
            s1, s2 = robust_scale(g1), robust_scale(g2)
            row["stability"] = abs(np.nanmedian(g1) - np.nanmedian(g2)) / np.sqrt(s1**2 + s2**2) if s1 + s2 > 0 else np.inf
        else:
            row["stability"] = np.nan
        row["noise_ratio"] = robust_scale(deg[c]) / robust_scale(raw[c]) if robust_scale(raw[c]) > 0 else np.inf
        row["pass_stability"] = bool(np.isnan(row["stability"]) or row["stability"] < V2_STABILITY_MAX)
        row["pass_noise"] = bool(row["noise_ratio"] < V2_NOISE_MAX)
        row["selected"] = row["pass_stability"] and row["pass_noise"]
        rows.append(row)
    table = pd.DataFrame(rows).assign(**counts)
    out = ROOT / "results" / "public" / "v2_selection"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{ACTIVE_BRIDGE}_v2_selection.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(f"[{ACTIVE_BRIDGE}] v2 Feature 선택 (정상 Window: 열화본 {len(deg)}개, 원본 {len(raw)}개)")
    print(f"  안정성 비교: {how}")
    print(f"  선택에 쓴 Window: train {counts['windows_train']}, val {counts['windows_val']}, 판정 보류로 제외 {counts['excluded_held_deg']} (ai2_calib·test 미사용)")
    print(table.drop(columns=list(counts)).round(3).to_string(index=False))
    n = int(table["selected"].sum())
    print(f"  선택 {n}개" + ("" if n else " → 통과 Feature 없음: 이 데이터셋은 v2 를 실행하지 않고 v1 만 보고 (기준값은 낮추지 않음)"))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--select-v2"]:
        select_v2()
    else:
        main()
