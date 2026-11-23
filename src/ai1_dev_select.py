"""
AI① 개선 방법 고르기 (docs/AI1_IMPROVEMENT_PLAN.md). 개발용 HBTA·Ponte Moesa 1.1 만 쓰고 LUMO 는 읽지 않는다.

실행: python src/ai1_dev_select.py          → results/public/ai1_dev_select/{candidates.csv, by_stage.csv}
      python src/ai1_dev_select.py --test   → 합성 데이터 점검만 (실제 데이터 읽지 않음)
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from config import ROOT

DATASETS = ("hbta", "ponte_moesa")
FEATURE_SETS, MODELS, POOLS = ("F1", "F2"), ("M1", "M2"), ("P1", "P2")
SENSORS, AXIS = ("acc01", "acc02"), "z"
BANDS = ("band_2_8", "band_8_15", "band_15_25")
FA_MAX, AUC_MIN, TIE, MIN_N = 0.10, 0.60, 0.02, 30
OUT = ROOT / "results" / "public" / "ai1_dev_select"


def features(df, kind):
    out = {}
    for s in SENSORS:
        if kind == "F1":
            for k in ("rms", "std", "max", "min", "p2p", "crest", "dom_freq", *BANDS):
                out[f"{s}_{k}"] = df[f"{s}_{k}_{AXIS}"]
        else:
            total = sum(df[f"{s}_{b}_{AXIS}"] for b in BANDS)
            out[f"{s}_crest"], out[f"{s}_dom_freq"] = df[f"{s}_crest_{AXIS}"], df[f"{s}_dom_freq_{AXIS}"]
            for b in BANDS:
                out[f"{s}_{b}_ratio"] = df[f"{s}_{b}_{AXIS}"] / total
    return pd.DataFrame(out, index=df.index)


def fit_score(x, train, kind):
    mu, sd = x[train].mean(), x[train].std(ddof=1).clip(lower=1e-6)
    z = ((x - mu) / sd).to_numpy()
    if kind == "M1":
        return -IsolationForest(n_estimators=200, max_samples="auto", random_state=42).fit(z[train]).score_samples(z)
    return LedoitWolf().fit(z[train]).mahalanobis(z)


def pool(df, score, kind):
    """같은 실험에서 시작 시각이 이어진 Window 만 묶는다. P1 최근 3개 중 두 번째로 큰 값(=중앙값), P2 최근 5개 중앙값. 모자라면 NaN"""
    n = 3 if kind == "P1" else 5
    d = df.assign(s=score).sort_values(["experiment_id", "window_start_us"])
    step = d.groupby("experiment_id")["window_start_us"].diff()
    run = (step != step.median()).cumsum()
    r = d.groupby(run)["s"].rolling(n).median().reset_index(level=0, drop=True)
    return r.reindex(df.index)


def auc_ci(y, s, unit, n=1000):
    """측정 단위 재표집 95% 구간 (정상·손상 측정을 따로 뽑는다)"""
    rng, ids = np.random.default_rng(20261123), {}
    for c in (0, 1):
        ids[c] = [np.flatnonzero((unit == u) & (y == c)) for u in np.unique(unit[y == c])]
    v = []
    for _ in range(n):
        ix = np.concatenate([ids[c][k] for c in (0, 1) for k in rng.integers(0, len(ids[c]), len(ids[c]))])
        v.append(roc_auc_score(y[ix], s[ix]))
    return np.percentile(v, [2.5, 97.5])


def evaluate(df, fs, m, p):
    use = df[~df["hold_main"].astype(bool)].copy()
    x = features(use, fs)
    full = x.notna().all(axis=1)
    use, x = use[full], x[full]
    train = (use["ai1_split"] == "train").to_numpy()
    s = pool(use, fit_score(x, train, m), p)
    ok = s.notna()
    val = ok & (use["ai1_split"] == "val")
    normal = ok & use["ai1_split"].isin(["ai2_calib", "test"]) & use["normal"]
    damage = ok & (use["ai1_split"] == "test") & ~use["normal"]
    row = {"n_val": int(val.sum()), "n_normal": int(normal.sum()), "n_damage": int(damage.sum())}
    if row["n_val"] == 0 or row["n_normal"] < MIN_N or row["n_damage"] < MIN_N:
        return {**row, "eligible_n": False}, []
    thr = float(np.percentile(s[val], 99))
    part = normal | damage
    y, sc = damage[part].astype(int).to_numpy(), s[part].to_numpy()
    lo, hi = auc_ci(y, sc, use.loc[part, "experiment_id"].to_numpy(dtype=object).astype(str))
    row.update(eligible_n=True, auc=roc_auc_score(y, sc), auc_lo=lo, auc_hi=hi,
               false_alarm=float((s[normal] > thr).mean()), detection=float((s[damage] > thr).mean()))
    stages = []
    for st in sorted(use.loc[damage, "damage_stage"].unique()):
        g = damage & (use["damage_stage"] == st)
        stages.append({"damage_stage": st, "n": int(g.sum()), "detection": float((s[g] > thr).mean())})
    return row, stages


def choose(table):
    """계획 3절: 두 데이터 오경보 ≤ 10% → 작은 AUC 최대. 0.02 이내는 단순한 것 (F2, M2, P1 순 우선)"""
    c = table.groupby(["features", "model", "pool"]).agg(
        ok=("eligible_n", "all"), fa=("false_alarm", "max"), auc=("auc", "min"), n=("dataset", "size")).reset_index()
    c = c[c["ok"] & (c["fa"] <= FA_MAX) & (c["n"] == len(DATASETS))]
    if c.empty or c["auc"].max() < AUC_MIN:
        return None
    near = c[c["auc"] >= c["auc"].max() - TIE]
    near = near.assign(a=near["features"].map({"F2": 0, "F1": 1}), b=near["model"].map({"M2": 0, "M1": 1}),
                       d=near["pool"].map({"P1": 0, "P2": 1})).sort_values(["a", "b", "d"])
    return near.iloc[0]


def run(load):
    rows, stages = [], []
    for source in ("deg", "raw"):
        for name in DATASETS:
            df = load(name, source)
            for fs, m, p in itertools.product(FEATURE_SETS, MODELS, POOLS):
                key = {"source": source, "dataset": name, "features": fs, "model": m, "pool": p}
                r, st = evaluate(df, fs, m, p)
                rows.append({**key, **r})
                stages += [{**key, **q} for q in st]
    return pd.DataFrame(rows), pd.DataFrame(stages)


def load_public(name, source):
    d = ROOT / "results" / "public" / "for_AI2"
    df = pd.read_csv(d / f"{name}_anomaly_results_{source}.csv", encoding="utf-8-sig", low_memory=False)
    if source == "raw":   # 원본도 열화본 보류 Window 를 뺀다 (같은 Window 비교)
        deg = pd.read_csv(d / f"{name}_anomaly_results_deg.csv", encoding="utf-8-sig", low_memory=False)
        key = ["experiment_id", "window_start_us"]
        df = df.drop(columns="hold_main", errors="ignore").merge(deg[key + ["hold_main"]], on=key, validate="1:1")
    keep = df["ai1_split"].isin(["train", "val", "ai2_calib", "test"]) & (df["label_status"] == "확실")
    df = df[keep].reset_index(drop=True)
    df["normal"] = df["damage_stage"] == ("UDS" if name == "hbta" else "DS2")
    assert df.loc[df["ai1_split"].isin(["train", "val", "ai2_calib"]), "normal"].all()
    return df


def synthetic(name, source):
    rng, rows = np.random.default_rng(1), []
    for e, (sp, norm, shift) in enumerate([("train", 1, 0), ("val", 1, 0), ("ai2_calib", 1, 0), ("test", 1, 0), ("test", 0, 4)]):
        for k in range(60):
            r = {"experiment_id": f"E{e}", "window_start_us": k * 10_000_000, "ai1_split": sp, "normal": bool(norm),
                 "damage_stage": "UDS" if norm else "DS1", "hold_main": k == 30}
            for s in SENSORS:
                for f in ("rms", "std", "max", "min", "p2p", "crest", "dom_freq"):
                    r[f"{s}_{f}_z"] = rng.normal() + (shift if f == "dom_freq" else 0)
                for b in BANDS:
                    r[f"{s}_{b}_z"] = rng.uniform(1, 2)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    test = "--test" in sys.argv
    table, stages = run(synthetic if test else load_public)
    best = choose(table[table["source"] == "deg"])
    if test:
        df = synthetic("x", "deg").query("experiment_id == 'E0' and not hold_main")
        s = pool(df, np.arange(len(df), dtype=float), "P1")
        assert s.isna().sum() == 4 and s.iloc[2] == 1.0   # 시작 2개 + 보류 뒤 2개는 묶지 못함
        assert table["auc"].min() > 0.8 and stages["n"].min() > 0
        print("합성 데이터 점검 통과")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT / "candidates.csv", index=False, encoding="utf-8-sig")
        stages.to_csv(OUT / "by_stage.csv", index=False, encoding="utf-8-sig")
        print(table.round(3).to_string())
        print("선택:", "개선 방법을 찾지 못함" if best is None else best.to_dict())
