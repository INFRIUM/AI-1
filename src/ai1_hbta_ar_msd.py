"""
HBTA AR(5) + MSD 재현 (docs/AI1_HBTA_AR_MSD_SETUP.md, Svendsen et al. 2021)

실행: python src/ai1_hbta_ar_msd.py           → results/public/hbta_ar_msd/{summary.csv, by_state.csv, setup.json}
      python src/ai1_hbta_ar_msd.py --test    → 합성 데이터 점검만
"""
import json
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import ORIGINAL_DIR, ROOT

FS, WIN, START_S, N_WIN, N_TRAIN, ORDER = 100, 1024, 90.0, 80, 40, 5
COPIES, SNR, TRIALS, ALPHA, SEED = 20, 20.0, 1000, 0.01, 20261126
STATES = ["UDS1", "UDS2"] + [f"DS{k}" for k in range(1, 9)]
RECORD = {"UDS1": "MVS_P2_UDS_NM_Z_01", "UDS2": "MVS_P2_UDS_NM_Z_02", **{f"DS{k}": f"MVS_P2_DS{k}_NM_Z_01" for k in range(1, 9)}}
GROUPS = {
    "group1": [(f"AL{i:02d}", "z") for i in range(1, 41)],
    "group2": [(f"AG{i:02d}", a) for i in range(1, 19) for a in ("y", "z") if not (i == 9 and a == "y")],
    "two_sensors": [("AG03", "y"), ("AG03", "z"), ("AG05", "y"), ("AG05", "z")],
}
OUT = ROOT / "results" / "public" / "hbta_ar_msd"


def ar_coef(x, p=ORDER):
    """평균을 뺀 신호의 AR(p) 계수 (최소제곱)"""
    x = x - x.mean()
    X = np.column_stack([x[p - j - 1:len(x) - j - 1] for j in range(p)])
    return np.linalg.lstsq(X, x[p:], rcond=None)[0]


def features(f, channels):
    """상태 → (80, 채널×5) AR 특징"""
    out = {}
    s0 = int(START_S * FS)
    for st, rec in RECORD.items():
        g = f[rec]["acceleration"]
        sig = np.stack([g[s][a][s0:s0 + N_WIN * WIN] for s, a in channels])
        if sig.shape[1] < N_WIN * WIN:
            raise SystemExit(f"{rec}: 길이 부족")
        out[st] = np.stack([np.concatenate([ar_coef(sig[c, w * WIN:(w + 1) * WIN]) for c in range(len(channels))])
                            for w in range(N_WIN)])
    return out


def augment(base, rng):
    sd = np.abs(base).mean(axis=0) / SNR
    return np.concatenate([base + rng.normal(size=base.shape) * sd for _ in range(COPIES)])


def msd(train, test):
    mu, S = train.mean(axis=0), np.cov(train, rowvar=False)
    d = test - mu
    return np.einsum("ij,ij->i", d @ np.linalg.inv(S), d)


def mc_threshold(n, k, rng):
    m = []
    for _ in range(TRIALS):
        z = rng.standard_normal((n, k))
        m.append(msd(z, z).max())
    return float(np.quantile(m, 1 - ALPHA))


def evaluate(feat, run, rng, thr_cache):
    if run == "R3":
        base, normal = feat["UDS1"][:N_TRAIN], {"UDS2": feat["UDS2"]}
    else:
        base = np.concatenate([feat["UDS1"][:N_TRAIN], feat["UDS2"][:N_TRAIN]])
        normal = {"UDS1": feat["UDS1"][N_TRAIN:], "UDS2": feat["UDS2"][N_TRAIN:]}
    train = augment(base, rng)
    key = train.shape
    if key not in thr_cache:
        thr_cache[key] = mc_threshold(*key, rng)
    thr = thr_cache[key]
    score = {st: msd(train, x) for st, x in {**normal, **{s: feat[s] for s in STATES[2:]}}.items()}
    nor = np.concatenate([score[s] for s in normal])
    dam = np.concatenate([score[s] for s in STATES[2:]])
    y = np.r_[np.zeros(len(nor)), np.ones(len(dam))]
    summ = {"threshold": thr, "n_train": len(train), "dims": train.shape[1], "n_normal": len(nor), "n_damage": len(dam),
            "false_alarm": float((nor > thr).mean()), "missed": float((dam <= thr).mean()),
            "auc": float(roc_auc_score(y, np.r_[nor, dam]))}
    by = [{"state": s, "n": len(v), "flagged": float((v > thr).mean()), "median_msd": float(np.median(v))} for s, v in score.items()]
    return summ, by


def run(source):
    rng = np.random.default_rng(SEED)
    rows, by_rows, cache = [], [], {}
    for g, ch in GROUPS.items():
        feat = source(ch)
        for r in ("R1", "R3") if g != "two_sensors" else ("R2", "R3"):
            s, b = evaluate(feat, r, rng, cache)
            rows.append({"run": r, "group": g, **s})
            by_rows += [{"run": r, "group": g, **x} for x in b]
    return pd.DataFrame(rows), pd.DataFrame(by_rows)


def synthetic(ch):
    rng = np.random.default_rng(1)
    return {st: rng.normal(size=(N_WIN, len(ch) * ORDER)) + (0 if st.startswith("UDS") else 3) for st in STATES}


if __name__ == "__main__":
    if "--test" in sys.argv:
        t = np.random.default_rng(0)
        x = np.zeros(4000)
        for i in range(2, 4000):
            x[i] = 1.5 * x[i - 1] - 0.7 * x[i - 2] + t.normal()
        c = ar_coef(x)
        assert abs(c[0] - 1.5) < 0.05 and abs(c[1] + 0.7) < 0.05 and np.abs(c[2:]).max() < 0.05
        globals()["TRIALS"] = 50
        s, b = run(synthetic)
        assert (s["auc"] > 0.99).all() and (s.loc[s["run"] != "R3", "missed"] < 0.05).all()
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    with h5py.File(ORIGINAL_DIR / "hbta" / "data_100Hz.h5", "r") as f:
        s, b = run(lambda ch: features(f, ch))
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "by_state.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({"records": RECORD, "start_s": START_S, "win": WIN, "n_win": N_WIN,
                                                "copies": COPIES, "snr": SNR, "trials": TRIALS, "seed": SEED}, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
