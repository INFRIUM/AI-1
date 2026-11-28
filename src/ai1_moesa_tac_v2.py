"""
E1 v2 Ponte Moesa 1-TAC, 10분 평균 전달률 (docs/AI1_MOESA_TAC_V2_SETUP.md) — 개발 시험

실행: python src/ai1_moesa_tac_v2.py         → results/public/moesa_tac_v2/{summary.csv, by_state.csv, by_record.csv, setup.json}
      python src/ai1_moesa_tac_v2.py --test  → 합성 데이터 점검만
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from ai1_moesa_tac import (BAND, DAMAGE, FS, NPS, PAIRS, RMS_MAX, SEED, SENSORS, WIN, load_records,
                           window_stats)
from config import ROOT

BLOCK = 10                                       # 60초 구간 10개 = 10분
OUT = ROOT / "results" / "public" / "moesa_tac_v2"
V1_AUC = 0.637                                   # v1 결합 AUC (조건표 9절)


def tac(a, b):
    return np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def blocks(records):
    """측정마다 유효(세 센서 RMS ≤ 0.008g) 60초 구간을 순서대로 10개씩 묶어 G 합"""
    out = []
    for amb, ds, rl, x in records:
        ws = [w for w in window_stats(x) if w["rms"].max() <= RMS_MAX]
        for j in range(len(ws) // BLOCK):
            grp = ws[j * BLOCK:(j + 1) * BLOCK]
            G = {p: (sum(w["G"][p][0] for w in grp), sum(w["G"][p][1] for w in grp)) for p in PAIRS}
            out.append({"amb": amb, "ds": ds, "role": rl, "j": j, "G": G, "f": grp[0]["f"]})
    return out


def evaluate(records):
    B = blocks(records)
    T = pd.DataFrame([{k: b[k] for k in ("amb", "ds", "role", "j")} for b in B])
    f = B[0]["f"]
    fb = f[(f >= BAND[0]) & (f <= BAND[1])]
    ref = [i for i, b in enumerate(B) if b["role"] == "ref"]
    feats = []
    for p in PAIRS:
        tr = sum(B[i]["G"][p][0] for i in ref) / sum(B[i]["G"][p][1] for i in ref)
        td = [b["G"][p][0] / b["G"][p][1] for b in B]
        T[f"score_{p}"] = [1 - tac(np.abs(d), np.abs(tr)) for d in td]          # 크기 |T|
        T[f"cplx_{p}"] = [1 - tac(d, tr) for d in td]                            # 보고만: 복소
        feats.append(np.array([[np.log(np.abs(d[(fb >= lo) & (fb < lo + 1)])).mean() for lo in range(3, 12)] for d in td]))
    T["score_E1v2"] = T[[f"score_{p}" for p in PAIRS]].mean(axis=1)
    T["score_E1v2_complex"] = T[[f"cplx_{p}" for p in PAIRS]].mean(axis=1)
    X = np.hstack(feats)
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[T.role == "ref"])
    T["score_IForest"] = -iso.score_samples(X)
    return T


def judge(T):
    rows, states = [], []
    for model in ["E1v2", "IForest", "E1v2_complex", *PAIRS]:
        sc = f"score_{model}"
        thr = float(np.quantile(T.loc[T.role == "thr", sc], 0.99))
        nor = T.loc[T.role == "test_normal", sc]
        dm = T[T.ds.isin(DAMAGE)]
        y = np.r_[np.zeros(len(nor)), np.ones(len(dm))]
        rows.append({"model": model, "threshold": thr, "n_ref": int((T.role == "ref").sum()),
                     "n_thr": int((T.role == "thr").sum()), "n_test_normal": len(nor),
                     "false_alarm": float((nor > thr).mean()),
                     **{f"detect_{d}": float((T.loc[T.ds == d, sc] > thr).mean()) for d in DAMAGE},
                     **{f"n_{d}": int((T.ds == d).sum()) for d in DAMAGE},
                     "detect_all": float((dm[sc] > thr).mean()), "auc": float(roc_auc_score(y, np.r_[nor, dm[sc]]))})
        for (rl, ds), g in T.groupby(["role", "ds"]):
            states.append({"model": model, "role": rl, "ds": ds, "n": len(g), "flagged": float((g[sc] > thr).mean()),
                           "median_score": float(g[sc].median())})
        T[f"flag_{model}"] = T[sc] > thr
    return pd.DataFrame(rows), pd.DataFrame(states)


def synthetic():
    """v1 합성과 같은 모형 (모드 4.9·7.3Hz, 손상 = 1번 2차 모드 형상 0.6배·2% 낮음), 측정당 120구간 = 묶음 12개"""
    rng = np.random.default_rng(5)
    n = 120 * WIN

    def rec(amb, ds, rl, damaged):
        q = []
        for f0 in (4.9, 7.3 * (0.98 if damaged else 1.0)):
            b, a = signal.iirpeak(f0, 30, fs=FS)
            q.append(signal.lfilter(b, a, rng.normal(size=n)))
        phi2 = np.array([0.6 if damaged else 1.0, 0.2, -0.9])
        x = np.outer(q[0], [0.7, 1.0, 0.8]) + 0.8 * np.outer(q[1], phi2) + 0.05 * rng.normal(size=(n, 3))
        return (amb, ds, rl, 0.01 * x)
    return ([rec("AMB_10", "DS2", "ref", False), rec("AMB_11", "DS2", "thr", False),
             rec("AMB_18", "DS2", "test_normal", False)]
            + [rec(f"AMB_{20 + i}", d, "damage", True) for i, d in enumerate(DAMAGE)])


if __name__ == "__main__":
    if "--test" in sys.argv:
        T = evaluate(synthetic())
        s, _ = judge(T)
        print(s[["model", "false_alarm", "detect_all", "auc"]].round(3).to_string())
        r = s.set_index("model")
        assert r.loc["E1v2", "auc"] > 0.9 and r.loc["E1v2", "false_alarm"] <= 0.1, "합성 손상을 못 잡음"
        assert (T.groupby("role").size() >= 12).all()
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    print("데이터 읽는 중 (.mat 4개)", flush=True)
    T = evaluate(load_records())
    s, b = judge(T)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "by_state.csv", index=False, encoding="utf-8-sig")
    T.drop(columns=[c for c in T if c.startswith("cplx_")]).to_csv(OUT / "by_block.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "block_windows": BLOCK, "win_s": WIN / FS, "nperseg": NPS, "band_hz": BAND, "rms_max_ms2": RMS_MAX,
        "pairs": PAIRS, "sensors_z": SENSORS, "tac_vector": "|T| (크기)", "threshold": "thr 역할 99% 분위수",
        "v1_auc": V1_AUC, "seed": SEED, "note": "개발 시험"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s.round(4).to_string())
