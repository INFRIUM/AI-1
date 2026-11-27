"""
B′ Old ADA 자유 감쇠 고유진동수 감시 (docs/AI1_OLDADA_FREEDECAY_SETUP.md) — 개발 시험

실행: python src/ai1_oldada_freedecay.py         → results/public/oldada_freedecay_model/{summary.csv, by_run.csv, setup.json}
      python src/ai1_oldada_freedecay.py --test  → 합성 데이터 점검만
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ai1_oldada_freedecay_check import FS, load_free_decay, peak, psd
from config import OLD_ADA, ROOT

F1_REF = OLD_ADA["f1_hz"]
F1_BAND, F2_BAND = (0.8, 1.2), (1.9, 2.6)            # × F1_REF
THRESHOLD_PCT = 1.0                                  # 개발 시험용 (설정 문서 3절)
SPLIT = OLD_ADA["int40_split"]
CONFIGS = {"S8": {"f1": [1, 2, 3, 4, 5, 6, 7, 8], "f2": [1, 2, 4, 5]},
           "S2": {"f1": [1, 3], "f2": [1]}}           # 열 번호 1부터, S2 = A1(L/4)·A3(L/2)
DMG = ("DMG1", "DMG2", "DMG3")
OUT = ROOT / "results" / "public" / "oldada_freedecay_model"


def features(seg, cfg):
    f, p = psd(seg)
    s1 = p[:, [c - 1 for c in cfg["f1"]]].sum(1)
    s2 = p[:, [c - 1 for c in cfg["f2"]]].sum(1)
    return (peak(f, s1, F1_BAND[0] * F1_REF, F1_BAND[1] * F1_REF),
            peak(f, s2, F2_BAND[0] * F1_REF, F2_BAND[1] * F1_REF))


def role(state, no):
    if state != "INT40":
        return state
    return next(k for k, v in SPLIT.items() if no in v)        # train / ai2_calib / test


def evaluate(runs):
    rows, summ = [], []
    for name, cfg in CONFIGS.items():
        d = pd.DataFrame([{"sensors": name, "state": st, "run": no, "role": role(st, no),
                           **dict(zip(("f1", "f2"), features(seg, cfg)))} for st, no, seg in runs])
        base = d[d.role == "train"]
        mu1, mu2 = base.f1.mean(), base.f2.mean()
        d["d_f1_pct"] = 100 * (d.f1 / mu1 - 1)
        d["d_f2_pct"] = 100 * (d.f2 / mu2 - 1)
        d["score_pct"] = np.maximum(d.d_f1_pct.abs(), d.d_f2_pct.abs())
        d["anomaly"] = d.score_pct > THRESHOLD_PCT
        rows.append(d)
        nor = d[d.role == "test"]
        dm = d[d.state.isin(DMG)]
        y = np.r_[np.zeros(len(nor)), np.ones(len(dm))]
        summ.append({"sensors": name, "mu_f1": mu1, "mu_f2": mu2, "threshold_pct": THRESHOLD_PCT,
                     "n_test_normal": len(nor), "false_alarm": nor.anomaly.mean(),
                     **{f"detect_{s}": d[d.state == s].anomaly.mean() for s in DMG},
                     "detect_all": dm.anomaly.mean(),
                     "detect_DMG2_3": d[d.state.isin(("DMG2", "DMG3"))].anomaly.mean(),
                     "auc": roc_auc_score(y, np.r_[nor.score_pct, dm.score_pct]),
                     **{f"flag_{k}": d[(d.role == k) if k == "ai2_calib" else (d.state == k)].anomaly.mean()
                        for k in ("ai2_calib", "INT30", "INT50", "RCV")}})
    return pd.DataFrame(summ), pd.concat(rows, ignore_index=True)


def synthetic():
    """감쇠 2모드 (f1, 2.33 f1, 감쇠비 1%), 손상: f1 −3%. 2.26 f1 에 A3·A6–A8 만 보이는 다른 모드"""
    rng = np.random.default_rng(3)
    t = np.arange(int(30 * FS)) / FS
    xL = np.array(OLD_ADA["layout_xL"])
    other = np.array([0, 0, 1, 0, 0, 1, 1, 1], float)
    out = []
    for st, n in [("INT40", 10), ("DMG1", 4), ("DMG2", 4), ("DMG3", 4), ("RCV", 2), ("INT30", 2), ("INT50", 2)]:
        for k in range(1, n + 1):
            f1 = F1_REF * (0.97 if st.startswith("DMG") else 1.0) * (1 + 0.0005 * rng.normal())
            f2 = 2.33 * F1_REF * (1 + 0.0005 * rng.normal())
            m1 = np.exp(-0.01 * 2 * np.pi * f1 * t) * np.sin(2 * np.pi * f1 * t + rng.uniform(0, 6))
            m2 = np.exp(-0.01 * 2 * np.pi * f2 * t) * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 6))
            m3 = np.exp(-0.01 * 2 * np.pi * 2.26 * F1_REF * t) * np.sin(2 * np.pi * 2.26 * F1_REF * t)
            sig = (np.outer(m1, np.sin(np.pi * xL)) + 0.4 * np.outer(m2, np.sin(2 * np.pi * xL))
                   + 0.6 * np.outer(m3, other) + 0.02 * rng.normal(size=(len(t), 8)))
            out.append((st, k, sig))
    return out


if __name__ == "__main__":
    if "--test" in sys.argv:
        s, r = evaluate(synthetic())
        print(s[["sensors", "false_alarm", "detect_all", "auc"]].round(3).to_string())
        assert (s.false_alarm == 0).all() and (s.detect_all == 1).all(), "합성 손상(f1 −3%)을 못 잡거나 오경보"
        assert (r[r.state == "INT40"].f2.sub(2.33 * F1_REF).abs() < 0.05).all(), "f2가 다른 모드(2.26 f1)를 잡음"
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    s, r = evaluate(load_free_decay())
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    r.to_csv(OUT / "by_run.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "f1_ref_hz": F1_REF, "f1_band": F1_BAND, "f2_band": F2_BAND, "threshold_pct": THRESHOLD_PCT,
        "configs": CONFIGS, "split": SPLIT, "free_decay_s": OLD_ADA["free_decay_s"],
        "welch": {"nperseg": 2048, "nfft": 16384}, "note": "개발 시험 (손상 데이터를 본 뒤 임계값 결정)"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s.round(4).to_string())
