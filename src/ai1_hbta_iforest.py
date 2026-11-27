"""
HBTA Isolation Forest 비교 (docs/AI1_HBTA_AR_MSD_SETUP.md "추가: Isolation Forest 비교")
AR+MSD 와 같은 AR(5) 특징·그룹·시험(R1·R2·R3)을 쓰고 탐지기만 Isolation Forest 로 바꾼다.

실행: python src/ai1_hbta_iforest.py          → results/public/hbta_ar_msd/{iforest_summary.csv, iforest_by_state.csv}
      python src/ai1_hbta_iforest.py --test   → 합성 데이터 점검만
"""
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from ai1_hbta_ar_msd import GROUPS, N_TRAIN, OUT, STATES, features, synthetic
from config import IFOREST, ORIGINAL_DIR


def evaluate(feat, run):
    if run == "R3":
        base, normal = feat["UDS1"][:N_TRAIN], {"UDS2": feat["UDS2"]}
    else:
        base = np.concatenate([feat["UDS1"][:N_TRAIN], feat["UDS2"][:N_TRAIN]])
        normal = {"UDS1": feat["UDS1"][N_TRAIN:], "UDS2": feat["UDS2"][N_TRAIN:]}
    iso = IsolationForest(n_estimators=IFOREST["n_estimators"], max_samples=IFOREST["max_samples"],
                          random_state=IFOREST["random_state"]).fit(base)
    thr = float(np.quantile(-iso.score_samples(base), IFOREST["threshold_quantile"]))
    score = {st: -iso.score_samples(x) for st, x in {**normal, **{s: feat[s] for s in STATES[2:]}}.items()}
    nor = np.concatenate([score[s] for s in normal])
    dam = np.concatenate([score[s] for s in STATES[2:]])
    y = np.r_[np.zeros(len(nor)), np.ones(len(dam))]
    summ = {"threshold": thr, "n_train": len(base), "dims": base.shape[1], "n_normal": len(nor), "n_damage": len(dam),
            "false_alarm": float((nor > thr).mean()), "missed": float((dam <= thr).mean()),
            "auc": float(roc_auc_score(y, np.r_[nor, dam]))}
    by = [{"state": s, "n": len(v), "flagged": float((v > thr).mean()), "median_score": float(np.median(v))} for s, v in score.items()]
    return summ, by


def run(source):
    rows, by_rows = [], []
    for g, ch in GROUPS.items():
        feat = source(ch)
        for r in ("R1", "R3") if g != "two_sensors" else ("R2", "R3"):
            s, b = evaluate(feat, r)
            rows.append({"run": r, "group": g, "model": "IForest", **s})
            by_rows += [{"run": r, "group": g, "model": "IForest", **x} for x in b]
    return pd.DataFrame(rows), pd.DataFrame(by_rows)


if __name__ == "__main__":
    if "--test" in sys.argv:
        s, b = run(synthetic)
        assert (s["auc"] > 0.99).all() and (s.loc[s["run"] != "R3", "missed"] < 0.05).all(), s
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "iforest_summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    with h5py.File(ORIGINAL_DIR / "hbta" / "data_100Hz.h5", "r") as f:
        s, b = run(lambda ch: features(f, ch))
    s.to_csv(OUT / "iforest_summary.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "iforest_by_state.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
