"""
C1 Vänersborg 개폐 재하 변형률 감시 (docs/AI1_VANERSBORG_STRAIN_SETUP.md) — 개발 시험

실행: python src/ai1_vanersborg_strain.py         → results/public/vanersborg_strain/{summary.csv, by_file.csv, setup.json}
      python src/ai1_vanersborg_strain.py --test  → 합성 데이터 점검만
"""
import json
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from config import ORIGINAL_DIR, ROOT, VANERSBORG

SG = {"SG9": "ch_9", "SG10": "ch_10"}
REPORT_SG = {"SG11": "ch_11"}
INCLINE, TEMP = "ch_25", "ch_30"
Z_STAR, Z_PAPER, SEED = 3.0, 2.0, 20261128
OUT = ROOT / "results" / "public" / "vanersborg_strain"


def role(k):
    if k < 20:
        return "train"
    if k < 40:
        return "calib"
    if k < 57:
        return "test_normal"
    return "damage"                                   # 57 = 진전 중, 58–63 = 진전 후


def file_features(d):
    inc = d[INCLINE].to_numpy(float)
    closed = inc >= np.nanmax(inc) - 1.0
    opened = inc <= np.nanmin(inc) + 1.0
    row = {"temp": float(d.loc[closed, TEMP].mean()), "n_closed": int(closed.sum()), "n_open": int(opened.sum())}
    for name, ch in {**SG, **REPORT_SG}.items():
        x = d[ch].to_numpy(float)
        row[f"open_{name}"] = float(np.median(x[opened]) - np.median(x[closed]))
        row[f"static_{name}"] = float(x[closed].mean())
    return row


def load_files():
    cols = [INCLINE, TEMP, *SG.values(), *REPORT_SG.values()]
    rows = []
    with zipfile.ZipFile(ORIGINAL_DIR / VANERSBORG["zip"]) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".csv"))
        assert len(names) == 64 and names[57][:-4] == VANERSBORG["dur_event"]
        for k, n in enumerate(names):
            with z.open(n) as f:
                rows.append({"k": k, "file": n[:-4], **file_features(pd.read_csv(f, usecols=cols))})
            print(f"  {k:2d} {n}", flush=True)
    return pd.DataFrame(rows)


def zscores(D, feat, fit_mask):
    """특징마다 기온 1차 회귀(fit_mask 파일) → 잔차 / 학습 잔차 σ"""
    Z = {}
    for name in SG:
        y, t = D[f"{feat}_{name}"], D["temp"]
        b, a = np.polyfit(t[fit_mask], y[fit_mask], 1)
        r = y - (a + b * t)
        Z[name] = r / r[fit_mask].std(ddof=1)
    return pd.DataFrame(Z)


def evaluate(D):
    D = D.copy()
    D["role"] = [role(k) for k in D.k]
    tr = (D.role == "train").to_numpy()
    for model, feat in (("C1", "open"), ("C1_static", "static")):
        Z = zscores(D, feat, tr)
        for name in SG:
            D[f"z_{model}_{name}"] = Z[name]
        D[f"score_{model}"] = Z.abs().max(axis=1)
        D[f"flag_{model}"] = D[f"score_{model}"] > Z_STAR
    X = D[[f"open_{n}" for n in SG] + [f"static_{n}" for n in SG] + ["temp"]].to_numpy()
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[tr])
    D["score_IForest"] = -iso.score_samples(X)
    thr_if = float(D.loc[D.role == "calib", "score_IForest"].max())
    D["flag_IForest"] = D["score_IForest"] > thr_if
    # 논문 재현 R0: 처음 5개 파일, SG10 닫힘 평균, Z ≥ 2
    first5 = (D.k < 5).to_numpy()
    y, t = D["static_SG10"], D["temp"]
    b, a = np.polyfit(t[first5], y[first5], 1)
    r = y - (a + b * t)
    D["z_paper_R0"] = r / r[first5].std(ddof=1)
    D["flag_paper_R0"] = D["z_paper_R0"] >= Z_PAPER
    summ = []
    for model, thr in (("C1", Z_STAR), ("C1_static", Z_STAR), ("IForest", thr_if)):
        nor, dm = D[D.role == "test_normal"], D[D.role == "damage"]
        y = np.r_[np.zeros(len(nor)), np.ones(len(dm))]
        summ.append({"model": model, "threshold": thr, "n_test_normal": len(nor), "n_damage": len(dm),
                     "false_alarm": nor[f"flag_{model}"].mean(), "flag_calib": D.loc[D.role == "calib", f"flag_{model}"].mean(),
                     "detect": dm[f"flag_{model}"].mean(), "detect_DUR": float(D.loc[D.k == 57, f"flag_{model}"].iloc[0]),
                     "detect_POST": dm[dm.k > 57][f"flag_{model}"].mean(),
                     "auc": roc_auc_score(y, np.r_[nor[f"score_{model}"], dm[f"score_{model}"]])})
    pre = D[D.k < 57]
    summ.append({"model": "paper_R0", "threshold": Z_PAPER, "n_test_normal": int((pre.k >= 5).sum()),
                 "false_alarm": pre[pre.k >= 5]["flag_paper_R0"].mean(), "detect": D[D.k >= 57]["flag_paper_R0"].mean()})
    return pd.DataFrame(summ), D


def synthetic():
    """기온 −8–8°C, 닫힘 변형률 = 700 + 39×기온 + 잡음, 재하 응답 280 ± 8. 손상(57–63): 재하 응답 +60%"""
    rng = np.random.default_rng(9)
    rows = []
    for k in range(64):
        t = rng.uniform(-8, 8)
        row = {"k": k, "file": f"syn_{k}", "temp": t}
        for i, n in enumerate(SG):
            row[f"static_{n}"] = 700 + 39 * t + rng.normal(0, 15)
            row[f"open_{n}"] = (280 + 20 * i) * (1.6 if k >= 57 else 1.0) + 2 * t + rng.normal(0, 8)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "--test" in sys.argv:
        s, D = evaluate(synthetic())
        print(s.round(3).to_string())
        r = s.set_index("model")
        assert r.loc["C1", "false_alarm"] == 0 and r.loc["C1", "detect"] == 1, "합성 손상을 못 잡거나 오경보"
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    print("파일 읽는 중 (64개, 약 2.6GB)", flush=True)
    s, D = evaluate(load_files())
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    D.to_csv(OUT / "by_file.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "sg": SG, "report_sg": REPORT_SG, "incline": INCLINE, "temp": TEMP, "z_star": Z_STAR, "z_paper": Z_PAPER,
        "closed": "incline >= max - 1°", "open": "incline <= min + 1°", "roles": "0–19 train / 20–39 calib / 40–56 test / 57–63 damage",
        "seed": SEED, "note": "개발 시험 (설계 중 손상 파일 2개 값을 봄)"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())
