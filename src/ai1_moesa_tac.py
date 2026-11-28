"""
E1 Ponte Moesa 1-TAC (docs/AI1_MOESA_TAC_SETUP.md, Garcia-Ramonda et al. 2025) — 개발 시험

실행: python src/ai1_moesa_tac.py         → results/public/moesa_tac/{summary.csv, by_state.csv, by_record.csv, setup.json}
      python src/ai1_moesa_tac.py --test  → 합성 데이터 점검만
"""
import io
import json
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from config import MOESA, ORIGINAL_DIR, ROOT
from convert_public import moesa_day, moesa_time_us

FS = MOESA["sample_rate_hz"]
WIN = 60 * FS                                   # 60초 구간, 겹침 없음
NPS = 2048                                      # Welch 10.24초, 50% 겹침
BAND = (3.0, 12.0)                              # TAC 대역 [Hz]
RMS_MAX = 0.008 * 9.80665                       # 0.008g [m/s²]
F1_BAND = (4.4, 5.4)                            # 수직 1차 4.9Hz (보고만)
SENSORS = (1, 2, 3)                             # 고정 센서, Z축
PAIRS = {"P12": (1, 2), "P23": (3, 2), "P13": (1, 3)}   # (a, b): T_ab = G_ab / G_bb
SEED = 20261128
DAMAGE = ("DS3", "DS4", "DS5", "DS6", "DS7")
OUT = ROOT / "results" / "public" / "moesa_tac"


def role(no, ds):
    if no == 10:
        return "ref"
    if no == 11:
        return "thr"
    if 12 <= no <= 18:
        return "test_normal"
    if ds in DAMAGE:
        return "damage"
    return "report"                              # AMB_1–9 (DS0·DS1, DS2 저녁)


def load_records():
    """[(amb, ds, role, (T, 3) Z축 1·2·3번)] — 공식 측정 목록 Ambient.txt 시각 그대로"""
    with zipfile.ZipFile(ORIGINAL_DIR / "moesa" / "Acceleration_Data.zip") as z:
        amb = pd.read_csv(io.BytesIO(z.read("Ambient.txt")), sep="\t")
    amb.columns = [c.strip() for c in amb.columns]
    out = []
    for day in sorted({int(d[:2]) for d in amb["DAY"]}):
        f, t_us, ch = moesa_day(day)
        for _, r in amb[amb["DAY"].str.startswith(f"{day:02d}")].iterrows():
            no, ds = int(r["ID"].split("_")[1]), f"DS{int(r['DAMAGE STATE'])}"
            s, e = moesa_time_us(f"{day} {r['Time Start']}"), moesa_time_us(f"{day} {r['Time End']}")
            i0, i1 = np.searchsorted(t_us, [s, e])
            x = np.stack([ch[f"Z{n}"][0, i0:i1].astype(float) for n in SENSORS], axis=1)
            out.append((r["ID"], ds, role(no, ds), x))
            print(f"  {r['ID']:7s} {ds} {role(no, ds):11s} {len(x) / FS / 60:7.1f}분", flush=True)
        f.close()
    return out


# ---------------------------------------------------------------- 구간 계산
SOS = signal.butter(4, (2.0, 25.0), btype="bandpass", fs=FS, output="sos")


def window_stats(x):
    """(T, 3) 측정 → 구간별 {rms (3,), G_ab {쌍: 복소 벡터}, G_bb, psd 합, f}"""
    y = signal.sosfiltfilt(SOS, x, axis=0)
    rows = []
    for k in range(len(y) // WIN):
        w = y[k * WIN:(k + 1) * WIN]
        f, P = signal.welch(w, FS, nperseg=NPS, axis=0)
        m = (f >= BAND[0]) & (f <= BAND[1])
        G = {}
        for name, (a, b) in PAIRS.items():
            ia, ib = SENSORS.index(a), SENSORS.index(b)
            _, gab = signal.csd(w[:, ib], w[:, ia], FS, nperseg=NPS)   # conj(X_b) X_a = G_ab
            G[name] = (gab[m], P[m, ib])
        rows.append({"k": k, "rms": np.sqrt((w ** 2).mean(0)), "G": G, "f": f, "psd": P.sum(1)})
    return rows


def tac(td, tr):
    return np.abs(np.vdot(td, tr)) ** 2 / (np.vdot(td, td).real * np.vdot(tr, tr).real)


def peak(f, p, lo, hi):
    m = (f >= lo) & (f <= hi)
    fm, pm = f[m], p[m]
    i = int(np.argmax(pm))
    if 0 < i < len(fm) - 1:
        a, b, c = np.log(pm[i - 1:i + 2])
        return float(fm[i] + 0.5 * (a - c) / (a - 2 * b + c) * (fm[1] - fm[0]))
    return float(fm[i])


def evaluate(records):
    W = []                                                    # 구간 표
    for amb, ds, rl, x in records:
        for s in window_stats(x):
            W.append({"amb": amb, "ds": ds, "role": rl, **s})
    fb = W[0]["f"][(W[0]["f"] >= BAND[0]) & (W[0]["f"] <= BAND[1])]
    edges = np.arange(3, 13)
    valid = lambda w, idx: max(w["rms"][i] for i in idx) <= RMS_MAX
    tab = []
    for w in W:
        row = {"amb": w["amb"], "ds": w["ds"], "role": w["role"], "k": w["k"],
               "valid_all": valid(w, range(3)), "f1": peak(w["f"], w["psd"], *F1_BAND)}
        for name, (a, b) in PAIRS.items():
            row[f"valid_{name}"] = valid(w, (SENSORS.index(a), SENSORS.index(b)))
        tab.append(row)
    T = pd.DataFrame(tab)
    ref = [i for i, w in enumerate(W) if w["role"] == "ref"]
    feats = {}
    for name in PAIRS:
        ok = [i for i in ref if T.loc[i, f"valid_{name}"]]
        tr = sum(W[i]["G"][name][0] for i in ok) / sum(W[i]["G"][name][1] for i in ok)
        td = [w["G"][name][0] / w["G"][name][1] for w in W]
        T[f"score_{name}"] = [1 - tac(d, tr) for d in td]
        feats[name] = np.array([[np.log(np.abs(d[(fb >= lo) & (fb < lo + 1)])).mean() for lo in edges[:-1]] for d in td])
    T["score_E1"] = T[[f"score_{p}" for p in PAIRS]].mean(axis=1)
    X = np.hstack([feats[p] for p in PAIRS])
    okref = T.index[(T.role == "ref") & T.valid_all]
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[okref])
    T["score_IForest"] = -iso.score_samples(X)
    mu = T.loc[okref, "f1"].mean()
    T["score_f1"] = (T.f1 / mu - 1).abs() * 100
    return T


def judge(T):
    rows, states = [], []
    models = {"E1": "valid_all", "IForest": "valid_all", "f1": "valid_all",
              **{p: f"valid_{p}" for p in PAIRS}}
    for model, vcol in models.items():
        sc = f"score_{model}"
        V = T[T[vcol]]
        thr = float(np.quantile(V.loc[V.role == "thr", sc], 0.99))
        nor = V.loc[V.role == "test_normal", sc]
        dm = V[V.ds.isin(DAMAGE)]
        y = np.r_[np.zeros(len(nor)), np.ones(len(dm))]
        rows.append({"model": model, "threshold": thr, "n_ref": int((V.role == "ref").sum()),
                     "n_thr": int((V.role == "thr").sum()), "n_test_normal": len(nor),
                     "false_alarm": float((nor > thr).mean()),
                     **{f"detect_{d}": float((V.loc[V.ds == d, sc] > thr).mean()) for d in DAMAGE},
                     "detect_all": float((dm[sc] > thr).mean()), "auc": float(roc_auc_score(y, np.r_[nor, dm[sc]]))})
        for (rl, ds), g in V.groupby(["role", "ds"]):
            states.append({"model": model, "role": rl, "ds": ds, "n": len(g), "flagged": float((g[sc] > thr).mean()),
                           "median_score": float(g[sc].median())})
        T[f"flag_{model}"] = (T[sc] > thr) & T[vcol]
    excl = T.groupby(["role", "ds"]).agg(n=("k", "size"), kept_all=("valid_all", "mean")).reset_index()
    return pd.DataFrame(rows), pd.DataFrame(states), excl


def synthetic():
    """3센서, 모드 2개(4.9·7.3Hz) 랜덤 가진. 손상: 1번 센서의 2차 모드 형상 0.6배, 2차 2% 낮음"""
    rng = np.random.default_rng(5)
    n = 30 * WIN

    def rec(amb, ds, rl, damaged):
        f2 = 7.3 * (0.98 if damaged else 1.0)
        q = []
        for f0 in (4.9, f2):
            b, a = signal.iirpeak(f0, 30, fs=FS)
            q.append(signal.lfilter(b, a, rng.normal(size=n)))
        phi1 = np.array([0.7, 1.0, 0.8])
        phi2 = np.array([0.6 if damaged else 1.0, 0.2, -0.9])
        x = np.outer(q[0], phi1) + 0.8 * np.outer(q[1], phi2) + 0.05 * rng.normal(size=(n, 3))
        return (amb, ds, rl, 0.01 * x)
    return ([rec("AMB_10", "DS2", "ref", False), rec("AMB_11", "DS2", "thr", False),
             rec("AMB_18", "DS2", "test_normal", False)]
            + [rec(f"AMB_{20 + i}", d, "damage", True) for i, d in enumerate(DAMAGE)])


if __name__ == "__main__":
    if "--test" in sys.argv:
        T = evaluate(synthetic())
        s, _, _ = judge(T)
        print(s[["model", "false_alarm", "detect_all", "auc"]].round(3).to_string())
        r = s.set_index("model")
        assert r.loc["E1", "auc"] > 0.9 and r.loc["E1", "false_alarm"] <= 0.1, "1-TAC 가 합성 손상을 못 잡음"
        assert r.loc["IForest", "auc"] > 0.8, "IForest 가 합성 손상을 못 잡음"
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    print("데이터 읽는 중 (.mat 4개)", flush=True)
    T = evaluate(load_records())
    s, b, excl = judge(T)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "by_state.csv", index=False, encoding="utf-8-sig")
    excl.to_csv(OUT / "rms_kept.csv", index=False, encoding="utf-8-sig")
    rec = T.groupby(["amb", "ds", "role"]).agg(n=("k", "size"), kept=("valid_all", "mean"),
                                               **{f"flag_{m}": (f"flag_{m}", "mean") for m in ("E1", "IForest", "f1")},
                                               f1_median=("f1", "median")).reset_index()
    rec.to_csv(OUT / "by_record.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "win_s": WIN / FS, "nperseg": NPS, "band_hz": BAND, "rms_max_ms2": RMS_MAX, "pairs": PAIRS,
        "sensors_z": SENSORS, "threshold": "thr 역할 99% 분위수", "f1_band": F1_BAND, "seed": SEED,
        "note": "개발 시험 (Moesa 라벨·논문 결과를 이미 봄)"}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s.round(4).to_string())
