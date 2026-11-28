"""
A2 HBTA 고유진동수 감시, B2 규칙 이전 (docs/AI1_HBTA_FREQ_SETUP.md)

실행: python src/ai1_hbta_freq.py         → results/public/hbta_freq/{summary.csv, by_record.csv, modes.csv, setup.json}
      python src/ai1_hbta_freq.py --test  → 합성 데이터 점검만
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import roc_auc_score

from config import ORIGINAL_DIR, ROOT

FS, START_S = 100, 90.0
NPS, NFFT = 4096, 32768
SEARCH, PROM, BAND_PCT = (1.0, 30.0), 0.05, 5.0
THRESHOLD_PCT = 1.0                               # B2 그대로
CONFIGS = ["P2_NM_Z", "P1_SM_Y", "P1_SM_Z", "P2_SM_Y", "P2_SM_Z"]
STATES = [f"DS{k}" for k in range(1, 9)]
SETS = {"S2": lambda ax: [("AG03", ax), ("AG05", ax)],
        "SAG": lambda ax: [(f"AG{i:02d}", ax) for i in range(1, 19) if not (i == 9 and ax == "y")]}
OUT = ROOT / "results" / "public" / "hbta_freq"


def name(cfg, state, rep):
    pos, mode, ax = cfg.split("_")
    return f"MVS_{pos}_{state}_{mode}_{ax}_{rep}"


def spectrum(sigs):
    P = 0
    for x in sigs:
        fr, p = signal.welch(signal.detrend(x), FS, nperseg=NPS, nfft=NFFT)
        P = P + p / p.max()
    return fr, P


def peak_in(fr, P, lo, hi):
    m = (fr >= lo) & (fr <= hi)
    fm, pm = fr[m], P[m]
    i = int(np.argmax(pm))
    if 0 < i < len(fm) - 1:
        a, b, c = np.log(pm[i - 1:i + 2])
        return float(fm[i] + 0.5 * (a - c) / (a - 2 * b + c) * (fm[1] - fm[0]))
    return float(fm[i])


def ref_modes(fr, P):
    """기준 측정에서 prominence ≥ PROM 인 봉우리 중 가장 큰 2개"""
    m = (fr >= SEARCH[0]) & (fr <= SEARCH[1])
    Pm = P[m] / P[m].max()
    pk, _ = signal.find_peaks(Pm, prominence=PROM)
    top = sorted(pk, key=lambda i: -Pm[i])[:2]
    return sorted(peak_in(fr, P, fr[m][i] * 0.999, fr[m][i] * 1.001) for i in top)


def evaluate(load):
    """load(record_name, channels) → [채널 신호]"""
    rows, modes = [], []
    for sname, chans in SETS.items():
        for cfg in CONFIGS:
            ax = cfg[-1].lower()
            fr, P = spectrum(load(name(cfg, "UDS", "01"), chans(ax)))
            fa, fb = ref_modes(fr, P)
            modes.append({"sensors": sname, "config": cfg, "f_a": fa, "f_b": fb})
            for st, rep, rl in [("UDS", "02", "test_normal")] + [(s, "01", "damage") for s in STATES]:
                fr, P = spectrum(load(name(cfg, st, rep), chans(ax)))
                ga = peak_in(fr, P, fa * (1 - BAND_PCT / 100), fa * (1 + BAND_PCT / 100))
                gb = peak_in(fr, P, fb * (1 - BAND_PCT / 100), fb * (1 + BAND_PCT / 100))
                da, db = 100 * (ga / fa - 1), 100 * (gb / fb - 1)
                rows.append({"sensors": sname, "config": cfg, "state": st, "role": rl, "f_a": ga, "f_b": gb,
                             "d_a_pct": da, "d_b_pct": db, "score_pct": max(abs(da), abs(db)),
                             "anomaly": max(abs(da), abs(db)) > THRESHOLD_PCT})
    R = pd.DataFrame(rows)
    summ = []
    for sname, g in R.groupby("sensors"):
        nor, dm = g[g.role == "test_normal"], g[g.role == "damage"]
        y = np.r_[np.zeros(len(nor)), np.ones(len(dm))]
        summ.append({"sensors": sname, "threshold_pct": THRESHOLD_PCT, "n_test_normal": len(nor),
                     "false_alarm": nor.anomaly.mean(), "detect_all": dm.anomaly.mean(),
                     "detect_DS1_4": dm[dm.state.isin(STATES[:4])].anomaly.mean(),
                     "detect_DS5_8": dm[dm.state.isin(STATES[4:])].anomaly.mean(),
                     **{f"detect_{s}": dm[dm.state == s].anomaly.mean() for s in STATES},
                     "auc": roc_auc_score(y, np.r_[nor.score_pct, dm.score_pct]),
                     "max_normal_score": nor.score_pct.max()})
    return pd.DataFrame(summ), R, pd.DataFrame(modes)


def synthetic_loader():
    """모드 3.1·7.2Hz(Y) / 6.8·16.9Hz(Z) 랜덤 가진. UDS_02 는 +0.2%, DS 는 두 모드 −3%"""
    rng = np.random.default_rng(7)
    n = int(600 * FS)

    def load(rec, chans):
        st, ax = rec.split("_")[2], rec.split("_")[-2]
        base = (3.1, 7.2) if ax == "Y" else (6.8, 16.9)
        s = 0.97 if st.startswith("DS") else (1.002 if rec.endswith("02") else 1.0)
        out = []
        for k, _ in enumerate(chans):
            x = 0.05 * rng.normal(size=n)
            for j, f0 in enumerate(base):
                b, a = signal.iirpeak(f0 * s, 40, fs=FS)
                x += (1.0 - 0.4 * j) * signal.lfilter(b, a, rng.normal(size=n))
            out.append(x)
        return out
    return load


if __name__ == "__main__":
    if "--test" in sys.argv:
        s, r, m = evaluate(synthetic_loader())
        print(m.round(3).to_string())
        print(s[["sensors", "false_alarm", "detect_all", "auc"]].round(3).to_string())
        assert (s.false_alarm == 0).all() and (s.detect_all == 1).all(), "합성 손상(−3%)을 못 잡거나 오경보"
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    import h5py
    with h5py.File(ORIGINAL_DIR / "hbta" / "data_100Hz.h5", "r") as f:
        s0 = int(START_S * FS)
        s, r, m = evaluate(lambda rec, chans: [f[rec]["acceleration"][c][a][s0:] for c, a in chans])
    meta = pd.read_csv(ROOT / "results" / "public" / "hbta_record_list.csv")
    r["record"] = [name(c, st, "02" if rl == "test_normal" else "01") for c, st, rl in zip(r.config, r.state, r.role)]
    r = r.merge(meta[["측정 이름", "date", "temperature"]].rename(columns={"측정 이름": "record"}), on="record", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    r.to_csv(OUT / "by_record.csv", index=False, encoding="utf-8-sig")
    m.to_csv(OUT / "modes.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "start_s": START_S, "nperseg": NPS, "nfft": NFFT, "search_hz": SEARCH, "prominence": PROM,
        "band_pct": BAND_PCT, "threshold_pct": THRESHOLD_PCT, "configs": CONFIGS,
        "sets": {k: [f"{c}_{a}" for c, a in v("y")] for k, v in SETS.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(m.round(3).to_string())
    print(s.round(3).to_string())
