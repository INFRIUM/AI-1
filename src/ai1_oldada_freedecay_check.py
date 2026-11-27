"""
Old ADA 자유 감쇠 구간 주행별 1·2차 고유진동수 확인 (모델 없음, 서술 확인만)

v1·v2 TCN-GAT(차량 통과 구간)가 손상을 구분하지 못해, 하중이 빠진 뒤 구간에 손상 정보가 있는지 확인한다.
손상 주행까지 모두 보는 확인이므로, 이 뒤에 Old ADA로 하는 자유 감쇠 모델 시험은 독립 검증이 아니라 개발 시험이다.

실행: python src/ai1_oldada_freedecay_check.py → results/public/oldada_freedecay/{freq_by_run.csv, freq_by_state.csv}
"""
import io
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

from config import OLD_ADA, ORIGINAL_DIR, ROOT

FS = OLD_ADA["sample_rate_hz"]
OUT = ROOT / "results" / "public" / "oldada_freedecay"
ORDER = ["INT30", "INT40", "INT50", "DMG1", "DMG2", "RCV", "DMG3"]
# f2: A3·A6–A8 에는 6.26Hz 다른 모드가 섞여 보여, 같은 쪽 중앙이 아닌 A1·A2·A4·A5 로도 따로 계산 (열 번호 1부터)
F2_SIDE = [1, 2, 4, 5]


def load_free_decay():
    """[(state, run_no, (T, 8) 진출 후 자유 감쇠)]"""
    with zipfile.ZipFile(ORIGINAL_DIR / OLD_ADA["zip"]) as z:
        name = next(n for n in z.namelist() if n.endswith(".mat"))
        vba = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)["VBA"]
    t0, t1 = OLD_ADA["free_decay_s"]
    out = []
    for r in range(1, vba.shape[0]):
        case, speed = str(vba[r, 0]).split()
        state = f"INT{speed.removesuffix('km')}" if case == "INT" else case
        for c in range(1, vba.shape[1]):
            x = vba[r, c]
            if not x.size:
                continue
            on = np.flatnonzero(np.diff((x[:, OLD_ADA["optical_exit_col"]] > 0.5).astype(int)) == 1)[0]
            out.append((state, c, x[on + int(t0 * FS): on + int(t1 * FS), 1:9].astype(float)))
    return out


def psd(seg):
    return signal.welch(signal.detrend(seg, axis=0), FS, nperseg=2048, nfft=16384, axis=0)


def peak(f, p, lo, hi):
    """lo–hi Hz 최대 봉우리, 포물선 보간"""
    m = (f >= lo) & (f <= hi)
    fm, pm = f[m], p[m]
    i = int(np.argmax(pm))
    if 0 < i < len(fm) - 1:
        a, b, c = np.log(pm[i - 1:i + 2])
        return float(fm[i] + 0.5 * (a - c) / (a - 2 * b + c) * (fm[1] - fm[0]))
    return float(fm[i])


if __name__ == "__main__":
    rows = []
    for st, no, seg in load_free_decay():
        f, p = psd(seg)
        rows.append({"state": st, "run": no, "f1_all": peak(f, p.sum(1), 2.0, 4.0),
                     "f2_all": peak(f, p.sum(1), 5.5, 8.0),
                     "f2_side": peak(f, p[:, [c - 1 for c in F2_SIDE]].sum(1), 5.5, 8.0),
                     "rms": float(np.sqrt((seg ** 2).mean()))})
    d = pd.DataFrame(rows)
    ref = d[d.state == "INT40"]
    summ = []
    for st in ORDER:
        s = d[d.state == st]
        row = {"state": st, "n": len(s)}
        for k in ("f1_all", "f2_all", "f2_side"):
            row |= {f"{k}_mean": s[k].mean(), f"{k}_std": s[k].std(), f"{k}_min": s[k].min(), f"{k}_max": s[k].max(),
                    f"{k}_vs_INT40_pct": 100 * (s[k].mean() / ref[k].mean() - 1)}
        summ.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "freq_by_run.csv", index=False, encoding="utf-8-sig")
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "freq_by_state.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(S[["state", "n"] + [c for c in S if c.endswith(("_mean", "_std", "_pct"))]].round(4).to_string())
