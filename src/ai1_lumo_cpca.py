"""
D1 LUMO 조건부 PCA + MEWMA, 함부르크 방법 (docs/AI1_LUMO_CPCA_SETUP.md, Neumann et al. 2026)

실행: python src/ai1_lumo_cpca.py         → results/public/lumo_cpca/{summary.csv, by_segment.csv, by_hour.csv, setup.json}
      python src/ai1_lumo_cpca.py --test  → 합성 데이터 점검만
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import ORIGINAL_DIR, ROOT

X_COLS, Z_COL = ["s1", "s2", "s3"], "temp"
KAPPA, ARL0, N_BOOT, L_MAX, SEED = 0.2, 720, 2000, 6000, 20261128
MIN_BLOCKS, EDGE_H = 4, 6
OUT = ROOT / "results" / "public" / "lumo_cpca"


# ---------------------------------------------------------------- 데이터
def load_blocks():
    import h5py
    rows = []
    for p in sorted((ORIGINAL_DIR / "lumo" / "meta_struct").glob("SHMTS_*_meta_struct.mat")):
        with h5py.File(p, "r") as f:
            d = f["Dat"]
            s = lambda r: "".join(chr(int(c)) for c in f[r][()].flatten())
            nm = [s(r) for r in d["ChannelNames"][()].flatten()]
            D = d["Data"][()]
            if D.shape[0] != len(nm):
                D = D.T
            I = d["Info"][()]
            for j in range(I.shape[1]):
                rows.append({"file": s(I[0, j]), "code": int(f[I[1, j]][()].flatten()[0]),
                             "s1": D[nm.index("mean-strain01"), j], "s2": D[nm.index("mean-strain02"), j],
                             "s3": D[nm.index("mean-strain03"), j], "temp": D[nm.index("mean-temp01"), j]})
        print(f"  {p.name}", flush=True)
    B = pd.DataFrame(rows)
    B["t"] = pd.to_datetime(B.file.str.extract(r"(\d{12})")[0], format="%Y%m%d%H%M")
    return B.sort_values("t", ignore_index=True)


def to_hours(B):
    """10분 블록 → 1시간 평균, 구간 이름·역할"""
    B = B.copy()
    B["seg"] = (B.code != B.code.shift()).cumsum()
    cv = pd.read_csv(ORIGINAL_DIR / "lumo" / "lumo_v4_blocks.csv", parse_dates=["t_end"])
    names = {}
    for sid, g in B.groupby("seg"):
        hit = cv[(cv.t_end >= g.t.min()) & (cv.t_end <= g.t.max())].cv_group
        if len(hit):
            n = hit.iloc[0]
            names[sid] = n.split("_")[0] if n.startswith("N") else "_".join(n.split("_")[:2])
    B["name"] = B.seg.map(names)
    change = B.t[B.code != B.code.shift()].iloc[1:].to_numpy()
    B["hour"] = B.t.dt.floor("h")
    H = B.groupby("hour").agg(n=("t", "size"), code_min=("code", "min"), code_max=("code", "max"),
                              name=("name", "first"), name_n=("name", "nunique"),
                              **{c: (c, "mean") for c in X_COLS + [Z_COL]}).reset_index()
    near = np.zeros(len(H), bool)
    for c in change:
        near |= np.abs((H.hour + pd.Timedelta(minutes=30) - c).dt.total_seconds()) <= EDGE_H * 3600
    keep = (H.n >= MIN_BLOCKS) & (H.code_min == H.code_max) & (H.code_min != 1) & ~near & H.name.notna() & (H.name_n == 1)
    keep &= H[X_COLS + [Z_COL]].notna().all(axis=1)
    H = H[keep].reset_index(drop=True)
    H["role"] = np.select([H.name == "N1", H.name.str.startswith("N"), H.name.str.endswith("_111"),
                           H.name.str.endswith("_010")], ["phase1", "normal", "big", "small"], "other")
    return H.rename(columns={"hour": "t"})


# ---------------------------------------------------------------- 방법
def kernel_w(zq, z, h):
    return np.exp(-0.5 * ((zq[:, None] - z[None, :]) / h) ** 2)


def reconstruct(H, method):
    """Phase I 표준화 공간에서 x̃ (n, 3)"""
    P = (H.role == "phase1").to_numpy()
    X = H[X_COLS].to_numpy(float)
    mu, sd = X[P].mean(0), X[P].std(0)
    X = (X - mu) / sd
    if method == "unsupervised":
        return X
    z, zP = H[Z_COL].to_numpy(float), H.loc[P, Z_COL].to_numpy(float)
    h = 1.06 * zP.std() * P.sum() ** (-0.2)                       # Silverman
    W = kernel_w(z, zP, h)
    W /= W.sum(1, keepdims=True)
    m = W @ X[P]                                                  # m(z) 모든 시간
    R = X - m
    xbar = X[P].mean(0)
    if method == "partial":
        return xbar + R
    RP = R[P]
    lam, A = np.linalg.eigh(np.cov(RP, rowvar=False))
    lam, A = lam[::-1], A[:, ::-1]
    S = np.einsum("qi,ij,ik->qjk", W, RP, RP)                     # Σ(z) 모든 시간
    out = np.empty_like(X)
    for i in range(len(X)):
        lz, Az = np.linalg.eigh(S[i])
        lz, Az = lz[::-1], Az[:, ::-1]
        Az = Az * np.sign(np.sum(Az * A, axis=0))                 # 고유벡터 부호를 일반 PCA에 맞춤
        out[i] = xbar + A @ (np.sqrt(lam) / np.sqrt(np.maximum(lz, 1e-12)) * (Az.T @ R[i]))
    return out


def mewma(Xt, mu0, Sinv):
    Z = np.zeros(Xt.shape[-1])
    T = np.empty(len(Xt))
    for i, x in enumerate(Xt):
        Z = KAPPA * (x - mu0) + (1 - KAPPA) * Z
        T[i] = Z @ Sinv @ Z
    return T


def calibrate_h(XP, days, rng):
    """Phase I 하루 단위 블록 부트스트랩 → 첫 경보까지 평균 시간 = ARL0 인 h"""
    mu0, S = XP.mean(0), np.cov(XP, rowvar=False)
    Sinv = np.linalg.inv(KAPPA / (2 - KAPPA) * S)
    groups = [np.flatnonzero(days == d) for d in np.unique(days)]
    T = np.empty((N_BOOT, L_MAX))
    Z = np.zeros((N_BOOT, XP.shape[1]))
    seqs = []
    for b in range(N_BOOT):
        idx = []
        while len(idx) < L_MAX:
            idx.extend(groups[rng.integers(len(groups))])
        seqs.append(idx[:L_MAX])
    seqs = np.array(seqs)
    for t in range(L_MAX):
        Z = KAPPA * (XP[seqs[:, t]] - mu0) + (1 - KAPPA) * Z
        T[:, t] = np.einsum("bi,ij,bj->b", Z, Sinv, Z)
    runmax = np.maximum.accumulate(T, axis=1)

    def arl(h):
        hit = runmax > h
        return np.where(hit.any(1), hit.argmax(1) + 1, L_MAX).mean()
    lo, hi = 0.0, float(T.max()) * 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if arl(mid) < ARL0 else (lo, mid)
    return hi, mu0, Sinv, arl(hi)


def evaluate(H):
    rng = np.random.default_rng(SEED)
    P = (H.role == "phase1").to_numpy()
    days = H.t.dt.date.to_numpy()
    zP = H.loc[P, Z_COL]
    H = H.copy()
    H["outside_phase1_temp"] = (H[Z_COL] < zP.min()) | (H[Z_COL] > zP.max())
    rows, segs = [], []
    for method in ("conditional", "partial", "unsupervised", "IForest"):
        if method == "IForest":
            F = H[X_COLS + [Z_COL]].to_numpy(float)
            F = (F - F[P].mean(0)) / F[P].std(0)
            iso = IsolationForest(n_estimators=300, random_state=SEED).fit(F[P])
            score = -iso.score_samples(F)
            h, arl = float(np.quantile(score[P], 0.99)), np.nan
        else:
            Xt = reconstruct(H, method)
            h, mu0, Sinv, arl = calibrate_h(Xt[P], days[P], rng)
            score = mewma(Xt, mu0, Sinv)
        H[f"score_{method}"] = score
        H[f"flag_{method}"] = score > h
        f = lambda m: float(H.loc[m, f"flag_{method}"].mean())
        fa, big, small = f(H.role == "normal"), f(H.role == "big"), f(H.role == "small")
        rows.append({"method": method, "h": h, "arl_check": arl, "n_phase1": int(P.sum()),
                     "flag_phase1": f(H.role == "phase1"), "false_alarm": fa, "detect_big": big, "detect_small": small,
                     "detect_minus_fa": big - fa})
        for name, g in H.groupby("name"):
            segs.append({"method": method, "segment": name, "role": g.role.iloc[0], "t0": g.t.min(), "t1": g.t.max(),
                         "hours": len(g), "flagged": float(g[f"flag_{method}"].mean()),
                         "temp_min": g[Z_COL].min(), "temp_max": g[Z_COL].max(),
                         "outside_phase1_temp": float(g.outside_phase1_temp.mean())})
    return pd.DataFrame(rows), pd.DataFrame(segs).sort_values(["method", "t0"]), H


def synthetic():
    """1년 시간 자료. 온도 = 계절 + 일교차, 변형률 = 온도 함수 + 온도에 따라 바뀌는 상관 잡음. 큰 손상: s2 +4σ"""
    rng = np.random.default_rng(11)
    t = pd.date_range("2020-08-01", "2021-07-31 23:00", freq="h")
    doy = np.arange(len(t)) / 24
    # 계절 변화를 작게 해 Phase II 온도가 Phase I 범위 안에 들도록 (코드 점검용. 범위 밖 외삽 오경보는 실제 데이터에서 본다)
    temp = 15 + 1.5 * np.cos(2 * np.pi * doy / 365) + 5 * np.sin(2 * np.pi * doy)
    rho = np.clip((temp - 5) / 30, -0.5, 0.9)
    e1, e2 = rng.normal(size=len(t)), rng.normal(size=len(t))
    s1 = 0.3 * temp + e1
    s2 = -0.2 * temp + rho * e1 + np.sqrt(1 - rho ** 2) * e2
    s3 = 0.05 * temp ** 1.5 + 0.5 * rng.normal(size=len(t))
    H = pd.DataFrame({"t": t, "s1": s1, "s2": s2, "s3": s3, "temp": temp})
    m = t.month
    H["name"] = np.select([(m >= 8) & (m <= 9), m == 10, m == 11, m == 12, (m >= 1) & (m <= 2)],
                          ["N1", "DAM6_111", "N2", "DAM4_010", "N3"], "N4")
    H.loc[H.name == "DAM6_111", "s2"] += 4.0
    H.loc[H.name == "DAM4_010", "s2"] += 0.5
    H["role"] = np.select([H.name == "N1", H.name.str.startswith("N"), H.name.str.endswith("_111"),
                           H.name.str.endswith("_010")], ["phase1", "normal", "big", "small"], "other")
    return H


if __name__ == "__main__":
    if "--test" in sys.argv:
        globals()["N_BOOT"], globals()["L_MAX"] = 300, 3000
        s, g, _ = evaluate(synthetic())
        print(s.round(3).to_string())
        r = s.set_index("method")
        assert r.loc["conditional", "false_alarm"] <= 0.1 and r.loc["conditional", "detect_big"] >= 0.8, "합성 점검 실패"
        assert abs(r.loc["conditional", "arl_check"] - ARL0) / ARL0 < 0.1
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    print("월별 메타 파일 읽는 중 (12개)", flush=True)
    H = to_hours(load_blocks())
    print(H.groupby("role").size().to_string(), flush=True)
    s, g, H = evaluate(H)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    g.to_csv(OUT / "by_segment.csv", index=False, encoding="utf-8-sig")
    H.to_csv(OUT / "by_hour.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "x": ["mean-strain01", "mean-strain02", "mean-strain03"], "z": "mean-temp01", "kappa": KAPPA, "arl0_h": ARL0,
        "n_boot": N_BOOT, "l_max": L_MAX, "min_blocks_per_hour": MIN_BLOCKS, "edge_exclusion_h": EDGE_H,
        "bandwidth": "Silverman (Phase I temp)", "seed": SEED}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(s.round(4).to_string())
