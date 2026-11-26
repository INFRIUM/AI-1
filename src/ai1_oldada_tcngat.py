"""
Old ADA TCN-GAT / TCN 단독 / Isolation Forest (docs/AI1_OLDADA_TCNGAT_SETUP.md, Ni et al. 2025)

실행: python src/ai1_oldada_tcngat.py           → results/public/oldada_tcngat/{summary.csv, by_state.csv, by_run.csv, setup.json}
      python src/ai1_oldada_tcngat.py --test    → 합성 데이터 점검만
"""
import io
import json
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.stats import kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from torch import nn

from config import OLD_ADA, ORIGINAL_DIR, ROOT

FS, WIN, STRIDE, SEED = 200, 128, 4, 20261126
EPOCHS, BATCH, LR, DROPOUT = 60, 32, 1e-3, 0.3      # LR: 논문 5.68e-3은 합성 점검에서 출력이 0으로 붕괴 → 1e-3 (설정 문서 참고)
TCN_CH, TCN_K, POOL, GAT_HEADS, GAT_H1, GAT_H2 = 64, 32, 16, 4, 64, 8
SQ = 8                                              # 인코더: 1×1 합성곱으로 채널 64 → 8, 시간 128점 유지 (평균 풀링은 고주파를 지워 사용 안 함)
SPLIT = OLD_ADA["int40_split"]                      # train (1–4) / ai2_calib → 임계값 (5–7) / test (8–10)
SENSORS = {"S8": list(range(1, 9)), "S2": [1, 3]}
JUDGED = ["INT40", "DMG1", "DMG2", "DMG3"]
REPORT = ["RCV", "INT30", "INT50"]
OUT = ROOT / "results" / "public" / "oldada_tcngat"


# ---------------------------------------------------------------- 데이터
def load_runs():
    """[(state, run_no, (T, 8) 통과 구간 가속도)]"""
    with zipfile.ZipFile(ORIGINAL_DIR / OLD_ADA["zip"]) as z:
        name = next(n for n in z.namelist() if n.endswith(".mat"))
        vba = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)["VBA"]
    runs = []
    for r in range(1, vba.shape[0]):
        case, speed = str(vba[r, 0]).split()
        state = f"INT{speed.removesuffix('km')}" if case == "INT" else case
        for c in range(1, vba.shape[1]):
            x = vba[r, c]
            if not x.size:
                continue
            on = np.flatnonzero(np.diff((x[:, 11] > 0.5).astype(int)) == 1)
            if not len(on):
                raise SystemExit(f"{state} #{c}: 진출 광센서 없음")
            runs.append((state, c, x[:on[0] + 1, 1:9].astype(np.float32)))
    return runs


def windows(sig):
    idx = np.arange(0, len(sig) - WIN + 1, STRIDE)
    return np.stack([sig[i:i + WIN].T for i in idx])          # (n, C, WIN)


# ---------------------------------------------------------------- 신경망
class TCN(nn.Module):
    """센서별 공유 인과 TCN: (B·N, 1, T) → (B·N, TCN_CH, T)"""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        for i, d in enumerate((1, 2, 4)):
            self.convs.append(nn.Conv1d(1 if i == 0 else TCN_CH, TCN_CH, TCN_K, dilation=d))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        for conv in self.convs:
            pad = (TCN_K - 1) * conv.dilation[0]
            x = self.drop(torch.relu(conv(nn.functional.pad(x, (pad, 0)))))
        return x


class GATLayer(nn.Module):
    """완전 연결 그래프(자기 연결 포함) 다중 헤드 주의"""
    def __init__(self, fin, fout, heads, concat):
        super().__init__()
        self.W = nn.Linear(fin, fout * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, fout) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, fout) * 0.1)
        self.heads, self.fout, self.concat = heads, fout, concat
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, h):                                          # (B, N, fin)
        B, N, _ = h.shape
        z = self.W(h).view(B, N, self.heads, self.fout)
        e = (z * self.a_src).sum(-1).unsqueeze(2) + (z * self.a_dst).sum(-1).unsqueeze(1)   # (B, N, N, H)
        att = self.drop(torch.softmax(nn.functional.leaky_relu(e, 0.2), dim=2))
        out = torch.einsum("bijh,bjhf->bihf", att, z)
        return out.reshape(B, N, -1) if self.concat else out.mean(2)


class AE(nn.Module):
    def __init__(self, use_gat):
        super().__init__()
        self.tcn, self.use_gat = TCN(), use_gat
        self.squeeze = nn.Conv1d(TCN_CH, SQ, 1)
        fin = SQ * WIN
        if use_gat:
            self.g1 = GATLayer(fin, GAT_H1, GAT_HEADS, concat=True)
            self.g2 = GATLayer(GAT_H1 * GAT_HEADS, GAT_H2, GAT_HEADS, concat=True)
        else:
            self.lin = nn.Sequential(nn.Linear(fin, GAT_H1 * GAT_HEADS), nn.ELU(), nn.Dropout(DROPOUT),
                                     nn.Linear(GAT_H1 * GAT_HEADS, GAT_H2 * GAT_HEADS))
        self.up = nn.Linear(GAT_H2 * GAT_HEADS, TCN_CH * POOL)
        self.dec = nn.Sequential(nn.ConvTranspose1d(TCN_CH, 32, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1))

    def forward(self, x):                                          # (B, N, T)
        B, N, T = x.shape
        h = self.tcn(x.reshape(B * N, 1, T))
        h = self.squeeze(h).reshape(B, N, -1)
        h = self.g2(nn.functional.elu(self.g1(h))) if self.use_gat else self.lin(h)
        h = self.up(nn.functional.elu(h)).reshape(B * N, TCN_CH, POOL)
        return self.dec(h).reshape(B, N, T)


PROG = {"done": 0, "total": 0, "t0": None, "log": None}


def hm(sec):
    sec = int(sec)
    return f"{sec // 3600}시간 {sec % 3600 // 60}분" if sec >= 3600 else f"{sec // 60}분 {sec % 60}초"


def say(msg):
    """터미널과 progress.log에 진행 상황 출력"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if PROG["log"]:
        with open(PROG["log"], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def fit_nn(train, use_gat, seed, label=""):
    torch.manual_seed(seed)
    m = AE(use_gat)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    X = torch.from_numpy(train)
    g = torch.Generator().manual_seed(seed)
    PROG["done"] += 1
    t_start = time.time()
    for ep in range(EPOCHS):
        m.train()
        tot = []
        for b in torch.randperm(len(X), generator=g).split(BATCH):
            loss = nn.functional.mse_loss(m(X[b]), X[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot.append(loss.item())
        per = (time.time() - t_start) / (ep + 1)
        say(f"학습 {PROG['done']}/{PROG['total']} {label} | epoch {ep + 1}/{EPOCHS} | 손실 {np.mean(tot):.4f} | "
            f"이번 학습 남은 약 {hm(per * (EPOCHS - ep - 1))} | 전체 경과 {hm(time.time() - PROG['t0'])}")
    m.eval()
    return m


@torch.no_grad()
def score_nn(m, X):
    out = []
    for b in torch.from_numpy(X).split(512):
        out.append(((m(b) - b) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- Isolation Forest
EDGES = np.linspace(0, 100, 9)


def if_features(X):
    """(n, C, WIN) → (n, C·11)"""
    f = np.fft.rfftfreq(WIN, 1 / FS)
    P = np.abs(np.fft.rfft(X, axis=2)) ** 2
    bands = [np.log(P[:, :, (f >= lo) & (f < hi if hi < 100 else f <= hi)].sum(2) + 1e-12) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    rms = np.sqrt((X ** 2).mean(2))
    p2p = X.max(2) - X.min(2)
    ku = kurtosis(X, axis=2)
    return np.concatenate([np.stack([rms, p2p, ku] + bands, axis=2).reshape(len(X), -1)], axis=1)


# ---------------------------------------------------------------- 평가
def build(runs, ch, rng):
    """역할별 Window (정규화 전) 와 주행 목록"""
    W = {}
    for st, no, sig in runs:
        W[(st, no)] = windows(sig[:, [c - 1 for c in ch]])
    roles = {}
    # R1 주행 단위
    roles["R1"] = {"train": np.concatenate([W[("INT40", n)] for n in SPLIT["train"]]),
                   "thr": np.concatenate([W[("INT40", n)] for n in SPLIT["ai2_calib"]]),
                   "normal": {("INT40", n): W[("INT40", n)] for n in SPLIT["test"]}}
    # R0 논문식 무작위 8:2
    allint = np.concatenate([W[("INT40", n)] for n in range(1, 11)])
    perm = rng.permutation(len(allint))
    k = int(round(0.8 * len(allint)))
    roles["R0"] = {"train": allint[perm[:k]], "thr": allint[perm[k:]], "normal": {("INT40_val", 0): allint[perm[k:]]}}
    others = {key: w for key, w in W.items() if key[0] != "INT40"}
    return roles, others


def evaluate(roles, others, models_seed, label=""):
    rows, by_state, by_run = [], [], []
    for r, d in roles.items():
        mu = d["train"].mean(axis=(0, 2), keepdims=True)
        sd = d["train"].std(axis=(0, 2), keepdims=True) + 1e-8
        z = lambda a: ((a - mu) / sd).astype(np.float32)
        sets = {**d["normal"], **others}
        for name in ("TCN-GAT", "TCN", "IForest"):
            t0 = time.time()
            if name == "IForest":
                iso = IsolationForest(n_estimators=300, random_state=models_seed).fit(if_features(z(d["train"])))
                sc = lambda a: -iso.score_samples(if_features(z(a)))
            else:
                m = fit_nn(z(d["train"]), name == "TCN-GAT", models_seed, f"{label} {r} {name}")
                sc = lambda a: score_nn(m, z(a))
            s_thr = sc(d["thr"])
            thr = float(s_thr.mean() + 3 * s_thr.std())
            S = {key: sc(w) for key, w in sets.items()}
            nor = np.concatenate([v for key, v in S.items() if key[0].startswith("INT40")])
            dmg = {st: np.concatenate([v for key, v in S.items() if key[0] == st]) for st in ("DMG1", "DMG2", "DMG3")}
            dall = np.concatenate(list(dmg.values()))
            y = np.r_[np.zeros(len(nor)), np.ones(len(dall))]
            rows.append({"run": r, "model": name, "threshold": thr, "n_train": len(d["train"]), "n_normal": len(nor),
                         "false_alarm": float((nor > thr).mean()),
                         **{f"detect_{st}": float((v > thr).mean()) for st, v in dmg.items()},
                         "detect_all": float((dall > thr).mean()), "auc": float(roc_auc_score(y, np.r_[nor, dall])),
                         "seconds": round(time.time() - t0, 1)})
            for st in sorted({key[0] for key in S}):
                v = np.concatenate([S[key] for key in S if key[0] == st])
                by_state.append({"run": r, "model": name, "state": st, "n": len(v), "flagged": float((v > thr).mean()),
                                 "median_score": float(np.median(v))})
            for key, v in S.items():
                by_run.append({"run": r, "model": name, "state": key[0], "run_no": key[1], "n": len(v),
                               "flagged": float((v > thr).mean()), "run_anomaly": bool((v > thr).mean() > 0.5)})
            say(f"완료: {label} {r} {name} ({hm(time.time() - t0)})")
    return rows, by_state, by_run


def run_all(runs):
    PROG.update(done=0, total=len(SENSORS) * 2 * 2, t0=time.time())
    say(f"시작: 센서 구성 {list(SENSORS)} × 분할 R1·R0 × 신경망 2개 = 학습 {PROG['total']}번, epoch {EPOCHS}")
    rows, bs, br = [], [], []
    for sname, ch in SENSORS.items():
        rng = np.random.default_rng(SEED)
        roles, others = build(runs, ch, rng)
        a, b, c = evaluate(roles, others, SEED, sname)
        rows += [{"sensors": sname, **x} for x in a]
        bs += [{"sensors": sname, **x} for x in b]
        br += [{"sensors": sname, **x} for x in c]
    return pd.DataFrame(rows), pd.DataFrame(bs), pd.DataFrame(br)


def synthetic():
    """정상: 모드 2개 (7Hz, 40Hz) / 손상: 두 번째 모드 30Hz (특징 대역이 바뀌도록)"""
    rng = np.random.default_rng(1)
    t = np.arange(1100) / FS
    out = []
    shape = np.linspace(0.5, 1.5, 8)
    for st, n in [("INT40", 10), ("DMG1", 4), ("DMG2", 4), ("DMG3", 4), ("RCV", 2), ("INT30", 2), ("INT50", 2)]:
        f2 = 30.0 if st.startswith("DMG") else 40.0
        for k in range(1, n + 1):
            base = np.sin(2 * np.pi * 7 * t) + 0.8 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 6))
            sig = base[:, None] * shape[None, :] + 0.2 * rng.normal(size=(len(t), 8))
            out.append((st, k, sig.astype(np.float32)))
    return out


if __name__ == "__main__":
    if "--test" in sys.argv:
        globals()["EPOCHS"] = 8
        globals()["STRIDE"] = 16
        globals()["SENSORS"] = {"S2": [1, 3]}
        x = torch.randn(3, 2, WIN)
        for g in (True, False):
            assert AE(g)(x).shape == x.shape
        s, b, r = run_all(synthetic())
        print(s[["run", "model", "false_alarm", "detect_all", "auc"]].round(3).to_string())
        assert (s.loc[s["model"] == "IForest", "auc"] > 0.9).all()
        assert (s.loc[(s["model"] != "IForest") & (s["run"] == "R1"), "auc"] > 0.8).all(), "신경망이 합성 손상을 못 잡음"
        print("합성 데이터 점검 통과")
        sys.exit()
    if (OUT / "summary.csv").exists():
        raise SystemExit("결과가 이미 있음: 설정을 바꿔 다시 돌리지 않는다")
    OUT.mkdir(parents=True, exist_ok=True)
    PROG["log"] = OUT / "progress.log"
    say("데이터 읽는 중 (1–2분)")
    s, b, r = run_all(load_runs())
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT / "by_state.csv", index=False, encoding="utf-8-sig")
    r.to_csv(OUT / "by_run.csv", index=False, encoding="utf-8-sig")
    (OUT / "setup.json").write_text(json.dumps({
        "win": WIN, "stride": STRIDE, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "dropout": DROPOUT,
        "tcn": [TCN_CH, TCN_K, [1, 2, 4]], "squeeze": SQ, "dec_start": POOL, "gat": [GAT_HEADS, GAT_H1, GAT_H2], "split": SPLIT,
        "sensors": SENSORS, "seed": SEED, "torch": torch.__version__}, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(s.round(4).to_string())
