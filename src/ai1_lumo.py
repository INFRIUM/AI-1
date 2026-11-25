"""
AI① LUMO 이상탐지와 AI①→AI② 연결 평가 (docs/AI1_LUMO_ANOMALY_DESIGN.md, AI② 보완 INFRIUM/AI-2#15 댓글 7092337355)

실행: python src/ai1_lumo.py --test                     합성 데이터 점검만 (LUMO 를 읽지 않음)
      python src/ai1_lumo.py detect                     3단계. 1회만 (결과 파일이 있으면 멈춤)
      python src/ai1_lumo.py link <AI② 폴더> <AI② 요약 JSON>   5단계. 1회만
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import ROOT

KEY = ["experiment_id", "window_start_us"]
SENSORS = {"acc01": "ML5", "acc02": "ML1"}
NAMES = ("rms", "std", "max", "min", "p2p", "crest", "dom_freq", "band_2_8", "band_8_15", "band_15_25")
FEATURES = [f"{s}_{n}_{a}" for s in SENSORS for a in ("x", "y") for n in NAMES]
TRAIN = ("N1_B1_0803", "N1_B1_0905")
VAL = ("N1_B2_0814", "N1_B2_0825", "N1_B2_0916", "N1_B2_0927")
STEP_US, POOL_N, POOL_MIN_VAL, MIN_FIT = 30_000_000, 5, 3, 200
FA_MAX, DET_MIN, MIN_JUDGED, N_ROWS = 0.05, 0.95, 200, 30096
MD5_LF = {"raw": "fef7accb07147acc3c3087430d396053", "deg": "1df914fd20ac14ccd199a61cfd5b207a"}   # 전달 파일 (LF 기준)
SRC = ROOT / "results" / "public" / "for_AI2_1.4"
OUT = ROOT / "results" / "public" / "ai1_lumo"
FA_ROLE, DET_ROLE = "오경보 확인", "위험 탐지"


def md5_lf(path):
    return hashlib.md5(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------- AI① 이상탐지
def pooled(df, score):
    """같은 블록에서 30초씩 이어진 보류 아닌 Window 의 최근 최대 5개 점수 중앙값. 반환: (값, 묶인 Window 수). 보류는 NaN, 0"""
    d = df[["experiment_id", "window_start_us"]].assign(s=score)[~df["held"].to_numpy()].sort_values(KEY)
    run = (d.groupby("experiment_id")["window_start_us"].diff() != STEP_US).cumsum()
    r = d.groupby(run)["s"].rolling(POOL_N, min_periods=1)
    val = r.median().reset_index(level=0, drop=True).reindex(df.index)
    num = r.count().reset_index(level=0, drop=True).reindex(df.index).fillna(0)
    return val, num


def detect_frame(df):
    """df: 키, cv_group, held, FEATURES. 학습·임계값 구간은 cv_group 이름으로만 고른다 (손상 라벨을 읽지 않음)"""
    if df[FEATURES].isna().any().any():
        raise SystemExit("특징에 빈 값이 있음")
    fit = (df["cv_group"].isin(TRAIN) & ~df["held"]).to_numpy()
    val = (df["cv_group"].isin(VAL) & ~df["held"]).to_numpy()
    if fit.sum() < MIN_FIT or val.sum() < MIN_FIT:
        raise SystemExit(f"보류 아닌 학습 {fit.sum()}개 / 임계값 {val.sum()}개: {MIN_FIT}개 미만이라 판정하지 않음")
    mu, sd = df.loc[fit, FEATURES].mean(), df.loc[fit, FEATURES].std(ddof=1).clip(lower=1e-6)
    z = ((df[FEATURES] - mu) / sd).to_numpy()
    score = LedoitWolf().fit(z[fit]).mahalanobis(z)
    pool, num = pooled(df, score)
    cal = val & (num >= POOL_MIN_VAL).to_numpy()
    if cal.sum() < MIN_FIT:
        raise SystemExit(f"임계값 구간의 묶은 점수 {cal.sum()}개: {MIN_FIT}개 미만이라 판정하지 않음")
    t_win, t_pool = float(np.percentile(score[val], 99)), float(np.percentile(pool[cal], 99))
    ratio = {}
    for s in SENSORS:
        ix = [i for i, c in enumerate(FEATURES) if c.startswith(s)]
        part = LedoitWolf().fit(z[fit][:, ix]).mahalanobis(z[:, ix])
        ratio[SENSORS[s]] = part / np.percentile(part[val], 99)
    out = df[KEY].copy()
    out["ai1_window_anomaly"] = (score > t_win).astype(int)
    out["ai1_final_anomaly"] = (pool > t_pool).fillna(False).astype(int)   # 보류 Window 는 0 (채점하지 않음)
    out["anomaly_score"], out["pooled_score"] = score, pool
    out["pooled_n"], out["max_abs_z"] = num.astype(int), np.abs(z).max(axis=1)
    out["top_sensor"] = pd.DataFrame(ratio, index=df.index).idxmax(axis=1)
    info = {"n_fit": int(fit.sum()), "n_val": int(val.sum()), "n_val_pooled": int(cal.sum()),
            "threshold_window": t_win, "threshold_pooled": t_pool, "n_held": int(df["held"].sum())}
    return out, info


def detect():
    if any((OUT / f"lumo_ai1_predictions{s}.csv").exists() for s in ("", "_raw")):
        raise SystemExit("결과 파일이 이미 있음: AI① LUMO 판정은 한 번만 한다")
    path = {k: SRC / f"lumo_anomaly_results_{k}.csv" for k in MD5_LF}
    for k, p in path.items():
        if md5_lf(p) != MD5_LF[k]:
            raise SystemExit(f"{p.name}: 전달 MD5 와 다름")
    cols = KEY + ["cv_group"] + FEATURES
    deg = pd.read_csv(path["deg"], encoding="utf-8-sig", usecols=cols + ["hold_main"]).rename(columns={"hold_main": "held"})
    deg["held"] = deg["held"].astype(str).eq("True")
    raw = pd.read_csv(path["raw"], encoding="utf-8-sig", usecols=cols).merge(deg[KEY + ["held"]], on=KEY, validate="1:1")
    info, frames = {}, {}
    for k, df in (("deg", deg), ("raw", raw)):
        if len(df) != N_ROWS or df.duplicated(KEY).any():
            raise SystemExit(f"{k}: 행 수 또는 키 이상")
        frames[k], info[k] = detect_frame(df)
    OUT.mkdir(parents=True, exist_ok=True)
    for k, s in (("deg", ""), ("raw", "_raw")):
        p = OUT / f"lumo_ai1_predictions{s}.csv"
        frames[k].to_csv(p, index=False, encoding="utf-8-sig", lineterminator="\n")
        print(p.name, len(frames[k]), "MD5", md5_lf(p))
    # 임계값·개수는 파일로만 남기고 화면에 출력하지 않는다
    (OUT / "detect_info.json").write_text(json.dumps({"inputs": MD5_LF, **info}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 연결 평가
def as_bool(s):
    v = s.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if v.isna().any():
        raise SystemExit(f"{s.name}: True/False 가 아닌 값")
    return v.astype(bool)


def rate(mask, flag):
    n, k = int(mask.sum()), int((mask & flag).sum())
    return {"n": n, "k": k, "rate": k / n if n else None}


def link_frames(ai1, ai2, n_rows=N_ROWS):
    for name, f in (("AI①", ai1), ("AI②", ai2)):
        if len(f) != n_rows or f.duplicated(KEY).any():
            raise SystemExit(f"{name}: 행 수 {len(f)} 또는 키 중복")
    m = ai1[KEY + ["ai1_final_anomaly", "pooled_n", "top_sensor"]].merge(ai2, on=KEY, how="outer", indicator=True, validate="1:1")
    if (m["_merge"] != "both").any():
        raise SystemExit("한쪽에만 있는 키가 있음")
    judged, two = as_bool(m["judged"]), as_bool(m["abnormal"])
    if (judged & (m["pooled_n"] == 0)).any():
        raise SystemExit("AI② 판정 Window 중 AI① 보류가 있음")
    one = as_bool(m["ai1_final_anomaly"])
    flags = {"ai1": one, "ai2_check": two, "link": one & two}
    fa, det = judged & m["role"].eq(FA_ROLE), judged & m["role"].eq(DET_ROLE)
    res = {}
    for k, v in flags.items():
        a, b = rate(fa, v), rate(det, v)
        ok = a["rate"] is not None and b["rate"] is not None
        res[k] = {"false_alarm": a, "detection": b, "pass": bool(ok and a["rate"] <= FA_MAX and b["rate"] >= DET_MIN),
                  "tentative": bool(min(a["n"], b["n"]) < MIN_JUDGED)}
    res["loss_111"] = {"ai1_miss": rate(det, ~one), "ai1_hit_ai2_miss": rate(det, one & ~two), "both_hit": rate(det, one & two)}
    res["false_alarm_split"] = {"ai1_only": rate(fa, one & ~two), "ai2_only": rate(fa, ~one & two), "both": rate(fa, one & two)}
    res["by_span"] = {}
    for g in sorted(m["cv_group"].astype(str).unique()):
        j = judged & m["cv_group"].astype(str).eq(g)
        res["by_span"][g] = {"role": str(m.loc[m["cv_group"].astype(str).eq(g), "role"].iloc[0]),
                             **{k: rate(j, v) for k, v in flags.items()}}
    res["top_sensor"] = {r: m.loc[judged & one & m["role"].eq(r), "top_sensor"].value_counts().to_dict()
                         for r in sorted(m["role"].astype(str).unique())}
    res["pooled_n_judged"] = {str(int(k)): int(v) for k, v in m.loc[judged, "pooled_n"].value_counts().sort_index().items()}
    res["windows"] = {"all": len(m), "judged": int(judged.sum())}
    return res


def link(folder, summary):
    dest = OUT / "link_result.json"
    if dest.exists():
        raise SystemExit("link_result.json 이 이미 있음: 연결 평가는 한 번만 한다")
    files = {"deg": (OUT / "lumo_ai1_predictions.csv", Path(folder) / "lumo_deg_grades.csv"),
             "raw": (OUT / "lumo_ai1_predictions_raw.csv", Path(folder) / "lumo_raw_grades.csv")}
    res = {"inputs_md5_lf": {}, "ai2_summary_md5_lf": md5_lf(summary)}
    s = json.loads(Path(summary).read_text(encoding="utf-8"))["result"]
    res["ai2_alone_v4_summary"] = {k: s[k] for k in ("false_alarm", "detection", "pass", "tentative")}   # 옮겨 적기만
    deg2 = pd.read_csv(files["deg"][1], encoding="utf-8-sig", low_memory=False)
    for k, (p1, p2) in files.items():
        res["inputs_md5_lf"] |= {p1.name: md5_lf(p1), p2.name: md5_lf(p2)}
        a2 = pd.read_csv(p2, encoding="utf-8-sig", low_memory=False)
        if k == "raw":   # v4 8.2절: 열화본 판정 Window 와 겹치는 것만
            j = deg2[KEY + ["judged"]].rename(columns={"judged": "dj"})
            a2 = a2.merge(j, on=KEY, how="left", validate="1:1")
            a2["judged"] = as_bool(a2["judged"]) & as_bool(a2["dj"])
        res[k] = link_frames(pd.read_csv(p1, encoding="utf-8-sig"), a2)
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dest, "MD5", md5_lf(dest))


# ---------------------------------------------------------------- 합성 데이터 점검
def synthetic():
    rng, rows = np.random.default_rng(7), []
    for g in TRAIN + VAL + ("N2_A", "DAM6_111_A"):
        for b in range(20):
            for w in range(18):
                rows.append({"experiment_id": f"{g}_{b}", "window_start_us": w * STEP_US, "cv_group": g, "held": b == 0 and w == 5})
    df = pd.DataFrame(rows)
    x = rng.normal(size=(len(df), len(FEATURES)))
    x[df["cv_group"].eq("DAM6_111_A").to_numpy(), :20] += 5   # acc01 만 변화
    return pd.concat([df, pd.DataFrame(x, columns=FEATURES)], axis=1)


def self_test():
    df = synthetic()
    out, info = detect_frame(df)
    assert list(out["pooled_n"][:9]) == [1, 2, 3, 4, 5, 0, 1, 2, 3]            # 보류에서 끊김
    assert out.loc[df["held"], "ai1_final_anomaly"].eq(0).all()
    s = np.arange(len(df), dtype=float)
    assert pooled(df, s)[0].iloc[[0, 2, 4]].tolist() == [0.0, 1.0, 2.0]
    dam, nor = df["cv_group"].eq("DAM6_111_A") & ~df["held"], df["cv_group"].eq("N2_A") & ~df["held"]
    assert out.loc[dam, "ai1_final_anomaly"].mean() > 0.95 and out.loc[nor, "ai1_final_anomaly"].mean() < 0.10
    assert (out.loc[dam, "top_sensor"] == "ML5").mean() > 0.95
    perm = df.sample(frac=1, random_state=0)                                   # 행 순서와 무관
    o2, _ = detect_frame(perm)
    assert o2["pooled_n"].reindex(df.index).equals(out["pooled_n"])
    # 연결
    role = df["cv_group"].map({"N2_A": FA_ROLE, "DAM6_111_A": DET_ROLE}).fillna("기준 적합")
    a2 = df[KEY + ["cv_group"]].assign(role=role, judged=(~df["held"]) & (out["pooled_n"] >= 3), abnormal=dam)
    r = link_frames(out, a2, len(df))
    assert r["link"]["detection"]["k"] <= min(r["ai1"]["detection"]["k"], r["ai2_check"]["detection"]["k"])
    assert r["ai2_check"]["detection"]["rate"] == 1.0 and r["link"]["false_alarm"]["k"] == 0
    tot = sum(r["loss_111"][k]["k"] for k in r["loss_111"])
    assert tot == r["ai1"]["detection"]["n"]
    for bad in (a2.iloc[1:], pd.concat([a2.iloc[1:], a2.iloc[[1]]]), a2.assign(window_start_us=a2["window_start_us"] + 1)):
        try:
            link_frames(out, bad, len(df))
        except SystemExit:
            continue
        raise AssertionError("잘못된 키를 걸러내지 못함")
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "a.csv"
        p.write_bytes(b"a\r\nb\r\n")
        assert md5_lf(p) == hashlib.md5(b"a\nb\n").hexdigest()
    print("합성 데이터 점검 통과")


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg == ["--test"]:
        self_test()
    elif arg == ["detect"]:
        detect()
    elif len(arg) == 3 and arg[0] == "link":
        link(arg[1], arg[2])
    else:
        raise SystemExit(__doc__)
