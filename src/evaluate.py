"""
11. 성능 평가 / 12. 그래프 — 이상탐지 성능만 평가한다 (위험도 평가는 AI② 담당).

입력: results/<source>/<bridge>/predictions.csv
      data/raw/<source>/<bridge>/experiments.csv   (정답: damage_type, severity_level, damage_x_mm)
      models/<source>/<bridge>/split.csv           (어느 실험이 test 인지)
출력: results/<source>/<bridge>/evaluation_summary.csv     판정 방법별 Confusion Matrix·Accuracy·Precision·Recall·F1·AUC
      results/<source>/<bridge>/evaluation_by_state.csv    D0~D3 × L1~L3 별 Anomaly 판정 비율
      results/<source>/<bridge>/evaluation_location.csv    주 이상 센서가 실제 이상 위치에 가장 가까운 센서인지
      results/<source>/<bridge>/score_distribution.csv     D0~D3, L1~L3 별 Anomaly Score 분포
      results/<source>/<bridge>/figures/*.png              그래프 7종

평가 기준
- test 실험만 사용 (train·val 은 학습·임계값 결정에 쓰였으므로 평가에 쓰면 성능이 부풀려짐)
- 정답: D0 → Normal, D1~D3 → Anomaly (이상 실험은 실험 전체가 이상 상태)
- Isolation Forest 는 L1/L2/L3 를 분류하지 않는다. 심각도별로는 점수 분포만 비교한다.

실행: python src/evaluate.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from config import MODEL_SUFFIX, ACTIVE_BRIDGE, BRIDGES, GROUND_TRUTH, MODELS_DIR, PREDICTION_LABELS, RAW_DIR, RESULTS_DIR, SPAN_MM
from preprocess import load_experiment, sensor_short_name

OUT_DIR = RESULTS_DIR / ACTIVE_BRIDGE
FIG_DIR = OUT_DIR / "figures"
NORMAL, ANOMALY = PREDICTION_LABELS
D0 = GROUND_TRUTH["normal_code"]
DAMAGE_ORDER = [D0, *GROUND_TRUTH["damage_codes"]]
SEV_ORDER = [GROUND_TRUTH["normal_severity"], *GROUND_TRUTH["severity_levels"]]

# 판정 방법: (표시 이름, 판정 열을 0/1 로 만드는 함수, AUC 용 점수 열)
METHODS = {
    "bridge_final": ("교량 판정 (최종, 연속 판정)", lambda d: d["prediction"] == ANOMALY, "anomaly_score"),
    "bridge_raw": ("교량 판정 (Window 1개)", lambda d: d["raw_prediction"] == ANOMALY, "anomaly_score"),
    "sensor_any": ("센서 판정 (하나라도 Anomaly)", lambda d: d["anomaly_sensors"] != "", None),
}

# 차트 색 (dataviz 기본 팔레트). 범주 색은 고정 순서로 사용
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, BAND, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f0efec", "#fcfcfb"
DAMAGE_COLORS = dict(zip(DAMAGE_ORDER, [BLUE, ORANGE, AQUA, YELLOW]))
SEV_BLUES = ["#c3c2b7", "#86b6ef", "#2a78d6", "#104281"]  # NORMAL, L1, L2, L3 (진할수록 심각)
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def load_inputs():
    pred = pd.read_csv(OUT_DIR / "predictions.csv")
    pred["anomaly_sensors"] = pred["anomaly_sensors"].fillna("")
    pred["main_sensor"] = pred["main_sensor"].fillna("")
    exps = pd.read_csv(RAW_DIR / ACTIVE_BRIDGE / "experiments.csv")
    split = pd.read_csv(MODELS_DIR / ACTIVE_BRIDGE / "split.csv", usecols=["experiment_id", "split"])
    sensors = pd.DataFrame(BRIDGES[ACTIVE_BRIDGE]["sensors"])
    labels = exps[["experiment_id", "damage_type", "severity_level", "damage_name", "damage_x_mm", "file"]]
    pred = pred.merge(labels, on="experiment_id").merge(split, on="experiment_id")
    test = pred[pred["split"] == "test"].copy()
    test["y_true"] = (test["damage_type"] != D0).astype(int)
    return test, exps, sensors


# ---------------------------------------------------------------------------
# 11. 성능 평가
# ---------------------------------------------------------------------------
def classification_metrics(test):
    rows = []
    for key, (label, judge, score_col) in METHODS.items():
        y_pred = judge(test).astype(int)
        tn, fp, fn, tp = confusion_matrix(test["y_true"], y_pred, labels=[0, 1]).ravel()
        rows.append({
            "방법": label, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Accuracy": accuracy_score(test["y_true"], y_pred),
            "Precision": precision_score(test["y_true"], y_pred, zero_division=0),
            "Recall": recall_score(test["y_true"], y_pred, zero_division=0),
            "F1": f1_score(test["y_true"], y_pred, zero_division=0),
            "오탐률(D0)": fp / (fp + tn) if fp + tn else np.nan,
            "AUC(점수)": roc_auc_score(test["y_true"], test[score_col]) if score_col else np.nan,
        })
    return pd.DataFrame(rows)


def rate_by_state(test):
    """D0~D3 × L 별 Anomaly 판정 비율. D0 행은 오탐률, 나머지는 탐지율(Recall)."""
    table = pd.DataFrame({label: judge(test) for key, (label, judge, _) in METHODS.items()})
    table[["damage_type", "severity_level"]] = test[["damage_type", "severity_level"]]
    return table.groupby(["damage_type", "severity_level"]).mean()


def score_distribution(test):
    """D0~D3, L1~L3 별 Anomaly Score 분포. 심각도가 커질수록 점수가 커지는지(경향)도 확인."""
    dist = test.groupby(["damage_type", "severity_level"])["anomaly_score"].describe()[["count", "mean", "50%", "75%"]]
    dist = dist.rename(columns={"50%": "median", "75%": "q75"})
    trend = {}
    for d in GROUND_TRUTH["damage_codes"]:
        med = [dist.loc[(d, s), "median"] for s in GROUND_TRUTH["severity_levels"] if (d, s) in dist.index]
        trend[d] = "증가" if all(a < b for a, b in zip(med, med[1:])) else "일정하지 않음"
    return dist, trend


def location_accuracy(test, sensors):
    """주 이상 센서가 실제 이상 위치(damage_x_mm)에 가장 가까운 센서인지.
    같은 위치(x)에 센서가 여러 개면 그중 어느 것이어도 정답으로 본다."""
    cand = sensors[sensors["sensor_type"] == "acceleration"].assign(x_mm=lambda d: d["x_ratio"] * SPAN_MM)
    x_of = cand.set_index("sensor_id")["x_mm"]
    hit = test[(test["y_true"] == 1) & (test["main_sensor"] != "")].copy()
    hit["sensor_x_mm"] = hit["main_sensor"].map(x_of)
    nearest = hit["damage_x_mm"].apply(lambda dx: np.min(np.abs(x_of.to_numpy() - dx)))
    hit["correct"] = np.isclose(np.abs(hit["sensor_x_mm"] - hit["damage_x_mm"]), nearest)
    return hit.groupby(["damage_type", "damage_name"]).agg(
        이상위치_mm=("damage_x_mm", "first"), 판정Window수=("correct", "size"), 위치정확도=("correct", "mean"),
        가장많이나온센서=("main_sensor", lambda s: s.mode().iat[0]),
    )


# ---------------------------------------------------------------------------
# 12. 그래프
# ---------------------------------------------------------------------------
def plain(text):
    """그래프 글꼴(맑은고딕)에 없는 기호를 바꾼다."""
    return str(text).replace("≈", "~")


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(test):
    cm = confusion_matrix(test["y_true"], (test["prediction"] == ANOMALY).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", [BAND, "#86b6ef", BLUE]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13, color="#ffffff" if v > cm.max() * 0.6 else INK)
    ax.set_xticks([0, 1], [f"예측 {NORMAL}", f"예측 {ANOMALY}"])
    ax.set_yticks([0, 1], [f"실제 {NORMAL}(D0)", f"실제 {ANOMALY}(D1~D3)"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion Matrix (교량 최종 판정, test)")
    save(fig, "1_confusion_matrix.png")


def plot_score_hist(test, threshold):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(test["anomaly_score"].min(), test["anomaly_score"].max(), 50)
    for y, name, color in [(0, f"실제 {NORMAL} (D0)", BLUE), (1, f"실제 {ANOMALY} (D1~D3)", ORANGE)]:
        ax.hist(test.loc[test["y_true"] == y, "anomaly_score"], bins=bins, density=True, alpha=0.55,
                color=color, label=name, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(threshold, color=INK2, ls="--", lw=1)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" 임계값 {threshold:.3f}", color=INK2, fontsize=9, va="top")
    ax.set_xlabel("Anomaly Score (0~1, 클수록 이상)")
    ax.set_ylabel("비율 (밀도)")
    ax.legend(frameon=False)
    ax.set_title("Normal vs Anomaly — Anomaly Score 분포")
    save(fig, "2_score_hist_normal_vs_anomaly.png")


def plot_score_by_damage(test, threshold):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data = [test.loc[test["damage_type"] == d, "anomaly_score"] for d in DAMAGE_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False, medianprops={"color": INK})
    for patch, d in zip(bp["boxes"], DAMAGE_ORDER):
        patch.set(facecolor=DAMAGE_COLORS[d], alpha=0.7, edgecolor=SURFACE)
    names = test.drop_duplicates("damage_type").set_index("damage_type")["damage_name"]
    ax.set_xticks(range(1, len(DAMAGE_ORDER) + 1), [f"{d}\n{plain(names.get(d, ''))}" for d in DAMAGE_ORDER])
    ax.axhline(threshold, color=INK2, ls="--", lw=1)
    ax.set_ylabel("Anomaly Score")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("이상 유형별 Anomaly Score (D0 / D1 / D2 / D3)")
    save(fig, "3_score_by_damage_type.png")


def plot_score_by_severity(test, threshold):
    codes = GROUND_TRUTH["damage_codes"]
    fig, axes = plt.subplots(1, len(codes), figsize=(4.2 * len(codes), 3.8), sharey=True)
    d0 = test.loc[test["damage_type"] == D0, "anomaly_score"]
    for ax, d in zip(axes, codes):
        sub = test[test["damage_type"] == d]
        data = [d0] + [sub.loc[sub["severity_level"] == s, "anomaly_score"] for s in GROUND_TRUTH["severity_levels"]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": INK})
        for patch, c in zip(bp["boxes"], SEV_BLUES):
            patch.set(facecolor=c, edgecolor=SURFACE)
        ax.set_xticks(range(1, 5), ["D0", *GROUND_TRUTH["severity_levels"]])
        ax.axhline(threshold, color=INK2, ls="--", lw=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(f"{d} {plain(sub['damage_name'].iat[0]) if len(sub) else ''}", fontsize=11)
    axes[0].set_ylabel("Anomaly Score")
    fig.suptitle("심각도별 Anomaly Score (L1 → L3 로 갈수록 커지는지)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "4_score_by_severity.png")


def pick_example(test):
    """그래프 예시 실험: 이상 실험 중 점수가 가장 높은 것과, 같은 하중 조건(크기 × 위치)의 D0 실험."""
    ab = test[test["y_true"] == 1].groupby("experiment_id")["anomaly_score"].mean().idxmax()
    cond = test.loc[test["experiment_id"] == ab, "condition"].iat[0]
    d0 = test[(test["damage_type"] == D0) & (test["condition"] == cond)]["experiment_id"].iat[0]
    return ab, d0


def plot_raw_timeseries(exps, sensors, experiment_id):
    """원본 센서 데이터: 가속도는 축마다, 하중·온도는 1개씩."""
    file = exps.set_index("experiment_id").loc[experiment_id, "file"]
    raw = load_experiment(file)
    raw = raw[raw["quality_flag"] == "OK"]
    t0 = raw["timestamp_us"].min()
    chans = [(s.sensor_id, axis, s.unit) for s in sensors.itertuples() for axis in s.axes]
    fig, axes = plt.subplots(len(chans), 1, figsize=(11, 1.3 * len(chans)), sharex=True)
    for ax, (sid, axis, unit) in zip(axes, chans):
        g = raw[(raw["sensor_id"] == sid) & (raw["axis"].astype(str) == axis)]
        ax.plot((g["timestamp_us"] - t0) / 1e6, g["value"], color=BLUE, lw=0.5,
                marker="o" if len(g) < 50 else None, ms=3)
        label = sensor_short_name(sid) + (f" {axis}" if axis != "NA" else "")
        ax.set_ylabel(f"{label}\n[{unit}]", fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[-1].set_xlabel("시간 [초]")
    fig.suptitle(f"원본 센서 데이터 (OK 만) — {experiment_id}", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "5_raw_sensor_timeseries.png")


def plot_score_timeline(test, threshold):
    """test 실험을 순서대로 이어 붙인 시간축: 위 = 교량 Anomaly Score, 아래 = 센서별 상태판."""
    t = test.sort_values(["experiment_id", "window_start_us"]).reset_index(drop=True)
    x = np.arange(len(t))
    state_cols = [c for c in t.columns if c.endswith("_state")]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 0.35 * (len(state_cols) + 1)]})

    blocks = t.groupby((t["experiment_id"] != t["experiment_id"].shift()).cumsum()).agg(
        start=("experiment_id", lambda s: s.index[0]), end=("experiment_id", lambda s: s.index[-1]),
        d=("damage_type", "first"), sev=("severity_level", "first"))
    state_blocks = blocks.groupby(((blocks["d"] + blocks["sev"]) != (blocks["d"] + blocks["sev"]).shift()).cumsum()).agg(
        start=("start", "min"), end=("end", "max"), d=("d", "first"), sev=("sev", "first"))
    for b in state_blocks.itertuples():
        if b.d != D0:
            ax1.axvspan(b.start - 0.5, b.end + 0.5, color=BAND, zorder=0)
        label = "D0" if b.d == D0 else f"{b.d}_{b.sev}"
        ax1.text((b.start + b.end) / 2, 1.01, label, transform=ax1.get_xaxis_transform(), ha="center",
                 fontsize=8, color=MUTED if b.d == D0 else INK)
    ax1.plot(x, t["anomaly_score"], color=BLUE, lw=0.7)
    ax1.axhline(threshold, color=INK2, ls="--", lw=1)
    ax1.set_ylabel("교량 Anomaly Score")
    ax1.grid(axis="y", color=GRID, lw=0.5)

    rows = state_cols + ["prediction"]
    grid = np.array([(t[c] == ANOMALY).to_numpy() for c in rows], dtype=float)
    ax2.imshow(grid, aspect="auto", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap([BAND, ORANGE]), vmin=0, vmax=1)
    ax2.set_yticks(range(len(rows)), [c.replace("_state", "") for c in state_cols] + ["교량 최종"], fontsize=9)
    ax2.tick_params(length=0)
    ax2.set_xlabel("test Window 순서 (실험 순서대로 이어 붙임, 회색 음영 = 실제 이상 구간, 주황 = Anomaly 판정)")
    fig.suptitle("시간축 Anomaly Score + 센서별 상태판", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "6_score_timeline_sensor_board.png")


def plot_fft(exps, sensors, ab_id, d0_id):
    """가속도 FFT: 같은 하중 조건의 D0 실험과 이상 실험 비교 (L/2 가속도계 수직 Z축)."""
    acc = sensors[sensors["sensor_type"] == "acceleration"].iloc[-1]
    fs = acc["sample_rate_hz"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    peaks = []
    for exp_id, color in [(d0_id, BLUE), (ab_id, ORANGE)]:
        raw = load_experiment(exps.set_index("experiment_id").loc[exp_id, "file"])
        sel = (raw["sensor_id"] == acc["sensor_id"]) & (raw["axis"] == "Z") & (raw["quality_flag"] == "OK")
        v = raw.loc[sel, "value"].to_numpy(dtype=float)
        # Welch 방식: 2초 구간 FFT(Hann 창)를 평균 → 120초 전체 FFT 보다 깔끔한 스펙트럼
        freqs, psd = signal.welch(v - v.mean(), fs=fs, nperseg=int(2 * fs))
        peak = freqs[np.argmax(np.where(freqs > 2, psd, 0))]
        peaks.append(peak)
        ax.plot(freqs, psd / psd.max(), color=color, lw=1.5, label=f"{exp_id}  (최대 {peak:.1f} Hz)")
        ax.axvline(peak, color=color, lw=0.8, ls=":")
    ax.set_xlim(0, min(fs / 2, 3 * max(peaks)))
    ax.set_xlabel("진동수 [Hz]")
    ax.set_ylabel("크기 (최댓값 = 1)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_title(f"가속도 FFT ({acc['sensor_id']} 수직 Z, {acc['sensor_location']}) — 같은 하중 조건의 D0 vs 이상")
    save(fig, "7_acceleration_fft.png")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def main():
    test, exps, sensors = load_inputs()
    threshold = test["threshold"].iat[0]

    summary = classification_metrics(test)
    by_state = rate_by_state(test)
    dist, trend = score_distribution(test)
    loc = location_accuracy(test, sensors)

    summary.to_csv(OUT_DIR / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    by_state.to_csv(OUT_DIR / "evaluation_by_state.csv", encoding="utf-8-sig")
    loc.to_csv(OUT_DIR / "evaluation_location.csv", encoding="utf-8-sig")
    dist.to_csv(OUT_DIR / "score_distribution.csv", encoding="utf-8-sig")

    ab_id, d0_id = pick_example(test)
    plot_confusion(test)
    plot_score_hist(test, threshold)
    plot_score_by_damage(test, threshold)
    plot_score_by_severity(test, threshold)
    plot_raw_timeseries(exps, sensors, ab_id)
    plot_score_timeline(test, threshold)
    plot_fft(exps, sensors, ab_id, d0_id)

    pd.set_option("display.width", 220)
    n_norm = (test["y_true"] == 0).sum()
    print(f"[{ACTIVE_BRIDGE}] test Window: Normal(D0) {n_norm}개 / Anomaly(D1~D3) {len(test) - n_norm}개")
    print("\n[11-1] 판정 방법별 성능")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print("\n[11-2] 상태별 Anomaly 판정 비율 (D0 = 오탐률, 나머지 = 탐지율)")
    print(by_state.to_string(float_format=lambda v: f"{v:.0%}"))
    print("\n[11-3] 심각도(L1→L3)가 커질수록 Anomaly Score 중앙값이 커지는가")
    print("  " + ", ".join(f"{d}: {v}" for d, v in trend.items()))
    print("\n[11-4] 이상 위치: 주 이상 센서가 실제 이상 위치에 가장 가까운 센서인가")
    print(loc.to_string(formatters={"이상위치_mm": "{:.0f}".format, "위치정확도": "{:.0%}".format}))
    print(f"\n[12] 그래프 7종 → {FIG_DIR}")
    print(f"     예시 실험: 이상 {ab_id} / 같은 하중 정상 {d0_id}")


def evaluate_public():
    """공개 데이터 (합의서 v3): AI① 판정을 손상 단계(damage_stage)별로 센다.
    등급 통과 판정은 AI② 몫이라 여기서는 하지 않는다. AI① test 실험만 보고, 판정 보류·불가는 따로 센다."""
    pred = pd.read_csv(OUT_DIR / f"predictions{MODEL_SUFFIX}.csv")
    exps = pd.read_csv(RAW_DIR / ACTIVE_BRIDGE / "experiments.csv", keep_default_na=False)
    cols = [c for c in ("split", "damage_stage", "label_status", "cv_group", "subset", "condition") if c in exps]
    df = pred.drop(columns=[c for c in ("condition",) if c in pred]).merge(exps[["experiment_id", *cols]], on="experiment_id")
    keys = [c for c in ("split", "condition", "damage_stage", "label_status", "subset", "cv_group") if c in df]
    table = (df.groupby(keys)["prediction"].value_counts().unstack(fill_value=0))
    judged = [c for c in (NORMAL, ANOMALY) if c in table]
    if judged:
        table["이상 비율 (판정한 Window 기준)"] = (table.get(ANOMALY, 0) / table[judged].sum(axis=1)).round(3)
    if "max_abs_z" in df:
        table["max_abs_z 중앙값"] = df.groupby(keys)["max_abs_z"].median().round(2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"evaluation_by_stage{MODEL_SUFFIX}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"[{ACTIVE_BRIDGE}] 손상 단계별 AI① 판정 Window 수 (전체 split, 참고 라벨 포함)")
    print(table.to_string())
    print(f"\n저장: {OUT_DIR / 'evaluation_by_stage.csv'}")


if __name__ == "__main__":
    if BRIDGES[ACTIVE_BRIDGE].get("public"):
        evaluate_public()
    else:
        main()
