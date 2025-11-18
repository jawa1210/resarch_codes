import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib
import os

# ── フォント設定 ────────────────────────────────
mpl.rcParams['font.family']        = 'IPAexGothic'
mpl.rcParams['text.usetex']        = False
mpl.rcParams['font.size']          = 16
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['lines.linewidth']    = 2
mpl.rcParams['axes.titlesize']     = 20
mpl.rcParams['axes.labelsize']     = 20
mpl.rcParams['legend.fontsize']    = 14
mpl.rcParams['xtick.labelsize']    = 16
mpl.rcParams['ytick.labelsize']    = 16


def _nice_label_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _ensure_run_column(df: pd.DataFrame) -> pd.DataFrame:
    if "run" not in df.columns:
        df = df.copy()
        df["run"] = 0
    return df


def plot_two_results_files(
    file1: str,
    file2: str,
    label1: str | None = None,
    label2: str | None = None,
    dt: float = 0.1,
):
    # ── CSV 読み込み ───────────────────────────
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    print("=== file1 head ===")
    print(df1.head())
    print("=== file2 head ===")
    print(df2.head())

    df1 = _ensure_run_column(df1)
    df2 = _ensure_run_column(df2)

    # NaN 行は落とす
    df1 = df1.dropna(subset=["step", "J", "true_crop_sum"])
    df2 = df2.dropna(subset=["step", "J", "true_crop_sum"])

    if label1 is None:
        label1 = _nice_label_from_path(file1)
    if label2 is None:
        label2 = _nice_label_from_path(file2)

    runs1 = sorted(df1["run"].unique())
    runs2 = sorted(df2["run"].unique())
    print(f"{label1}: runs = {runs1}")
    print(f"{label2}: runs = {runs2}")

    # ── 平均曲線（step ごと）─────────────────────
    mean1 = df1.groupby("step")[["J", "true_crop_sum"]].mean().reset_index()
    mean2 = df2.groupby("step")[["J", "true_crop_sum"]].mean().reset_index()

    if mean1.empty or mean2.empty:
        print("平均用のデータが空です。CSV の中身を確認してください。")
        return

    t1_mean = mean1["step"].to_numpy() * dt
    J1_mean = mean1["J"].to_numpy()
    C1_mean = mean1["true_crop_sum"].to_numpy()

    t2_mean = mean2["step"].to_numpy() * dt
    J2_mean = mean2["J"].to_numpy()
    C2_mean = mean2["true_crop_sum"].to_numpy()

    color1 = "tab:blue"
    color2 = "tab:red"

    # =====================================================
    # 1. Objective J(t)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # file1: 各 run
    for i, r in enumerate(runs1):
        sub = df1[df1["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        J = sub["J"].to_numpy()
        lab = f"{label1} 各試行" if i == 0 else None
        ax.plot(t, J, color=color1, alpha=0.3, label=lab)

    # file1: 平均
    ax.plot(
        t1_mean, J1_mean,
        color=color1, linewidth=3.0,
        label=f"{label1} 平均",
    )

    # file2: 各 run
    for i, r in enumerate(runs2):
        sub = df2[df2["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        J = sub["J"].to_numpy()
        lab = f"{label2} 各試行" if i == 0 else None
        ax.plot(t, J, color=color2, alpha=0.3, label=lab)

    # file2: 平均
    ax.plot(
        t2_mean, J2_mean,
        color=color2, linewidth=3.0,
        label=f"{label2} 平均",
    )

    ax.set_xlabel(f"時間 (step×{dt:.2f}s)")
    ax.set_ylabel("目的関数 $J$")
    ax.set_title("Objective $J$ の推移（2条件比較）")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 2. true_crop_sum(t)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # file1: 各 run
    for i, r in enumerate(runs1):
        sub = df1[df1["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        C = sub["true_crop_sum"].to_numpy()
        lab = f"{label1} 各試行" if i == 0 else None
        ax.plot(t, C, color=color1, alpha=0.3, label=lab)

    # file1: 平均
    ax.plot(
        t1_mean, C1_mean,
        color=color1, linewidth=3.0,
        label=f"{label1} 平均",
    )

    # file2: 各 run
    for i, r in enumerate(runs2):
        sub = df2[df2["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        C = sub["true_crop_sum"].to_numpy()
        lab = f"{label2} 各試行" if i == 0 else None
        ax.plot(t, C, color=color2, alpha=0.3, label=lab)

    # file2: 平均
    ax.plot(
        t2_mean, C2_mean,
        color=color2, linewidth=3.0,
        label=f"{label2} 平均",
    )

    ax.set_xlabel(f"時間 (step×{dt:.2f}s)")
    ax.set_ylabel("累積真値の合計")
    ax.set_title("UGV 訪問セルの累積真値（2条件比較）")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 実際のファイル名に合わせて書き換え
    file_A = "multi_uav_multi_ugv_suenaga_results_10runs.csv"
    file_B = "multi_uav_multi_ugv_results_10runs.csv"

    plot_two_results_files(
        file_A,
        file_B,
        label1="条件A",
        label2="条件B",
        dt=0.1,
    )
