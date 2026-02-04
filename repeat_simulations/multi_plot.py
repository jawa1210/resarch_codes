import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib  # noqa: F401
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


def _infer_param_file(results_path: str) -> str | None:
    """
    results ファイルパスから、対応する params ファイルパスを推定する。
    例: xxx_results_10runs.csv → xxx_params_10runs.csv
    """
    base = os.path.basename(results_path)
    if "results" not in base:
        return None
    param_base = base.replace("results", "params")
    param_path = os.path.join(os.path.dirname(results_path), param_base)
    if os.path.exists(param_path):
        return param_path
    return None


def _load_num_uavs_from_param(param_path: str) -> int | None:
    """
    params CSV から num_uavs を読む。
    読み込みに失敗したら None を返す。
    """
    try:
        dfp = pd.read_csv(param_path)
        if "num_uavs" not in dfp.columns:
            print(f"[WARN] {param_path} に 'num_uavs' 列がありません。")
            return None
        N = int(dfp["num_uavs"].iloc[0])
        print(f"[INFO] {param_path} から num_uavs = {N} を取得しました。")
        return N
    except Exception as e:
        print(f"[WARN] パラメータファイル読み込み失敗: {param_path} -> {e}")
        return None


def _sanitize_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    以前の表示は維持しつつ、縦縞の主因だけ潰す最小整形:
    - run 列を保証
    - step を数値化（文字列事故防止）
    - (run, step) 重複があれば mean で集約（重複0なら無影響）
    """
    df = _ensure_run_column(df)

    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step", "J", "true_crop_sum"])
    df["step"] = df["step"].astype(int)

    dup = df.duplicated(subset=["run", "step"]).sum()
    if dup > 0:
        print(f"[WARN] {name}: duplicated (run, step) = {dup} -> mean で集約します（描画の縦縞対策）")
        df = (
            df.groupby(["run", "step"], as_index=False)[["J", "true_crop_sum"]]
              .mean()
        )
    return df


def plot_two_results_files(
    file1: str,
    file2: str | None = None,
    label1: str | None = None,
    label2: str | None = None,
    dt: float = 0.1,
    ds: float = 0.5,
    gamma: float | None = 3.0,
):
    """
    file2 を None にすると単独ファイルのプロットになる。
    file2 を指定すると 2条件比較プロットになる。

    gamma: 1 UAV あたりの「1秒あたりの減少量 γ」
    → 理論線: J_ideal(t) = J0 - N_uav * (gamma/ds) * t   （★以前仕様）
    """

    # ── CSV 読み込み ───────────────────────────
    df1_raw = pd.read_csv(file1)
    df1 = _sanitize_df(df1_raw, "file1")

    if label1 is None:
        label1 = _nice_label_from_path(file1)

    if file2 is not None:
        df2_raw = pd.read_csv(file2)
        df2 = _sanitize_df(df2_raw, "file2")

        if label2 is None:
            label2 = _nice_label_from_path(file2)

        has_second = not df2.empty
    else:
        df2 = None
        has_second = False

    runs1 = sorted(df1["run"].unique())
    print(f"{label1}: runs = {runs1}")
    if has_second:
        runs2 = sorted(df2["run"].unique())
        print(f"{label2}: runs = {runs2}")

    # ── 平均曲線（step ごと）─────────────────────
    mean1 = df1.groupby("step")[["J", "true_crop_sum"]].mean().reset_index()
    if mean1.empty:
        print("file1 の平均用データが空です。CSV を確認してください。")
        return

    t1_mean = mean1["step"].to_numpy() * dt
    J1_mean = mean1["J"].to_numpy()
    C1_mean = mean1["true_crop_sum"].to_numpy()

    if has_second:
        mean2 = df2.groupby("step")[["J", "true_crop_sum"]].mean().reset_index()
        if mean2.empty:
            print("file2 の平均用データが空です。file1 だけを描画します。")
            has_second = False
        else:
            t2_mean = mean2["step"].to_numpy() * dt
            J2_mean = mean2["J"].to_numpy()
            C2_mean = mean2["true_crop_sum"].to_numpy()

    # 色（以前仕様）
    color1 = "tab:blue"
    color2 = "tab:red"

    # =====================================================
    # 1) Objective J(t)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # file1: 各 run
    for i, r in enumerate(runs1):
        sub = df1[df1["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        J = sub["J"].to_numpy()
        ax.plot(
            t, J,
            color=color1,
            alpha=0.3,
            label=(f"{label1} 各試行" if i == 0 else None),
        )

    # file1: 平均（太線）
    ax.plot(
        t1_mean, J1_mean,
        color=color1,
        linewidth=3.0,
        label=f"{label1} 平均",
    )

    # file2
    if has_second:
        for i, r in enumerate(runs2):
            sub = df2[df2["run"] == r].sort_values("step")
            if sub.empty:
                continue
            t = sub["step"].to_numpy() * dt
            J = sub["J"].to_numpy()
            ax.plot(
                t, J,
                color=color2,
                alpha=0.3,
                label=(f"{label2} 各試行" if i == 0 else None),
            )

        ax.plot(
            t2_mean, J2_mean,
            color=color2,
            linewidth=3.0,
            label=f"{label2} 平均",
        )

    # ── 理論直線（★以前仕様：gamma/ds を掛ける） ──────────────
    if gamma is not None:
        t_common = t1_mean
        J0 = float(J1_mean[0])

        param_path = _infer_param_file(file1)
        N = _load_num_uavs_from_param(param_path) if (param_path is not None) else None
        if N is None:
            N = 1
            print(f"[WARN] {file1}: num_uavs が取得できなかったので N=1 とみなして理論線を描画します。")

        # ★以前仕様を維持
        J_ideal = J0 - N * (gamma / ds) * t_common

        ax.plot(
            t_common, J_ideal,
            "k--", linewidth=2.5,
            label=rf"理論直線 $J_0 - N_{{\rm UAV}}\gamma t$ (N={N}, $\gamma={gamma:.2f}$)"
        )

    ax.set_xlabel(f"時間 (s, step×{dt:.2f}s)")
    ax.set_ylabel("目的関数 $J$")
    ax.set_title("Objective $J$ の推移（2条件比較）" if has_second else f"Objective $J$ の推移（{label1}）")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================
    # 2) true_crop_sum(t)
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, r in enumerate(runs1):
        sub = df1[df1["run"] == r].sort_values("step")
        if sub.empty:
            continue
        t = sub["step"].to_numpy() * dt
        C = sub["true_crop_sum"].to_numpy()
        ax.plot(
            t, C,
            color=color1,
            alpha=0.3,
            label=(f"{label1} 各試行" if i == 0 else None),
        )

    ax.plot(
        t1_mean, C1_mean,
        color=color1,
        linewidth=3.0,
        label=f"{label1} 平均",
    )

    if has_second:
        for i, r in enumerate(runs2):
            sub = df2[df2["run"] == r].sort_values("step")
            if sub.empty:
                continue
            t = sub["step"].to_numpy() * dt
            C = sub["true_crop_sum"].to_numpy()
            ax.plot(
                t, C,
                color=color2,
                alpha=0.3,
                label=(f"{label2} 各試行" if i == 0 else None),
            )

        ax.plot(
            t2_mean, C2_mean,
            color=color2,
            linewidth=3.0,
            label=f"{label2} 平均",
        )

    ax.set_xlabel(f"時間 (s, step×{dt:.2f}s)")
    ax.set_ylabel("累積真値の合計")
    ax.set_title("UGV 訪問セルの累積真値（2条件比較）" if has_second else f"UGV 訪問セルの累積真値（{label1}）")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ① 単独ファイル（以前表示）
    # file_single = "test_results_3runs.csv"
    # plot_two_results_files(
    #     file_single,
    #     file2=None,
    #     label1="条件",
    #     dt=0.1,
    #     ds=0.5,
    #     gamma=3.0,
    # )

    # ② 2条件比較（以前表示）
    file_A = "multi_uav_multi_ugv_use_path_suenaga_results_10runs.csv"
    file_B = "multi_uav_multi_ugv_results_10runs.csv"
    plot_two_results_files(
        file_A, file_B,
        label1="条件A",
        label2="条件B",
        dt=0.1,
        ds=0.5,
        gamma=3.0,
    )
