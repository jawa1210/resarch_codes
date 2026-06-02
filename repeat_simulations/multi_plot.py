import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib  # noqa: F401
import os
import numpy as np
import matplotlib.patheffects as pe
import hashlib

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
    df = df.copy()
    if "run" in df.columns:
        return df
    if "run_idx" in df.columns:
        df["run"] = df["run_idx"]
        return df
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

def _load_threshold_from_param(param_path: str, default: float = 0.7) -> float:
    try:
        dfp = pd.read_csv(param_path)
        if "calibrator_threshold" in dfp.columns:
            th = float(dfp["calibrator_threshold"].iloc[0])
            print(f"[INFO] {param_path} から threshold = {th} を取得しました。")
            return th
    except Exception as e:
        print(f"[WARN] threshold 読み込み失敗: {e}")

    print(f"[WARN] threshold が読めないので default={default} を使います。")
    return default


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

def generate_ground_truth_map_scalar(
    grid_size=20,
    num_blobs=4,
    amp_range=(1.0, 1.5),
    sigma_range=(2.5, 4.5),
    background=0.05,
    noise_std=0.03,
    max_value=1.0,
    seed=None,
):
    rng = np.random.default_rng(seed)

    H = W = grid_size
    I, J = np.indices((H, W))
    gt = np.full((H, W), background, dtype=float)

    for _ in range(num_blobs):
        cy = rng.uniform(0, H - 1)
        cx = rng.uniform(0, W - 1)
        amp = rng.uniform(*amp_range)
        sig = rng.uniform(*sigma_range)

        dist2 = (I - cy) ** 2 + (J - cx) ** 2
        gt += amp * np.exp(-dist2 / (2.0 * sig ** 2))

    gt += rng.normal(0.0, noise_std, size=(H, W))
    gt = np.clip(gt, 0.0, None)

    m = float(gt.max())
    if m > 1e-12:
        gt = gt / m * max_value

    return gt

def _stable_int_seed(*items) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)


def compute_gt_stats_from_params(
    params_csv_path: str,
    threshold: float | None = None,
):
    dfp = pd.read_csv(params_csv_path)

    rows = []

    for _, row in dfp.iterrows():
        run_idx = int(row["run_idx"])
        master_seed = int(row["master_seed"])
        grid_size = int(row["grid_size"])
        num_uavs = int(row["num_uavs"])
        num_ugvs = int(row["num_ugvs"])

        th = float(row["calibrator_threshold"]) if threshold is None else float(threshold)

        gt_seed = _stable_int_seed(
            master_seed, run_idx, grid_size, num_uavs, num_ugvs, "gt"
        )

        gt = generate_ground_truth_map_scalar(
            grid_size=grid_size,
            seed=gt_seed,
            num_blobs=20,   # 本体コードと一致
        )

        rows.append({
            "run_idx": run_idx,
            "gt_seed": gt_seed,
            "gt_sum": float(np.sum(gt)),
            "gt_count_ge_threshold": int(np.sum(gt >= th)),
            "gt_sum_ge_threshold": float(np.sum(gt[gt >= th])),
            "threshold": th,
        })

    return pd.DataFrame(rows)


def infer_params_path_from_data_path(data_path: str) -> str | None:
    cand = data_path.replace("_data_", "_params_")
    if os.path.exists(cand):
        return cand
    return None


def plot_two_results_files(
        file1: str,
        file2: str | None = None,
        label1: str | None = None,
        label2: str | None = None,
        dt: float = 0.1,
        ds: float = 0.5,
        gamma: float | None = 3.0,
        max_step: int | None = None,
        max_time: float | None = None,
        plot_eval: bool = True,
        plot_sogp_case: bool = True,
        plot_threshold_count: bool = True,
        threshold: float | None = None,
    ):
    """
    file2 を None にすると単独ファイルのプロットになる。
    file2 を指定すると 2条件比較プロットになる。

    gamma: 1 UAV あたりの「1秒あたりの減少量 γ」
    → 理論線: J_ideal(t) = J0 - N_uav * (gamma/ds) * t   （★以前仕様）
    """

    has_second=None
    # ── CSV 読み込み ───────────────────────────
    df1_raw = pd.read_csv(file1)
    df1 = _sanitize_df(df1_raw, "file1")

    # =====================================================
    # GT情報をparams CSVから自動再生成して結合
    # =====================================================
    param1 = infer_params_path_from_data_path(file1)
    if param1 is not None:
        gt_stats1 = compute_gt_stats_from_params(param1, threshold=threshold)
        df1 = df1.merge(
            gt_stats1,
            left_on="run",
            right_on="run_idx",
            how="left",
            suffixes=("", "_gt")
        )
        print("[INFO] file1 に gt_sum / gt_count_ge_threshold を追加しました。")
    else:
        print(f"[WARN] file1 の params CSV が見つかりません: {file1}")

    # ── 表示区間のカット ─────────────────────────
    if max_time is not None:
        max_step_eff = int(np.floor(max_time / dt))
    else:
        max_step_eff = max_step

    if max_step_eff is not None:
        df1 = df1[df1["step"] <= max_step_eff].copy()
        if has_second:
            df2 = df2[df2["step"] <= max_step_eff].copy()


    if label1 is None:
        label1 = _nice_label_from_path(file1)

    if file2 is not None:
        df2_raw = pd.read_csv(file2)
        df2 = _sanitize_df(df2_raw, "file2")

        param2 = infer_params_path_from_data_path(file2)
        if param2 is not None:
            gt_stats2 = compute_gt_stats_from_params(param2, threshold=threshold)
            df2 = df2.merge(
                gt_stats2,
                left_on="run",
                right_on="run_idx",
                how="left",
                suffixes=("", "_gt")
            )
            print("[INFO] file2 に gt_sum / gt_count_ge_threshold を追加しました。")
        else:
            print(f"[WARN] file2 の params CSV が見つかりません: {file2}")

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

    # =====================================================
    # Harvest ratio summary
    # =====================================================
    def print_harvest_ratio_summary(df, label):
        if "gt_sum" not in df.columns:
            print(f"[WARN] {label}: gt_sum がないため収穫率を計算できません。")
            return

        # UGVが通ったpath上の真値合計 = 最終的な累積収穫量
        final_crop = (
            df.sort_values("step")
            .groupby("run")["true_crop_sum"]
            .last()
        )

        # GT map全体の真値合計
        gt_sum = df.groupby("run")["gt_sum"].first()

        ratio = 100.0 * final_crop / gt_sum

        # UGVが通ったセルのうち、真値がthreshold以上だった回数
        target_cols = [
            c for c in df.columns
            if c.startswith("ugv") and c.endswith("_target_true")
        ]

        if len(target_cols) > 0:
            th = df.groupby("run")["threshold"].first()
            path_threshold_count = {}

            for run_id, sub in df.groupby("run"):
                sub = sub.sort_values("step").copy()
                threshold_run = float(th.loc[run_id])

                # true_crop_sum が増えたstepだけを「収穫発生step」とする
                sub["crop_delta"] = sub["true_crop_sum"].diff().fillna(sub["true_crop_sum"])
                harvest_mask = sub["crop_delta"] > 1e-9

                # 収穫が発生したstepだけで、target_true >= threshold のUGV数を数える
                vals = sub.loc[harvest_mask, target_cols].to_numpy()
                path_threshold_count[run_id] = int(np.sum(vals >= threshold_run))

            path_threshold_count = pd.Series(path_threshold_count)
        else:
            path_threshold_count = None

        print("\n" + "=" * 70)
        print(f"{label}: UGV path harvest summary")
        print("=" * 70)

        print(f"UGV path 真値合計 平均: {final_crop.mean():.3f}")
        print(f"UGV path 真値合計 最小: {final_crop.min():.3f}")
        print(f"UGV path 真値合計 最大: {final_crop.max():.3f}")

        print(f"収穫量 / GT真値合計 平均: {ratio.mean():.2f}%")
        print(f"収穫量 / GT真値合計 最小: {ratio.min():.2f}%")
        print(f"収穫量 / GT真値合計 最大: {ratio.max():.2f}%")

        print(f"GT真値合計 平均: {gt_sum.mean():.3f}")

        if path_threshold_count is not None:
            print(f"UGV path 閾値以上セル収穫数 平均: {path_threshold_count.mean():.3f}")
            print(f"UGV path 閾値以上セル収穫数 最小: {path_threshold_count.min():.0f}")
            print(f"UGV path 閾値以上セル収穫数 最大: {path_threshold_count.max():.0f}")

    print_harvest_ratio_summary(df1, label1)

    if has_second:
        print_harvest_ratio_summary(df2, label2)

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

        # 終点に run_index を表示
        ax.text(
            t[-1], J[-1],
            f"{r}",
            fontsize=10,
            color=color1,
            ha="left",
            va="center"
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

            ax.text(
                t[-1], J[-1],
                f"{r}",
                fontsize=10,
                color=color2,
                ha="left",
                va="center"
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
    ax.set_ylim(bottom=0.0)
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

        # ★追加：終点にrun番号
        ax.text(
        t[-1], C[-1],
        f"{r}",
        fontsize=10,
        color=color1,
        ha="left",
        va="center"
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

            ax.text(
                t[-1], C[-1],
                f"{r}",
                fontsize=10,
                color=color2,
                ha="left",
                va="center"
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

    # =====================================================
    # 3) Evaluation metrics at UGV planning timing
    # =====================================================
    if plot_eval:
        eval_metrics = [
            ("reachable_rmse", "Reachable領域RMSE", "推定誤差 RMSE"),
            ("reachable_mae", "Reachable領域MAE", "推定誤差 MAE"),
            ("reachable_var", "Reachable領域平均分散", "GP分散"),
            ("reachable_calib", "Reachable領域Calibration誤差", r"$|e^2-\sigma^2|$"),

            ("path_rmse", "Planned path RMSE", "経路上の推定誤差 RMSE"),
            ("path_mae", "Planned path MAE", "経路上の推定誤差 MAE"),
            ("path_var", "Planned path 平均分散", "経路上のGP分散"),

            ("target_error", "Target cell error", "次セルの絶対誤差"),
            ("target_var", "Target cell variance", "次セルのGP分散"),
            ("path_true_sum", "Planned path 真値合計", "経路上の真値合計"),
            ("path_mu_sum", "Planned path 推定平均合計", "経路上の推定平均合計"),
            ("path_var_sum", "Planned path 分散合計", "経路上の分散合計"),
            ("path_prob_sum", "Planned path 確率合計", "経路上の確率合計"),
            ("path_expected_sum", "Planned path 期待収穫量合計", r"$\sum \mu p$"),
            ("target_true", "Target cell 真値", "次セルの真値"),
            ("target_mu", "Target cell 推定平均", "次セルの推定平均"),
            ("target_prob", "Target cell 確率", "次セルの確率"),
            ("target_expected", "Target cell 期待収穫量", r"$\mu p$"),
        ]

        for metric_key, title, ylabel in eval_metrics:
            cols1 = [c for c in df1.columns if c.startswith("ugv") and c.endswith(f"_{metric_key}")]

            if len(cols1) == 0:
                print(f"[WARN] {metric_key} の列がCSVにありません。シミュレーションコード側で保存してください。")
                continue

            df1_eval = df1.copy()
            df1_eval[metric_key] = df1_eval[cols1].mean(axis=1)

            mean1_eval = df1_eval.groupby("step")[metric_key].mean().reset_index()
            t1_eval = mean1_eval["step"].to_numpy() * dt
            y1_eval = mean1_eval[metric_key].to_numpy()

            fig, ax = plt.subplots(figsize=(10, 6))

            for i, r in enumerate(runs1):
                sub = df1_eval[df1_eval["run"] == r].sort_values("step")
                if sub.empty:
                    continue
                t = sub["step"].to_numpy() * dt
                y = sub[metric_key].to_numpy()

                ax.plot(
                    t, y,
                    color=color1,
                    alpha=0.25,
                    label=(f"{label1} 各試行" if i == 0 else None),
                )

            ax.plot(
                t1_eval, y1_eval,
                color=color1,
                linewidth=3.0,
                label=f"{label1} 平均",
            )

            if has_second:
                cols2 = [c for c in df2.columns if c.startswith("ugv") and c.endswith(f"_{metric_key}")]

                if len(cols2) == 0:
                    print(f"[WARN] file2 に {metric_key} の列がありません。")
                else:
                    df2_eval = df2.copy()
                    df2_eval[metric_key] = df2_eval[cols2].mean(axis=1)

                    mean2_eval = df2_eval.groupby("step")[metric_key].mean().reset_index()
                    t2_eval = mean2_eval["step"].to_numpy() * dt
                    y2_eval = mean2_eval[metric_key].to_numpy()

                    for i, r in enumerate(runs2):
                        sub = df2_eval[df2_eval["run"] == r].sort_values("step")
                        if sub.empty:
                            continue
                        t = sub["step"].to_numpy() * dt
                        y = sub[metric_key].to_numpy()

                        ax.plot(
                            t, y,
                            color=color2,
                            alpha=0.25,
                            label=(f"{label2} 各試行" if i == 0 else None),
                        )

                    ax.plot(
                        t2_eval, y2_eval,
                        color=color2,
                        linewidth=3.0,
                        label=f"{label2} 平均",
                    )

            ax.set_xlabel(f"時間 (s, step×{dt:.2f}s)")
            ax.set_ylabel(ylabel)
            ax.set_title(title + "（UGV経路計画時点）")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

    # =====================================================
    # 4) UGV threshold passing count
    # =====================================================
    if plot_threshold_count:
        param_path = _infer_param_file(file1)
        th = threshold
        if th is None:
            th = _load_threshold_from_param(param_path, default=0.7) if param_path is not None else 0.7

        def add_threshold_count(df: pd.DataFrame, name: str) -> pd.DataFrame | None:
            df = df.copy()

            target_cols = [
                c for c in df.columns
                if c.startswith("ugv") and c.endswith("_target_true")
            ]

            if len(target_cols) == 0:
                print(f"[WARN] {name}: ugv*_target_true 列がありません。")
                print("       シミュレーション側で ugv0_target_true, ugv1_target_true を保存してください。")
                return None

            df = df.sort_values(["run", "step"]).copy()

            # 収穫量が増えたstepだけを採用
            df["crop_delta"] = df.groupby("run")["true_crop_sum"].diff().fillna(df["true_crop_sum"])
            df["harvest_step"] = df["crop_delta"] > 1e-9

            # 収穫が発生したstepのみ、閾値以上の target_true を数える
            df["threshold_hit_step"] = 0
            df.loc[df["harvest_step"], "threshold_hit_step"] = (
                df.loc[df["harvest_step"], target_cols] >= th
            ).sum(axis=1)

            df["threshold_hit_cumsum"] = (
                df.groupby("run")["threshold_hit_step"].cumsum()
            )

            return df

        df1_hit = add_threshold_count(df1, "file1")

        if df1_hit is not None:
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, r in enumerate(runs1):
                sub = df1_hit[df1_hit["run"] == r].sort_values("step")
                if sub.empty:
                    continue

                t = sub["step"].to_numpy() * dt
                y = sub["threshold_hit_cumsum"].to_numpy()

                ax.plot(
                    t, y,
                    color=color1,
                    alpha=0.25,
                    label=(f"{label1} 各試行" if i == 0 else None),
                )

            mean1_hit = (
                df1_hit.groupby("step")["threshold_hit_cumsum"]
                .mean()
                .reset_index()
            )

            ax.plot(
                mean1_hit["step"].to_numpy() * dt,
                mean1_hit["threshold_hit_cumsum"].to_numpy(),
                color=color1,
                linewidth=3.0,
                label=f"{label1} 平均",
            )

            if has_second:
                df2_hit = add_threshold_count(df2, "file2")

                if df2_hit is not None:
                    for i, r in enumerate(runs2):
                        sub = df2_hit[df2_hit["run"] == r].sort_values("step")
                        if sub.empty:
                            continue

                        t = sub["step"].to_numpy() * dt
                        y = sub["threshold_hit_cumsum"].to_numpy()

                        ax.plot(
                            t, y,
                            color=color2,
                            alpha=0.25,
                            label=(f"{label2} 各試行" if i == 0 else None),
                        )

                    mean2_hit = (
                        df2_hit.groupby("step")["threshold_hit_cumsum"]
                        .mean()
                        .reset_index()
                    )

                    ax.plot(
                        mean2_hit["step"].to_numpy() * dt,
                        mean2_hit["threshold_hit_cumsum"].to_numpy(),
                        color=color2,
                        linewidth=3.0,
                        label=f"{label2} 平均",
                    )

            ax.set_xlabel(f"時間 (s, step×{dt:.2f}s)")
            ax.set_ylabel("閾値以上セル収穫回数の累積")
            ax.set_title(rf"UGVが真値 $\geq {th:.2f}$ のセルを収穫した累積回数")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()

    # =====================================================
    # 4) SOGP case count
    # =====================================================
    if plot_sogp_case:
        case_keys = [
            ("case1_total", "Case1: 新規basis追加"),
            ("case2_total", "Case2: 既存basis更新"),
            ("case3_total", "Case3: basis削除/置換"),
            ("basis_total", "Basis total"),
            ("sparse_ratio_total", "Sparse ratio"),
        ]

        for col, title in case_keys:
            if col not in df1.columns:
                print(f"[WARN] file1 に {col} がありません。")
                continue

            mean1_case = df1.groupby("step")[col].mean().reset_index()
            t1_case = mean1_case["step"].to_numpy() * dt
            y1_case = mean1_case[col].to_numpy()

            fig, ax = plt.subplots(figsize=(10, 6))

            for i, r in enumerate(runs1):
                sub = df1[df1["run"] == r].sort_values("step")
                if sub.empty:
                    continue

                t = sub["step"].to_numpy() * dt
                y = sub[col].to_numpy()

                ax.plot(
                    t, y,
                    color=color1,
                    alpha=0.25,
                    label=(f"{label1} 各試行" if i == 0 else None),
                )

            ax.plot(
                t1_case, y1_case,
                color=color1,
                linewidth=3.0,
                label=f"{label1} 平均",
            )

            if has_second:
                if col not in df2.columns:
                    print(f"[WARN] file2 に {col} がありません。")
                else:
                    mean2_case = df2.groupby("step")[col].mean().reset_index()
                    t2_case = mean2_case["step"].to_numpy() * dt
                    y2_case = mean2_case[col].to_numpy()

                    for i, r in enumerate(runs2):
                        sub = df2[df2["run"] == r].sort_values("step")
                        if sub.empty:
                            continue

                        t = sub["step"].to_numpy() * dt
                        y = sub[col].to_numpy()

                        ax.plot(
                            t, y,
                            color=color2,
                            alpha=0.25,
                            label=(f"{label2} 各試行" if i == 0 else None),
                        )

                    ax.plot(
                        t2_case, y2_case,
                        color=color2,
                        linewidth=3.0,
                        label=f"{label2} 平均",
                    )

            ax.set_xlabel(f"時間 (s, step×{dt:.2f}s)")
            ax.set_ylabel(col)
            ax.set_title(f"{title} の推移")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # ① 単独ファイル（以前表示）
    # file_single = "miyashita_future_ucb_poster_seed1234_data_1runs_004.csv"
    # plot_two_results_files(
    #     file_single,
    #     file2=None,
    #     label1="条件",
    #     dt=0.1,
    #     ds=0.5,
    #     gamma=8.0,
    #     max_step=1000
    # )

    # ② 2条件比較（以前表示）
    file_A = "collab_mu_only_ugv_future_collab_mu_only_ugv_future_seed1234_data_15runs.csv"
    file_B = "collab_mu_only_prob_collab_mu_only_prob_seed1234_data_15runs.csv"
    plot_two_results_files(
        file_A, file_B,
        label1="条件A",
        label2="条件B",
        dt=0.1,
        ds=0.5,
        gamma=6.0,
        plot_eval=True,   # Falseにすれば評価plotをOFF
        plot_sogp_case=True, # FalseにすればSOGP case plotをOFF
        plot_threshold_count=True,
        threshold=None, ) 