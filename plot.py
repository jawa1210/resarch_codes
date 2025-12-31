import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib

# ── フォント／mathtext 設定 ────────────────────────────────
mpl.rcParams['font.family']        = 'IPAexGothic'
mpl.rcParams['text.usetex']        = False
mpl.rcParams['font.size']          = 16
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['lines.linewidth']    = 4
mpl.rcParams['axes.titlesize']     = 20
mpl.rcParams['axes.labelsize']     = 20
mpl.rcParams['legend.fontsize']    = 16
mpl.rcParams['xtick.labelsize']    = 16
mpl.rcParams['ytick.labelsize']    = 16

def _nice_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _infer_dt(df: pd.DataFrame, default_dt: float = 0.1) -> float:
    # 元コードの t = step/10 から dt=0.1 を想定。列があればそれを優先。
    if "cfg.control_period" in df.columns:
        try:
            return float(df["cfg.control_period"].iloc[0])
        except Exception:
            pass
    return default_dt

def _gather_ugv_indices(df: pd.DataFrame) -> list[int]:
    # ugv0_..., ugv1_... のような列から k を拾う
    ks = set()
    for c in df.columns:
        if c.startswith("ugv") and "_visited_count" in c:
            head = c.split("_")[0]  # "ugv0"
            try:
                ks.add(int(head.replace("ugv", "")))
            except Exception:
                pass
    return sorted(ks)

def _compute_fleet_visited_means(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    複数UGVの visited_* を合算して fleet の mean を作る。
    fleet_mu_mean[t] = sum_k visited_mu_sum_k[t] / sum_k visited_count_k[t]
    """
    ks = _gather_ugv_indices(df)
    if len(ks) == 0:
        raise ValueError("CSVに ugv{k}_visited_count 等の列が見つかりません。UGVログ追加後のCSVを指定してください。")

    cnt = None
    mu_sum = None
    var_sum = None
    prob_sum = None

    for k in ks:
        c_cnt = f"ugv{k}_visited_count"
        c_mu  = f"ugv{k}_visited_mu_sum"
        c_var = f"ugv{k}_visited_var_sum"
        c_pr  = f"ugv{k}_visited_prob_sum"
        for cc in [c_cnt, c_mu, c_var, c_pr]:
            if cc not in df.columns:
                raise ValueError(f"列 {cc} がCSVにありません（UGVログの保存が未反映かも）。")

        cnt_k = df[c_cnt].to_numpy(dtype=float)
        mu_k  = df[c_mu].to_numpy(dtype=float)
        var_k = df[c_var].to_numpy(dtype=float)
        pr_k  = df[c_pr].to_numpy(dtype=float)

        cnt     = cnt_k if cnt is None else (cnt + cnt_k)
        mu_sum  = mu_k  if mu_sum is None else (mu_sum + mu_k)
        var_sum = var_k if var_sum is None else (var_sum + var_k)
        prob_sum= pr_k  if prob_sum is None else (prob_sum + pr_k)

    eps = 1e-12
    mu_mean   = mu_sum   / np.maximum(cnt, eps)
    var_mean  = var_sum  / np.maximum(cnt, eps)
    prob_mean = prob_sum / np.maximum(cnt, eps)

    return mu_mean, var_mean, prob_mean

def _annotate_overall_mean(ax, label: str, y: np.ndarray, xpos: float, ypos: float, unit: str = ""):
    y_mean = float(np.nanmean(y))
    txt = f"{label}: 平均={y_mean:.4f}{unit}"
    ax.text(
        xpos, ypos, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )

def _end_label(ax, t: np.ndarray, y: np.ndarray, text: str):
    # 線の右端にラベル（凡例の代わり）
    ax.text(
        float(t[-1]), float(y[-1]),
        f"  {text}",
        va="center", ha="left",
        fontsize=12,
        alpha=0.85,
        clip_on=True,
    )

def plot_files_extended(file_list, gamma=3.0, time_scale_div=10.0):
    """
    time_scale_div=10.0 なら t=step/10 (元コード互換)。dtが欲しければ time_scale_div を None にして dt推定にしてもOK。
    """
    data = []
    for path in file_list:
        df = pd.read_csv(path)

        step = df["step"].to_numpy(dtype=float)

        # 時間軸（元コード互換：/10）
        if time_scale_div is not None:
            t = step / float(time_scale_div)
        else:
            dt = _infer_dt(df, default_dt=0.1)
            t = step * dt

        J        = df["J"].to_numpy(dtype=float)
        true_sum = df["true_crop_sum"].to_numpy(dtype=float)

        J0     = float(J[0]) if len(J) else 0.0
        theory = J0 - gamma * (float(time_scale_div) if time_scale_div is not None else (1.0/_infer_dt(df))) * t
        label  = _nice_label(path)

        # ★ 追加：UGV通過セル集合の mean/var/prob の「平均」の時系列
        mu_mean, var_mean, prob_mean = _compute_fleet_visited_means(df)

        data.append((label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean))

    # ───────────────────────────────────────────────
    # 1) J 推移
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, J, theory, *_ in data:
        ax.plot(t, J)
        ax.plot(t, theory, "--")
        _end_label(ax, t, J, f"{label} 実測J")
    ax.set_xlabel("時間 (Step/10)")
    ax.set_ylabel("目的関数 $J$")
    ax.set_title("Objective $J$ の推移と理論直線")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ───────────────────────────────────────────────
    # 2) crops 推移
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, _, _, true_sum, *_ in data:
        ax.plot(t, true_sum)
        _end_label(ax, t, true_sum, f"{label} crops")
    ax.set_xlabel("時間 (Step/10)")
    ax.set_ylabel("真値の合計")
    ax.set_title("UGV 訪問セルの累積真値")
    ax.grid(True)
    ax.set_ylim(0,)
    plt.tight_layout()
    plt.show()

    # ───────────────────────────────────────────────
    # 3) visited mean (mu)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, *rest in data:
        mu_mean = rest[3]  # (true_sum, mu_mean, var_mean, prob_mean) の位置関係に注意
        ax.plot(t, mu_mean)
        _end_label(ax, t, mu_mean, f"{label}")
    ax.set_xlabel("時間 (Step/10)")
    ax.set_ylabel("visited 平均の平均値（μ）")
    ax.set_title("UGVが通ったセル集合：GP平均 μ の推移（平均）")
    ax.grid(True)

    # 図中に「系列平均」を表示（上から順に並べる）
    y0 = 0.98
    dy = 0.08
    for i, (label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean) in enumerate(data):
        _annotate_overall_mean(ax, label, mu_mean, xpos=0.02, ypos=y0 - i*dy)
    plt.tight_layout()
    plt.show()

    # ───────────────────────────────────────────────
    # 4) visited mean (var)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean in data:
        ax.plot(t, var_mean)
        _end_label(ax, t, var_mean, f"{label}")
    ax.set_xlabel("時間 (Step/10)")
    ax.set_ylabel("visited 平均の平均値（分散）")
    ax.set_title("UGVが通ったセル集合：GP分散 σ² の推移（平均）")
    ax.grid(True)

    y0 = 0.98
    dy = 0.08
    for i, (label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean) in enumerate(data):
        _annotate_overall_mean(ax, label, var_mean, xpos=0.02, ypos=y0 - i*dy)
    plt.tight_layout()
    plt.show()

    # ───────────────────────────────────────────────
    # 5) visited mean (prob)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean in data:
        ax.plot(t, prob_mean)
        _end_label(ax, t, prob_mean, f"{label}")
    ax.set_xlabel("時間 (Step/10)")
    ax.set_ylabel("visited 平均の平均値（確率）")
    ax.set_title("UGVが通ったセル集合：ロジスティック確率 p の推移（平均）")
    ax.grid(True)
    ax.set_ylim(0, 1)

    y0 = 0.98
    dy = 0.08
    for i, (label, t, J, theory, true_sum, mu_mean, var_mean, prob_mean) in enumerate(data):
        _annotate_overall_mean(ax, label, prob_mean, xpos=0.02, ypos=y0 - i*dy)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    files = [
        "multi_uav_multi_ugv_results_off_cbf.csv",
        "multi_uav_multi_ugv_results_off_cbf_logistic.csv",
    ]
    plot_files_extended(files, gamma=3.0)
