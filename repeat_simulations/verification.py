import pandas as pd
import numpy as np
from scipy import stats
import os
import hashlib

# =========================
# 設定
# =========================
file_A = "not_collab_nom_amb_two_re_not_collab_prob_init_change_all_nom_amb_grid_30_seed1234_data_15runs.csv"
file_B = "collab_amb_two_collab_prob_init_change_all_nom_amb_grid_seed1234_data_15runs.csv"

name_A = "A"
name_B = "B"

# 大きいほど良い指標は +1
# 小さいほど良い指標は -1
metrics = {
    # 最終成果
    "path_true_sum_total": +1,
    "final_true_crop_sum": +1,
    "harvest_ratio": +1,

    # 時間全体の評価
    "path_prob_sum_total_mean": +1,
    "path_prob_sum_total_auc": +1,

    "path_expected_sum_total_mean": +1,
    "path_expected_sum_total_auc": +1,

    "path_rmse_mean_timeavg": -1,
    "path_rmse_mean_auc": -1,

    "path_var_mean_timeavg": -1,
    "path_var_mean_auc": -1,

    # 最終状態
    "basis_total": None,
    "sparse_ratio_total": None,
}

#===========
# gt sum は全 run 共通なので、Aの方から取る
#===========
def _stable_int_seed(*items) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)


def _infer_param_file(results_path: str) -> str | None:
    base = os.path.basename(results_path)

    candidates = []

    # xxx_data_15runs.csv -> xxx_params_15runs.csv
    if "_data_" in base:
        candidates.append(base.replace("_data_", "_params_"))

    # 念のため: results -> params
    if "results" in base:
        candidates.append(base.replace("results", "params"))

    for cand in candidates:
        param_path = os.path.join(os.path.dirname(results_path), cand)
        if os.path.exists(param_path):
            return param_path

    return None

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

def infer_artifact_path_from_data_path(data_path: str) -> str:
    d = os.path.dirname(data_path)

    cand = os.path.join(
        d,
        "analysis_artifacts_allruns.csv"
    )

    if not os.path.exists(cand):
        raise FileNotFoundError(
            f"analysis_artifacts_allruns.csv が見つかりません: {cand}"
        )

    return cand


def compute_gt_stats_from_artifacts(
    artifact_csv_path: str,
    threshold: float | None = None,
):
    art = pd.read_csv(artifact_csv_path)

    gt_df = art[
        art["type"] == "gt_initial"
    ].copy()

    rows = []

    for run_idx, g in gt_df.groupby("run_idx"):

        vals = g["value"].to_numpy()

        th = 0.7 if threshold is None else float(threshold)

        rows.append({
            "run_idx": int(run_idx),
            "gt_sum": float(vals.sum()),
            "gt_count_ge_threshold": int((vals >= th).sum()),
            "gt_sum_ge_threshold": float(vals[vals >= th].sum()),
            "threshold": th,
        })

    return pd.DataFrame(rows)

# =========================
# 読み込み
# =========================
df_A = pd.read_csv(file_A)
df_B = pd.read_csv(file_B)

artifact_A = infer_artifact_path_from_data_path(file_A)
artifact_B = infer_artifact_path_from_data_path(file_B)

gt_A = compute_gt_stats_from_artifacts(artifact_A)
gt_B = compute_gt_stats_from_artifacts(artifact_B)

df_A = df_A.merge(gt_A, on="run_idx", how="left")
df_B = df_B.merge(gt_B, on="run_idx", how="left")

def make_run_summary(df):
    """
    各runについて、
    最終結果系の指標は最終stepから取得し、
    時系列で変化する指標は全step平均/AUCで評価する。
    """

    df = df.sort_values(["run_idx", "step"]).copy()

    # =========================
    # stepごとの指標を作る
    # =========================
    ugv_prob_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_prob_sum")]
    ugv_expected_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_expected_sum")]
    ugv_rmse_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_rmse")]
    ugv_var_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_var")]

    df["path_prob_sum_total_step"] = df[ugv_prob_cols].sum(axis=1) if ugv_prob_cols else np.nan
    df["path_expected_sum_total_step"] = df[ugv_expected_cols].sum(axis=1) if ugv_expected_cols else np.nan
    df["path_rmse_mean_step"] = df[ugv_rmse_cols].mean(axis=1) if ugv_rmse_cols else np.nan
    df["path_var_mean_step"] = df[ugv_var_cols].mean(axis=1) if ugv_var_cols else np.nan
    # =========================
    # 時間平均・AUC
    # =========================
    time_summary = (
        df.groupby("run_idx")
        .agg(
            path_prob_sum_total_mean=("path_prob_sum_total_step", "mean"),
            path_prob_sum_total_auc=("path_prob_sum_total_step", "sum"),

            path_expected_sum_total_mean=("path_expected_sum_total_step", "mean"),
            path_expected_sum_total_auc=("path_expected_sum_total_step", "sum"),

            path_rmse_mean_timeavg=("path_rmse_mean_step", "mean"),
            path_rmse_mean_auc=("path_rmse_mean_step", "sum"),

            path_var_mean_timeavg=("path_var_mean_step", "mean"),
            path_var_mean_auc=("path_var_mean_step", "sum"),
        )
        .reset_index()
    )

    # =========================
    # 最終stepの指標
    # =========================
    last = df.groupby("run_idx").tail(1).copy()

    ugv_true_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_true_sum")]
    last["path_true_sum_total"] = last[ugv_true_cols].sum(axis=1) if ugv_true_cols else np.nan

    last["final_true_crop_sum"] = last["true_crop_sum"]

    last["harvest_ratio"] = (
        last["true_crop_sum"] / last["gt_sum"]
    )

    # 最終値も残したい場合
    last["path_prob_sum_total_final"] = last["path_prob_sum_total_step"]
    last["path_expected_sum_total_final"] = last["path_expected_sum_total_step"]
    last["path_rmse_mean_final"] = last["path_rmse_mean_step"]
    last["path_var_mean_final"] = last["path_var_mean_step"]

    # =========================
    # 結合
    # =========================
    summary = last.merge(time_summary, on="run_idx", how="left")

    return summary

A = make_run_summary(df_A)
B = make_run_summary(df_B)

# run_idxで対応づけ
merged = pd.merge(
    A,
    B,
    on="run_idx",
    suffixes=(f"_{name_A}", f"_{name_B}")
)

print(f"対応できたrun数: {len(merged)}")
print("run_idx:", sorted(merged["run_idx"].tolist()))

# =========================
# 対応あり検定
# =========================
results = []

for metric, direction in metrics.items():
    col_A = f"{metric}_{name_A}"
    col_B = f"{metric}_{name_B}"

    if col_A not in merged.columns or col_B not in merged.columns:
        print(f"[SKIP] {metric}: column not found")
        continue

    x_A = merged[col_A].to_numpy()
    x_B = merged[col_B].to_numpy()

    # 差 d_i = B - A
    diff = x_B - x_A

    n = len(diff)
    mean_A = np.mean(x_A)
    std_A = np.std(x_A, ddof=1)

    mean_B = np.mean(x_B)
    std_B = np.std(x_B, ddof=1)

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # 95% confidence interval
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # paired t-test
    t_stat, p_t = stats.ttest_rel(x_B, x_A)

    # Wilcoxon signed-rank test
    try:
        w_stat, p_w = stats.wilcoxon(diff)
    except ValueError:
        w_stat, p_w = np.nan, np.nan

    # Cohen's dz
    cohen_dz = mean_diff / std_diff if std_diff != 0 else np.nan

    # 勝率
    if direction == +1:
        win_rate = np.mean(diff > 0)
    elif direction == -1:
        win_rate = np.mean(diff < 0)
    else:
        win_rate = np.nan

    # 改善率
    if direction == +1:
        improvement_rate = mean_diff / abs(mean_A) * 100
    elif direction == -1:
        improvement_rate = -mean_diff / abs(mean_A) * 100
    else:
        improvement_rate = np.nan

    results.append({
        "metric": metric,
        f"{name_A}_mean±std": f"{mean_A:.4f} ± {std_A:.4f}",
        f"{name_B}_mean±std": f"{mean_B:.4f} ± {std_B:.4f}",
        "mean_diff_B-A": mean_diff,
        "std_diff": std_diff,
        "95%CI_low": ci_low,
        "95%CI_high": ci_high,
        "paired_t_p": p_t,
        "wilcoxon_p": p_w,
        "cohen_dz": cohen_dz,
        "win_rate": win_rate,
        "improvement_%": improvement_rate,
    })

result_df = pd.DataFrame(results)



print("\n" + "="*80)
print("PAIRED STATISTICAL TEST RESULTS")
print("="*80)

for _, row in result_df.iterrows():
    print(f"\n■ {row['metric']}")
    print("-"*60)

    print(f"A : {row[f'{name_A}_mean±std']}")
    print(f"B : {row[f'{name_B}_mean±std']}")

    print(f"平均差(B-A)      : {row['mean_diff_B-A']:.4f}")
    print(f"差の標準偏差      : {row['std_diff']:.4f}")

    print(
        f"95% CI          : "
        f"[{row['95%CI_low']:.4f}, {row['95%CI_high']:.4f}]"
    )

    print(f"paired t p      : {row['paired_t_p']:.6f}")
    print(f"Wilcoxon p      : {row['wilcoxon_p']:.6f}")

    print(f"Cohen dz        : {row['cohen_dz']:.3f}")

    if not pd.isna(row['win_rate']):
        print(f"勝率             : {row['win_rate']*100:.1f}%")

    if not pd.isna(row['improvement_%']):
        print(f"改善率           : {row['improvement_%']:.2f}%")

# =========================
# Bが負けているrun 上位3つ
# ＋ 各run内で負け幅が大きいstep 上位3つ
# =========================

def add_step_metrics(df):
    df = df.sort_values(["run_idx", "step"]).copy()

    prob_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_prob_sum")]
    exp_cols  = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_expected_sum")]
    rmse_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_rmse")]
    var_cols  = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_var")]
    true_cols = [c for c in df.columns if c.startswith("ugv") and c.endswith("_path_true_sum")]

    df["path_prob_sum_total_step"] = df[prob_cols].sum(axis=1) if prob_cols else np.nan
    df["path_expected_sum_total_step"] = df[exp_cols].sum(axis=1) if exp_cols else np.nan
    df["path_rmse_mean_step"] = df[rmse_cols].mean(axis=1) if rmse_cols else np.nan
    df["path_var_mean_step"] = df[var_cols].mean(axis=1) if var_cols else np.nan
    df["path_true_sum_total_step"] = df[true_cols].sum(axis=1) if true_cols else np.nan
    df["final_true_crop_sum_step"] = df["true_crop_sum"]
    df["harvest_ratio_step"] = df["true_crop_sum"] / df["gt_sum"]

    return df

step_A = add_step_metrics(df_A)
step_B = add_step_metrics(df_B)

step_merged = pd.merge(
    step_A,
    step_B,
    on=["run_idx", "step"],
    suffixes=(f"_{name_A}", f"_{name_B}")
)

step_metric_map = {
    "path_true_sum_total": "path_true_sum_total_step",
    "final_true_crop_sum": "final_true_crop_sum_step",
    "harvest_ratio": "harvest_ratio_step",

    "path_prob_sum_total_mean": "path_prob_sum_total_step",
    "path_prob_sum_total_auc": "path_prob_sum_total_step",

    "path_expected_sum_total_mean": "path_expected_sum_total_step",
    "path_expected_sum_total_auc": "path_expected_sum_total_step",

    "path_rmse_mean_timeavg": "path_rmse_mean_step",
    "path_rmse_mean_auc": "path_rmse_mean_step",

    "path_var_mean_timeavg": "path_var_mean_step",
    "path_var_mean_auc": "path_var_mean_step",
}

print("\n" + "="*80)
print("Bが負けているrun 上位3つ ＋ 負け幅が大きいstep 上位3つ")
print("="*80)

for metric, direction in metrics.items():

    if direction is None:
        continue

    col_A = f"{metric}_{name_A}"
    col_B = f"{metric}_{name_B}"

    if col_A not in merged.columns or col_B not in merged.columns:
        continue

    tmp = merged[["run_idx", col_A, col_B]].copy()
    tmp["diff_B_minus_A"] = tmp[col_B] - tmp[col_A]

    # Bの負け幅
    # 大きいほど良い指標: A - B
    # 小さいほど良い指標: B - A
    if direction == +1:
        tmp["B_loss_amount"] = tmp[col_A] - tmp[col_B]
    elif direction == -1:
        tmp["B_loss_amount"] = tmp[col_B] - tmp[col_A]

    lose_runs = (
        tmp[tmp["B_loss_amount"] > 0]
        .sort_values("B_loss_amount", ascending=False)
        .head(3)
    )

    print(f"\n■ {metric}")
    print("-"*80)

    if lose_runs.empty:
        print("Bが負けているrunはありません。")
        continue

    for _, r in lose_runs.iterrows():
        run_idx = int(r["run_idx"])

        print(
            f"\nrun_idx={run_idx} | "
            f"A={r[col_A]:.6f}, B={r[col_B]:.6f}, "
            f"B負け幅={r['B_loss_amount']:.6f}"
        )

        step_base = step_metric_map.get(metric)

        if step_base is None:
            print("  step別比較なし")
            continue

        sA = f"{step_base}_{name_A}"
        sB = f"{step_base}_{name_B}"

        if sA not in step_merged.columns or sB not in step_merged.columns:
            print("  step別比較に必要な列がありません。")
            continue

        ss = step_merged[step_merged["run_idx"] == run_idx][
            ["step", sA, sB]
        ].copy()

        ss["diff_B_minus_A"] = ss[sB] - ss[sA]

        if direction == +1:
            ss["B_loss_amount"] = ss[sA] - ss[sB]
        elif direction == -1:
            ss["B_loss_amount"] = ss[sB] - ss[sA]

        top_steps = (
            ss[ss["B_loss_amount"] > 0]
            .sort_values("B_loss_amount", ascending=False)
            .head(3)
        )

        print("  負け幅が大きいstep 上位3つ:")

        if top_steps.empty:
            print("    なし")
        else:
            for _, sr in top_steps.iterrows():
                print(
                    f"    step={int(sr['step'])} | "
                    f"A={sr[sA]:.6f}, B={sr[sB]:.6f}, "
                    f"B負け幅={sr['B_loss_amount']:.6f}"
                )
# 保存
# result_df.to_csv("paired_test_result.csv", index=False)
# print("\nSaved: paired_test_result.csv")