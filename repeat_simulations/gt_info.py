import numpy as np


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

    # 最大値を max_value に揃える
    m = float(gt.max())
    if m > 1e-12:
        gt = gt / m * max_value

    return gt


def analyze_ground_truth(
    num_runs=10,
    base_seed=1234,
    threshold=0.7,

    # generate_ground_truth_map_scalar の設定
    grid_size=30,
    num_blobs=4,
    amp_range=(1.0, 1.5),
    sigma_range=(2.5, 4.5),
    background=0.05,
    noise_std=0.03,
    max_value=1.0,
):
    total_sums = []
    threshold_counts = []

    print("=" * 70)
    print(f"Ground Truth Analysis")
    print("=" * 70)
    print(f"base_seed   : {base_seed}")
    print(f"num_runs    : {num_runs}")
    print(f"threshold   : {threshold}")
    print()

    for run in range(num_runs):

        seed = base_seed + run

        gt = generate_ground_truth_map_scalar(
            grid_size=grid_size,
            num_blobs=num_blobs,
            amp_range=amp_range,
            sigma_range=sigma_range,
            background=background,
            noise_std=noise_std,
            max_value=max_value,
            seed=seed,
        )

        total_sum = float(np.sum(gt))
        threshold_count = int(np.sum(gt >= threshold))

        total_sums.append(total_sum)
        threshold_counts.append(threshold_count)

        print(
            f"Run {run:02d} | "
            f"seed={seed} | "
            f"sum={total_sum:.3f} | "
            f"cells >= {threshold}: {threshold_count}"
        )

    total_sums = np.array(total_sums)
    threshold_counts = np.array(threshold_counts)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(
        f"True map sum:\n"
        f"  mean = {total_sums.mean():.3f}\n"
        f"  std  = {total_sums.std():.3f}\n"
        f"  min  = {total_sums.min():.3f}\n"
        f"  max  = {total_sums.max():.3f}"
    )

    print()

    print(
        f"Cells >= {threshold}:\n"
        f"  mean = {threshold_counts.mean():.3f}\n"
        f"  std  = {threshold_counts.std():.3f}\n"
        f"  min  = {threshold_counts.min()}\n"
        f"  max  = {threshold_counts.max()}"
    )

    return {
        "total_sums": total_sums,
        "threshold_counts": threshold_counts,
    }


# ============================================================
# 使用例
# ============================================================

analyze_ground_truth(
    num_runs=15,
    base_seed=1234,
    threshold=0.7,

    grid_size=30,
    max_value=1.0,
)