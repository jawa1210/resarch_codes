#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

from multi_repeat_run import (
    load_params,
    deep_get,
    _stable_int_seed,
    generate_ground_truth_map_scalar,
)

def build_gt(params, master_seed, run_idx):

    grid_size = int(deep_get(params, "grid_size", 30))
    num_uavs = int(deep_get(params, "num_uavs", 3))
    num_ugvs = int(deep_get(params, "num_ugvs", 2))

    gt_seed = _stable_int_seed(
        master_seed,
        run_idx,
        grid_size,
        num_uavs,
        num_ugvs,
        "gt",
    )

    gt_params = deep_get(params, "ground_truth", {})

    gt = generate_ground_truth_map_scalar(
        grid_size=grid_size,
        seed=gt_seed,
        num_blobs=int(gt_params.get("num_blobs", 30)),
        amp_range=tuple(gt_params.get("amp_range", [1.0, 3.0])),
        sigma_range=tuple(gt_params.get("sigma_range", [0.8, 1.8])),
        background=float(gt_params.get("background", 0.02)),
        noise_std=float(gt_params.get("noise_std", 0.02)),
        max_value=float(gt_params.get("max_value", 10.0)),
    )

    return gt, gt_seed


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--master_seed", type=int, default=1234)

    args = parser.parse_args()

    params = load_params(args.config)

    num_runs = int(deep_get(params, "num_runs", 20))

    grid_size = int(deep_get(params, "grid_size", 30))

    cols = 5
    rows = int(np.ceil(num_runs / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows)
    )

    axes = np.array(axes).reshape(-1)

    vmax_global = 0.0
    gt_list = []

    # まず全部生成して vmax を統一
    for run_idx in range(num_runs):

        gt, gt_seed = build_gt(
            params=params,
            master_seed=args.master_seed,
            run_idx=run_idx,
        )

        gt_list.append((gt, gt_seed))

        vmax_global = max(vmax_global, float(np.max(gt)))

    # 描画
    for run_idx in range(num_runs):

        ax = axes[run_idx]

        gt, gt_seed = gt_list[run_idx]

        im = ax.imshow(
            gt,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_global,
        )

        ax.set_title(
            f"run={run_idx}\nseed={gt_seed}",
            fontsize=10
        )

        ax.set_xticks([])
        ax.set_yticks([])

        print("=" * 60)
        print(f"Run {run_idx}")
        print("seed:", gt_seed)
        print("GT min :", float(np.min(gt)))
        print("GT max :", float(np.max(gt)))
        print("GT mean:", float(np.mean(gt)))
        print("GT sum :", float(np.sum(gt)))
        print("cells >= 0.7:", int(np.sum(gt >= 0.7)))
        print("cells >= 2.0:", int(np.sum(gt >= 2.0)))
        print("cells >= 5.0:", int(np.sum(gt >= 5.0)))

    # 余ったsubplot削除
    for k in range(num_runs, len(axes)):
        fig.delaxes(axes[k])

    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        fraction=0.02,
        pad=0.02
    )
    cbar.set_label("Yield")

    plt.suptitle(
        f"Ground Truth Samples (0 ~ {num_runs-1})",
        fontsize=16
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()