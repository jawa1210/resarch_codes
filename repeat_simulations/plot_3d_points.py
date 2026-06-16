#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_all_field_3d(csv_path, run_idx, step, show_gp_points=True):
    df = pd.read_csv(csv_path)

    gt = df[
        (df["run_idx"] == run_idx) &
        (df["type"] == "gt_initial")
    ].copy()

    if len(gt) == 0:
        raise ValueError(f"gt_initial がありません: run_idx={run_idx}")

    d_step = df[
        (df["run_idx"] == run_idx) &
        (df["step"] == step)
    ].copy()

    reach = d_step[d_step["type"] == "reachable_map_cell"].copy()
    gp = d_step[d_step["type"] == "gp_point"].copy()
    ugv = d_step[d_step["type"] == "ugv_position"].copy()

    if len(reach) == 0:
        raise ValueError(
            f"reachable_map_cell がありません: run_idx={run_idx}, step={step}"
        )

    grid_size = int(max(gt["cell_y"].max(), gt["cell_x"].max()) + 1)

    Y, X = np.indices((grid_size, grid_size))

    mean_grid = np.full((grid_size, grid_size), np.nan)
    true_grid = np.full((grid_size, grid_size), np.nan)

    for _, r in reach.iterrows():
        y = int(r["cell_y"])
        x = int(r["cell_x"])
        mean_grid[y, x] = float(r["mu"])
        true_grid[y, x] = float(r["true"])

    if len(gp) > 0:
        gp["inside_reachable"] = pd.to_numeric(
            gp["inside_reachable"],
            errors="coerce"
        ).fillna(0).astype(int)

        gp_inside = gp[gp["inside_reachable"] == 1].copy()
        gp_outside = gp[gp["inside_reachable"] == 0].copy()
    else:
        gp_inside = pd.DataFrame()
        gp_outside = pd.DataFrame()

    mu_lookup = {
        (int(r["cell_y"]), int(r["cell_x"])): float(r["mu"])
        for _, r in reach.iterrows()
    }

    def get_mu_height(row):
        key = (int(row["cell_y"]), int(row["cell_x"]))
        return mu_lookup.get(key, 0.0)

    if len(gp_inside) > 0:
        gp_inside["_plot_z"] = gp_inside.apply(get_mu_height, axis=1)

    if len(gp_outside) > 0:
        gp_outside["_plot_z"] = 0.0

    print(f"[INFO] grid_size: {grid_size}")
    print(f"[INFO] reachable_map_cell: {len(reach)}")
    print(f"[INFO] gp total: {len(gp)}")
    print(f"[INFO] gp inside reachable: {len(gp_inside)}")
    print(f"[INFO] gp outside reachable: {len(gp_outside)}")
    print(f"[INFO] ugv positions: {len(ugv)}")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        mean_grid,
        alpha=0.45,
        linewidth=0,
        antialiased=True
    )

    ax.plot_wireframe(
        X,
        Y,
        true_grid,
        linewidth=0.8,
        alpha=0.6
    )

    for ugv_id, g in reach.groupby("ugv_id"):
        ax.scatter(
            g["x"],
            g["y"],
            np.zeros(len(g)),
            marker="s",
            s=35,
            alpha=0.18,
            depthshade=False,
            label=f"UGV{int(ugv_id)} reachable floor"
        )

    ax.scatter(
        reach["x"],
        reach["y"],
        reach["true"],
        marker="^",
        s=25,
        alpha=0.45,
        depthshade=False,
        label="Reachable true points"
    )

    ax.scatter(
        reach["x"],
        reach["y"],
        reach["mu"],
        marker="o",
        s=20,
        alpha=0.45,
        depthshade=False,
        label="Reachable GP mean μ"
    )

    if show_gp_points:
        if len(gp_outside) > 0:
            ax.scatter(
                gp_outside["x"],
                gp_outside["y"],
                gp_outside["_plot_z"],
                c="gray",
                marker="o",
                s=45,
                alpha=0.55,
                depthshade=False,
                label="GP basis outside reachable"
            )

        if len(gp_inside) > 0:
            ax.scatter(
                gp_inside["x"],
                gp_inside["y"],
                gp_inside["_plot_z"],
                c="black",
                marker="o",
                s=80,
                alpha=1.0,
                depthshade=False,
                label="GP basis inside reachable"
            )

    for _, r in ugv.iterrows():
        ax.scatter(
            r["x"],
            r["y"],
            0.0,
            marker="X",
            s=150,
            c="red",
            edgecolors="black",
            depthshade=False
        )
        ax.text(
            r["x"],
            r["y"],
            0.05,
            f"UGV{int(r['ugv_id'])}",
            fontsize=10
        )

    ax.set_title(
        f"All reachable 3D | run={run_idx}, step={step}\n"
        f"GP basis inside/outside = {len(gp_inside)}/{len(gp_outside)}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")

    ax.set_xlim(0, grid_size - 1)
    ax.set_ylim(0, grid_size - 1)
    ax.set_zlim(0, 1.05)

    legend_handles = [
        Line2D([0], [0], linestyle="-", label="GP mean surface"),
        Line2D([0], [0], linestyle="--", label="GT wireframe"),
        Line2D([0], [0], marker="s", linestyle="None", label="Reachable floor"),
        Line2D([0], [0], marker="^", linestyle="None", label="Reachable true points"),
        Line2D([0], [0], marker="o", linestyle="None", label="Reachable GP mean μ"),
        Line2D([0], [0], marker="o", linestyle="None", color="black",
               label="GP basis inside reachable"),
        Line2D([0], [0], marker="o", linestyle="None", color="gray",
               label="GP basis outside reachable"),
        Line2D([0], [0], marker="X", linestyle="None", color="red",
               label="UGV position"),
    ]

    ax.legend(handles=legend_handles)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--run_idx", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--hide_gp_points", action="store_true")

    args = parser.parse_args()

    plot_all_field_3d(
        csv_path=args.csv,
        run_idx=args.run_idx,
        step=args.step,
        show_gp_points=not args.hide_gp_points,
    )