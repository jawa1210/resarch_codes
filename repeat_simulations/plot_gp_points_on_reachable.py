#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
python3 plot_gp_points_on_reachable.py \
--csv analysis_artifacts_allruns.csv \
--run_idx 0 \
--step 50
"""


def plot_gp_points_on_reachable(csv_path, run_idx, step):
    df = pd.read_csv(csv_path)

    d = df[(df["run_idx"] == run_idx) & (df["step"] == step)].copy()

    if len(d) == 0:
        raise ValueError(f"データなし: run_idx={run_idx}, step={step}")

    gp = d[d["type"] == "gp_point"].copy()
    reach = d[d["type"] == "reachable_cell"].copy()
    ugv = d[d["type"] == "ugv_position"].copy()

    din = gp[gp["inside_reachable"] == 1].copy()
    dout = gp[gp["inside_reachable"] == 0].copy()

    total = len(gp)
    inside = int(gp["inside_reachable"].sum())
    outside = total - inside

    inside_ratio = inside / total * 100 if total > 0 else 0.0
    outside_ratio = outside / total * 100 if total > 0 else 0.0

    print("================================")
    print(f"run_idx = {run_idx}, step = {step}")
    print(f"GP points total      : {total}")
    print(f"Inside reachable     : {inside} ({inside_ratio:.2f}%)")
    print(f"Outside reachable    : {outside} ({outside_ratio:.2f}%)")
    print("================================")

    max_x = int(d["cell_x"].max())
    max_y = int(d["cell_y"].max())
    grid_size = max(max_x, max_y) + 1

    fig, ax = plt.subplots(figsize=(8, 8))

    # reachable全体
    ax.scatter(
        reach["cell_x"], reach["cell_y"],
        marker="s",
        s=160,
        alpha=0.18,
        color="tab:blue",
        label="Reachable region",
        zorder=1
    )

    ax.scatter(
        dout["x"], dout["y"],
        marker="x",
        s=80,
        alpha=1.0,
        color="green",
        linewidths=2.0,
        label=f"GP points outside reachable: {outside}",
        zorder=5
    )

    ax.scatter(
        din["x"], din["y"],
        marker="o",
        s=70,
        alpha=1.0,
        color="orange",
        edgecolors="black",
        label=f"GP points inside reachable: {inside}",
        zorder=6
    )

    ax.scatter(
        ugv["x"], ugv["y"],
        marker="X",
        s=220,
        color="red",
        edgecolors="black",
        linewidths=1.5,
        label="UGV position",
        zorder=10
    )

    for _, r in ugv.iterrows():
        ax.text(
            r["x"] + 0.2,
            r["y"] + 0.2,
            f"UGV{int(r['ugv_id'])}",
            fontsize=11
        )

    ax.set_title(
        f"GP basis points and reachable region\n"
        f"run={run_idx}, step={step}, inside={inside}/{total} ({inside_ratio:.1f}%)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--run_idx", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    plot_gp_points_on_reachable(
        csv_path=args.csv,
        run_idx=args.run_idx,
        step=args.step,
    )