#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.lines import Line2D


def interp_surface(d, z_col, resolution=80, method="linear"):
    xi = np.linspace(d["x"].min(), d["x"].max(), resolution)
    yi = np.linspace(d["y"].min(), d["y"].max(), resolution)
    XI, YI = np.meshgrid(xi, yi)

    points = d[["x", "y"]].to_numpy()
    values = d[z_col].to_numpy()

    ZI = griddata(points, values, (XI, YI), method=method)
    return XI, YI, ZI


def plot_reachable_3d(
    csv_path,
    run_idx,
    step,
    ugv_id=None,
    smooth=True,
    interp_method="linear",
    show_gp_points=True,
):
    df = pd.read_csv(csv_path)

    d = df[
        (df["run_idx"] == run_idx) &
        (df["step"] == step) &
        (df["type"] == "reachable_map_cell")
    ].copy()

    if ugv_id is not None:
        d = d[d["ugv_id"] == ugv_id].copy()

    if len(d) == 0:
        raise ValueError(
            f"該当データがありません: run_idx={run_idx}, step={step}, ugv_id={ugv_id}"
        )

    gp = df[
        (df["run_idx"] == run_idx) &
        (df["step"] == step) &
        (df["type"] == "gp_point")
    ].copy()

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

    # reachable_map_cell の座標 -> mu 高さ
    mu_lookup = {
        (int(r["cell_y"]), int(r["cell_x"])): float(r["mu"])
        for _, r in d.iterrows()
    }

    def get_mu_height(row):
        key = (int(row["cell_y"]), int(row["cell_x"]))
        return mu_lookup.get(key, 0.0)

    if len(gp_inside) > 0:
        gp_inside["_plot_z"] = gp_inside.apply(get_mu_height, axis=1)

    if len(gp_outside) > 0:
        gp_outside["_plot_z"] = 0.0

    print(f"[INFO] reachable_map_cell: {len(d)}")
    print(f"[INFO] gp_point total: {len(gp)}")
    print(f"[INFO] gp inside reachable: {len(gp_inside)}")
    print(f"[INFO] gp outside reachable: {len(gp_outside)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if smooth and len(d) >= 4:
        try:
            XI, YI, MU_smooth = interp_surface(
                d,
                z_col="mu",
                resolution=80,
                method=interp_method,
            )

            _, _, TRUE_smooth = interp_surface(
                d,
                z_col="true",
                resolution=80,
                method=interp_method,
            )

            ax.plot_surface(
                XI,
                YI,
                MU_smooth,
                alpha=0.45,
                linewidth=0,
                antialiased=True
            )

            ax.plot_wireframe(
                XI,
                YI,
                TRUE_smooth,
                linewidth=0.8,
                alpha=0.6
            )

        except Exception as e:
            print(f"[WARN] surface interpolation failed: {e}")
            print("[WARN] 点表示のみで描画します。")

    ax.scatter(
        d["x"], d["y"], d["mu"],
        marker="o",
        s=30,
        alpha=0.65,
        label="GP mean μ points"
    )

    ax.scatter(
        d["x"], d["y"], d["true"],
        marker="^",
        s=30,
        alpha=0.65,
        label="Ground truth points"
    )

    for _, r in d.iterrows():
        ax.plot(
            [r["x"], r["x"]],
            [r["y"], r["y"]],
            [r["mu"], r["true"]],
            linewidth=0.7,
            alpha=0.35
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
                s=85,
                alpha=1.0,
                depthshade=False,
                label="GP basis inside reachable"
            )

    for k, g in d.groupby("ugv_id"):
        r0 = g.iloc[0]
        ax.scatter(
            r0["ugv_x"],
            r0["ugv_y"],
            0.0,
            marker="X",
            s=140,
            c="red",
            depthshade=False,
            label=f"UGV{int(k)} position"
        )

    title = f"Reachable cells 3D | run={run_idx}, step={step}"
    if ugv_id is not None:
        title += f", UGV{ugv_id}"
    title += f"\nGP basis inside/outside = {len(gp_inside)}/{len(gp_outside)}"

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="black",
               label="GP basis inside reachable"),
        Line2D([0], [0], marker="o", linestyle="None", color="gray",
               label="GP basis outside reachable"),
        Line2D([0], [0], marker="o", linestyle="None",
               label="GP mean points"),
        Line2D([0], [0], marker="^", linestyle="None",
               label="Ground truth points"),
        Line2D([0], [0], linestyle="-",
               label="GP mean surface"),
        Line2D([0], [0], linestyle="--",
               label="GT wireframe"),
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
    parser.add_argument("--ugv_id", type=int, default=None)
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--interp_method", type=str, default="linear")
    parser.add_argument("--hide_gp_points", action="store_true")

    args = parser.parse_args()

    plot_reachable_3d(
        csv_path=args.csv,
        run_idx=args.run_idx,
        step=args.step,
        ugv_id=args.ugv_id,
        smooth=not args.no_smooth,
        interp_method=args.interp_method,
        show_gp_points=not args.hide_gp_points,
    )