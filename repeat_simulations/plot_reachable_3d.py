#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.lines import Line2D

"""
例:
python3 plot_reachable_3d.py \
  --csv analysis_artifacts_allruns.csv \
  --run_idx 0 \
  --step 50 \
  --ugv_id 0
"""


def interp_surface(d, z_col, resolution=80, method="linear"):
    xi = np.linspace(d["x"].min(), d["x"].max(), resolution)
    yi = np.linspace(d["y"].min(), d["y"].max(), resolution)
    XI, YI = np.meshgrid(xi, yi)

    points = d[["x", "y"]].to_numpy()
    values = d[z_col].to_numpy()

    ZI = griddata(
        points,
        values,
        (XI, YI),
        method=method
    )

    return XI, YI, ZI


def plot_reachable_3d(
    csv_path,
    run_idx,
    step,
    ugv_id=None,
    smooth=True,
    interp_method="linear",
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

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # =========================
    # Smooth surface / wireframe
    # =========================
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

    # =========================
    # Original points
    # =========================
    ax.scatter(
        d["x"], d["y"], d["mu"],
        marker="o",
        s=35,
        alpha=0.75,
        label="GP mean μ points"
    )

    ax.scatter(
        d["x"], d["y"], d["true"],
        marker="^",
        s=35,
        alpha=0.75,
        label="Ground truth points"
    )

    # 同じセルの mu と true を縦線で結ぶ
    for _, r in d.iterrows():
        ax.plot(
            [r["x"], r["x"]],
            [r["y"], r["y"]],
            [r["mu"], r["true"]],
            linewidth=0.7,
            alpha=0.35
        )

    # UGV位置
    for k, g in d.groupby("ugv_id"):
        r0 = g.iloc[0]
        ax.scatter(
            r0["ugv_x"], r0["ugv_y"], 0.0,
            marker="X",
            s=120,
            label=f"UGV{int(k)} position"
        )

    title = f"Reachable cells 3D | run={run_idx}, step={step}"
    if ugv_id is not None:
        title += f", UGV{ugv_id}"

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("value")

    # 3D surfaceはlegendで落ちることがあるので、ダミーlegendを使う
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", label="GP mean points"),
        Line2D([0], [0], marker="^", linestyle="None", label="Ground truth points"),
        Line2D([0], [0], linestyle="-", label="GP mean surface"),
        Line2D([0], [0], linestyle="--", label="GT wireframe"),
        Line2D([0], [0], marker="X", linestyle="None", label="UGV position"),
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

    args = parser.parse_args()

    plot_reachable_3d(
        csv_path=args.csv,
        run_idx=args.run_idx,
        step=args.step,
        ugv_id=args.ugv_id,
        smooth=not args.no_smooth,
        interp_method=args.interp_method,
    )