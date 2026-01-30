#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_various_plot.py (final)

- results/data CSV（1条件=1ファイル）を1個または複数指定して、
  ・各runの軌跡（薄線）+ 平均（太線）
  ・J, true_crop_sum
  ・UGVごとの mu/var/prob
  ・UGVごとの mu/prob の累積（step方向のcumsum）
  を描画する。

- J には理論線を重ねる：
    J_ideal(t) = J0 - N_uav * gamma * t

  dt は params の control_period から
  gamma は params の cbf_j_gamma から
  N_uav は params の num_uavs から
  可能な限り自動取得する。

優先順位（全条件共通の上書き）
- CLI で --dt / --gamma / --num_uav を指定した場合は、params より優先して使用。

params の推定
- xxx_results_*.csv -> xxx_params_*.csv
- xxx_data_*.csv    -> xxx_params_*.csv

「取れるパラメータはすべて取ってきたい」
- params CSV を読み、1行目の全カラムを dict として保持
- 各Figureの右側に params を注釈表示（長い場合は自動トリム）

Usage examples
  python3 multi_various_plot.py --files A_data.csv --labels A
  python3 multi_various_plot.py --files A_data.csv B_data.csv --labels A B
  python3 multi_various_plot.py --files A_data.csv B_data.csv --labels A B --num_uav 3
  python3 multi_various_plot.py --files A_data.csv --labels A --param_files A_params.csv
  python3 multi_various_plot.py --files A_data.csv B_data.csv --labels A B --gamma 3.0 --dt 0.1 --num_uav 3
"""

import os
import re
import argparse
import textwrap
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib  # noqa: F401


# ── Plot style ────────────────────────────────────────────────
mpl.rcParams["font.family"] = "IPAexGothic"
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16


# ============================================================
# Helpers: filenames / columns
# ============================================================

def _nice_label_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absorb schema differences across results CSVs.
    - run: use run_idx if present
    - J: use estimated_J if present
    - true_crop_sum: accept aliases
    """
    df = df.copy()

    if "run" not in df.columns:
        if "run_idx" in df.columns:
            df["run"] = df["run_idx"]
        else:
            df["run"] = 0

    if "J" not in df.columns and "estimated_J" in df.columns:
        df["J"] = df["estimated_J"]

    if "true_crop_sum" not in df.columns:
        for cand in ["total_crops", "total_crop_sum", "crop_sum"]:
            if cand in df.columns:
                df["true_crop_sum"] = df[cand]
                break

    return df


def _infer_ugv_ids(df: pd.DataFrame) -> list[int]:
    """
    Infer UGV ids from column names like ugv0_mu, ugv1_var, ugv2_prob, ...
    """
    ugv_ids = set()
    pat = re.compile(r"^ugv(\d+)_(mu|var|prob|visited_mu_sum|visited_var_sum|visited_prob_sum|visited_count)$")
    for c in df.columns:
        m = pat.match(c)
        if m:
            ugv_ids.add(int(m.group(1)))
    return sorted(ugv_ids)


def _mean_by_step(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols2 = [c for c in cols if c in df.columns]
    if len(cols2) == 0:
        return pd.DataFrame()
    return df.groupby("step")[cols2].mean().reset_index()


def _add_cumsum_per_run(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df2 = df.sort_values(["run", "step"]).copy()
    df2[out_col] = df2.groupby("run")[col].cumsum()
    return df2


# ============================================================
# Params: infer & load (grab "everything")
# ============================================================

def _infer_param_file(results_path: str) -> str | None:
    """
    Infer params CSV path from results/data CSV path.

    Supported:
      xxx_results_*.csv -> xxx_params_*.csv
      xxx_data_*.csv    -> xxx_params_*.csv
    """
    base = os.path.basename(results_path)
    d = os.path.dirname(results_path) or "."

    candidates = []
    if "results" in base:
        candidates.append(base.replace("results", "params"))
    if "data" in base:
        candidates.append(base.replace("data", "params"))

    if not candidates:
        return None

    for pb in candidates:
        pp = os.path.join(d, pb)
        if os.path.exists(pp):
            return pp
    return None


def _load_params_row(param_path: str) -> Dict[str, Any] | None:
    """
    Load params CSV and return first row as a dict.
    (We "take everything": all columns.)
    """
    try:
        dfp = pd.read_csv(param_path)
        if dfp.empty:
            return None
        row = dfp.iloc[0].to_dict()
        # normalize numpy types to python scalars where possible
        for k, v in list(row.items()):
            if isinstance(v, (np.generic,)):
                row[k] = v.item()
        return row
    except Exception:
        return None


def _find_first_existing_param_key(params: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in params and params[k] is not None and not (isinstance(params[k], float) and np.isnan(params[k])):
            return k
    return None


def _extract_num_uav(params: Dict[str, Any]) -> Optional[int]:
    key = _find_first_existing_param_key(params, [
        "num_uavs", "num_uav", "uav_num", "n_uav", "N_uav", "N_uavs", "uavs", "n_uavs"
    ])
    if key is None:
        return None
    try:
        return int(params[key])
    except Exception:
        return None


def _extract_dt(params: Dict[str, Any]) -> Optional[float]:
    """
    dt = control_period
    """
    key = _find_first_existing_param_key(params, [
        "control_period",
        "cfg.control_period", "cfg_control_period",
        "dt", "cfg.dt", "cfg_dt",
    ])
    if key is None:
        return None
    try:
        return float(params[key])
    except Exception:
        return None


def _extract_gamma(params: Dict[str, Any]) -> Optional[float]:
    """
    gamma = cbf_j_gamma
    """
    key = _find_first_existing_param_key(params, [
        "cbf_j_gamma",
        "cfg.cbf_j_gamma", "cfg_cbf_j_gamma",
        "cfg.j_gamma", "cfg_j_gamma",
    ])
    if key is None:
        return None
    try:
        return float(params[key])
    except Exception:
        return None


def _format_params_for_annotation(params: Dict[str, Any], max_chars: int = 650, width: int = 92) -> str:
    """
    Convert params dict into a readable annotation.
    If too long, truncate.
    """
    items = []
    for k in sorted(params.keys(), key=lambda x: str(x)):
        v = params[k]
        sv = str(v)
        if len(sv) > 140:
            sv = sv[:137] + "..."
        items.append(f"{k}={sv}")

    text = ", ".join(items)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return "\n".join(textwrap.wrap(text, width=width))


def _add_params_box(ax, label: str, params: Optional[Dict[str, Any]]):
    if not params:
        return
    txt = _format_params_for_annotation(params)
    ax.text(
        1.02, 0.98,
        f"[{label} params]\n{txt}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


# ============================================================
# Plot core
# ============================================================

def _resolve_dt_gamma_numuav(
    c: dict,
    dt_cli: Optional[float],
    gamma_cli: Optional[float],
    num_uav_cli: Optional[int],
) -> tuple[float, Optional[float], int]:
    """
    Decide dt, gamma, num_uav for a given condition c with CLI overrides.

    Returns:
      dt_use (float)      : always returns a value (fallback to 0.1 with warning)
      gamma_use (float|None): if None, theory line should be skipped
      num_uav_use (int)   : always returns a value (fallback to 1 with warning)
    """
    # dt
    dt_use = dt_cli if dt_cli is not None else c.get("dt_param", None)
    if dt_use is None:
        dt_use = 0.1
        print(f"[WARN] {c['label']}: control_period 不明 → dt=0.1 を仮定")

    # gamma
    gamma_use = gamma_cli if gamma_cli is not None else c.get("gamma_param", None)

    # num_uav
    num_uav_use = num_uav_cli if num_uav_cli is not None else c.get("num_uav_param", None)
    if num_uav_use is None:
        num_uav_use = 1
        print(f"[WARN] {c['label']}: num_uav 不明 → N=1 を仮定")

    return float(dt_use), (float(gamma_use) if gamma_use is not None else None), int(num_uav_use)


def plot_conditions(
    files: list[str],
    labels: list[str] | None = None,
    param_files: list[str] | None = None,
    dt: float | None = None,          # ← paramsのcontrol_periodを使えるようにNone許可
    gamma: float | None = None,       # ← paramsのcbf_j_gammaを使えるようにNone許可
    num_uav: int | None = None,       # ← paramsのnum_uavsを使えるようにNone許可
    ugv_ids: list[int] | None = None,
    alpha_each: float = 0.25,
    show_params_on_fig: bool = True,
):
    if labels is None:
        labels = [_nice_label_from_path(p) for p in files]
    if len(labels) != len(files):
        raise ValueError("labels の数が files と一致していません")

    if param_files is not None and len(param_files) != len(files):
        raise ValueError("param_files を指定する場合、files と同じ数を指定してください")

    conds: list[dict] = []
    all_ugvs = set()

    for i, (path, lab) in enumerate(zip(files, labels)):
        df = pd.read_csv(path)
        df = _standardize_columns(df)

        df = df.dropna(subset=["step"])
        df["step"] = df["step"].astype(int)

        for uid in _infer_ugv_ids(df):
            all_ugvs.add(uid)

        # params: explicit > inferred
        ppath = None
        if param_files is not None:
            ppath = param_files[i]
        else:
            ppath = _infer_param_file(path)

        params = _load_params_row(ppath) if (ppath is not None and os.path.exists(ppath)) else None

        dt_p = _extract_dt(params) if params else None
        gamma_p = _extract_gamma(params) if params else None
        num_uav_p = _extract_num_uav(params) if params else None

        conds.append({
            "path": path,
            "label": lab,
            "df": df,
            "param_path": ppath,
            "params": params,
            "dt_param": dt_p,
            "gamma_param": gamma_p,
            "num_uav_param": num_uav_p,
        })

    if ugv_ids is None:
        ugv_ids = sorted(all_ugvs)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # =====================================================
    # 1) J(t) + theoretical line
    # =====================================================
    if any("J" in c["df"].columns for c in conds):
        fig, ax = plt.subplots(figsize=(10, 6))

        # plot runs and means
        for k, c in enumerate(conds):
            df = c["df"]
            if "J" not in df.columns:
                continue

            dt_use, _, _ = _resolve_dt_gamma_numuav(c, dt, gamma, num_uav)
            color = color_cycle[k % len(color_cycle)]

            runs = sorted(df["run"].unique())
            for j, r in enumerate(runs):
                sub = df[df["run"] == r].sort_values("step")
                t = sub["step"].to_numpy() * dt_use
                y = sub["J"].to_numpy()
                ax.plot(t, y, color=color, alpha=alpha_each, label=(f"{c['label']} 各試行" if j == 0 else None))

            mean = _mean_by_step(df, ["J"])
            if not mean.empty:
                t = mean["step"].to_numpy() * dt_use
                ax.plot(t, mean["J"].to_numpy(), color=color, linewidth=3.0, label=f"{c['label']} 平均")

        # theoretical line(s): per condition, using dt/gamma/N resolved
        any_theory = False
        for c in conds:
            df = c["df"]
            if "J" not in df.columns:
                continue

            mean = _mean_by_step(df, ["J"])
            if mean.empty:
                continue

            dt_use, gamma_use, N_use = _resolve_dt_gamma_numuav(c, dt, gamma, num_uav)
            if gamma_use is None:
                continue

            t = mean["step"].to_numpy() * dt_use
            J_mean = mean["J"].to_numpy()
            J0 = float(J_mean[0])

            J_ideal = J0 - N_use * gamma_use * t
            any_theory = True

            ax.plot(
                t, J_ideal,
                linestyle="--",
                linewidth=2.5,
                color="k",
                alpha=0.85,
                label=rf"{c['label']} 理論線 $J_0-N_{{\rm UAV}}\gamma t$ (N={N_use}, $\gamma={gamma_use}$)",
            )

        ax.set_xlabel("時間 (s)")
        ax.set_ylabel("目的関数 $J$")
        ax.set_title("Objective $J$ の推移（条件比較）" if len(conds) > 1 else "Objective $J$ の推移")
        ax.grid(True)
        ax.legend()

        if show_params_on_fig:
            for c in conds:
                _add_params_box(ax, c["label"], c.get("params", None))

        plt.tight_layout()
        plt.show()

        if not any_theory:
            print("[INFO] 理論線が描画されませんでした。")
            print("      gamma が CLI(--gamma) でも params(cbf_j_gamma) でも取得できていない可能性があります。")
            print("      paramsに cbf_j_gamma 列があるか確認してください。")

    # =====================================================
    # 2) total crops (true_crop_sum)
    # =====================================================
    if any("true_crop_sum" in c["df"].columns for c in conds):
        fig, ax = plt.subplots(figsize=(10, 6))

        for k, c in enumerate(conds):
            df = c["df"]
            if "true_crop_sum" not in df.columns:
                continue

            dt_use, _, _ = _resolve_dt_gamma_numuav(c, dt, gamma, num_uav)
            color = color_cycle[k % len(color_cycle)]

            runs = sorted(df["run"].unique())
            for j, r in enumerate(runs):
                sub = df[df["run"] == r].sort_values("step")
                t = sub["step"].to_numpy() * dt_use
                y = sub["true_crop_sum"].to_numpy()
                ax.plot(t, y, color=color, alpha=alpha_each, label=(f"{c['label']} 各試行" if j == 0 else None))

            mean = _mean_by_step(df, ["true_crop_sum"])
            if not mean.empty:
                t = mean["step"].to_numpy() * dt_use
                ax.plot(t, mean["true_crop_sum"].to_numpy(), color=color, linewidth=3.0, label=f"{c['label']} 平均")

        ax.set_xlabel("時間 (s)")
        ax.set_ylabel("累積真値の合計（total crops）")
        ax.set_title("UGV 訪問セルの累積真値（条件比較）" if len(conds) > 1 else "UGV 訪問セルの累積真値")
        ax.grid(True)
        ax.legend()

        if show_params_on_fig:
            for c in conds:
                _add_params_box(ax, c["label"], c.get("params", None))

        plt.tight_layout()
        plt.show()

    # =====================================================
    # 3) UGV metrics: mu/var/prob (instantaneous)
    # 4) UGV cumulative: mu/prob cumsum over steps
    # =====================================================
    base_metrics = ["mu", "var", "prob"]
    cum_metrics = ["mu", "prob"]

    for ugv in ugv_ids:
        # instantaneous
        for met in base_metrics:
            col = f"ugv{ugv}_{met}"
            if not any(col in c["df"].columns for c in conds):
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            for k, c in enumerate(conds):
                df = c["df"]
                if col not in df.columns:
                    continue

                dt_use, _, _ = _resolve_dt_gamma_numuav(c, dt, gamma, num_uav)
                color = color_cycle[k % len(color_cycle)]

                runs = sorted(df["run"].unique())
                for j, r in enumerate(runs):
                    sub = df[df["run"] == r].sort_values("step")
                    t = sub["step"].to_numpy() * dt_use
                    y = sub[col].to_numpy()
                    ax.plot(t, y, color=color, alpha=alpha_each, label=(f"{c['label']} 各試行" if j == 0 else None))

                mean = _mean_by_step(df, [col])
                if not mean.empty:
                    t = mean["step"].to_numpy() * dt_use
                    ax.plot(t, mean[col].to_numpy(), color=color, linewidth=3.0, label=f"{c['label']} 平均")

            ax.set_xlabel("時間 (s)")
            ax.set_ylabel(f"UGV{ugv} {met}")
            ax.set_title(f"UGV{ugv}: {met}（条件比較）" if len(conds) > 1 else f"UGV{ugv}: {met}")
            ax.grid(True)
            ax.legend()

            if show_params_on_fig:
                for c in conds:
                    _add_params_box(ax, c["label"], c.get("params", None))

            plt.tight_layout()
            plt.show()

        # cumulative (mu/prob)
        for met in cum_metrics:
            col = f"ugv{ugv}_{met}"
            if not any(col in c["df"].columns for c in conds):
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            for k, c in enumerate(conds):
                df0 = c["df"]
                if col not in df0.columns:
                    continue

                dt_use, _, _ = _resolve_dt_gamma_numuav(c, dt, gamma, num_uav)
                color = color_cycle[k % len(color_cycle)]

                cum_col = f"{col}_cumsum"
                df = _add_cumsum_per_run(df0, col=col, out_col=cum_col)

                runs = sorted(df["run"].unique())
                for j, r in enumerate(runs):
                    sub = df[df["run"] == r].sort_values("step")
                    t = sub["step"].to_numpy() * dt_use
                    y = sub[cum_col].to_numpy()
                    ax.plot(t, y, color=color, alpha=alpha_each, label=(f"{c['label']} 各試行" if j == 0 else None))

                mean = _mean_by_step(df, [cum_col])
                if not mean.empty:
                    t = mean["step"].to_numpy() * dt_use
                    ax.plot(t, mean[cum_col].to_numpy(), color=color, linewidth=3.0, label=f"{c['label']} 平均")

            ax.set_xlabel("時間 (s)")
            ax.set_ylabel(f"UGV{ugv} {met} の累積（cumsum）")
            ax.set_title(f"UGV{ugv}: {met} 累積（条件比較）" if len(conds) > 1 else f"UGV{ugv}: {met} 累積")
            ax.grid(True)
            ax.legend()

            if show_params_on_fig:
                for c in conds:
                    _add_params_box(ax, c["label"], c.get("params", None))

            plt.tight_layout()
            plt.show()


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="results/data CSV を複数指定（条件）")
    ap.add_argument("--labels", nargs="*", default=None, help="各条件のラベル（省略可）")
    ap.add_argument("--param_files", nargs="*", default=None, help="各条件の params CSV を明示指定（省略可）")

    ap.add_argument("--dt", type=float, default=None,
                    help="1 step あたりの秒（省略すると params の control_period を使用）")
    ap.add_argument("--gamma", type=float, default=None,
                    help="1 UAV あたり 1 秒あたりの J 減少率 γ（省略すると params の cbf_j_gamma を使用）")
    ap.add_argument("--num_uav", type=int, default=None,
                    help="UAV 台数（省略すると params の num_uavs を使用）")

    ap.add_argument("--ugvs", nargs="*", type=int, default=None, help="表示するUGV ID（例: --ugvs 0 1）")
    ap.add_argument("--alpha_each", type=float, default=0.25, help="各runの薄線の透明度")
    ap.add_argument("--no_params_box", action="store_true", help="params 注釈を図に表示しない")
    args = ap.parse_args()

    labels = args.labels if (args.labels is not None and len(args.labels) > 0) else None
    param_files = args.param_files if (args.param_files is not None and len(args.param_files) > 0) else None

    if param_files is not None and len(param_files) != len(args.files):
        raise ValueError("--param_files を指定する場合、--files と同じ数にしてください")

    plot_conditions(
        files=args.files,
        labels=labels,
        param_files=param_files,
        dt=args.dt,
        gamma=args.gamma,
        num_uav=args.num_uav,
        ugv_ids=args.ugvs,
        alpha_each=args.alpha_each,
        show_params_on_fig=(not args.no_params_box),
    )


if __name__ == "__main__":
    main()
