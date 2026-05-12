#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-run (N=10) simulation runner with:
- External parameter config (YAML/JSON) + CLI overrides
- Per-run different UAV/UGV initial positions
- reproducible across executions (master_seed + scenario_id)
- OUTPUT:
    (1) ONE big data CSV  (all runs, all steps)
    (2) ONE params CSV    (one row per run)

Usage:
  pip install pyyaml qpsolvers quadprog numpy pandas matplotlib

  python3 sim_multi.py --config params.yaml --num_runs 10 --master_seed 1234 --scenario_id base
  python3 sim_multi.py --config params.yaml --num_runs 1 --master_seed 1234 --scenario_id debug --visualize
"""

import os
from datetime import datetime
import math
import numpy as np
from typing import Callable, Tuple, List, Optional, Literal
from dataclasses import dataclass
from collections import deque

import argparse
import json
import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
import time
from tqdm import tqdm


from qpsolvers import solve_qp

try:
    import yaml
except ImportError:
    yaml = None
    
from matplotlib.colors import ListedColormap, BoundaryNorm

GT_CMAP = ListedColormap(["#1f77b4", "#d62728"])  # 0=blue, 1=red
GT_NORM = BoundaryNorm([-0.5, 0.5, 1.5], GT_CMAP.N)

def normalize_01(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a_min = float(np.nanmin(arr))
    a_max = float(np.nanmax(arr))
    if a_max - a_min < eps:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min)

def is_binary_map(arr: np.ndarray, tol: float = 1e-8) -> bool:
    vals = np.unique(np.asarray(arr))
    if vals.size == 0:
        return False
    return np.all(np.isclose(vals, 0.0, atol=tol) | np.isclose(vals, 1.0, atol=tol))


def compute_display_limits(arr: np.ndarray, binary: bool, q_low: float = 1.0, q_high: float = 99.0):
    a = np.asarray(arr, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0

    if binary:
        return -0.5, 1.5

    vmin = float(np.percentile(finite, q_low))
    vmax = float(np.percentile(finite, q_high))

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return vmin, vmax


def compute_std_map(var_map: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(var_map, dtype=float), 0.0))


def get_gt_plot_style(gt: np.ndarray):
    if is_binary_map(gt):
        return {
            "cmap": GT_CMAP,
            "vmin": -0.5,
            "vmax": 1.5,
            "title": "Ground truth (binary)",
            "contour_levels": [0.5],
            "colorbar_ticks": [0, 1],
        }
    else:
        vmin, vmax = compute_display_limits(gt, binary=False)
        return {
            "cmap": "viridis",
            "vmin": vmin,
            "vmax": vmax,
            "title": "Ground truth",
            "contour_levels": None,
            "colorbar_ticks": None,
        }



# ============================================================
# 0) Param IO + reproducible seeds / initial positions
# ============================================================

def load_params(path: Optional[str]) -> dict:
    if path is None:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yml", ".yaml"]:
            if yaml is None:
                raise RuntimeError("PyYAML が入っていません。`pip install pyyaml` してください。")
            return yaml.safe_load(f) or {}
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {ext}")


def deep_set(d: dict, key: str, value):
    """Support dotted keys like 'cfg.use_cbf'."""
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_get(d: dict, key: str, default=None):
    parts = key.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _stable_int_seed(*items) -> int:
    s = "|".join(map(str, items)).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)


def make_rng(master_seed: int, scenario_id: str, run_idx: int,
             grid_size: int, num_uavs: int, num_ugvs: int) -> np.random.Generator:
    seed = _stable_int_seed(master_seed, run_idx, grid_size, num_uavs, num_ugvs)
    return np.random.default_rng(seed)


def sample_initial_positions(
    rng: np.random.Generator,
    grid_size: int,
    num_uavs: int,
    num_ugvs: int,
    ugv_radius_ratio: float = 1/6,
    uav_radius_ratio: float = 1/4,
    angle_jitter: float = 0.6,   # rad
    radius_jitter: float = 0.15  # ratio
):
    center = np.array([grid_size / 2, grid_size / 2], dtype=float)

    # --- UGV ---
    Rg_base = grid_size * ugv_radius_ratio
    thetas_g = np.linspace(0, 2*np.pi, num_ugvs, endpoint=False)
    thetas_g = thetas_g + rng.uniform(-angle_jitter, angle_jitter, size=num_ugvs)
    Rg = Rg_base * (1.0 + rng.uniform(-radius_jitter, radius_jitter, size=num_ugvs))

    ugv_init = []
    for k in range(num_ugvs):
        p0 = center + Rg[k] * np.array([np.cos(thetas_g[k]), np.sin(thetas_g[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size - 1)
        ugv_init.append(p0)

    # --- UAV ---
    Ru_base = grid_size * uav_radius_ratio
    thetas = np.linspace(0, 2*np.pi, num_uavs, endpoint=False)
    thetas = thetas + rng.uniform(-angle_jitter, angle_jitter, size=num_uavs)
    Ru = Ru_base * (1.0 + rng.uniform(-radius_jitter, radius_jitter, size=num_uavs))

    uav_init = []
    for k in range(num_uavs):
        p0 = center + Ru[k] * np.array([np.cos(thetas[k]), np.sin(thetas[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size - 1)
        uav_init.append(p0)

    return np.array(uav_init, dtype=float), np.array(ugv_init, dtype=float)


# ============================================================
# 0) QP Solver (CBF)
# ============================================================

class solver:
    def __init__(self):
        self.cbf_list: list[np.ndarray] = []     # each: [bJ, gx, gy]
        self.slack_list: list[float] = []        # each: slack coef
        self.P_co: float = 1.0               # weight for u part
        self.P_slack: float = 1.0       # weight for slack vars

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        self.cbf_list.append(np.array([bJ, dbJ_du_x, dbJ_du_y], dtype=float))
        self.slack_list.append(float(slack))

    def add_cbfs(self, cbfs: List[Tuple[float, float, float, float]]):
        for (bJ, gx, gy, slack) in cbfs:
            self.add_cbf(bJ, gx, gy, slack)

    def solve(self, nominal: Optional[np.ndarray] = None) -> np.ndarray:
        nominal_input = np.zeros(2, dtype=float) if nominal is None else np.asarray(nominal, dtype=float).reshape(2,)

        m = len(self.cbf_list)
        if m == 0:
            return nominal_input.copy()

        dim = 2 + m
        G = np.zeros((m, dim), dtype=float)
        h = np.zeros((m,), dtype=float)

        # NOTE:
        for i, ((bJ, gx, gy), slack_coef) in enumerate(zip(self.cbf_list, self.slack_list)):
            G[i, 0] = -gx
            G[i, 1] = -gy
            G[i, 2 + i] = slack_coef   # スラック変数 s_i の係数
            h[i,] = bJ

        P = np.zeros((dim, dim), dtype=float)
        P[0, 0] = 2.0 * self.P_co
        P[1, 1] = 2.0 * self.P_co
        for i in range(m):
            P[2 + i, 2 + i] = 2.0 * self.P_slack

        q = np.zeros((dim,), dtype=float)
        q[0:2] = -2.0 * self.P_co * nominal_input

        sol = solve_qp(P, q, G, h, solver="quadprog")
        #self.slack = sol[2]
        if sol is None:
            print("[QP] infeasible -> stop")
            return np.zeros(2, dtype=float)
        return np.asarray(sol[:2], dtype=float)


class VelocityLimitation:
    def __init__(self, max_velocity: float = 5.0, num_poly: int = 8, slack: float = 0.0):
        self.max_velocity = float(max_velocity)*math.cos(math.pi/num_poly)
        self.num_poly = int(num_poly)
        self.slack = float(slack)

    def calc_cbf(self):
        """
        constraints: n_k^T u <= v_max  (k=1..K)
        solver expects tuples: (bJ, gx, gy, slack)
        where constraint is: -gx*u_x -gy*u_y <= bJ
        so choose gx=-n_x, gy=-n_y, bJ=v_max
        """
        theta0=0.0
        cbfs = []
        for k in range(self.num_poly):
            th = theta0 + 2.0 * np.pi * k / self.num_poly
            n_x = float(np.cos(th))
            n_y = float(np.sin(th))
            cbfs.append((self.max_velocity, -n_x, -n_y, 0.0))  # slackなし
        return cbfs
    
class FieldLimitation:
    def __init__(self, grid_size: int, position: np.ndarray, slack: float = 0.0):
        self.center = np.array([grid_size/2, grid_size/2], dtype=float)
        self.radius = float(grid_size/2)
        self.position = position
        self.slack = float(slack)
    
    def calc_cbf(self):
        L4_norm=np.sum(((self.center - self.position)/self.radius)**4)
        cbf=1-L4_norm

        grad=4*((self.center - self.position)**3)/(self.radius**4)

        return [(cbf, -grad[0], -grad[1], self.slack)]


class HarvestLogitCalibrator:
    """
    収穫量 g を logit a = w*g + b に変換するオンライン学習器。
    p = sigmoid(a) として「価値ある収穫セル確率」を作る。
    """

    def __init__(
        self,
        threshold: float = 2.0,
        init_w: float = 1.0,
        lr: float = 0.03,
        l2: float = 1e-4,
        min_samples: int = 20,
    ):
        self.threshold = float(threshold)
        self.w = float(init_w)
        self.b = -float(init_w) * float(threshold)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.min_samples = int(min_samples)

        self.g_list = []
        self.y_list = []

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))
    
    @property
    def learned_threshold(self):
        if abs(self.w) < 1e-12:
            return np.nan
        return float(-self.b / self.w)

    def add_sample(self, harvested_amount: float):
        g = float(harvested_amount)

        # 「価値がある収穫量」を教師ラベル化
        y = 1.0 if g >= self.threshold else 0.0

        self.g_list.append(g)
        self.y_list.append(y)

    def fit_step(self, n_iter: int = 20):
        if len(self.g_list) < self.min_samples:
            return

        g = np.asarray(self.g_list, dtype=float)
        y = np.asarray(self.y_list, dtype=float)

        # 0/1の両方がないと分類境界を学べない
        if np.unique(y).size < 2:
            return

        for _ in range(n_iter):
            z = self.w * g + self.b
            p = self.sigmoid(z)
            err = p - y

            grad_w = np.mean(err * g) + self.l2 * self.w
            grad_b = np.mean(err)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def prob_from_mu_var(self, mu_map: np.ndarray, var_map: np.ndarray) -> np.ndarray:
        """
        g ~ N(mu, var)
        a = w*g + b
        a ~ N(w*mu+b, w^2*var)

        MacKay近似:
        p ≈ sigmoid( a_mu / sqrt(1 + pi/8 * a_var) )
        """
        var_map = np.maximum(var_map, 0.0)

        a_mu = self.w * mu_map + self.b
        a_var = (self.w ** 2) * var_map

        z_raw = a_mu / np.sqrt(1.0 + (np.pi / 8.0) * a_var)
        return self.sigmoid(z_raw)


# ============================================================
# 1) Sparse Online GP (SOGP) + kernel
# ============================================================

def rbf_kernel(x, y, rbf_sigma=2.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * rbf_sigma ** 2))


class SparseOnlineGP:
    def __init__(self, sigma0: float, kernel=rbf_kernel, max_basis: int = None, delta: float = 0.1):
        self.sigma0 = sigma0
        self.kernel = kernel
        self.max_basis = max_basis
        self.delta = delta
        self.count_case1 = 0
        self.count_case2 = 0
        self.count_case3 = 0

        self.X = np.zeros((0, 2))
        self.a = np.zeros((0,))
        self.C = np.zeros((0, 0))
        self.Q = np.zeros((0, 0))

    def init_first(self, x, y):
        k00 = self.kernel(x, x)
        denom = k00 + self.sigma0 ** 2
        self.X = x.reshape(1, 2)
        self.a = np.array([y / denom])
        self.C = np.array([[-1.0 / denom]])
        self.Q = np.array([[1.0 / k00]])

    def update(self, x: np.ndarray, y: float):
        if self.X.shape[0] == 0:
            self.init_first(x, y)
            return

        k_vec = np.array([self.kernel(xi, x) for xi in self.X])  # (N,)
        k_tt = self.kernel(x, x)
        f_star = float(self.a.dot(k_vec))
        var_star = float(k_tt + k_vec.dot(self.C.dot(k_vec)))

        denom = var_star + self.sigma0 ** 2
        q_t = (y - f_star) / denom
        r_t = -1.0 / denom
        self.r_t=r_t

        h_t = k_tt - k_vec.dot(self.Q.dot(k_vec))

        n = self.X.shape[0]

        if h_t < self.delta:
            ehat = self.Q.dot(k_vec)
            s_short = self.C.dot(k_vec) + ehat
            self.a += q_t * s_short
            self.C += r_t * np.outer(s_short, s_short)
            self.count_case2 += 1

        else:
            if self.max_basis is None or n < self.max_basis:
                s_t = np.concatenate([self.C.dot(k_vec), [1.0]])
                a_ext = np.concatenate([self.a, [0.0]])
                C_ext = np.pad(self.C, ((0, 1), (0, 1)), 'constant')
                self.a = a_ext + q_t * s_t
                self.C = C_ext + r_t * np.outer(s_t, s_t)

                ehat = self.Q.dot(k_vec)
                ehat_full = np.concatenate([ehat, [0.0]])
                efull = np.zeros(n + 1)
                efull[-1] = 1.0
                Q_ext = np.pad(self.Q, ((0, 1), (0, 1)), 'constant')
                self.Q = Q_ext + (1.0 / h_t) * np.outer(ehat_full - efull, ehat_full - efull)

                self.X = np.vstack([self.X, x])
                self.count_case1 += 1

            else:
                s_t = np.concatenate([self.C.dot(k_vec), [1.0]])
                a_ext = np.concatenate([self.a, [0.0]])
                C_ext = np.pad(self.C, ((0, 1), (0, 1)), 'constant')
                self.a = a_ext + q_t * s_t
                self.C = C_ext + r_t * np.outer(s_t, s_t)

                ehat = self.Q.dot(k_vec)
                ehat_full = np.concatenate([ehat, [0.0]])
                efull = np.zeros(n + 1)
                efull[-1] = 1.0
                Q_ext = np.pad(self.Q, ((0, 1), (0, 1)), 'constant')
                self.Q = Q_ext + (1.0 / h_t) * np.outer(ehat_full - efull, ehat_full - efull)

                self.X = np.vstack([self.X, x])
                self._prune_basis()
                self.count_case3 += 1

    def _prune_basis(self):
        phi = np.abs(self.a) / np.diag(self.Q)
        j = int(np.argmin(phi))
        idx = [i for i in range(len(self.a)) if i != j]

        a_new = self.a.copy()
        C_new = self.C.copy()
        Q_new = self.Q.copy()

        a_j = a_new[j]
        Q_jj = Q_new[j, j]
        C_jj = C_new[j, j]
        Q_jcol = Q_new[idx, j]
        C_jcol = C_new[idx, j]

        a_old = a_new[idx]
        C_old = C_new[np.ix_(idx, idx)]
        Q_old = Q_new[np.ix_(idx, idx)]

        a_hat = a_old - (a_j / Q_jj) * Q_jcol
        term1 = C_jj * np.outer(Q_jcol, Q_jcol) / (Q_jj ** 2)
        term2 = (np.outer(Q_jcol, C_jcol) + np.outer(C_jcol, Q_jcol)) / Q_jj
        C_hat = C_old + term1 - term2
        Q_hat = Q_old - np.outer(Q_jcol, Q_jcol) / Q_jj

        self.a = a_hat
        self.C = C_hat
        self.Q = Q_hat
        self.X = self.X[idx]

    def predict(self, x: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        return: mu, var, logistic_prob, k_vec, C
        """
        if self.X.shape[0] == 0:
            mu = 0.0
            var = float(self.kernel(x, x))
            z = 0.5
            return mu, var, z, np.zeros(0), np.zeros((0, 0))

        k_vec = np.array([self.kernel(xi, x) for xi in self.X])
        mu = float(self.a.dot(k_vec))
        var = float(self.kernel(x, x) + k_vec.dot(self.C.dot(k_vec)))

        z_raw = mu / np.sqrt(1.0 + (np.pi / 8.0) * var)
        z = float(1.0 / (1.0 + np.exp(-z_raw)))
        self.var=var
        return mu, var, z, k_vec, self.C
    
    def _calc_now_C(self, position: np.ndarray):
        """
        Virtually update C by appending 'position' as a new basis vector.
        Returns Ctilde of size (N+1, N+1).
        """
        N = self.X.shape[0]

        # --- If no basis yet, Ztilde = {position} (size 1) ---
        if N == 0:
            k_tt = float(self.kernel(position, position))
            sigma2 = k_tt  # k_xx
            denom = sigma2 + self.sigma0**2
            # C = -(K + sigma^2 I)^-1 for 1x1
            return np.array([[-1.0 / denom]], dtype=float)

        # --- standard case ---
        k_vec = np.array([self.kernel(xi, position) for xi in self.X], dtype=float)  # (N,)
        k_tt = float(self.kernel(position, position))

        # predictive variance sigma^2(x*)
        sigma2 = float(k_tt + k_vec @ (self.C @ k_vec))

        denom = sigma2 + self.sigma0**2
        r_t = -1.0 / denom
        self.r_t=r_t

        # s_t = [C k_vec ; 1]   (W(Ck)+e)
        s_t = np.concatenate([self.C @ k_vec, [1.0]])  # (N+1,)

        # U(C) : pad last row/col with zeros
        C_ext = np.pad(self.C, ((0, 1), (0, 1)), mode="constant")  # (N+1,N+1)

        # Ctilde = U(C) + r s s^T
        C_next = C_ext + r_t * np.outer(s_t, s_t)
        return C_next
    
    def _calc_now_C_multi(self, points: np.ndarray) -> np.ndarray:
        """
        Virtually update C by appending multiple points as new basis vectors.
        points: (M,2)
        Returns Ctilde of size (N+M, N+M).

        NOTE:
        - これは「新規点を順に append した」ときの C の仮想更新（rank-1更新の繰り返し）。
        - あなたの _calc_now_C(p) を多点に拡張したもの。
        """
        P = np.asarray(points, dtype=float)
        if P.ndim == 1:
            P = P.reshape(1, 2)
        M = int(P.shape[0])
        if M == 0:
            # 追加点が無いなら現状の C を返す（サイズ N×N）
            return np.asarray(self.C, dtype=float).copy()

        # 現在の basis
        Z = np.asarray(self.X, dtype=float)  # (N,2)
        N = int(Z.shape[0])

        # ベースが空のとき
        if N == 0:
            # 1点目から順に「空→append」を繰り返す
            Ccur = np.zeros((0, 0), dtype=float)
            Zcur = np.zeros((0, 2), dtype=float)
        else:
            Ccur = np.asarray(self.C, dtype=float).copy()
            Zcur = Z.copy()

        for m in range(M):
            x_new = P[m]

            Nc = int(Zcur.shape[0])
            if Nc == 0:
                k_tt = float(self.kernel(x_new, x_new))
                denom = k_tt + self.sigma0**2
                Ccur = np.array([[-1.0 / denom]], dtype=float)
                Zcur = x_new.reshape(1, 2)
                continue

            # k_vec between existing (Zcur) and x_new
            k_vec = np.array([float(self.kernel(Zcur[i], x_new)) for i in range(Nc)], dtype=float)  # (Nc,)
            k_tt = float(self.kernel(x_new, x_new))

            # predictive variance sigma^2(x_new) using current Ccur
            sigma2 = float(k_tt + k_vec @ (Ccur @ k_vec))

            denom = sigma2 + self.sigma0**2
            r_t = -1.0 / denom

            s_t = np.concatenate([Ccur @ k_vec, [1.0]])  # (Nc+1,)
            C_ext = np.pad(Ccur, ((0, 1), (0, 1)), mode="constant")  # (Nc+1,Nc+1)
            Ccur = C_ext + r_t * np.outer(s_t, s_t)

            Zcur = np.vstack([Zcur, x_new.reshape(1, 2)])

        return Ccur



# ============================================================
# 2) Env / Map utils
# ============================================================

def environment_function(pos: np.ndarray, true_map: np.ndarray,
                         rng: np.random.Generator,
                         noise_std: float = 0.5) -> List[Tuple[np.ndarray, float]]:
    """
    3×3の観測（中心あり＝9点）。
    """
    i0, j0 = int(round(pos[0])), int(round(pos[1]))
    H, W = true_map.shape
    observations: List[Tuple[np.ndarray, float]] = []

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0 + di, j0 + dj
            if not (0 <= i < H and 0 <= j < W):
                continue

            val = float(true_map[i, j])
            noisy = float(val + rng.normal(loc=0.0, scale=noise_std))
            observations.append((np.array([i, j], dtype=float), noisy))

    return observations


def generate_ground_truth_map(grid_size=20):
    gt = np.zeros((grid_size, grid_size))
    H = W = grid_size

    def rect(y0, y1, x0, x1, val=1.0):
        i0 = int(max(0, min(H, round(y0 * H))))
        i1 = int(max(0, min(H, round(y1 * H))))
        j0 = int(max(0, min(W, round(x0 * W))))
        j1 = int(max(0, min(W, round(x1 * W))))
        if i1 > i0 and j1 > j0:
            gt[i0:i1, j0:j1] = val

    rect(0.30, 0.70, 0.06, 0.26, 1.0)
    rect(0.20, 0.50, 0.60, 0.92, 1.0)
    rect(0.70, 0.85, 0.60, 0.80, 1.0)
    rect(0.10, 0.30, 0.10, 0.30, 1.0)

    return gt

def generate_ground_truth_map_scalar(
    grid_size=20,
    num_blobs=4,
    amp_range=(1.0, 1.5),
    sigma_range=(2.5, 4.5),
    background=0.05,
    noise_std=0.03,
    max_value=10.0,
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


def mask_to_rgba(mask: np.ndarray, color: str, alpha: float = 0.18) -> np.ndarray:
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    r, g, b, _ = to_rgba(color, alpha)
    rgba[mask] = [r, g, b, alpha]
    return rgba


def compute_voronoi_masks(positions: list[np.ndarray], H: int, W: int) -> list[np.ndarray]:
    K = len(positions)
    masks = [np.zeros((H, W), dtype=bool) for _ in range(K)]
    if K == 0:
        return masks

    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid = np.stack([grid_y, grid_x], axis=-1).astype(float)

    dists = []
    for p in positions:
        p2 = np.asarray(p, dtype=float).reshape(1, 1, 2)
        d = np.linalg.norm(grid - p2, axis=-1)
        dists.append(d)
    dists = np.stack(dists, axis=-1)

    owner = np.argmin(dists, axis=-1)
    for k in range(K):
        masks[k] = (owner == k)
    return masks


# ============================================================
# 3) UGV Controller + Fleet
# ============================================================

class UGVController:
    def __init__(
        self,
        grid_size: int = 20,
        reward_type: int = 0,
        discount_factor: float = 0.95,
        revisit_penalty: float = 0.3,
    ):
        self.grid_size = grid_size
        self.reward_type = reward_type
        self.position = np.array([grid_size // 2, grid_size // 2], dtype=int)
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited[self.position[0], self.position[1]] = True
        self.discount_factor = discount_factor
        self.revisit_penalty = float(revisit_penalty)

    def _calculate_reward(
        self,
        pos: np.ndarray,
        current_pos: np.ndarray,
        remaining_map: np.ndarray,
        variance_map: np.ndarray,
        ambiguity_map: Optional[np.ndarray],
        k1, k2, k3, k4, k5, k6, epsilon,
        reward_type: int,
        step: int,
    ) -> float:
        d = float(np.linalg.norm(pos - current_pos))
        E = float(remaining_map[pos[0], pos[1]])
        V = float(variance_map[pos[0], pos[1]])
        U = 0.0 if ambiguity_map is None else float(ambiguity_map[pos[0], pos[1]])

        if reward_type == 0:
            return (k1 * E - V) / (d ** 2 + epsilon)
        elif reward_type == 1:
            return float(np.exp(-(d ** 2) / k4) * np.exp((k2 * E - V) / k3))
        elif reward_type == 2:
            return (np.tanh(k5 * E) - k6 * V) / (d ** 2 + epsilon)
        elif reward_type == 3:
            delta = 0.1
            beta = 2 * np.log((np.pi ** 2) * (step ** 2) / (6 * delta))
            return E - float(np.sqrt(beta * V))
        elif reward_type == 4:
            return (k1 * E - k2 * U) / (d ** 2 + epsilon)
        elif reward_type == 5:
            return (k1 * E + k2 * U) / (d ** 2 + epsilon)
        elif reward_type == 6:
            return E
        elif reward_type == 7:
            delta = 0.1
            beta = 2 * np.log((np.pi ** 2) * (step ** 2) / (6 * delta))
            eps2 = 1e-8
            sigma_tilde = np.sqrt(V) / (np.median(np.sqrt(variance_map)) + eps2)
            return E - np.sqrt(beta) * sigma_tilde
        else:
            return 0.0

    def _recursive_search(
        self,
        pos: np.ndarray,
        remaining_map: np.ndarray,
        variance_map: np.ndarray,
        ambiguity_map: Optional[np.ndarray],
        depth: int,
        visited: np.ndarray,
        step: int,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        k1, k2, k3, k4, k5, k6, epsilon = 2, 2, 0.1, 0.1, 1, 1, 1e-3

        best_reward = -np.inf
        best_move = pos.copy()

        def is_allowed(cell):
            if allowed_mask is None:
                return True
            i, j = int(cell[0]), int(cell[1])
            return bool(allowed_mask[i, j])

        for a in actions:
            nxt = pos + a
            if not (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size):
                continue
            if not is_allowed(nxt):
                continue

            yi, xj = int(nxt[0]), int(nxt[1])

            # その枝での「まだ残っている価値」
            harvest_reward = self._calculate_reward(
                nxt, pos, remaining_map, variance_map, ambiguity_map,
                k1, k2, k3, k4, k5, k6, epsilon,
                self.reward_type, step
            )

            # 再訪は軽いコストだけ
            revisit_cost = self.revisit_penalty if visited[yi, xj] else 0.0

            # この枝では nxt を収穫済みにする
            new_remaining = remaining_map.copy()
            new_remaining[yi, xj] = 0.0

            new_visited = visited.copy()
            new_visited[yi, xj] = True

            total = harvest_reward - revisit_cost

            if depth > 1:
                _, fut = self._recursive_search(
                    nxt,
                    new_remaining,
                    variance_map,
                    ambiguity_map,
                    depth - 1,
                    new_visited,
                    step + 1,
                    allowed_mask=allowed_mask
                )
                total += self.discount_factor * fut

            if total > best_reward:
                best_reward = total
                best_move = nxt.copy()

        return best_move, float(best_reward)

    def calc(
        self,
        expectation_map: np.ndarray,
        variance_map: np.ndarray,
        ambiguity_map: Optional[np.ndarray] = None,
        depth: int = 10,
        step: int = 1,
        allowed_mask: Optional[np.ndarray] = None
    ):
        remaining_map = expectation_map.copy()
        new_pos, _ = self._recursive_search(
            self.position,
            remaining_map,
            variance_map,
            ambiguity_map,
            depth,
            self.visited.copy(),
            step,
            allowed_mask=allowed_mask
        )
        if not np.array_equal(new_pos, self.position):
            self.position = new_pos
            self.visited[new_pos[0], new_pos[1]] = True

    def get_planned_path(
        self,
        expectation_map: np.ndarray,
        variance_map: np.ndarray,
        ambiguity_map: Optional[np.ndarray] = None,
        depth: int = 10,
        allowed_mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        path = []
        pos = self.position.copy()
        visited = self.visited.copy()
        remaining_map = expectation_map.copy()

        for t in range(depth):
            nxt, _ = self._recursive_search(
                pos,
                remaining_map,
                variance_map,
                ambiguity_map,
                depth - t,
                visited,
                t + 1,
                allowed_mask=allowed_mask
            )
            path.append(nxt.copy())

            yi, xj = int(nxt[0]), int(nxt[1])
            remaining_map[yi, xj] = 0.0
            visited[yi, xj] = True
            pos = nxt

        return path

    def reachable_unvisited_mask(
        self,
        depth: int,
        allowed_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        H = W = self.grid_size
        sy, sx = int(self.position[0]), int(self.position[1])

        def ok(i, j):
            if not (0 <= i < H and 0 <= j < W):
                return False
            if allowed_mask is not None and (not bool(allowed_mask[i, j])):
                return False
            return True

        dist = -np.ones((H, W), dtype=int)
        q = deque()

        if ok(sy, sx):
            dist[sy, sx] = 0
            q.append((sy, sx))

        while q:
            y, x = q.popleft()
            d = dist[y, x]
            if d >= depth:
                continue
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if not ok(ny, nx):
                    continue
                if dist[ny, nx] != -1:
                    continue
                dist[ny, nx] = d + 1
                q.append((ny, nx))

        reach = (dist != -1)
        return reach & (~self.visited)


class UGVFleet:
    def __init__(self, ugvs: list[UGVController]):
        self.ugvs = ugvs
        self.planned_paths: list[list[np.ndarray]] = []
        self.voronoi_masks: list[np.ndarray] = []

    def positions(self) -> list[np.ndarray]:
        return [u.position.copy() for u in self.ugvs]

    def compute_voronoi(self, H: int, W: int):
        self.voronoi_masks = compute_voronoi_masks(self.positions(), H, W)

    def plan_all(self, mean_map: np.ndarray, var_map: np.ndarray, depth: int,
                 ambiguity_map: Optional[np.ndarray] = None):
        H, W = mean_map.shape
        if (not self.voronoi_masks) or (len(self.voronoi_masks) != len(self.ugvs)):
            self.compute_voronoi(H, W)

        self.planned_paths = []
        for u, mask in zip(self.ugvs, self.voronoi_masks):
            path = u.get_planned_path(mean_map, var_map, ambiguity_map=ambiguity_map, depth=depth, allowed_mask=mask)
            self.planned_paths.append(path)

    def step_all(self, mean_map: np.ndarray, var_map: np.ndarray, depth: int, step: int,
                 ambiguity_map: Optional[np.ndarray] = None):
        for u, mask in zip(self.ugvs, self.voronoi_masks):
            u.calc(mean_map, var_map, ambiguity_map=ambiguity_map, depth=depth, step=step, allowed_mask=mask)

    def target_cell_for_uav(self, uav_pos: np.ndarray, step_offset: int) -> tuple[int, int]:
        best_cell = None
        best_dist = np.inf

        for k, ugv in enumerate(self.ugvs):
            path = self.planned_paths[k] if k < len(self.planned_paths) else []
            if path:
                idx = min(max(step_offset - 1, 0), len(path) - 1)
                cy, cx = path[idx]
            else:
                cy, cx = ugv.position
            cell = np.array([cy, cx], dtype=float)
            d = float(np.linalg.norm(uav_pos - cell))
            if d < best_dist:
                best_dist = d
                best_cell = cell

        if best_cell is None:
            for ugv in self.ugvs:
                cell = ugv.position.astype(float)
                d = float(np.linalg.norm(uav_pos - cell))
                if d < best_dist:
                    best_dist = d
                    best_cell = cell

        return int(best_cell[0]), int(best_cell[1])

    def build_path_weight_map(self, H: int, W: int, ell: float, step_offset: int) -> np.ndarray:
        I, J = np.indices((H, W))
        weight = np.zeros((H, W), dtype=float)

        for k, ugv in enumerate(self.ugvs):
            path = self.planned_paths[k] if k < len(self.planned_paths) else []
            if path:
                idx = min(max(step_offset - 1, 0), len(path) - 1)
                cy, cx = path[idx]
            else:
                cy, cx = ugv.position

            dist2 = (I - float(cy)) ** 2 + (J - float(cx)) ** 2
            weight += np.exp(-dist2 / (2.0 * ell ** 2))

        max_w = float(weight.max())
        if max_w > 1e-8:
            weight /= max_w
        return weight

    def build_direction_weight_map(
        self,
        H: int,
        W: int,
        num_steps: int = 10,
        ell_u: float = 8.0,
        ell_perp: float = 4.0,
        eta_forward: float = 2.0,
        forward_shift: float = 0.0,
        eps: float = 1e-9,
        mode: str = "max",
        normalize: bool = True,
    ) -> np.ndarray:
        I, J = np.indices((H, W))
        W_all = np.zeros((H, W), dtype=float)

        if (not self.planned_paths) or (len(self.planned_paths) != len(self.ugvs)):
            for ugv in self.ugvs:
                p0 = ugv.position.astype(float)
                r0 = I - p0[0]
                r1 = J - p0[1]
                dist2 = r0**2 + r1**2
                Wk = np.exp(-dist2 / (2.0 * ell_u**2))
                W_all = np.maximum(W_all, Wk) if mode == "max" else (W_all + Wk)
            if normalize and float(W_all.max()) > 1e-9:
                W_all /= float(W_all.max())
            return W_all

        for k, ugv in enumerate(self.ugvs):
            path = self.planned_paths[k] if k < len(self.planned_paths) else []
            if not path:
                p0 = ugv.position.astype(float)
                pN = p0.copy()
            else:
                K = min(num_steps, len(path))
                p0 = np.array(path[0], dtype=float)
                pN = np.array(path[K - 1], dtype=float)

            dvec = pN - p0
            dn = float(np.linalg.norm(dvec))

            r0 = (I - p0[0])
            r1 = (J - p0[1])

            if dn < eps:
                dist2 = r0**2 + r1**2
                Wk = np.exp(-dist2 / (2.0 * ell_u**2))
            else:
                d_hat = dvec / (dn + eps)
                s = d_hat[0] * r0 + d_hat[1] * r1

                forward = np.maximum(0.0, s - forward_shift)
                forward_norm = forward / (dn + eps)
                w_forward = forward_norm ** eta_forward

                rperp0 = r0 - s * d_hat[0]
                rperp1 = r1 - s * d_hat[1]
                dperp2 = rperp0**2 + rperp1**2
                w_lane = np.exp(-dperp2 / (2.0 * ell_perp**2))

                dist2 = r0**2 + r1**2
                w_dist = np.exp(-dist2 / (2.0 * ell_u**2))

                Wk = w_forward * w_lane * w_dist

            if mode == "max":
                W_all = np.maximum(W_all, Wk)
            else:
                W_all += Wk

        if normalize and float(W_all.max()) > 1e-9:
            W_all /= float(W_all.max())
        return W_all

    def target_cell_max_Aeff_in_reachable(
        self,
        ugv_idx: int,
        A_eff: np.ndarray,
        depth: int,
    ) -> Optional[np.ndarray]:
        if ugv_idx < 0 or ugv_idx >= len(self.ugvs):
            return None

        ugv = self.ugvs[ugv_idx]
        H, W = A_eff.shape

        if (not self.voronoi_masks) or (len(self.voronoi_masks) != len(self.ugvs)):
            self.compute_voronoi(H, W)

        allowed = self.voronoi_masks[ugv_idx] if self.voronoi_masks else None
        m = ugv.reachable_unvisited_mask(depth=depth, allowed_mask=allowed)

        if not np.any(m):
            return None

        S = A_eff.copy()
        S[~m] = -np.inf
        if not np.isfinite(S).any():
            return None

        yi, xj = np.unravel_index(int(np.nanargmax(S)), S.shape)
        return np.array([float(yi), float(xj)], dtype=float)


# ============================================================
# 4) UAV config (all flags)
# ============================================================

WaypointMode = Literal["miyashita", "suenaga_dp", "ugv_future_point", "common_weighted"]
NominalMode = Literal[
    "to_waypoint",
    "to_ugv_future",
    "to_ugv_reachable_best_amb",
    "to_ugv_future_if_in_voronoi_else_waypoint",
]
CommonMapMode = Literal["point", "direction"]
SignalMode = Literal["gp_mean", "gp_logistic_prob"]
UAVWaypointSignal = Literal["gp_var", "prob_ambiguity"]


@dataclass
class UAVConfig:
    num_uavs:int=1
    use_cbf: bool = True
    waypoint_mode: WaypointMode = "common_weighted"
    nominal_mode: NominalMode = "to_waypoint"
    nominal_hold_steps: int = 5
    use_voronoi: bool = True

    use_common_map: bool = False
    common_map_mode: CommonMapMode = "direction"
    ugv_weight_eta: float = 0.3

    d0: float = 10.0
    ugv_future_path_sigma: float = 5.0
    step_of_ugv_path_used: int = 6

    ring_d_star: float = 10.0
    ring_sigma: float = 5.0

    suenaga_rho: float = 0.95
    suenaga_depth: int = 5

    k_pp: float = 2.0
    k_ugv: float = 2.0
    v_limit: float = 25.0
    num_poly: int = 8
    control_period: float = 0.1
    gp_update_period: float = 0.5

    cbf_j_alpha: float = 1.0
    cbf_j_gamma: float = 3.0

    signal_mode: SignalMode = "gp_mean"
    uav_waypoint_signal: UAVWaypointSignal = "gp_var"

    map_publish_period: float = 0.5

    dir_num_steps: int = 8
    dir_ell_u: float = 8.0
    dir_ell_perp: float = 4.0
    dir_eta_forward: float = 2.0
    dir_forward_shift: float = 0.0
    dir_mode: str = "sum"
    dir_normalize: bool = True

    wp_use_topk_centroid: bool = True
    wp_topk: int = 80
    wp_min_dist: float = 2.0
    wp_power: float = 1.0

    ugv_move_period: int = 1


# ============================================================
# 5) UAV Controller
# ============================================================

class UAVController:
    def __init__(
        self,
        train_data_x: np.ndarray,
        train_data_y: np.ndarray,
        grid_size: int,
        ugv_fleet: UGVFleet,
        cfg: UAVConfig,
        shared_gp: Optional[SparseOnlineGP] = None,
        uav_id: int = 0,
        gp_sensing_noise_sigma0: float = 0.4,
        gp_max_basis: int = 100,
        gp_threshold_delta: float = 0.05,
        rbf_sigma: float = 2.0
    ):
        self.cfg = cfg
        self.uav_id = uav_id
        self.grid_size = grid_size
        self.ugv_fleet = ugv_fleet
        self.rbf_sigma = rbf_sigma
        # UAVController.__init__
        self._nominal_hold_counter = 0
        self._nominal_hold_steps = 5   # ★ これが効く
        self._last_waypoint = None
        self._last_obs_points: Optional[np.ndarray] = None


        self.pos = train_data_x[0].copy().astype(float)
        self.v = np.zeros(2, dtype=float)

        self.gp = shared_gp if shared_gp is not None else SparseOnlineGP(
            sigma0=gp_sensing_noise_sigma0,
            kernel=lambda x, y, s=rbf_sigma: rbf_kernel(x, y, s),
            max_basis=gp_max_basis,
            delta=gp_threshold_delta
        )

        for x, y in zip(train_data_x, train_data_y):
            self.gp.update(np.asarray(x, dtype=float), float(y))

        self._publish_counter = 0.0
        self._cached_mean_map, self._cached_var_map, self._cached_prob_map = self.get_map_estimates()
        self.ugv_weighted_var_map: Optional[np.ndarray] = None
        self._voronoi_mask: Optional[np.ndarray] = None

        self.current_waypoint = self.pos.copy()
        self.current_chase_point = None

        self._V_eff_for_uav: Optional[np.ndarray] = None
        self._A_eff_for_uav: Optional[np.ndarray] = None
        self.J0: Optional[float] = None
        self._I0: Optional[float] = None
        self._I_prev: Optional[float] = None
        self._sample_count: int = 0

    def set_voronoi_mask(self, mask: np.ndarray):
        self._voronoi_mask = mask

    def update_maps_for_ugv(self):
        self._publish_counter += self.cfg.control_period
        if (self._publish_counter >= self.cfg.map_publish_period or
                self._cached_mean_map is None or self._cached_var_map is None or self._cached_prob_map is None):
            self._cached_mean_map, self._cached_var_map, self._cached_prob_map = self.get_map_estimates()
            self._publish_counter = 0.0

    def get_maps_for_ugv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cached_mean_map, self._cached_var_map, self._cached_prob_map

    def update_map(self, env_fn: Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]):
        obs_list = env_fn(self.pos)
        for p_i, y_i in obs_list:
            self.gp.update(np.asarray(p_i, dtype=float), float(y_i))

    def get_map_estimates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = W = self.grid_size
        mean_map = np.zeros((H, W), dtype=float)
        var_map = np.zeros((H, W), dtype=float)
        prob_map = np.zeros((H, W), dtype=float)

        for i in range(H):
            for j in range(W):
                x = np.array([i, j], dtype=float)
                mu, var, z, _, _ = self.gp.predict(x)
                mean_map[i, j] = mu
                var_map[i, j] = var
                prob_map[i, j] = z

        return mean_map, var_map, prob_map

    def calc_icbf_terms(self, u: np.ndarray, gamma: float, alpha: float) -> Tuple[np.ndarray, float]:
        r2 = self.rbf_sigma ** 2
        ns = len(self.gp.X)
        xi_J1 = np.zeros(2, dtype=float)
        xi_J2 = 0.0
        p = self.pos.copy()

        if ns == 0:
            xi_J2 = alpha * (np.dot(xi_J1, u) + gamma)
            return xi_J1, float(xi_J2)

        for l in range(ns):
            x_l = self.gp.X[l]
            k_vec_l = np.array([self.gp.kernel(xj, x_l) for xj in self.gp.X])
            z_l = self.gp.Q @ k_vec_l
            z_l_ns = float(z_l[-1])
            k_lp = float(self.gp.kernel(x_l, p))

            grad_k_lp = (k_lp / r2) * (p - x_l)

            grad_k_sum = np.zeros(2, dtype=float)
            for j in range(ns - 1):
                x_j = self.gp.X[j]
                k_jp = float(self.gp.kernel(x_j, p))
                grad_k_jp = (k_jp / r2) * (p - x_j)
                grad_k_sum += float(z_l[j]) * grad_k_jp

            # xi_J1 += z_l_ns * (grad_k_lp + grad_k_sum)
            xi_J1 += 2*z_l_ns * (grad_k_lp - grad_k_sum)

            norm_u2 = float(np.dot(u, u))
            delta_lp = (x_l - p) / r2
            inner_lp = float(np.dot(delta_lp, u))
            term1 = (-norm_u2 / r2 + inner_lp ** 2) * (2.0 * z_l_ns * k_lp)

            dot_k_l = np.array([
                float(np.dot((self.gp.kernel(self.gp.X[j], p) / r2) * (p - self.gp.X[j]), u))
                for j in range(ns)
            ])

            dot_K = np.zeros((ns, ns), dtype=float)
            for k in range(ns - 1):
                grad = (self.gp.kernel(p, self.gp.X[k]) / r2) * (p - self.gp.X[k])
                dot_K[-1, k] = float(np.dot(grad, u))
            for j in range(ns - 1):
                grad = (self.gp.kernel(self.gp.X[j], p) / r2) * (p - self.gp.X[j])
                dot_K[j, -1] = float(np.dot(grad, u))

            term2 = float(-4.0 * dot_k_l.T @ self.gp.Q @ dot_K @ z_l)
            term3 = float(2.0 * dot_k_l.T @ self.gp.Q @ dot_k_l)
            term4 = float(2.0 * z_l.T @ dot_K @ self.gp.Q @ dot_K @ z_l)

            cross_term = 0.0
            for j in range(ns - 1):
                x_j = self.gp.X[j]
                k_jp = float(self.gp.kernel(x_j, p))
                delta_jp = (x_j - p) / r2
                inner_jp = float(np.dot(delta_jp, u))
                scalar = (-norm_u2 / r2 + inner_jp ** 2)
                cross_term += float(z_l[j]) * k_jp * scalar
            term5 = float(-2.0 * z_l_ns * cross_term)

            xi_J2 += float(-(term1 + term2 + term3 + term4 + term5))

        xi_J2 += float(alpha * (np.dot(xi_J1, u) + gamma))
        return xi_J1, float(xi_J2)

    def _rbf(self, a: np.ndarray, b: np.ndarray) -> float:
        # あなたの kernel 定義に合わせる（rbf_sigma は self.rbf_sigma）
        self.r2 = float(self.rbf_sigma ** 2)
        return float(np.exp(-np.linalg.norm(a - b) ** 2 / (2.0 * self.r2)))

    def calc_cbf_terms(
        self,
        alpha_J: float,
        gamma: float,
        n_robots: int,
        ts_sampling: float,
        t_now: float,
        use_voronoi: bool = True,
        virtual_points: Optional[np.ndarray] = None,  # ★追加: (M,2)
    ) -> tuple[np.ndarray, float, float]:
        """
        Returns:
            xi1 (2,), xi2 (scalar), I_tilde (scalar)

        virtual_points:
            - None のとき: 従来通り p=self.pos を 1点仮想追加
            - 与えたとき: points(=観測点) を M点まとめて仮想追加して Ctilde を作る
        """

        # --- cells ---
        H = W = self.grid_size
        if use_voronoi and (self._voronoi_mask is not None) and self.cfg.use_voronoi:
            cells = np.argwhere(self._voronoi_mask.astype(bool))
        else:
            cells = np.argwhere(np.ones((H, W), dtype=bool))

        # --- basis Z ---
        Z = np.asarray(self.gp.X, dtype=float)     # (N,2)
        N = int(Z.shape[0])

        # --- virtual points P (M,2) ---
        if virtual_points is None:
            P = self.pos.astype(float).reshape(1, 2)   # 従来互換（中心1点）
        else:
            P = np.asarray(virtual_points, dtype=float)
            if P.ndim == 1:
                P = P.reshape(1, 2)
            # 念のため bounds 内に収める（grid座標前提）
            P[:, 0] = np.clip(P[:, 0], 0, H - 1)
            P[:, 1] = np.clip(P[:, 1], 0, W - 1)

        M = int(P.shape[0])

        # --- build Ctilde for Ztilde=[Z; P] ---
        if N == 0:
            # basisが空のときも multi が処理してくれる
            Ctilde = self.gp._calc_now_C_multi(P)   # (M,M)
        else:
            Ctilde = self.gp._calc_now_C_multi(P)   # (N+M, N+M)

        L2 = float(self.rbf_sigma ** 2)

        # --- precompute terms used in term_Z for each virtual point ---
        if N > 0:
            # for each m: p_m - Z, and k(p_m, Z)
            P_minus_Z = P[:, None, :] - Z[None, :, :]    # (M,N,2)
            k_PZ = np.zeros((M, N), dtype=float)
            for m in range(M):
                for j in range(N):
                    k_PZ[m, j] = float(self._rbf(P[m], Z[j]))
        else:
            P_minus_Z = np.zeros((M, 0, 2), dtype=float)
            k_PZ = np.zeros((M, 0), dtype=float)

        xi1 = np.zeros(2, dtype=float)
        I_tilde = 0.0

        for (i, j) in cells:
            xstar = np.array([float(i), float(j)], dtype=float)

            # kstar = [k(Z,x*); k(P,x*)]  length (N+M)
            if N > 0:
                k_Zx = np.array([float(self._rbf(Z[q], xstar)) for q in range(N)], dtype=float)  # (N,)
            else:
                k_Zx = np.zeros((0,), dtype=float)

            k_Px = np.array([float(self._rbf(P[m], xstar)) for m in range(M)], dtype=float)      # (M,)
            kstar = np.concatenate([k_Zx, k_Px])                                                 # (N+M,)

            # z = Ctilde k*
            z = Ctilde @ kstar  # (N+M,)

            # tilde sigma^2(x*)
            k_xx = float(self._rbf(xstar, xstar))
            sigma2 = k_xx + float(kstar.T @ Ctilde @ kstar)
            I_tilde += sigma2

            # xi1: sum over virtual points
            #   xi1 += Σ_m 2 z_{N+m} [ (k(p_m,x*)/L2)(p_m-x*) + Σ_j z_j k(p_m,Z_j)(p_m-Z_j)/L2 ]
            if M > 0:
                zj = z[:N] if N > 0 else np.zeros((0,), dtype=float)

                for m in range(M):
                    z_m = float(z[N + m])  # virtual point m's coefficient
                    k_pmx = float(k_Px[m])

                    term_x = (k_pmx / L2) * (P[m] - xstar)  # (2,)

                    if N > 0:
                        # term_Z = Σ_j (zj[j] * k(p_m,Z_j)) * (p_m - Z_j)/L2
                        w = (zj * k_PZ[m, :])[:, None]  # (N,1)
                        term_Z = (w * (P_minus_Z[m] / L2)).sum(axis=0)  # (2,)
                    else:
                        term_Z = np.zeros(2, dtype=float)

                    xi1 += 2.0 * z_m * (term_x + term_Z)

        # --- set I0 only once (fixed reference) ---
        if self._I0 is None:
            self._I0 = float(I_tilde)
            self._I_prev = float(I_tilde)
        I_i0 = float(self._I0)

        # xi2 (あなたの式のまま)
        xi2 = float(
            -gamma/(n_robots*ts_sampling)
            - alpha_J*(I_tilde + (gamma*t_now)/(n_robots*ts_sampling) - I_i0)
        )

        return xi1, xi2, float(I_tilde)



    def _ring_weight_map(self, H: int, W: int, center_ij: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        I, J = np.indices((H, W))
        d = np.hypot(I - float(center_ij[0]), J - float(center_ij[1]))
        d_star = float(cfg.ring_d_star)
        sig = float(max(cfg.ring_sigma, 1e-6))
        w = np.exp(-((d - d_star) ** 2) / (2.0 * sig ** 2))
        return w

    def path_generation_for_uav_suenaga(self, var_map: np.ndarray, rho=0.95, depth=8, use_voronoi=False):
        H, W = var_map.shape

        if use_voronoi and (self._voronoi_mask is not None):
            allowed = np.argwhere(self._voronoi_mask)
            cells = [tuple(x) for x in allowed]
            allowed_set = set(cells)
            def in_allowed(c): return c in allowed_set
        else:
            cells = [(i, j) for i in range(H) for j in range(W)]
            def in_allowed(c): return True

        V_prev = {c: 0.0 for c in cells}
        policy = {c: c for c in cells}

        def neighbors(c):
            i, j = c
            nbrs = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    c2 = (ni, nj)
                    if in_allowed(c2):
                        nbrs.append(c2)
            return nbrs

        for _ in range(depth):
            V_cur = {}
            for c in cells:
                best, best_n = -1e9, c
                for c2 in neighbors(c):
                    dist = math.hypot(c[0] - c2[0], c[1] - c2[1])
                    if dist < 1e-6:
                        continue
                    R = float(var_map[c2]) / dist
                    val = R + rho * V_prev[c2]
                    if val > best:
                        best, best_n = val, c2
                V_cur[c] = best
                policy[c] = best_n
            V_prev = V_cur

        ci = (int(round(self.pos[0])), int(round(self.pos[1])))
        if not in_allowed(ci):
            if use_voronoi and len(cells) > 0:
                arr = np.array(cells)
                d = np.hypot(arr[:, 0] - ci[0], arr[:, 1] - ci[1])
                ci = tuple(arr[int(np.argmin(d))])

        next_cell = policy.get(ci, ci)
        return np.array([next_cell[0], next_cell[1]], dtype=float)

    def path_generation_for_uav(self, var_map: np.ndarray, d0: float = 1.0, use_voronoi=False):
        H, W = var_map.shape
        ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=int)
        ci[0] = int(np.clip(ci[0], 0, H - 1))
        ci[1] = int(np.clip(ci[1], 0, W - 1))

        allowed_mask = None
        if use_voronoi and (self._voronoi_mask is not None):
            allowed_mask = self._voronoi_mask.astype(bool)
            if not np.any(allowed_mask):
                return ci.astype(float)
            if not allowed_mask[ci[0], ci[1]]:
                idxs = np.argwhere(allowed_mask)
                d = np.hypot(idxs[:, 0] - ci[0], idxs[:, 1] - ci[1])
                ci = idxs[int(np.argmin(d))]

        w_ring = self._ring_weight_map(H, W, center_ij=ci)
        score = var_map * w_ring

        if allowed_mask is not None:
            score[~allowed_mask] = -np.inf

        if not np.isfinite(score).any():
            return ci.astype(float)

        if self.cfg.wp_use_topk_centroid:
            return self._topk_centroid_waypoint(score, allowed_mask=allowed_mask)
        else:
            ti, tj = np.unravel_index(np.nanargmax(score), score.shape)
            return np.array([ti, tj], dtype=float)

    def path_generation_for_high_variance_point_including_effect_of_ugv(
        self,
        var_map: np.ndarray,
        ugv_future_point: Optional[np.ndarray],
        use_voronoi: bool = True,
        d0: float = 1.0,
        ell: float = 5.0,
    ) -> np.ndarray:
        H, W = var_map.shape

        ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=int)
        ci[0] = int(np.clip(ci[0], 0, H - 1))
        ci[1] = int(np.clip(ci[1], 0, W - 1))

        allowed_mask = None
        if use_voronoi and (self._voronoi_mask is not None):
            allowed_mask = self._voronoi_mask.astype(bool)
            if not np.any(allowed_mask):
                return ci.astype(float)
            if not allowed_mask[ci[0], ci[1]]:
                idxs = np.argwhere(allowed_mask)
                d = np.hypot(idxs[:, 0] - ci[0], idxs[:, 1] - ci[1])
                ci = idxs[int(np.argmin(d))]

        I, J = np.indices((H, W))
        w_ring = self._ring_weight_map(H, W, center_ij=ci)
        sigma2 = var_map

        if ugv_future_point is None:
            Score = sigma2 * w_ring
        else:
            ugv_i = float(ugv_future_point[0])
            ugv_j = float(ugv_future_point[1])
            dist2_ugv = (I - ugv_i) ** 2 + (J - ugv_j) ** 2
            Score = (sigma2 * w_ring) * np.exp(-dist2_ugv / (2.0 * ell ** 2))

        if allowed_mask is not None:
            Score[~allowed_mask] = -np.inf

        if not np.isfinite(Score).any():
            return ci.astype(float)

        if self.cfg.wp_use_topk_centroid:
            return self._topk_centroid_waypoint(Score, allowed_mask=allowed_mask)
        else:
            ti, tj = np.unravel_index(np.nanargmax(Score), Score.shape)
            return np.array([ti, tj], dtype=float)

    def _topk_centroid_waypoint(
        self,
        Score: np.ndarray,
        allowed_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        H, W = Score.shape
        S = Score.copy()

        if allowed_mask is not None:
            S[~allowed_mask] = -np.inf

        if cfg.wp_min_dist is not None and cfg.wp_min_dist > 0:
            I, J = np.indices((H, W))
            d2 = (I - self.pos[0])**2 + (J - self.pos[1])**2
            S[d2 < (cfg.wp_min_dist**2)] = -np.inf

        if not np.isfinite(S).any():
            ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=float)
            ci[0] = float(np.clip(ci[0], 0, H - 1))
            ci[1] = float(np.clip(ci[1], 0, W - 1))
            return ci

        flat = S.ravel()
        K = int(max(1, min(cfg.wp_topk, flat.size)))

        idx_topk = np.argpartition(flat, -K)[-K:]
        vals = flat[idx_topk]

        ok = np.isfinite(vals)
        idx_topk = idx_topk[ok]
        vals = vals[ok]

        if len(vals) == 0:
            ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=float)
            return ci

        ii, jj = np.unravel_index(idx_topk, (H, W))
        pts = np.stack([ii.astype(float), jj.astype(float)], axis=1)

        vmin = float(np.min(vals))
        w = vals - vmin + 1e-12
        if cfg.wp_power is not None and cfg.wp_power != 1.0:
            w = w ** float(cfg.wp_power)

        wsum = float(np.sum(w))
        if wsum <= 1e-12:
            ti, tj = np.unravel_index(int(np.nanargmax(flat)), (H, W))
            return np.array([float(ti), float(tj)], dtype=float)

        centroid = (w[:, None] * pts).sum(axis=0) / wsum
        centroid[0] = float(np.clip(centroid[0], 0, H - 1))
        centroid[1] = float(np.clip(centroid[1], 0, W - 1))
        return centroid.astype(float)

    def _choose_waypoint(self, V_map):
        # ---- 1) ヒステリシス ----
        if self._nominal_hold_counter > 0 and self._last_waypoint is not None:
            self._nominal_hold_counter -= 1
            return self._last_waypoint

        cfg = self.cfg

        # ---- 2) 新規 waypoint 計算 ----
        if cfg.waypoint_mode == "suenaga_dp":
            wp = self.path_generation_for_uav_suenaga(
                var_map=V_map,
                rho=cfg.suenaga_rho,
                depth=cfg.suenaga_depth,
                use_voronoi=cfg.use_voronoi
            )

        elif cfg.waypoint_mode == "miyashita":
            V_use = self.ugv_weighted_var_map if (self.ugv_weighted_var_map is not None) else V_map
            wp = self.path_generation_for_uav(
                var_map=V_use,
                d0=cfg.d0,
                use_voronoi=cfg.use_voronoi
            )

        elif cfg.waypoint_mode == "ugv_future_point":
            cy, cx = self.ugv_fleet.target_cell_for_uav(
                self.pos, step_offset=cfg.step_of_ugv_path_used
            )
            ugv_future = np.array([cy, cx], dtype=float)
            wp = self.path_generation_for_high_variance_point_including_effect_of_ugv(
                var_map=V_map,
                ugv_future_point=ugv_future,
                d0=cfg.d0,
                ell=cfg.ugv_future_path_sigma,
                use_voronoi=cfg.use_voronoi
            )

        else:
            V_use = self.ugv_weighted_var_map if (self.ugv_weighted_var_map is not None) else V_map
            wp = self.path_generation_for_high_variance_point_including_effect_of_ugv(
                var_map=V_use,
                ugv_future_point=None,
                d0=cfg.d0,
                ell=cfg.ugv_future_path_sigma,
                use_voronoi=cfg.use_voronoi
            )

        # ---- 3) 保存 & hold セット ----
        self._last_waypoint = wp
        self._nominal_hold_counter = cfg.nominal_hold_steps  # 例: 5〜10

        return wp


    def _compute_nominal(self, waypoint: np.ndarray, fused_amb: Optional[np.ndarray] = None) -> np.ndarray:
        cfg = self.cfg
        self.current_chase_point = None

        if cfg.nominal_mode == "to_ugv_reachable_best_amb":
            if fused_amb is None:
                return -cfg.k_pp * (self.pos - waypoint)
            ugv_idx = self._find_ugv_in_my_voronoi()
            if ugv_idx is None:
                return -cfg.k_pp * (self.pos - waypoint)
            tgt = self.ugv_fleet.target_cell_max_Aeff_in_reachable(
                ugv_idx=ugv_idx,
                A_eff=fused_amb,
                depth=cfg.dir_num_steps
            )
            if tgt is None:
                return -cfg.k_pp * (self.pos - waypoint)
            self.current_chase_point = tgt.copy()
            return -cfg.k_ugv * (self.pos - tgt)

        if cfg.nominal_mode == "to_ugv_future":
            cy, cx = self.ugv_fleet.target_cell_for_uav(self.pos, step_offset=cfg.step_of_ugv_path_used)
            ugv_future = np.array([cy, cx], dtype=float)
            self.current_chase_point = ugv_future.copy()
            return -cfg.k_ugv * (self.pos - ugv_future)

        if cfg.nominal_mode == "to_ugv_future_if_in_voronoi_else_waypoint":
            ugv_idx = self._find_ugv_in_my_voronoi()
            if ugv_idx is None:
                return -cfg.k_pp * (self.pos - waypoint)
            # ここでは fleet の planned_path から future を取るので
            cy, cx = self.ugv_fleet.target_cell_for_uav(self.pos, step_offset=cfg.step_of_ugv_path_used)
            ugv_future = np.array([cy, cx], dtype=float)
            self.current_chase_point = ugv_future.copy()
            return -cfg.k_ugv * (self.pos - ugv_future)

        return -cfg.k_pp * (self.pos - waypoint)

    def _find_ugv_in_my_voronoi(self) -> Optional[int]:
        if (self._voronoi_mask is None) or (not self.cfg.use_voronoi):
            return None
        H = W = self.grid_size
        best_k = None
        best_d = np.inf
        for k, ugv in enumerate(self.ugv_fleet.ugvs):
            yi, xj = int(ugv.position[0]), int(ugv.position[1])
            if 0 <= yi < H and 0 <= xj < W and bool(self._voronoi_mask[yi, xj]):
                d = float(np.linalg.norm(self.pos - ugv.position.astype(float)))
                if d < best_d:
                    best_d = d
                    best_k = k
        return best_k

    def set_effective_maps(self, V_eff: Optional[np.ndarray], A_eff: Optional[np.ndarray]):
        self._V_eff_for_uav = None if V_eff is None else V_eff.copy()
        self._A_eff_for_uav = None if A_eff is None else A_eff.copy()

    def calc(
        self,
        env_fn: Callable[[np.ndarray], List[Tuple[np.ndarray, float]]],
        fused_amb: Optional[np.ndarray] = None,
        fused_var: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        use_icbf: bool = False,
    ):

        cfg = self.cfg
        use_icbf=False

        # --- GP観測の間引き（例: 5stepに1回）---
        steps_per_gp = max(1, int(round(cfg.gp_update_period / cfg.control_period)))
        do_update = (step is None) or ((step % steps_per_gp) == 0)

        if do_update:
            obs_list = env_fn(self.pos)  # 3×3（中心あり）の観測点9つ
            self._last_obs_points = np.vstack([p for p, _ in obs_list]) if len(obs_list) > 0 else None

            # GP更新（obs_list を使い回す）
            for p_i, y_i in obs_list:
                self.gp.update(np.asarray(p_i, dtype=float), float(y_i))

            self._sample_count += 1
            self._cached_mean_map, self._cached_var_map, self._cached_prob_map = self.get_map_estimates()
            self._publish_counter = 0.0
        else:
            self.update_maps_for_ugv()

       # (1) まずマップはここまでのあなたの処理のまま取得済みとする
        _, V_map, P_map = self.get_maps_for_ugv()

        if cfg.uav_waypoint_signal == "prob_ambiguity":
            A_map = P_map * (1.0 - P_map)
            V_for_wp = A_map
        else:
            V_for_wp = V_map

        # (2) 先に J-CBF の enable 判定だけ作る（waypointはまだ作らない）
        enable_J_cbf = True
        if cfg.use_cbf and (not use_icbf):
            if fused_var is None or step is None:
                raise ValueError("fused_var and step are required for J-based CBF")

            if self._I0 is None:
                _, _, I0 = self.calc_cbf_terms(
                    alpha_J=cfg.cbf_j_alpha,
                    gamma=cfg.cbf_j_gamma,
                    n_robots=int(cfg.num_uavs),
                    ts_sampling=float(cfg.gp_update_period),
                    t_now=0.0,
                    use_voronoi=cfg.use_voronoi,
                    virtual_points=self._last_obs_points,
                )
                self._I0 = float(I0)
                self._I_prev = float(I0)

            t_now = float(step * cfg.control_period)
            xi_J1, xi_J2, I_tilde = self.calc_cbf_terms(
                alpha_J=cfg.cbf_j_alpha,
                gamma=cfg.cbf_j_gamma,
                n_robots=int(cfg.num_uavs),
                ts_sampling=float(cfg.gp_update_period),
                t_now=t_now,
                use_voronoi=cfg.use_voronoi,
                virtual_points=self._last_obs_points,
            )

            #print("分散変化量=", I_tilde - self._I_prev, "I_0=", self._I0, "I_tilde=", I_tilde)
            self._I_prev = I_tilde

            vlim = float(cfg.v_limit)
            if (xi_J2 + float(np.linalg.norm(xi_J1)) * vlim) < 0.0:
                #print(f"[UAV{self.uav_id}] J-CBF skip: necessary condition fails "
                #    f"(xi2 + ||xi1|| vlim < 0). "
                #    f"||xi1||={np.linalg.norm(xi_J1):.3e} xi2={xi_J2:.3e} vlim={vlim:.3f} "
                #    f"I_tilde={I_tilde:.3f} I0={self._I0:.3f}")
                enable_J_cbf = False

        # (3) waypoint を作る
        if enable_J_cbf:
            waypoint = self._choose_waypoint(V_for_wp)
        else:
            # ★このステップだけ waypoint_mode を miyashita 扱いで生成
            _orig = cfg.waypoint_mode
            try:
                cfg.waypoint_mode = "miyashita"
                waypoint = self._choose_waypoint(V_for_wp)
                #print("========================宮下モード=============================")
            finally:
                cfg.waypoint_mode = _orig

        self.current_waypoint = waypoint

        # (4) nominal をその waypoint から計算
        v_nom = self._compute_nominal(waypoint, fused_amb=fused_amb)


        cbf_J = [float(xi_J2), float(xi_J1[0]), float(xi_J1[1]), 0.1]

        velocity_limitation = VelocityLimitation(cfg.v_limit, cfg.num_poly, slack=0.0)
        speed_cbfs = velocity_limitation.calc_cbf()

        field_limitatiom=FieldLimitation(self.grid_size, self.pos, slack=0.0)
        field_cbfs=field_limitatiom.calc_cbf()

        qp = solver()
        if enable_J_cbf:
            qp.add_cbfs([tuple(cbf_J), *speed_cbfs, *field_cbfs])
        else:
            qp.add_cbfs([*speed_cbfs,*field_cbfs])
        #qp.add_cbfs([tuple(cbf_J)])
        if use_icbf:
            nu = qp.solve(nu_nom)
            self.v = self.v + nu * cfg.control_period
        else:
            u_star=qp.solve(v_nom)
            self.v = u_star
            u = self.v.copy()  # 今回は self.v = u_star にしてるので

            lhs = float(xi_J1 @ u + xi_J2)   # >= 0 なら制約OK
            #print("feasible@u=0?", (lhs >= 0.0))
            #print("Xi_J1=", xi_J1, " Xi_J2=", xi_J2)
            #print("velocity=", self.v)


        spd = float(np.linalg.norm(self.v))
        # if spd > cfg.v_limit:
        #     self.v = (cfg.v_limit / max(spd, 1e-12)) * self.v

        self.pos = self.pos + self.v * cfg.control_period
        H = W = self.grid_size
        self.pos[0] = float(np.clip(self.pos[0], 0, H - 1))
        self.pos[1] = float(np.clip(self.pos[1], 0, W - 1))


# ============================================================
# 6) common IO helpers
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_run_name(run_name: str | None) -> str:
    if run_name is not None and len(run_name) > 0:
        return run_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _unique_path(path: str, enable: bool = True) -> str:
    if (not enable) or (not os.path.exists(path)):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}_{k:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def build_result_paths(results_dir: str, run_name: str, scenario_id: str, num_runs: int,
                       master_seed: int | None = None, auto_increment: bool = True):
    os.makedirs(results_dir, exist_ok=True)
    seed_str = f"_seed{master_seed}" if master_seed is not None else ""
    base = f"{run_name}_{scenario_id}{seed_str}"
    data_path = os.path.join(results_dir, f"{base}_data_{num_runs}runs.csv")
    params_path = os.path.join(results_dir, f"{base}_params_{num_runs}runs.csv")
    data_path = _unique_path(data_path, enable=auto_increment)
    params_path = _unique_path(params_path, enable=auto_increment)
    return data_path, params_path

# ============================================================
# 7) 評価用のローカル指標計算（RMSE, MAE, 分散の平均, キャリブレーション誤差の平均）
# ============================================================

def calc_reachable_certainty_metrics(
    ugv: UGVController,
    mean_map: np.ndarray,
    var_map: np.ndarray,
    gt_ref: np.ndarray,
    depth: int,
    allowed_mask: Optional[np.ndarray] = None,
) -> dict:
    mask = ugv.reachable_unvisited_mask(
        depth=depth,
        allowed_mask=allowed_mask
    )

    # reachable が空の場合は、現在位置だけ評価
    if not np.any(mask):
        mask = np.zeros_like(gt_ref, dtype=bool)
        y, x = int(ugv.position[0]), int(ugv.position[1])
        mask[y, x] = True

    pred = mean_map[mask]
    true = gt_ref[mask]
    var = var_map[mask]

    err = pred - true

    return {
        "reachable_rmse": float(np.sqrt(np.mean(err ** 2))),
        "reachable_mae": float(np.mean(np.abs(err))),
        "reachable_var": float(np.mean(var)),
        "reachable_calib": float(np.mean(np.abs((err ** 2) - var))),
        "reachable_cell_count": int(np.sum(mask)),
    }


def calc_planned_path_certainty_metrics(
    path: list[np.ndarray],
    mean_map: np.ndarray,
    var_map: np.ndarray,
    gt_ref: np.ndarray,
    prob_map: Optional[np.ndarray] = None,
) -> dict:
    if path is None or len(path) == 0:
        return {
            "path_rmse": np.nan,
            "path_mae": np.nan,
            "path_var": np.nan,
            "target_error": np.nan,
            "target_var": np.nan,
            "path_true_sum": np.nan,
            "path_mu_sum": np.nan,
            "path_var_sum": np.nan,
            "path_prob_sum": np.nan,
            "target_true": np.nan,
            "target_mu": np.nan,
            "target_prob": np.nan,
            "path_expected_sum": np.nan,
            "target_expected": np.nan,
        }

    H, W = gt_ref.shape

    pts = []
    for p in path:
        y = int(np.clip(p[0], 0, H - 1))
        x = int(np.clip(p[1], 0, W - 1))
        pts.append((y, x))

    yy = np.array([p[0] for p in pts], dtype=int)
    xx = np.array([p[1] for p in pts], dtype=int)

    pred = mean_map[yy, xx]
    true = gt_ref[yy, xx]
    var = var_map[yy, xx]
    err = pred - true

    if prob_map is not None:
        prob = prob_map[yy, xx]
        path_prob_sum = float(np.sum(prob))
        path_expected_sum = float(np.sum(pred * prob))
    else:
        prob = None
        path_prob_sum = np.nan
        path_expected_sum = np.nan

    ty, tx = pts[0]

    target_error = abs(float(mean_map[ty, tx]) - float(gt_ref[ty, tx]))
    target_var = float(var_map[ty, tx])
    target_prob = float(prob_map[ty, tx]) if prob_map is not None else np.nan
    target_expected = (
        float(mean_map[ty, tx] * prob_map[ty, tx])
        if prob_map is not None else np.nan
    )

    return {
        "path_rmse": float(np.sqrt(np.mean(err ** 2))),
        "path_mae": float(np.mean(np.abs(err))),
        "path_var": float(np.mean(var)),
        "target_error": float(target_error),
        "target_var": float(target_var),

        "path_true_sum": float(np.sum(true)),
        "path_mu_sum": float(np.sum(pred)),
        "path_var_sum": float(np.sum(var)),
        "path_prob_sum": path_prob_sum,

        "target_true": float(gt_ref[ty, tx]),
        "target_mu": float(mean_map[ty, tx]),
        "target_prob": target_prob,
        "path_expected_sum": path_expected_sum,
        "target_expected": target_expected,
    }

# ============================================================
# 7) run_once: a single simulation run (returns data_df, params_row)
# ============================================================

def run_once(
    visualize: bool,
    params: dict,
    cfg: UAVConfig,
    run_idx: int,
    master_seed: int,
    scenario_id: str,
    run_root: str
) -> tuple[pd.DataFrame, dict]:
    grid_size = int(deep_get(params, "grid_size", 30))
    noise_std = float(deep_get(params, "noise_std", 0.5))
    num_uavs = int(deep_get(params, "num_uavs", 3))
    num_ugvs = int(deep_get(params, "num_ugvs", 2))
    steps = int(deep_get(params, "steps", 300))

    ugv_depth = int(deep_get(params, "ugv_depth", 8))
    reward_type = int(deep_get(params, "reward_type", 6))
    discount_factor = float(deep_get(params, "discount_factor", 0.95))

    gp_sensing_noise_sigma0 = float(deep_get(params, "gp_sensing_noise_sigma0", 0.4))
    gp_max_basis = int(deep_get(params, "gp_max_basis", 150))
    gp_threshold_delta = float(deep_get(params, "gp_threshold_delta", 0.05))
    rbf_sigma = float(deep_get(params, "rbf_sigma", 2.0))
    cfg.num_uavs = num_uavs

    # reproducible RNG per run
    rng = make_rng(master_seed, scenario_id, run_idx, grid_size, num_uavs, num_ugvs)

    uav_init, ugv_init = sample_initial_positions(rng, grid_size, num_uavs, num_ugvs)

    print(f"[RUN {run_idx}] init_uav={uav_init.tolist()} init_ugv={ugv_init.tolist()}")

    #gt = generate_ground_truth_map(grid_size)
    gt_seed = _stable_int_seed(master_seed, run_idx, grid_size, num_uavs, num_ugvs, "gt")
    gt = generate_ground_truth_map_scalar(grid_size, seed=gt_seed, num_blobs=20)

    # --- dynamic harvest states ---
    gt_initial = gt.copy()  # 評価・可視化用に元GTを保存したいなら残す
    harvested_mask = np.zeros_like(gt, dtype=bool)
    harvested_total = 0.0

    calibrator = HarvestLogitCalibrator(
        threshold=float(deep_get(params, "calibrator.threshold", 2.0)),
        init_w=float(deep_get(params, "calibrator.init_w", 1.0)),
        lr=float(deep_get(params, "calibrator.lr", 0.03)),
        l2=float(deep_get(params, "calibrator.l2", 1e-4)),
        min_samples=int(deep_get(params, "calibrator.min_samples", 20)),
    )


    ugvs = []
    for k in range(num_ugvs):
        ugv = UGVController(grid_size, reward_type=reward_type, discount_factor=discount_factor)
        p0 = ugv_init[k]
        ugv.position = p0.astype(int)
        ugv.visited[ugv.position[0], ugv.position[1]] = True
        ugvs.append(ugv)
    
    # UGVごとに「前ステップでいた場所」を記録
    prev_ugv_positions = [u.position.copy() for u in ugvs]

    ugv_fleet = UGVFleet(ugvs)

    uavs: list[UAVController] = []
    for k in range(num_uavs):
        p0 = uav_init[k]
        init_obs = environment_function(p0, gt, rng=rng, noise_std=noise_std)
        init_x = np.vstack([p for p, _ in init_obs])
        init_y = np.array([y for _, y in init_obs], dtype=float)

        uav = UAVController(
            train_data_x=init_x,
            train_data_y=init_y,
            grid_size=grid_size,
            ugv_fleet=ugv_fleet,
            cfg=cfg,
            shared_gp=None,
            uav_id=k,
            gp_sensing_noise_sigma0=gp_sensing_noise_sigma0,
            gp_max_basis=gp_max_basis,
            gp_threshold_delta=gp_threshold_delta,
            rbf_sigma=rbf_sigma
        )
        uav.pos = p0.astype(float)
        uavs.append(uav)

    # visualize
    colors = ['r', '#ff7f0e', 'm', 'y', 'g', 'b']
    ugv_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    trajs_uav = [[u.pos.copy()] for u in uavs]
    trajs_ugv = [[u.position.copy()] for u in ugvs]

    if visualize:
        show_legend = False
        plt.ion()

        gt_style = get_gt_plot_style(gt_initial)

        mean_vmin, mean_vmax = compute_display_limits(gt_initial, binary=False)

        std0 = np.zeros((grid_size, grid_size), dtype=float)
        std_vmin = 0.0
        std_vmax = 1.0

        fig, ax = plt.subplots(1, 4, figsize=(24, 5))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, wspace=0.30)

        # --- panel 0: mean ---
        im_mean = ax[0].imshow(
            np.zeros((grid_size, grid_size)),
            cmap="viridis",
            origin="lower",
            vmin=mean_vmin,
            vmax=mean_vmax,
            zorder=1,
        )
        cb0 = fig.colorbar(im_mean, ax=ax[0], fraction=0.046, pad=0.04)
        cb0.set_label("Estimated mean")

        # --- panel 1: std ---
        im_std = ax[1].imshow(
            std0,
            cmap="magma",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            zorder=1,
        )
        cb1 = fig.colorbar(im_std, ax=ax[1], fraction=0.046, pad=0.04)
        cb1.set_label("Estimated std")

        # --- panel 2: prob ---
        im_prob = ax[2].imshow(
            np.zeros((grid_size, grid_size)),
            cmap="jet",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            zorder=1,
        )
        cb2 = fig.colorbar(im_prob, ax=ax[2], fraction=0.046, pad=0.04)
        cb2.set_label("Logistic probability")

        # --- panel 3: GT ---
        im_gt = ax[3].imshow(
            gt_initial,
            cmap=gt_style["cmap"],
            origin="lower",
            vmin=gt_style["vmin"],
            vmax=gt_style["vmax"],
        )
        cb3 = fig.colorbar(im_gt, ax=ax[3], fraction=0.046, pad=0.04)
        if gt_style["colorbar_ticks"] is not None:
            cb3.set_ticks(gt_style["colorbar_ticks"])
        cb3.set_label(gt_style["title"])

        # パス類は mean パネルに重ねる
        uav_dots = [ax[0].plot([], [], 'o', color=colors[i % len(colors)], label=f'UAV{i}', zorder=20 + i)[0]
                    for i in range(num_uavs)]
        uav_lines = [ax[0].plot([], [], '-', color=colors[i % len(colors)], markersize=3, zorder=10 + i)[0]
                    for i in range(num_uavs)]

        ugv_lines = [ax[0].plot([], [], '-x', color=ugv_colors[i % len(ugv_colors)], markersize=3,
                                label=f'UGV{i} Path', zorder=30 + i)[0]
                    for i in range(num_ugvs)]
        ugv_dots = [ax[0].plot([], [], 'wo', markeredgecolor='k', label=f'UGV{i}', zorder=31 + i)[0]
                    for i in range(num_ugvs)]

        waypoint_dots = [ax[0].plot([], [], 's', color=colors[i % len(colors)], markersize=6,
                                    label=f'UAV{i} WP', zorder=40 + i)[0]
                        for i in range(num_uavs)]
        chase_dots = [ax[0].plot([], [], 'X', color=colors[i % len(colors)], markersize=8,
                                label=f'UAV{i} chase', zorder=45 + i)[0]
                    for i in range(num_uavs)]

        ugv_plan_lines = []
        ugv_plan_targets = []
        for i in range(num_ugvs):
            line, = ax[0].plot([], [], '--', alpha=1.0, linewidth=2.5, label=f'UGV{i} planned', zorder=35 + i)
            line.set_color('#FFFF00')
            line.set_path_effects([pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()])
            ugv_plan_lines.append(line)

            tgt, = ax[0].plot([], [], 'o', mfc='#FFFF00', mec='black', markersize=9,
                            label=f'UGV{i} target', zorder=36 + i)
            ugv_plan_targets.append(tgt)

        empty_mask = np.zeros((grid_size, grid_size), dtype=bool)
        vor_layers = [ax[0].imshow(mask_to_rgba(empty_mask, colors[i % len(colors)], alpha=0.18),
                                origin='lower', zorder=9)
                    for i in range(num_uavs)]
        vor_cnt_lines = [None for _ in range(num_uavs)]
        prob_threshold_contour = None
        prob_threshold_hatch = None
        mean_threshold_contour = None
        mean_threshold_hatch = None
        std_threshold_contour = None
        std_threshold_hatch = None

        for a in ax:
            a.set_xlim(0, grid_size - 1)
            a.set_ylim(0, grid_size - 1)
            a.set_aspect("equal")

        ax[0].set_title(f"[RUN {run_idx}] Mean")
        ax[1].set_title(f"[RUN {run_idx}] Std")
        ax[2].set_title(f"[RUN {run_idx}] Prob")
        ax[3].set_title(f"[RUN {run_idx}] {gt_style['title']}")

        handles, labels = ax[0].get_legend_handles_labels()

        if show_legend:
            ax[0].legend(
                    handles, labels,
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1),  # ← 右外に逃がす
                    borderaxespad=0,
                    frameon=True,
                    fontsize=9
                )

    else:
        fig = ax = im_mean = im_std = im_prob = im_gt = None
        cb0 = cb1 = cb2 = None
        uav_dots = uav_lines = ugv_lines = ugv_dots = waypoint_dots = chase_dots = []
        ugv_plan_lines = ugv_plan_targets = []
        vor_layers = []
        vor_cnt_lines = []
        prob_threshold_contour = None
        prob_threshold_hatch = None
        mean_threshold_contour = None
        mean_threshold_hatch = None

    # logs
    J_history = []
    true_sum_history = []
    calib_theta_history = []
    calib_w_history = []
    calib_b_history = []
    calib_learned_threshold_history = []
    calib_num_samples_history = []
    ugv_log = {"step": []}
    fixed_std_range_initialized = False

    def _ensure_ugv_cols(k: int):
        base_cols = [
            f"ugv{k}_y", f"ugv{k}_x",
            f"ugv{k}_mu", f"ugv{k}_var", f"ugv{k}_prob",
            f"ugv{k}_visited_count",
            f"ugv{k}_visited_mu_sum", f"ugv{k}_visited_var_sum", f"ugv{k}_visited_prob_sum",

            # 評価用
            f"ugv{k}_reachable_rmse",
            f"ugv{k}_reachable_mae",
            f"ugv{k}_reachable_var",
            f"ugv{k}_reachable_calib",
            f"ugv{k}_reachable_cell_count",

            f"ugv{k}_path_rmse",
            f"ugv{k}_path_mae",
            f"ugv{k}_path_var",
            f"ugv{k}_target_error",
            f"ugv{k}_target_var",
            f"ugv{k}_path_true_sum",
            f"ugv{k}_path_mu_sum",
            f"ugv{k}_path_var_sum",
            f"ugv{k}_path_prob_sum",
            f"ugv{k}_path_expected_sum",
            f"ugv{k}_target_true",
            f"ugv{k}_target_mu",
            f"ugv{k}_target_prob",
            f"ugv{k}_target_expected",
        ]
        for c in base_cols:
            if c not in ugv_log:
                ugv_log[c] = []


    def plot_and_save_maps_snapshot(
        step: int,
        fused_mean: np.ndarray,
        fused_var: np.ndarray,
        gt: np.ndarray,
        trajs_uav: list[list[np.ndarray]],
        trajs_ugv: list[list[np.ndarray]],
        out_dir: str,
        colors: list[str],
        ugv_colors: list[str],
        scenario_id: str,
        master_seed: int,
        run_idx: int,
        show: bool = False,
        signal_mode: str = "gp_mean",
        fused_prob: Optional[np.ndarray] = None,
        threshold_value: float = np.nan,
        learned_threshold: float = np.nan,
    ):
        os.makedirs(out_dir, exist_ok=True)

        save_path = os.path.join(
            out_dir,
            f"maps_{scenario_id}_seed{master_seed}_run{run_idx}_step{step:05d}.png"
        )

        was_interactive = plt.isinteractive()
        plt.ioff()

        gt_style = get_gt_plot_style(gt)

        mean_map_to_show = fused_mean
        mean_vmin = float(np.nanmin(gt))
        mean_vmax = float(np.nanmax(gt))
        mean_cmap = "viridis"

        prob_map_to_show = (
            fused_prob if fused_prob is not None
            else np.clip(1.0 / (1.0 + np.exp(-fused_mean)), 0.0, 1.0)
        )

        std_map_to_show = compute_std_map(fused_var)
        std_vmin = 0.0
        std_vmax = 1.0

        fig2, ax2 = plt.subplots(1, 4, figsize=(24, 5))
        for a in ax2:
            a.set_aspect("equal")
            a.set_xlim(0, fused_mean.shape[1] - 1)
            a.set_ylim(0, fused_mean.shape[0] - 1)

        # (1) Mean / Prob + UGV paths
        im0 = ax2[0].imshow(
            mean_map_to_show,
            cmap=mean_cmap,
            origin="lower",
            vmin=mean_vmin,
            vmax=mean_vmax
        )
        if gt_style["contour_levels"] is not None:
            ax2[0].contour(
                gt,
                levels=gt_style["contour_levels"],
                colors="white",
                linewidths=2,
                origin="lower"
            )
        else:
            ax2[0].imshow(gt, cmap="gray", origin="lower", alpha=0.18)

        ax2[0].set_title(f"Mean + UGV paths (step={step})")
        plt.colorbar(im0, ax=ax2[0], fraction=0.046, pad=0.04)
        threshold_mask = (mean_map_to_show >= learned_threshold).astype(float)
        if np.isfinite(learned_threshold) and np.any(threshold_mask > 0):
            ax2[0].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )
            ax2[0].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

        for i, tr in enumerate(trajs_ugv):
            arr = np.asarray(tr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                ax2[0].plot(
                    arr[:, 1], arr[:, 0],
                    "-", linewidth=3.0,
                    color=ugv_colors[i % len(ugv_colors)],
                    label=f"UGV{i}",
                    zorder=20,
                )
                y_cur, x_cur = float(arr[-1, 0]), float(arr[-1, 1])
                ax2[0].plot(
                    x_cur, y_cur,
                    marker="o", markersize=9,
                    mfc="white", mec=ugv_colors[i % len(ugv_colors)], mew=2.5,
                    linestyle="None",
                    zorder=50,
                )

        ax2[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=9,
            frameon=True,
        )

        # (2) Std + UAV paths
        im1 = ax2[1].imshow(
            std_map_to_show,
            cmap="magma",
            origin="lower",
            vmin=std_vmin,
            vmax=std_vmax
        )
        if gt_style["contour_levels"] is not None:
            ax2[1].contour(
                gt,
                levels=gt_style["contour_levels"],
                colors="white",
                linewidths=2,
                origin="lower"
            )
        else:
            ax2[1].imshow(gt, cmap="gray", origin="lower", alpha=0.18)

        ax2[1].set_title(f"Std + UAV paths (step={step})")
        plt.colorbar(im1, ax=ax2[1], fraction=0.046, pad=0.04)

        threshold_mask = (mean_map_to_show >= learned_threshold).astype(float)

        if np.isfinite(learned_threshold) and np.any(threshold_mask > 0):
            ax2[1].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )

            ax2[1].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

        for i, tr in enumerate(trajs_uav):
            arr = np.asarray(tr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                ax2[1].plot(
                    arr[:, 1], arr[:, 0],
                    "-", linewidth=2.8,
                    color=colors[i % len(colors)],
                    label=f"UAV{i}",
                    zorder=20,
                )
                y_cur, x_cur = float(arr[-1, 0]), float(arr[-1, 1])
                ax2[1].plot(
                    x_cur, y_cur,
                    marker="o", markersize=9,
                    mfc="white", mec=colors[i % len(colors)], mew=2.5,
                    linestyle="None",
                    zorder=50,
                )

        ax2[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=9,
            frameon=True,
        )
                # (3) Prob + threshold region
        im2 = ax2[2].imshow(
            prob_map_to_show,
            cmap="jet",
            origin="lower",
            vmin=0.0,
            vmax=1.0
        )
        ax2[2].set_title(
            f"Prob map (step={step})\n"
            f"theta={threshold_value:.2f}, learned={learned_threshold:.2f}"
        )
        plt.colorbar(im2, ax=ax2[2], fraction=0.046, pad=0.04)

        threshold_mask = (mean_map_to_show >= learned_threshold).astype(float)
        if np.isfinite(learned_threshold) and np.any(threshold_mask > 0):
            ax2[2].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )
            ax2[2].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

                # (4) Ground truth
        im3 = ax2[3].imshow(
            gt,
            cmap=gt_style["cmap"],
            origin="lower",
            vmin=gt_style["vmin"],
            vmax=gt_style["vmax"]
        )
        ax2[3].set_title(gt_style["title"])
        cb3 = plt.colorbar(im3, ax=ax2[3], fraction=0.046, pad=0.04)
        if gt_style["colorbar_ticks"] is not None:
            cb3.set_ticks(gt_style["colorbar_ticks"])

        fig2.subplots_adjust(bottom=0.23, wspace=0.35)
        fig2.savefig(save_path, dpi=200)
        print(f"[SAVE] snapshot -> {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig2)

        if was_interactive:
            plt.ion()



    # sim loop
    # sim loop
    run_start_time = time.time()

    pbar = tqdm(
        range(steps),
        desc=f"[RUN {run_idx}]",
        ncols=120,
        leave=True
    )

    for step in pbar:
        if step % 50 == 0:
            print(f"[RUN {run_idx}] === Step {step} ===")

        positions_now_uav = [u.pos.copy() for u in uavs]
        masks_uav = compute_voronoi_masks(positions_now_uav, grid_size, grid_size)
        for i, m in enumerate(masks_uav):
            uavs[i].set_voronoi_mask(m)

        if visualize:
            for ln in vor_cnt_lines:
                if ln is not None:
                    for c in ln.collections:
                        c.remove()
            for i, m in enumerate(masks_uav):
                vor_cnt_lines[i] = ax[0].contour(
                    m.astype(float), levels=[0.5],
                    colors=colors[i % len(colors)],
                    linewidths=1.2, origin='lower', zorder=11, alpha=0.9
                )
                vor_layers[i].set_data(mask_to_rgba(m, colors[i % len(colors)], alpha=0.18))

        mean_maps, var_maps, prob_maps = [], [], []
        for uav in uavs:
            m_map_i, v_map_i, p_map_i = uav.get_maps_for_ugv()
            mean_maps.append(m_map_i)
            var_maps.append(v_map_i)
            prob_maps.append(p_map_i)

        fused_mean = np.mean(np.stack(mean_maps, axis=0), axis=0)
        fused_var = np.mean(np.stack(var_maps, axis=0), axis=0)


        # 学習済み g -> logit 変換を用いて probability map を作る
        if cfg.signal_mode == "gp_logistic_prob":
            fused_prob = calibrator.prob_from_mu_var(fused_mean, fused_var)
        else:
            fused_prob = np.mean(np.stack(prob_maps, axis=0), axis=0)

        fused_amb = fused_var * fused_prob * (1.0 - fused_prob)


        if visualize and (not fixed_std_range_initialized):
            std_init = compute_std_map(fused_var)
            std_vmin = 0.0
            std_vmax = float(np.percentile(std_init, 99.0))
            if std_vmax < 1e-12:
                std_vmax = 1.0

            mean_vmin, mean_vmax = compute_display_limits(gt_initial, binary=False)
            fixed_std_range_initialized = True

        # UGV visited logging
        ugv_log["step"].append(step)
        for k, ugv in enumerate(ugvs):
            _ensure_ugv_cols(k)

            yi, xj = int(ugv.position[0]), int(ugv.position[1])

            mu_now = float(fused_mean[yi, xj])
            var_now = float(fused_var[yi, xj])
            prob_now = float(fused_prob[yi, xj])

            ugv_log[f"ugv{k}_y"].append(yi)
            ugv_log[f"ugv{k}_x"].append(xj)
            ugv_log[f"ugv{k}_mu"].append(mu_now)
            ugv_log[f"ugv{k}_var"].append(var_now)
            ugv_log[f"ugv{k}_prob"].append(prob_now)

            m = ugv.visited.astype(bool)
            cnt = int(np.sum(m))
            ugv_log[f"ugv{k}_visited_count"].append(cnt)
            ugv_log[f"ugv{k}_visited_mu_sum"].append(float(np.sum(fused_mean[m])) if cnt > 0 else 0.0)
            ugv_log[f"ugv{k}_visited_var_sum"].append(float(np.sum(fused_var[m])) if cnt > 0 else 0.0)
            ugv_log[f"ugv{k}_visited_prob_sum"].append(float(np.sum(fused_prob[m])) if cnt > 0 else 0.0)

        ugv_fleet.compute_voronoi(grid_size, grid_size)
        ugv_E_raw = fused_prob if (cfg.signal_mode == "gp_logistic_prob") else fused_mean
        ugv_E = ugv_E_raw.copy()
        ugv_E[harvested_mask] = 0.0

        # --- evaluation at UGV planning timing ---
        eval_radius = int(deep_get(params, "eval.local_radius", 3))

        # --- evaluation at UGV planning timing using reachable region ---
        for k, ugv in enumerate(ugvs):
            allowed = ugv_fleet.voronoi_masks[k] if (
                ugv_fleet.voronoi_masks and k < len(ugv_fleet.voronoi_masks)
            ) else None

            metrics = calc_reachable_certainty_metrics(
                ugv=ugv,
                mean_map=fused_mean,
                var_map=fused_var,
                gt_ref=gt_initial,
                depth=ugv_depth,
                allowed_mask=allowed,
            )

            ugv_log[f"ugv{k}_reachable_rmse"].append(metrics["reachable_rmse"])
            ugv_log[f"ugv{k}_reachable_mae"].append(metrics["reachable_mae"])
            ugv_log[f"ugv{k}_reachable_var"].append(metrics["reachable_var"])
            ugv_log[f"ugv{k}_reachable_calib"].append(metrics["reachable_calib"])
            ugv_log[f"ugv{k}_reachable_cell_count"].append(metrics["reachable_cell_count"])

        ugv_fleet.plan_all(ugv_E, fused_var, depth=ugv_depth, ambiguity_map=fused_amb)

        # --- evaluation on planned path and target cell ---
        for k, ugv in enumerate(ugvs):
            path = ugv_fleet.planned_paths[k] if k < len(ugv_fleet.planned_paths) else []

            metrics_path = calc_planned_path_certainty_metrics(
                path=path,
                mean_map=fused_mean,
                var_map=fused_var,
                gt_ref=gt_initial,
                prob_map=fused_prob,
            )

            ugv_log[f"ugv{k}_path_rmse"].append(metrics_path["path_rmse"])
            ugv_log[f"ugv{k}_path_mae"].append(metrics_path["path_mae"])
            ugv_log[f"ugv{k}_path_var"].append(metrics_path["path_var"])
            ugv_log[f"ugv{k}_target_error"].append(metrics_path["target_error"])
            ugv_log[f"ugv{k}_target_var"].append(metrics_path["target_var"])
            ugv_log[f"ugv{k}_path_true_sum"].append(metrics_path["path_true_sum"])
            ugv_log[f"ugv{k}_path_mu_sum"].append(metrics_path["path_mu_sum"])
            ugv_log[f"ugv{k}_path_var_sum"].append(metrics_path["path_var_sum"])
            ugv_log[f"ugv{k}_path_prob_sum"].append(metrics_path["path_prob_sum"])
            ugv_log[f"ugv{k}_path_expected_sum"].append(metrics_path["path_expected_sum"])
            ugv_log[f"ugv{k}_target_true"].append(metrics_path["target_true"])
            ugv_log[f"ugv{k}_target_mu"].append(metrics_path["target_mu"])
            ugv_log[f"ugv{k}_target_prob"].append(metrics_path["target_prob"])
            ugv_log[f"ugv{k}_target_expected"].append(metrics_path["target_expected"])

        V_eff = None
        A_eff = None
        if cfg.use_common_map:
            if cfg.common_map_mode == "direction":
                W_common = ugv_fleet.build_direction_weight_map(
                    H=grid_size, W=grid_size,
                    num_steps=cfg.dir_num_steps,
                    ell_u=cfg.dir_ell_u,
                    ell_perp=cfg.dir_ell_perp,
                    eta_forward=cfg.dir_eta_forward,
                    forward_shift=cfg.dir_forward_shift,
                    mode=cfg.dir_mode,
                    normalize=cfg.dir_normalize
                )
            else:
                W_common = ugv_fleet.build_path_weight_map(
                    H=grid_size, W=grid_size,
                    ell=cfg.ugv_future_path_sigma,
                    step_offset=cfg.step_of_ugv_path_used
                )

            V_eff = fused_var * (1.0 + cfg.ugv_weight_eta * W_common)
            A_eff = fused_amb * (1.0 + cfg.ugv_weight_eta * W_common)

        for uav in uavs:
            uav.set_effective_maps(V_eff=V_eff, A_eff=A_eff)
            if cfg.uav_waypoint_signal == "prob_ambiguity":
                uav.ugv_weighted_var_map = None if (A_eff is None) else A_eff.copy()
            else:
                uav.ugv_weighted_var_map = None if (V_eff is None) else V_eff.copy()

        for uav in uavs:
            env_fn = (lambda p, _gt=gt, _ns=noise_std, _rng=rng:
                      environment_function(p, _gt, rng=_rng, noise_std=_ns))
            uav.calc(env_fn, fused_amb=fused_amb, fused_var=fused_var, step=step)

        ugv_period = max(int(cfg.ugv_move_period), 1)
        if (step % ugv_period) == 0:
            ugv_fleet.step_all(ugv_E, fused_var, depth=ugv_depth, step=step + 1, ambiguity_map=fused_amb)
        else:
            ugv_fleet.compute_voronoi(grid_size, grid_size)

        # --- dynamic harvest update ---
        if (step % ugv_period) == 0:
            for k, ugv in enumerate(ugvs):
                yi, xj = int(ugv.position[0]), int(ugv.position[1])

                # 「そのステップで新しく入ったセル」だけ収穫する
                moved_to_new_cell = not np.array_equal(prev_ugv_positions[k], ugv.position)

                if moved_to_new_cell and (not harvested_mask[yi, xj]):
                    harvested_amount = float(gt[yi, xj])
                    harvested_total += harvested_amount

                    # # 収穫量 g を教師データとして g -> logit 変換を更新
                    # calibrator.add_sample(harvested_amount)
                    # calibrator.fit_step(n_iter=20)

                    # 真の環境を更新
                    gt[yi, xj] = 0.0
                    harvested_mask[yi, xj] = True

                # 次ステップ比較用に更新
                prev_ugv_positions[k] = ugv.position.copy()

        J = float(np.sum(fused_var))
        J_history.append(J)

        true_sum_history.append(harvested_total)

        calib_theta_history.append(float(calibrator.threshold))
        calib_w_history.append(float(calibrator.w))
        calib_b_history.append(float(calibrator.b))
        calib_learned_threshold_history.append(
            float(-calibrator.b / calibrator.w) if abs(calibrator.w) > 1e-12 else np.nan
        )
        calib_num_samples_history.append(len(calibrator.g_list))

        total_crop = 0.0
        for u in ugvs:
            total_crop += float(np.sum(gt[u.visited]))

        if visualize:
            mean_map_to_show = fused_mean
            std_map_to_show = compute_std_map(fused_var)
            prob_map_to_show = fused_prob

            # mean は GT スケールに合わせる
            im_mean.set_clim(mean_vmin, mean_vmax)

            # std は固定スケールにする
            im_std.set_clim(std_vmin, std_vmax)

            # prob は常に 0-1
            im_prob.set_clim(0.0, 1.0)

            # GT は動的収穫後の形をそのまま表示
            gt_style_now = get_gt_plot_style(gt)
            im_gt.set_data(gt)
            im_gt.set_cmap(gt_style_now["cmap"])
            im_gt.set_clim(gt_style_now["vmin"], gt_style_now["vmax"])

            im_mean.set_data(mean_map_to_show)
            im_std.set_data(std_map_to_show)
            im_prob.set_data(prob_map_to_show)

            if std_threshold_contour is not None:
                for c in std_threshold_contour.collections:
                    c.remove()
                std_threshold_contour = None

            if std_threshold_hatch is not None:
                for c in std_threshold_hatch.collections:
                    c.remove()
                std_threshold_hatch = None

            if mean_threshold_contour is not None:
                for c in mean_threshold_contour.collections:
                    c.remove()
                mean_threshold_contour = None

            if mean_threshold_hatch is not None:
                for c in mean_threshold_hatch.collections:
                    c.remove()
                mean_threshold_hatch = None

            if prob_threshold_contour is not None:
                for c in prob_threshold_contour.collections:
                    c.remove()
                prob_threshold_contour = None

            if prob_threshold_hatch is not None:
                for c in prob_threshold_hatch.collections:
                    c.remove()
                prob_threshold_hatch = None

            learned_th = calibrator.learned_threshold
            threshold_mask = (fused_mean >= learned_th).astype(float)

            if np.isfinite(learned_th) and np.any(threshold_mask > 0):
                mean_threshold_contour = ax[0].contour(
                    threshold_mask,
                    levels=[0.5],
                    colors="white",
                    linewidths=2.0,
                    origin="lower",
                    zorder=30,
                )
                mean_threshold_hatch = ax[0].contourf(
                    threshold_mask,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                    alpha=0.0,
                    origin="lower",
                    zorder=31,
                )
                
                std_threshold_contour = ax[1].contour(
                    threshold_mask,
                    levels=[0.5],
                    colors="white",
                    linewidths=2.0,
                    origin="lower",
                    zorder=30,
                )

                std_threshold_hatch = ax[1].contourf(
                    threshold_mask,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                    alpha=0.0,
                    origin="lower",
                    zorder=31,
                )

                prob_threshold_contour = ax[2].contour(
                    threshold_mask,
                    levels=[0.5],
                    colors="white",
                    linewidths=2.0,
                    origin="lower",
                    zorder=30,
                )
                prob_threshold_hatch = ax[2].contourf(
                    threshold_mask,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["///"],
                    alpha=0.0,
                    origin="lower",
                    zorder=31,
                )

            for i, uav in enumerate(uavs):
                trajs_uav[i].append(uav.pos.copy())
                pu = np.array(trajs_uav[i])
                uav_lines[i].set_data(pu[:, 1], pu[:, 0])
                uav_dots[i].set_data([uav.pos[1]], [uav.pos[0]])

                if uav.current_chase_point is not None:
                    cp = uav.current_chase_point
                    chase_dots[i].set_data(cp[1], cp[0])
                    waypoint_dots[i].set_data([], [])
                else:
                    wp = uav.current_waypoint
                    waypoint_dots[i].set_data(wp[1], wp[0])
                    chase_dots[i].set_data([], [])

            for i, ugv in enumerate(ugvs):
                trajs_ugv[i].append(ugv.position.copy())
                pv = np.array(trajs_ugv[i])
                ugv_lines[i].set_data(pv[:, 1], pv[:, 0])
                ugv_dots[i].set_data([ugv.position[1]], [ugv.position[0]])

            for i, path in enumerate(ugv_fleet.planned_paths):
                if path:
                    arr = np.array(path)
                    ugv_plan_lines[i].set_data(arr[:, 1], arr[:, 0])
                    tgt = path[0]
                    ugv_plan_targets[i].set_data([tgt[1]], [tgt[0]])
                else:
                    ugv_plan_lines[i].set_data([], [])
                    ugv_plan_targets[i].set_data([], [])

            ax[0].set_title(f"[RUN {run_idx}] Step {step} Mean")
            ax[1].set_title(f"[RUN {run_idx}] Step {step} Std")
            ax[2].set_title(
                f"[RUN {run_idx}] Step {step} Prob\n"
                f"theta={calibrator.threshold:.2f}, learned={calibrator.learned_threshold:.2f}"
            )
            ax[3].set_title(f"[RUN {run_idx}] {gt_style_now['title']}")

            fig.canvas.draw()
            plt.pause(0.01)

            if visualize and (step % 100 == 0):
                out_dir_vis = os.path.join(
                    run_root,
                    "snapshots",
                    f"run{run_idx:02d}"
                )
                plot_and_save_maps_snapshot(
                    step=step,
                    fused_mean=fused_mean,
                    fused_var=fused_var,
                    fused_prob=fused_prob,
                    gt=gt,
                    trajs_uav=trajs_uav,
                    trajs_ugv=trajs_ugv,
                    out_dir=out_dir_vis,
                    colors=colors,
                    ugv_colors=ugv_colors,
                    scenario_id=scenario_id,
                    master_seed=master_seed,
                    run_idx=run_idx,
                    show=False,
                    signal_mode=cfg.signal_mode,
                    threshold_value=calibrator.threshold,
                    learned_threshold=calibrator.learned_threshold,
                )


        elapsed = time.time() - run_start_time
        progress = (step + 1) / steps
        eta = elapsed / max(progress, 1e-12) - elapsed

        pbar.set_postfix({
            "crop": f"{harvested_total:.2f}",
            "J": f"{J:.1f}",
            "elapsed": f"{elapsed/60:.1f}m",
            "eta": f"{eta/60:.1f}m",
        })

    if visualize:
        plt.ioff()
        plt.close(fig)

    # ==========================
    # Final summary visualization (3 panels)
    # ==========================
    if visualize:
        final_mean = fused_mean
        final_var  = fused_var
        final_gt   = gt
        was_interactive = plt.isinteractive()
        plt.ioff()

        gt_style = get_gt_plot_style(final_gt)
        final_mean_map = final_mean
        final_prob_map = fused_prob
        final_std_map = compute_std_map(final_var)

        mean_vmin = float(np.nanmin(gt_initial))
        mean_vmax = float(np.nanmax(gt_initial))
        mean_cmap = "viridis"

        std_vmin = 0.0
        std_vmax = 1.0
        out_dir_vis = os.path.join(run_root, "final")
        os.makedirs(out_dir_vis, exist_ok=True)
        save_path = os.path.join(
            out_dir_vis,
            f"final_maps_{scenario_id}_seed{master_seed}_run{run_idx}.png"
        )

        fig2, ax2 = plt.subplots(1, 4, figsize=(24, 5))
        for a in ax2:
            a.set_aspect("equal")
            a.set_xlim(0, grid_size - 1)
            a.set_ylim(0, grid_size - 1)

        # (1) Mean / Prob + UGV paths
        im0 = ax2[0].imshow(final_mean_map, cmap=mean_cmap, origin="lower", vmin=mean_vmin, vmax=mean_vmax)
        if gt_style["contour_levels"] is not None:
            ax2[0].contour(final_gt, levels=gt_style["contour_levels"], colors="white", linewidths=2, origin="lower")
        else:
            ax2[0].imshow(final_gt, cmap="gray", origin="lower", alpha=0.18)
        ax2[0].set_title("Final mean/prob + UGV paths")
        threshold_mask = (final_mean_map >= calibrator.learned_threshold).astype(float)
        if np.isfinite(calibrator.learned_threshold) and np.any(threshold_mask > 0):
            ax2[0].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )
            ax2[0].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

        for i, tr in enumerate(trajs_ugv):
            arr = np.asarray(tr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                ax2[0].plot(
                    arr[:, 1], arr[:, 0],
                    "-", linewidth=3.0,
                    color=ugv_colors[i % len(ugv_colors)],
                    label=f"UGV{i}",
                    zorder=20,
                )
                y_cur, x_cur = float(arr[-1, 0]), float(arr[-1, 1])
                ax2[0].plot(
                    x_cur, y_cur,
                    marker="o", markersize=9,
                    mfc="white", mec=ugv_colors[i % len(ugv_colors)], mew=2.5,
                    linestyle="None",
                    zorder=50,
                )

        ax2[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=9,
            frameon=True,
        )

        # (2) Std + UAV paths
        im1 = ax2[1].imshow(final_std_map, cmap="magma", origin="lower", vmin=std_vmin, vmax=std_vmax)
        if gt_style["contour_levels"] is not None:
            ax2[1].contour(final_gt, levels=gt_style["contour_levels"], colors="white", linewidths=2, origin="lower")
        else:
            ax2[1].imshow(final_gt, cmap="gray", origin="lower", alpha=0.18)
        ax2[1].set_title("Final std + UAV paths")
        plt.colorbar(im1, ax=ax2[1], fraction=0.046, pad=0.04)

        threshold_mask = (final_mean_map >= calibrator.learned_threshold).astype(float)

        if np.isfinite(calibrator.learned_threshold) and np.any(threshold_mask > 0):
            ax2[1].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )

            ax2[1].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

        for i, tr in enumerate(trajs_uav):
            arr = np.asarray(tr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                ax2[1].plot(
                    arr[:, 1], arr[:, 0],
                    "-", linewidth=2.8,
                    color=colors[i % len(colors)],
                    label=f"UAV{i}",
                    zorder=20,
                )
                y_cur, x_cur = float(arr[-1, 0]), float(arr[-1, 1])
                ax2[1].plot(
                    x_cur, y_cur,
                    marker="o", markersize=9,
                    mfc="white", mec=colors[i % len(colors)], mew=2.5,
                    linestyle="None",
                    zorder=50,
                )

        ax2[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=9,
            frameon=True,
        )

        # (3) Prob + threshold region
        im2 = ax2[2].imshow(
            final_prob_map,
            cmap="jet",
            origin="lower",
            vmin=0.0,
            vmax=1.0
        )
        ax2[2].set_title(
            f"Final prob\n"
            f"theta={calibrator.threshold:.2f}, learned={calibrator.learned_threshold:.2f}"
        )
        plt.colorbar(im2, ax=ax2[2], fraction=0.046, pad=0.04)

        threshold_mask = (final_mean_map >= calibrator.learned_threshold).astype(float)
        if np.isfinite(calibrator.learned_threshold) and np.any(threshold_mask > 0):
            ax2[2].contour(
                threshold_mask,
                levels=[0.5],
                colors="white",
                linewidths=2.0,
                origin="lower",
            )
            ax2[2].contourf(
                threshold_mask,
                levels=[0.5, 1.5],
                colors="none",
                hatches=["///"],
                alpha=0.0,
                origin="lower",
            )

        # (4) GT
        im3 = ax2[3].imshow(
            final_gt,
            cmap=gt_style["cmap"],
            origin="lower",
            vmin=gt_style["vmin"],
            vmax=gt_style["vmax"]
        )
        ax2[3].set_title(gt_style["title"])
        cb3 = plt.colorbar(im3, ax=ax2[3], fraction=0.046, pad=0.04)
        if gt_style["colorbar_ticks"] is not None:
            cb3.set_ticks(gt_style["colorbar_ticks"])

        fig2.subplots_adjust(bottom=0.23, wspace=0.35)
        fig2.savefig(save_path, dpi=200)
        print(f"[SAVE] final maps figure -> {save_path}")

        plt.close(fig2)

        if was_interactive:
            plt.ion()



    visited_union = np.zeros_like(gt, dtype=bool)
    for u in ugvs:
        visited_union |= u.visited
    total_crop_union = float(harvested_total)
    print(f"[RUN {run_idx}] UGVs harvested crop sum: {total_crop_union:.3f}")
    learned_th = float(-calibrator.b / calibrator.w) if abs(calibrator.w) > 1e-12 else np.nan

    print(
        f"[RUN {run_idx}] Calibrator final: "
        f"theta_fixed={calibrator.threshold:.4f}, "
        f"w={calibrator.w:.4f}, "
        f"b={calibrator.b:.4f}, "
        f"learned_threshold=-b/w={learned_th:.4f}, "
        f"samples={len(calibrator.g_list)}"
    )

    # ---- build run data dataframe (with run_idx column) ----
    df = pd.DataFrame({
        'run_idx': run_idx,
        'step': np.arange(len(J_history)),
        'J': J_history,
        'true_crop_sum': true_sum_history,
        'calibrator_theta': calib_theta_history,
        'calibrator_w': calib_w_history,
        'calibrator_b': calib_b_history,
        'calibrator_learned_threshold': calib_learned_threshold_history,
        'calibrator_num_samples': calib_num_samples_history,
    })
    df_ugv = pd.DataFrame(ugv_log)
    df_ugv["run_idx"] = run_idx
    df = df.merge(df_ugv, on=["run_idx", "step"], how="left")

    gp0 = uavs[0].gp
    params_row = {
        'run_idx': run_idx,
        'master_seed': master_seed,
        'scenario_id': scenario_id,

        'grid_size': grid_size,
        'noise_std': noise_std,
        'num_uavs': num_uavs,
        'num_ugvs': num_ugvs,
        'steps': steps,

        'ugv_depth': ugv_depth,
        'reward_type': reward_type,
        'discount_factor': discount_factor,

        'cfg.use_cbf': cfg.use_cbf,
        'cfg.waypoint_mode': cfg.waypoint_mode,
        'cfg.nominal_mode': cfg.nominal_mode,
        'cfg.use_voronoi': cfg.use_voronoi,
        'cfg.use_common_map': cfg.use_common_map,
        'cfg.common_map_mode': cfg.common_map_mode,
        'cfg.signal_mode': cfg.signal_mode,
        'cfg.uav_waypoint_signal': cfg.uav_waypoint_signal,

        'cfg.d0': cfg.d0,
        'cfg.step_of_ugv_path_used': cfg.step_of_ugv_path_used,
        'cfg.ugv_future_path_sigma': cfg.ugv_future_path_sigma,
        'cfg.ugv_weight_eta': cfg.ugv_weight_eta,

        'cfg.ring_d_star': cfg.ring_d_star,
        'cfg.ring_sigma': cfg.ring_sigma,

        'cfg.suenaga_rho': cfg.suenaga_rho,
        'cfg.suenaga_depth': cfg.suenaga_depth,

        'cfg.k_pp': cfg.k_pp,
        'cfg.k_ugv': cfg.k_ugv,
        'cfg.v_limit': cfg.v_limit,
        'cfg.num_poly': cfg.num_poly,
        'cfg.control_period': cfg.control_period,
        'cfg.cbf_j_alpha': cfg.cbf_j_alpha,
        'cfg.cbf_j_gamma': cfg.cbf_j_gamma,
        'cfg.ugv_move_period': cfg.ugv_move_period,

        'gp_sigma0': gp0.sigma0,
        'gp_max_basis': gp0.max_basis,
        'gp_delta': gp0.delta,
        'rbf_sigma': rbf_sigma,

        'case1_count': gp0.count_case1,
        'case2_count': gp0.count_case2,
        'case3_count': gp0.count_case3,

        'final_total_crop_union': total_crop_union,
        'final_true_crop_sum': float(true_sum_history[-1]) if len(true_sum_history) > 0 else np.nan,
        'final_J': float(J_history[-1]) if len(J_history) > 0 else np.nan,

        'init_uav_positions': json.dumps(uav_init.tolist(), ensure_ascii=False),
        'init_ugv_positions': json.dumps(ugv_init.tolist(), ensure_ascii=False),
        'calibrator_threshold': calibrator.threshold,
        'calibrator_w': calibrator.w,
        'calibrator_b': calibrator.b,
        'calibrator_learned_threshold': float(-calibrator.b / calibrator.w) if abs(calibrator.w) > 1e-12 else np.nan,
        'calibrator_num_samples': len(calibrator.g_list),
    }

    return df, params_row


# ============================================================
# 8) multi-run main (N=10)  -> ONE data.csv + ONE params.csv
# ============================================================

def build_cfg_from_params(params: dict) -> UAVConfig:
    cfg = UAVConfig(
        use_cbf=bool(deep_get(params, "cfg.use_cbf", True)),
        waypoint_mode=str(deep_get(params, "cfg.waypoint_mode", "miyashita")),
        nominal_mode=str(deep_get(params, "cfg.nominal_mode", "to_waypoint")),
        nominal_hold_steps=int(deep_get(params, "cfg.nominal_hold_steps", 5)),
        use_voronoi=bool(deep_get(params, "cfg.use_voronoi", True)),

        use_common_map=bool(deep_get(params, "cfg.use_common_map", True)),
        common_map_mode=str(deep_get(params, "cfg.common_map_mode", "direction")),
        ugv_weight_eta=float(deep_get(params, "cfg.ugv_weight_eta", 0.3)),

        d0=float(deep_get(params, "cfg.d0", 5.0)),
        ugv_future_path_sigma=float(deep_get(params, "cfg.ugv_future_path_sigma", 5.0)),
        step_of_ugv_path_used=int(deep_get(params, "cfg.step_of_ugv_path_used", 6)),

        ring_d_star=float(deep_get(params, "cfg.ring_d_star", 8.0)),
        ring_sigma=float(deep_get(params, "cfg.ring_sigma", 4.0)),

        suenaga_rho=float(deep_get(params, "cfg.suenaga_rho", 0.95)),
        suenaga_depth=int(deep_get(params, "cfg.suenaga_depth", 5)),

        k_pp=float(deep_get(params, "cfg.k_pp", 2.0)),
        k_ugv=float(deep_get(params, "cfg.k_ugv", 2.0)),
        v_limit=float(deep_get(params, "cfg.v_limit", 5.0)),
        num_poly=int(deep_get(params, "cfg.num_poly", 8)),
        control_period=float(deep_get(params, "cfg.control_period", 0.1)),
        gp_update_period=float(deep_get(params, "cfg.gp_update_period", 0.5)),

        cbf_j_alpha=float(deep_get(params, "cfg.cbf_j_alpha", 1.0)),
        cbf_j_gamma=float(deep_get(params, "cfg.cbf_j_gamma", 3.0)),

        signal_mode=str(deep_get(params, "cfg.signal_mode", "gp_logistic_prob")),
        uav_waypoint_signal=str(deep_get(params, "cfg.uav_waypoint_signal", "gp_var")),

        map_publish_period=float(deep_get(params, "cfg.map_publish_period", 0.5)),

        dir_num_steps=int(deep_get(params, "cfg.dir_num_steps", deep_get(params, "ugv_depth", 8))),
        dir_ell_u=float(deep_get(params, "cfg.dir_ell_u", 6.0)),
        dir_ell_perp=float(deep_get(params, "cfg.dir_ell_perp", 3.0)),
        dir_eta_forward=float(deep_get(params, "cfg.dir_eta_forward", 2.0)),
        dir_forward_shift=float(deep_get(params, "cfg.dir_forward_shift", 0.0)),
        dir_mode=str(deep_get(params, "cfg.dir_mode", "sum")),
        dir_normalize=bool(deep_get(params, "cfg.dir_normalize", True)),

        wp_use_topk_centroid=bool(deep_get(params, "cfg.wp_use_topk_centroid", False)),
        wp_topk=int(deep_get(params, "cfg.wp_topk", 60)),
        wp_min_dist=float(deep_get(params, "cfg.wp_min_dist", 2.0)),
        wp_power=float(deep_get(params, "cfg.wp_power", 1.0)),

        ugv_move_period=int(deep_get(params, "cfg.ugv_move_period", 5)),
    )
    return cfg


def main_multi():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON params file")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--master_seed", type=int, default=1234)
    parser.add_argument("--scenario_id", type=str, default="base")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--set", action="append", default=[],
                        help="Override param: key=value (supports dotted keys, e.g. cfg.use_cbf=false)")
    parser.add_argument("--run_idx", type=int, default=None)
    args = parser.parse_args()

    params = load_params(args.config)

    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got: {kv}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ["true", "false"]:
            vv = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        deep_set(params, k, vv)

    cfg = build_cfg_from_params(params)

    if args.visualize and args.num_runs > 1:
        print("[WARN] visualize=True with num_runs>1 is heavy. Consider --num_runs 1.")

    base_results_dir = str(deep_get(params, "RESULTS_DIR", "results"))
    run_name = str(deep_get(params, "RUN_NAME", "direction_weighted"))

    run_root = os.path.join(
        base_results_dir,
        f"{run_name}_{args.scenario_id}_seed{args.master_seed}"
    )
    os.makedirs(run_root, exist_ok=True)

    auto_inc = bool(deep_get(params, "AUTO_INCREMENT", True))
    _ensure_dir(run_root)

    data_csv_path, params_csv_path = build_result_paths(
        results_dir=run_root,
        run_name=_make_run_name(run_name),
        scenario_id=args.scenario_id,
        num_runs=args.num_runs,
        master_seed=args.master_seed,
        auto_increment=auto_inc
    )


    all_data = []
    all_params_rows = []

    if args.run_idx is not None:
        run_indices = [args.run_idx]
    else:
        run_indices = range(args.num_runs)

    total_start_time = time.time()
    run_indices = list(run_indices)
    total_runs = len(run_indices)

    for run_count, run_idx in enumerate(run_indices, start=1):
        run_start_time = time.time()

        print(f"\n[ALL RUNS] start run {run_count}/{total_runs} (run_idx={run_idx})")

        df_run, row = run_once(
            visualize=args.visualize,
            params=params,
            cfg=cfg,
            run_idx=run_idx,
            master_seed=args.master_seed,
            scenario_id=args.scenario_id,
            run_root=run_root,
        )

        run_elapsed = time.time() - run_start_time
        total_elapsed = time.time() - total_start_time

        avg_per_run = total_elapsed / run_count
        remaining_runs = total_runs - run_count
        eta_total = avg_per_run * remaining_runs

        print(
            f"[ALL RUNS] finished {run_count}/{total_runs} | "
            f"this_run={run_elapsed/60:.1f} min | "
            f"total_elapsed={total_elapsed/60:.1f} min | "
            f"eta_total={eta_total/60:.1f} min"
        )

        all_data.append(df_run)
        all_params_rows.append(row)

    df_data = pd.concat(all_data, axis=0, ignore_index=True)
    df_params = pd.DataFrame(all_params_rows)

    df_data.to_csv(data_csv_path, index=False)
    df_params.to_csv(params_csv_path, index=False)

    print(f"[SAVE] data   -> {data_csv_path}")
    print(f"[SAVE] params -> {params_csv_path}")

    # quick stats
    if "final_J" in df_params.columns:
        print("[STATS] final_J mean/std:",
              float(df_params["final_J"].mean()),
              float(df_params["final_J"].std()))
    if "final_total_crop_union" in df_params.columns:
        print("[STATS] crop_union mean/std:",
              float(df_params["final_total_crop_union"].mean()),
              float(df_params["final_total_crop_union"].std()))


if __name__ == "__main__":
    main_multi()
