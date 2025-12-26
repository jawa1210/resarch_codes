#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
import math
import numpy as np
from typing import Callable, Tuple, List, Optional, Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba

from qpsolvers import solve_qp


# ============================================================
# 0) QP Solver (CBF)
# ============================================================

class solver:
    def __init__(self):
        self.cbf_list: list[np.ndarray] = []     # each: [bJ, gx, gy]
        self.slack_list: list[float] = []        # each: slack coef
        self.P_co: float = 1.0                   # weight for u part
        self.P_slack: float = 1.0                # weight for slack vars

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

        for i, ((bJ, gx, gy), slack_coef) in enumerate(zip(self.cbf_list, self.slack_list)):
            G[i, 0] = -gx
            G[i, 1] = -gy
            G[i, 2 + i] = slack_coef
            h[i] = bJ

        P = np.zeros((dim, dim), dtype=float)
        P[0, 0] = 2.0 * self.P_co
        P[1, 1] = 2.0 * self.P_co
        for i in range(m):
            P[2 + i, 2 + i] = 2.0 * self.P_slack

        q = np.zeros((dim,), dtype=float)
        q[0:2] = -2.0 * self.P_co * nominal_input

        sol = solve_qp(P, q, G, h, solver="quadprog")
        if sol is None:
            print("[QP] infeasible -> nominal")
            return nominal_input.copy()

        return np.asarray(sol[:2], dtype=float)


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

        # logistic with variance (Jaakkola-like)
        z_raw = mu / np.sqrt(1.0 + (np.pi / 8.0) * var)
        z = float(1.0 / (1.0 + np.exp(-z_raw)))
        return mu, var, z, k_vec, self.C


# ============================================================
# 2) Env / Map utils
# ============================================================

def environment_function(pos: np.ndarray, true_map: np.ndarray, noise_std: float = 0.5) -> List[Tuple[np.ndarray, float]]:

    i0, j0 = int(round(pos[0])), int(round(pos[1]))
    H, W = true_map.shape
    observations = []

    for di in (-4, 0, 4):
        for dj in (-4, 0, 4):
            i, j = i0 + di, j0 + dj
            if not (0 <= i < H and 0 <= j < W):
                continue

            imin, imax = max(0, i - 1), min(H, i + 2)
            jmin, jmax = max(0, j - 1), min(W, j + 2)
            local_patch = true_map[imin:imax, jmin:jmax]
            val = float(np.mean(local_patch))
            noisy = float(val + np.random.normal(loc=0.0, scale=noise_std))
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

    rect(0.30, 0.80, 0.06, 0.46, 1.0)
    rect(0.20, 0.70, 0.60, 0.92, 1.0)
    rect(0.80, 0.95, 0.70, 0.90, 1.0)
    rect(0.10, 0.40, 0.10, 0.40, 1.0)

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
    def __init__(self, grid_size: int = 20, reward_type: int = 0, discount_factor: float = 0.95):
        self.grid_size = grid_size
        self.reward_type = reward_type
        self.position = np.array([grid_size // 2, grid_size // 2], dtype=int)
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited[self.position[0], self.position[1]] = True
        self.discount_factor = discount_factor

    def _calculate_reward(self, pos: np.ndarray, current_pos: np.ndarray,
                          expectation_map: np.ndarray, variance_map: np.ndarray,
                          ambiguity_map: Optional[np.ndarray],
                          k1, k2, k3, k4, k5, k6, epsilon,
                          reward_type: int, step: int) -> float:
        d = float(np.linalg.norm(pos - current_pos))
        E = float(expectation_map[pos[0], pos[1]])
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

        # ---- logistic versions ----
        # reward_type=4: exploit p, penalize ambiguity p(1-p)
        elif reward_type == 4:
            return (k1 * E - k2 * U) / (d ** 2 + epsilon)

        # reward_type=5: exploit p, but also a bit explore ambiguity
        elif reward_type == 5:
            return (k1 * E + k2 * U) / (d ** 2 + epsilon)

        else:
            return 0.0

    def _recursive_search(self, pos: np.ndarray,
                          expectation_map: np.ndarray, variance_map: np.ndarray,
                          ambiguity_map: Optional[np.ndarray],
                          depth: int, visited: np.ndarray, step: int,
                          allowed_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
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

            new_visited = visited.copy()
            new_visited[nxt[0], nxt[1]] = True

            r = self._calculate_reward(
                nxt, pos, expectation_map, variance_map, ambiguity_map,
                k1, k2, k3, k4, k5, k6, epsilon,
                self.reward_type, step
            )
            if visited[nxt[0], nxt[1]]:
                r = -100

            total = r
            if depth > 1:
                _, fut = self._recursive_search(
                    nxt, expectation_map, variance_map, ambiguity_map,
                    depth - 1, new_visited, step + 1, allowed_mask=allowed_mask
                )
                total += self.discount_factor * fut

            if total > best_reward:
                best_reward = total
                best_move = nxt.copy()

        return best_move, float(best_reward)

    def calc(self, expectation_map: np.ndarray, variance_map: np.ndarray,
             ambiguity_map: Optional[np.ndarray] = None,
             depth: int = 10, step: int = 1, allowed_mask: Optional[np.ndarray] = None):
        new_pos, _ = self._recursive_search(
            self.position, expectation_map, variance_map, ambiguity_map,
            depth, self.visited.copy(), step, allowed_mask=allowed_mask
        )
        if not np.array_equal(new_pos, self.position):
            self.position = new_pos
            self.visited[new_pos[0], new_pos[1]] = True

    def get_planned_path(self, expectation_map: np.ndarray, variance_map: np.ndarray,
                         ambiguity_map: Optional[np.ndarray] = None,
                         depth: int = 10, allowed_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        path = []
        pos = self.position.copy()
        visited = self.visited.copy()

        for t in range(depth):
            nxt, _ = self._recursive_search(
                pos, expectation_map, variance_map, ambiguity_map,
                depth - t, visited, t + 1, allowed_mask=allowed_mask
            )
            path.append(nxt.copy())
            visited[nxt[0], nxt[1]] = True
            pos = nxt
        return path


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


# ============================================================
# 4) UAV config (all flags)
# ============================================================

WaypointMode = Literal["miyashita", "suenaga_dp", "ugv_future_point", "common_weighted"]
NominalMode = Literal["to_waypoint", "to_ugv_future"]
CommonMapMode = Literal["point", "direction"]
SignalMode = Literal["gp_mean", "gp_logistic_prob"]

# ★追加：UAV waypoint に使う信号
UAVWaypointSignal = Literal["gp_var", "prob_ambiguity"]


@dataclass
class UAVConfig:
    # core switches
    use_cbf: bool = True
    waypoint_mode: WaypointMode = "common_weighted"
    nominal_mode: NominalMode = "to_waypoint"

    # Voronoi
    use_voronoi: bool = True

    # common weighted var map (from UGV paths)
    use_common_map: bool = False
    common_map_mode: CommonMapMode = "direction"
    ugv_weight_eta: float = 0.3

    # waypoint params
    d0: float = 10.0
    ugv_future_path_sigma: float = 5.0
    step_of_ugv_path_used: int = 6

    # suenaga
    suenaga_rho: float = 0.95
    suenaga_depth: int = 5

    # gains + limits
    k_pp: float = 2.0
    k_ugv: float = 2.0
    v_limit: float = 25.0
    control_period: float = 0.1

    # cbf
    cbf_j_alpha: float = 1.0
    cbf_j_gamma: float = 3.0

    # UGV reward signal map choice
    signal_mode: SignalMode = "gp_mean"

    # ★追加：UAV waypoint に使う信号
    uav_waypoint_signal: UAVWaypointSignal = "gp_var"

    # map publish
    map_publish_period: float = 0.5

    # direction map params
    dir_num_steps: int = 8
    dir_ell_u: float = 8.0
    dir_ell_perp: float = 4.0
    dir_eta_forward: float = 2.0
    dir_forward_shift: float = 0.0
    dir_mode: str = "sum"
    dir_normalize: bool = True

    # ----- waypoint smoothing: Top-K centroid -----
    wp_use_topk_centroid: bool = True
    wp_topk: int = 80                 # 例: 30〜200くらいで調整
    wp_min_dist: float = 2.0          # 現在位置の近すぎを除外（スタック対策）
    wp_power: float = 1.0             # 重み = score^power（>1で尖らせる）



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
        rbf_sigma: float = 2.0,
    ):
        self.cfg = cfg
        self.uav_id = uav_id
        self.grid_size = grid_size
        self.ugv_fleet = ugv_fleet
        self.rbf_sigma = rbf_sigma

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

    def set_voronoi_mask(self, mask: np.ndarray):
        self._voronoi_mask = mask

    def _in_my_voronoi(self, cell: Tuple[int, int]) -> bool:
        if self.cfg.use_voronoi and (self._voronoi_mask is not None):
            i, j = cell
            return bool(self._voronoi_mask[i, j])
        return True

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

    def calc_cbf_terms(self, u: np.ndarray, gamma: float, alpha: float) -> Tuple[np.ndarray, float]:
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

            xi_J1 += z_l_ns * (grad_k_lp + grad_k_sum)

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

    # ---------------- waypoint generators ----------------

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

        I, J = np.indices((H, W))
        dist2 = (I - ci[0]) ** 2 + (J - ci[1]) ** 2
        score = var_map / (dist2 + d0 ** 2)

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
        dist2_uav = (I - ci[0]) ** 2 + (J - ci[1]) ** 2
        denom = dist2_uav + d0 ** 2

        sigma2 = var_map
        if ugv_future_point is None:
            Score = sigma2 / denom
        else:
            ugv_i = float(ugv_future_point[0])
            ugv_j = float(ugv_future_point[1])
            dist2_ugv = (I - ugv_i) ** 2 + (J - ugv_j) ** 2
            Score = (sigma2 / denom) * np.exp(-dist2_ugv / (2.0 * ell ** 2))

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
        """
        Score上位K点の重心を返す（Top-K centroid）
        - allowed_mask: Trueのセルのみ候補
        - cfg.wp_min_dist: 現在位置に近すぎる候補を除外（スタック対策）
        """
        cfg = self.cfg
        H, W = Score.shape
        S = Score.copy()

        # mask outside
        if allowed_mask is not None:
            S[~allowed_mask] = -np.inf

        # 近すぎる点を除外（居座り対策）
        if cfg.wp_min_dist is not None and cfg.wp_min_dist > 0:
            I, J = np.indices((H, W))
            d2 = (I - self.pos[0])**2 + (J - self.pos[1])**2
            S[d2 < (cfg.wp_min_dist**2)] = -np.inf

        # 有効点がなければフォールバック（今の位置）
        if not np.isfinite(S).any():
            ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=float)
            ci[0] = float(np.clip(ci[0], 0, H - 1))
            ci[1] = float(np.clip(ci[1], 0, W - 1))
            return ci

        # Top-K抽出
        flat = S.ravel()
        K = int(max(1, min(cfg.wp_topk, flat.size)))

        # np.argpartitionで上位Kのindexを取る（高速）
        idx_topk = np.argpartition(flat, -K)[-K:]
        vals = flat[idx_topk]

        # -inf混入を除去
        ok = np.isfinite(vals)
        idx_topk = idx_topk[ok]
        vals = vals[ok]

        if len(vals) == 0:
            ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=float)
            return ci

        ii, jj = np.unravel_index(idx_topk, (H, W))
        pts = np.stack([ii.astype(float), jj.astype(float)], axis=1)

        # 重み（scoreが負もあり得るので、最小を引いて正にしてから power）
        vmin = float(np.min(vals))
        w = vals - vmin + 1e-12
        if cfg.wp_power is not None and cfg.wp_power != 1.0:
            w = w ** float(cfg.wp_power)

        wsum = float(np.sum(w))
        if wsum <= 1e-12:
            # 最後の保険：最大点
            ti, tj = np.unravel_index(int(np.nanargmax(flat)), (H, W))
            return np.array([float(ti), float(tj)], dtype=float)

        centroid = (w[:, None] * pts).sum(axis=0) / wsum
        centroid[0] = float(np.clip(centroid[0], 0, H - 1))
        centroid[1] = float(np.clip(centroid[1], 0, W - 1))
        return centroid.astype(float)


    # ---------------- unified waypoint decision ----------------

    def _choose_waypoint(self, V_map: np.ndarray) -> np.ndarray:
        cfg = self.cfg

        if cfg.waypoint_mode == "suenaga_dp":
            return self.path_generation_for_uav_suenaga(
                var_map=V_map,
                rho=cfg.suenaga_rho,
                depth=cfg.suenaga_depth,
                use_voronoi=cfg.use_voronoi
            )

        if cfg.waypoint_mode == "miyashita":
            return self.path_generation_for_uav(
                var_map=V_map,
                d0=cfg.d0,
                use_voronoi=cfg.use_voronoi
            )

        if cfg.waypoint_mode == "ugv_future_point":
            cy, cx = self.ugv_fleet.target_cell_for_uav(self.pos, step_offset=cfg.step_of_ugv_path_used)
            ugv_future = np.array([cy, cx], dtype=float)
            return self.path_generation_for_high_variance_point_including_effect_of_ugv(
                var_map=V_map,
                ugv_future_point=ugv_future,
                d0=cfg.d0,
                ell=cfg.ugv_future_path_sigma,
                use_voronoi=cfg.use_voronoi
            )

        # common_weighted
        V_use = self.ugv_weighted_var_map if (self.ugv_weighted_var_map is not None) else V_map
        return self.path_generation_for_high_variance_point_including_effect_of_ugv(
            var_map=V_use,
            ugv_future_point=None,
            d0=cfg.d0,
            ell=cfg.ugv_future_path_sigma,
            use_voronoi=cfg.use_voronoi
        )

    def _compute_nominal(self, waypoint: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.nominal_mode == "to_ugv_future":
            cy, cx = self.ugv_fleet.target_cell_for_uav(self.pos, step_offset=cfg.step_of_ugv_path_used)
            ugv_future = np.array([cy, cx], dtype=float)
            return -cfg.k_ugv * (self.pos - ugv_future)

        return -cfg.k_pp * (self.pos - waypoint)

    # ---------------- main step ----------------

    def calc(self, env_fn: Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]):
        cfg = self.cfg

        # 1) observe + GP update
        self.update_map(env_fn)

        # 2) update cached maps
        self.update_maps_for_ugv()
        _, V_map, P_map = self.get_maps_for_ugv()

        # ★UAV waypoint に使うマップ切替（var or ambiguity）
        if cfg.uav_waypoint_signal == "prob_ambiguity":
            A_map = P_map * (1.0 - P_map)  # p(1-p)
            V_for_wp = A_map
        else:
            V_for_wp = V_map

        # 3) choose waypoint
        waypoint = self._choose_waypoint(V_for_wp)
        self.current_waypoint = waypoint

        # 4) nominal accel (nu_nom) from desired velocity
        v_nom = self._compute_nominal(waypoint)
        nu_nom = (v_nom - self.v) / cfg.control_period

        # 5) apply cbf or not
        if not cfg.use_cbf:
            self.v = v_nom
        else:
            xi_J1, xi_J2 = self.calc_cbf_terms(self.v, gamma=cfg.cbf_j_gamma, alpha=cfg.cbf_j_alpha)
            cbf_J = [float(-xi_J2), float(-xi_J1[0]), float(-xi_J1[1]), 0.0001]

            qp = solver()
            qp.add_cbfs([tuple(cbf_J)])
            nu = qp.solve(nu_nom)
            self.v = self.v + nu * cfg.control_period

        # 6) clip + integrate
        spd = float(np.linalg.norm(self.v))
        if spd > cfg.v_limit:
            self.v = (cfg.v_limit / max(spd, 1e-12)) * self.v

        self.pos = self.pos + self.v * cfg.control_period
        H = W = self.grid_size
        self.pos[0] = float(np.clip(self.pos[0], 0, H - 1))
        self.pos[1] = float(np.clip(self.pos[1], 0, W - 1))


# ============================================================
# 6) main
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _make_run_name(run_name: str | None) -> str:
    if run_name is not None and len(run_name) > 0:
        return run_name
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _unique_path(base_path_no_ext: str, ext: str, enable: bool = True) -> str:
    """ base_path_no_ext + ext が既に存在するなら _001, _002 を付けて回避 """
    if not enable:
        return base_path_no_ext + ext
    path = base_path_no_ext + ext
    if not os.path.exists(path):
        return path
    k = 1
    while True:
        cand = f"{base_path_no_ext}_{k:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

visualize=True

def main(visualize: bool = False):
    # ---------------- global settings ----------------
    grid_size = 50
    noise_std = 0.5
    num_uavs = 3
    num_ugvs = 2
    steps = 600

    # UGV planning
    ugv_depth = 8
    reward_type = 4          # ★例: 4(p - ambiguity) / 5(p + ambiguity) / 3(UCBなど従来)
    discount_factor = 0.95

    # GP
    gp_sensing_noise_sigma0 = 0.4
    gp_max_basis = 100
    gp_threshold_delta = 0.05
    rbf_sigma = 2.0

    # ===== output naming =====
    RESULTS_DIR = "results"
    RUN_NAME = None  # Noneなら日時で自動命名
    AUTO_INCREMENT = True # 同名があったら _001 みたいに回避


    # ---------------- all-flags config ----------------
    cfg = UAVConfig(
        # CBF
        use_cbf=True,

        # waypoint selection (switch here!)
        waypoint_mode="common_weighted",   # "miyashita" / "suenaga_dp" / "ugv_future_point" / "common_weighted"

        # nominal selection (switch here!)
        nominal_mode="to_waypoint",        # "to_waypoint" / "to_ugv_future"

        use_voronoi=True,

        # common map weighted var
        use_common_map=True,
        common_map_mode="direction",
        ugv_weight_eta=0.3,

        # waypoint params
        d0=10.0,
        ugv_future_path_sigma=5.0,
        step_of_ugv_path_used=6,

        # suenaga params
        suenaga_rho=0.95,
        suenaga_depth=5,

        # gains
        k_pp=2.0,
        k_ugv=2.0,
        v_limit=25.0,
        control_period=0.1,

        # cbf params
        cbf_j_alpha=1.0,
        cbf_j_gamma=3.0,

        # UGV reward map signal
        signal_mode="gp_logistic_prob",    # ★ "gp_mean" / "gp_logistic_prob"

        # ★UAV waypoint signal
        uav_waypoint_signal="prob_ambiguity",  # ★ "gp_var" / "prob_ambiguity"

        map_publish_period=0.5,

        # direction map params
        dir_num_steps=ugv_depth,
        dir_ell_u=8.0,
        dir_ell_perp=4.0,
        dir_eta_forward=2.0,
        dir_forward_shift=0.0,
        dir_mode="sum",
        dir_normalize=True,

        # ----- waypoint smoothing: Top-K centroid -----
        wp_use_topk_centroid = True,
        wp_topk = 80,                 # 例: 30〜200くらいで調整
        wp_min_dist = 5.0,          # 現在位置の近すぎを除外（スタック対策）
        wp_power = 1.0,             # 重み = score^power（>1で尖らせる）

        )

    run_name = _make_run_name(RUN_NAME)
    _ensure_dir(RESULTS_DIR)

    base = os.path.join(RESULTS_DIR, run_name)  # 例: results/20251225_143012
    csv_cbf_path   = _unique_path(base + "__cbf", ".csv", AUTO_INCREMENT)
    csv_param_path = _unique_path(base + "__params", ".csv", AUTO_INCREMENT)

    print(f"[SAVE] cbf csv   -> {csv_cbf_path}")
    print(f"[SAVE] params csv -> {csv_param_path}")

    # ---------------- init GT ----------------
    gt = generate_ground_truth_map(grid_size)

    # ---------------- init UGVs ----------------
    ugvs = []
    center = np.array([grid_size / 2, grid_size / 2], dtype=float)
    Rg = grid_size / 6
    thetas_g = np.linspace(0, 2 * np.pi, num_ugvs, endpoint=False)
    for k in range(num_ugvs):
        ugv = UGVController(grid_size, reward_type=reward_type, discount_factor=discount_factor)
        p0 = center + Rg * np.array([np.cos(thetas_g[k]), np.sin(thetas_g[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size - 1)
        ugv.position = p0.astype(int)
        ugv.visited[ugv.position[0], ugv.position[1]] = True
        ugvs.append(ugv)

    ugv_fleet = UGVFleet(ugvs)

    # ---------------- init UAVs ----------------
    Ru = grid_size / 4
    thetas = np.linspace(0, 2 * np.pi, num_uavs, endpoint=False)
    uavs: list[UAVController] = []
    for k in range(num_uavs):
        p0 = center + Ru * np.array([np.cos(thetas[k]), np.sin(thetas[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size - 1)

        init_obs = environment_function(p0, gt, noise_std=noise_std)
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

    # ---------------- visualization setup ----------------
    colors = ['r', 'c', 'm', 'y', 'g', 'b']
    ugv_colors = ['k', '#444444', '#111111', '#888888']

    trajs_uav = [[u.pos.copy()] for u in uavs]
    trajs_ugv = [[u.position.copy()] for u in ugvs]

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].contour(gt, levels=[0.5], colors='white', linewidths=2, origin='lower', zorder=12)

        im_mean = ax[0].imshow(np.zeros((grid_size, grid_size)),
                               vmin=0, vmax=1, cmap='jet', origin='lower')
        im_var = ax[1].imshow(np.zeros((grid_size, grid_size)),
                              vmin=0, vmax=1, cmap='jet', origin='lower')

        cb0 = fig.colorbar(im_mean, ax=ax[0], fraction=0.046, pad=0.04)
        cb0.set_label('Estimated mean/prob (fused)')
        cb1 = fig.colorbar(im_var, ax=ax[1], fraction=0.046, pad=0.04)
        cb1.set_label('Estimated variance (fused)')

        uav_dots = [ax[0].plot([], [], 'o', color=colors[i % len(colors)],
                               label=f'UAV{i}', zorder=12 + i)[0] for i in range(num_uavs)]
        uav_lines = [ax[0].plot([], [], '-', color=colors[i % len(colors)],
                                markersize=3, zorder=6 + i)[0] for i in range(num_uavs)]

        ugv_lines = [ax[0].plot([], [], '-x', color=ugv_colors[i % len(ugv_colors)],
                                markersize=3, label=f'UGV{i} Path', zorder=20 + i)[0]
                     for i in range(num_ugvs)]
        ugv_dots = [ax[0].plot([], [], 'wo', markeredgecolor='k',
                               label=f'UGV{i}', zorder=21 + i)[0]
                    for i in range(num_ugvs)]

        waypoint_dots = [ax[0].plot([], [], 's', color=colors[i % len(colors)],
                                    markersize=6, label=f'UAV{i} WP', zorder=30 + i)[0]
                         for i in range(num_uavs)]

        ugv_plan_lines = []
        ugv_plan_targets = []
        for i in range(num_ugvs):
            line, = ax[0].plot([], [], '--', alpha=1.0, linewidth=2.5,
                               label=f'UGV{i} planned', zorder=25 + i)
            line.set_color('#FFFF00')
            line.set_path_effects([pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()])
            ugv_plan_lines.append(line)

            tgt, = ax[0].plot([], [], 'o', mfc='#FFFF00', mec='black',
                              markersize=9, label=f'UGV{i} target', zorder=26 + i)
            ugv_plan_targets.append(tgt)

        empty_mask = np.zeros((grid_size, grid_size), dtype=bool)
        vor_layers = [ax[0].imshow(mask_to_rgba(empty_mask, colors[i % len(colors)], alpha=0.18),
                                   origin='lower', zorder=9) for i in range(num_uavs)]
        vor_cnt_lines = [None for _ in range(num_uavs)]

        ax[0].legend(loc='upper right')
        ax[0].set_xlim(0, grid_size - 1)
        ax[0].set_ylim(0, grid_size - 1)
        ax[0].autoscale(False)
    else:
        fig = ax = im_mean = im_var = None
        uav_dots = uav_lines = ugv_lines = ugv_dots = waypoint_dots = []
        ugv_plan_lines = ugv_plan_targets = []
        vor_layers = []
        vor_cnt_lines = []

    # ---------------- loop ----------------
    J_history = []
    true_sum_history = []

    for step in range(steps):
        print(f"=== Step {step} ===")

        # (A) UAV Voronoi
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

        # (B) fused maps from cached per UAV (before moving this step)
        mean_maps, var_maps, prob_maps = [], [], []
        for uav in uavs:
            m_map_i, v_map_i, p_map_i = uav.get_maps_for_ugv()
            mean_maps.append(m_map_i)
            var_maps.append(v_map_i)
            prob_maps.append(p_map_i)

        fused_mean = np.mean(np.stack(mean_maps, axis=0), axis=0)
        fused_var  = np.mean(np.stack(var_maps,  axis=0), axis=0)
        fused_prob = np.mean(np.stack(prob_maps, axis=0), axis=0)
        fused_amb  = fused_prob * (1.0 - fused_prob)  # ★追加

        # (C) UGV Voronoi among UGVs, plan all
        ugv_fleet.compute_voronoi(grid_size, grid_size)

        ugv_E = fused_prob if (cfg.signal_mode == "gp_logistic_prob") else fused_mean
        ugv_fleet.plan_all(ugv_E, fused_var, depth=ugv_depth, ambiguity_map=fused_amb)

        # (C') common weighted map (optional) for UAV waypoint
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
            if cfg.uav_waypoint_signal == "prob_ambiguity":
                uav.ugv_weighted_var_map = None if (A_eff is None) else A_eff.copy()
            else:
                uav.ugv_weighted_var_map = None if (V_eff is None) else V_eff.copy()

        # (D) UAV move (observe inside)
        for uav in uavs:
            env_fn = (lambda p, _gt=gt, _ns=noise_std: environment_function(p, _gt, noise_std=_ns))
            uav.calc(env_fn)

        # (E) UGV step after UAV moved
        ugv_fleet.step_all(ugv_E, fused_var, depth=ugv_depth, step=step + 1, ambiguity_map=fused_amb)

        # (F) log & visualize
        J = float(np.sum(fused_var))
        J_history.append(J)

        total_crop = 0.0
        for u in ugvs:
            total_crop += float(np.sum(gt[u.visited]))
        true_sum_history.append(total_crop)

        if visualize:
            # 左：表示は mean でも prob でもお好みで。ここは fused_mean を表示のままにする
            im_mean.set_data(fused_mean)
            im_var.set_data(fused_var)

            for i, uav in enumerate(uavs):
                trajs_uav[i].append(uav.pos.copy())
                pu = np.array(trajs_uav[i])
                uav_lines[i].set_data(pu[:, 1], pu[:, 0])
                uav_dots[i].set_data([uav.pos[1]], [uav.pos[0]])
                wp = uav.current_waypoint
                waypoint_dots[i].set_data(wp[1], wp[0])

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

            ax[0].set_title(f"Step {step} Mean (fused)")
            ax[1].set_title(f"Step {step} Var (fused)")
            fig.canvas.draw()
            plt.pause(0.05)

        if step % 20 == 0:
            print(f"J={J:.3f}, True crop sum={total_crop:.3f}")

    if visualize:
        plt.ioff()
        plt.close(fig)

    if visualize:
        # ===== 最終結果の並列表示（融合マップ） =====
        final_mean = fused_mean
        final_var  = fused_var

        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        im_true = axes[0].imshow(gt, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[0].set_title("True Map")
        plt.colorbar(im_true, ax=axes[0], fraction=0.046, pad=0.04)

        im_mean2 = axes[1].imshow(final_mean, origin='lower', cmap='jet')
        axes[1].set_title("Estimated Mean (fused)")
        plt.colorbar(im_mean2, ax=axes[1], fraction=0.046, pad=0.04)

        im_var2 = axes[2].imshow(final_var, origin='lower', cmap='jet')
        axes[2].set_title("Estimated Variance (fused)")
        plt.colorbar(im_var2,  ax=axes[2], fraction=0.046, pad=0.04)

        for ax_ in axes:
            # UGV paths
            for i in range(num_ugvs):
                pv = np.array(trajs_ugv[i])
                ax_.plot(pv[:, 1], pv[:, 0], '-x', color='black', markersize=4,
                        label=f'UGV{i} Path', zorder=5+i)

            # UAV paths
            for i in range(num_uavs):
                pu = np.array(trajs_uav[i])
                ax_.plot(pu[:, 1], pu[:, 0], '-', color=colors[i % len(colors)], linewidth=1,
                        label=f'UAV{i} Path', zorder=4+i)

        axes[0].legend(loc='upper right')
        plt.tight_layout()
        plt.show(block=True)

        # ===== Jの推移と理論直線 =====
        t = np.arange(len(J_history))
        J0 = float(J_history[0]) if len(J_history) > 0 else 0.0
        gamma = float(cfg.cbf_j_gamma)

        theory = J0 - gamma * t

        plt.figure(figsize=(6, 4))
        plt.plot(t, J_history, '-o', markersize=3, label='J (fused)')
        plt.plot(t, theory, '--', label=r'$J_0 - \gamma t$')
        plt.xlabel('Step')
        plt.ylabel('J (sum variance)')
        plt.title('Objective $J$ and linear bound')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    # ---------------- save results ----------------
    visited_union = np.zeros_like(gt, dtype=bool)
    for u in ugvs:
        visited_union |= u.visited
    total_crop = float(np.sum(gt[visited_union]))
    print(f"UGVs visited crop sum (GT): {total_crop:.3f}")

    df = pd.DataFrame({
        'step': np.arange(len(J_history)),
        'J': J_history,
        'true_crop_sum': true_sum_history
    })
    df.to_csv(csv_cbf_path, index=False)

    gp0 = uavs[0].gp
    params_info = {
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

        'cfg.suenaga_rho': cfg.suenaga_rho,
        'cfg.suenaga_depth': cfg.suenaga_depth,

        'cfg.k_pp': cfg.k_pp,
        'cfg.k_ugv': cfg.k_ugv,
        'cfg.v_limit': cfg.v_limit,
        'cfg.control_period': cfg.control_period,
        'cfg.cbf_j_alpha': cfg.cbf_j_alpha,
        'cfg.cbf_j_gamma': cfg.cbf_j_gamma,

        'gp_sigma0': gp0.sigma0,
        'gp_max_basis': gp0.max_basis,
        'gp_delta': gp0.delta,
        'rbf_sigma': rbf_sigma,

        'case1_count': gp0.count_case1,
        'case2_count': gp0.count_case2,
        'case3_count': gp0.count_case3,

        'final_total_crop': total_crop,
        'final_J': float(J_history[-1]) if len(J_history) > 0 else np.nan,
    }

    pd.DataFrame([params_info]).to_csv('cbf_flag_params.csv', index=False)
    print("Saved: cbf_flag_all.csv, cbf_flag_params.csv")


if __name__ == "__main__":
    main(visualize=visualize)
