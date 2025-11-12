import math
import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import pandas as pd
import matplotlib.patches as patches

# ─── CBF QP Solver ─────────────────────────────────────────────


class solver():
    def __init__(self):
        self.alpha = 1.0
        self.cbf = None
        self.slack = 0.0
        self.cbf_list=[]
        self.slack_list=[]
        self.P_co=1

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        self.cbf_list.append(np.array([bJ, dbJ_du_x, dbJ_du_y]))
        self.slack_list.append(slack)

    def add_cbfs(self, cbfs: List[Tuple[float, float, float]]):
        for cbf in cbfs:
            cbf_bJ= cbf[0]
            cbf_grad_x = cbf[1]
            cbf_grad_y = cbf[2]
            slack= cbf[3]
            self.add_cbf(cbf_bJ, cbf_grad_x, cbf_grad_y, slack)

    def solve(self, nominal: np.ndarray = None) -> np.ndarray:
        # 1) nominal input
        if nominal is None:
            nominal_input = np.zeros(2)
        else:
            nominal_input = nominal

        # 2) slack の数
        m = len(self.cbf_list)
        dim = 2 + m

        # 3) G, h の構築
        G = np.zeros((m, dim))
        h = np.zeros((m, 1))
        for i, ((bJ, gx, gy), slack_coef) in enumerate(zip(self.cbf_list, self.slack_list)):
            G[i, 0]      = -gx
            G[i, 1]      = -gy
            G[i, 2 + i]  =  slack_coef   # スラック変数 s_i の係数
            h[i, 0]      =  bJ

        P = np.zeros((dim, dim))
        P[0,0] = 2*self.P_co
        P[1,1] = 2*self.P_co
        for i in range(m):
            P[2+i, 2+i] = 2*1 #self.gad_co

        q = np.zeros(dim)
        q[0:2] = -2 * nominal_input

        # 5) QP を解く
        sol = solve_qp(P, q, G, h, solver="cvxopt")
        if sol is None:
            # 解が見つからなければ nominal
            print(f"ノミナルです")
            return nominal_input.copy()
        # 6) 先頭2要素(ν_x,ν_y) を返す
        return sol[:2]


# ─── Sparse Online Gaussian Process (論文準拠実装) ─────────────


def rbf_kernel(x, y, sigma=2.0):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))


class SparseOnlineGP:
    def __init__(self, sigma0: float, kernel=rbf_kernel, max_basis: int = None, delta: float = 0.1):
        self.sigma0 = sigma0         # 観測ノイズ分散
        self.kernel = kernel
        self.max_basis = max_basis      # 基底の最大数
        self.delta = delta          # ノベリティ閾値 ω
        # 初期化：データなし
        self.X = np.zeros((0, 2))
        self.a = np.zeros((0,))
        self.C = np.zeros((0, 0))
        self.Q = np.zeros((0, 0))

    def init_first(self, x, y):
        k00 = self.kernel(x, x)
        denom = k00 + self.sigma0**2
        self.X = x.reshape(1, 2)
        self.a = np.array([y/denom])
        self.C = np.array([[-1.0/denom]])
        self.Q = np.array([[1.0/k00]])

    def update(self, x: np.ndarray, y: float):
        # ―― 1) 事前予測 f*, var_* ――
        k_vec = np.array([self.kernel(xi, x) for xi in self.X])  # (N,)
        k_tt = self.kernel(x, x)
        f_star = float(self.a.dot(k_vec))
        var_star = float(k_tt + k_vec.dot(self.C.dot(k_vec)))

        # ―― 2) q_t, r_t の計算 ――
        denom = var_star + self.sigma0**2
        q_t = (y - f_star) / denom
        r_t = -1.0 / denom

        # ―― 3) ノベリティ h_t の計算 ――
        h_t = k_tt - k_vec.dot(self.Q.dot(k_vec))

        n = self.X.shape[0]
        # --- 基底数に余裕あり → 常に拡張 branch ---
        if self.max_basis is None or n < self.max_basis:
            # 2.17 の a,C,Q 拡張更新
            s_t = np.concatenate([self.C.dot(k_vec), [1.0]])
            # a, C の拡張
            a_ext = np.concatenate([self.a, [0.0]])
            C_ext = np.pad(self.C, ((0, 1), (0, 1)), 'constant')
            self.a = a_ext + q_t * s_t
            self.C = C_ext + r_t * np.outer(s_t, s_t)
            # Q の拡張
            ehat = self.Q.dot(k_vec)
            ehat_full = np.concatenate([ehat, [0.0]])
            efull = np.zeros(n+1)
            efull[-1] = 1.0
            Q_ext = np.pad(self.Q, ((0, 1), (0, 1)), 'constant')
            self.Q = Q_ext + (1.0/h_t)*np.outer(ehat_full-efull, ehat_full-efull)
            # X
            self.X = np.vstack([self.X, x])
            print(f"ケース1")

        else:
            # 基底数が上限到達 → h_t でさらに分岐
            if h_t < self.delta:
                # --- ケース２: discard branch (2.24) だけ更新 ---
                ehat = self.Q.dot(k_vec)
                s_short = self.C.dot(k_vec) + ehat     # Eq.(2.24)
                self.a += q_t * s_short
                self.C += r_t * np.outer(s_short, s_short)
                # self.Q, self.X はそのまま
                print(f"ケース2")

            else:
                # --- ケース３: 拡張 branch + prune (2.17→2.26) ---
                # (2.17) で拡張
                s_t = np.concatenate([self.C.dot(k_vec), [1.0]])
                a_ext = np.concatenate([self.a, [0.0]])
                C_ext = np.pad(self.C, ((0, 1), (0, 1)), 'constant')
                self.a = a_ext + q_t * s_t
                self.C = C_ext + r_t * np.outer(s_t, s_t)
                # Q も拡張
                ehat = self.Q.dot(k_vec)
                ehat_full = np.concatenate([ehat, [0.0]])
                efull = np.zeros(n+1)
                efull[-1] = 1.0
                Q_ext = np.pad(self.Q, ((0, 1), (0, 1)), 'constant')
                self.Q = Q_ext + (1.0/h_t)*np.outer(ehat_full-efull, ehat_full-efull)
                self.X = np.vstack([self.X, x])

                # (2.26) による prune
                self._prune_basis()
                print(f"ケース3")

    def _prune_basis(self):
        # 1) 最も寄与が小さい j を探す
        phi = np.abs(self.a) / np.diag(self.Q)
        j = np.argmin(phi)

        # 2) 残すインデックスを集める
        idx = [i for i in range(len(self.a)) if i != j]

        # 3) 取り除く前のパラメータ
        a_new = self.a.copy()
        C_new = self.C.copy()
        Q_new = self.Q.copy()

        a_j = a_new[j]
        Q_jj = Q_new[j, j]
        C_jj = C_new[j, j]
        Q_jcol = Q_new[idx, j]     # Q^j
        C_jcol = C_new[idx, j]     # c_j

        # 4) スライスしたパラメータ
        a_old = a_new[idx]                         # a^{(t)}
        C_old = C_new[np.ix_(idx, idx)]            # C^{(t)}
        Q_old = Q_new[np.ix_(idx, idx)]            # Q^{(t)}

        # 5) prune 更新
        a_hat = a_old - (a_j / Q_jj) * Q_jcol
        term1 = C_jj*np.outer(Q_jcol, Q_jcol) / (Q_jj**2)
        term2 = (np.outer(Q_jcol, C_jcol) + np.outer(C_jcol, Q_jcol)) / Q_jj
        C_hat = C_old + term1 - term2
        Q_hat = Q_old - np.outer(Q_jcol, Q_jcol) / Q_jj

        # 6) 置き換え
        self.a = a_hat
        self.C = C_hat
        self.Q = Q_hat
        self.X = self.X[idx]

    def predict(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        平均 μ(x), 分散 σ^2(x), k_vec, C_t を返す
        """
        if self.X.shape[0] == 0:
            return 0.0, self.kernel(x, x), np.zeros(0), np.zeros((0, 0))
        k_vec = np.array([self.kernel(xi, x) for xi in self.X])
        mu = float(self.a.dot(k_vec))
        var = self.kernel(x, x) + float(k_vec.dot(self.C.dot(k_vec)))

        # # Jaakkola & Jordan のロジスティック近似
        # z = mu / np.sqrt(1.0 + (np.pi/8.0) * var)
        # mu = 1.0 / (1.0 + np.exp(-z))      # 予測確率 ∈ [0,1]
        # var  = mu * (1.0 - mu)               # Bernoulli 分散 ∈ [0,0.25]
        return mu, var, k_vec, self.C

# ─── Environment / GT Map ────────────────────────────────────


def environment_function(pos: np.ndarray,
                              true_map: np.ndarray,
                              noise_std: float = 0.5
                             ) -> List[Tuple[np.ndarray, float]]:
    """
    pos に最も近い格子点 (i0,j0) を中心に、9点 (中心＋周囲8点) について
      1) その点を中心とした 3x3 の平均値 val
      2) val + N(0, noise_std^2) を [0,1] にクリップ
    を計算し [(位置ベクトル, noisy_value), ...] のリストで返す。
    """
    i0, j0 = int(round(pos[0])), int(round(pos[1]))
    H, W = true_map.shape
    observations = []

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0 + di, j0 + dj
            if not (0 <= i < H and 0 <= j < W):
                continue

            # そのセルを中心とした 3x3 の平均
            imin, imax = max(0, i-1), min(H, i+2)
            jmin, jmax = max(0, j-1), min(W, j+2)
            local_patch = true_map[imin:imax, jmin:jmax]
            val = np.mean(local_patch)

            # ガウスノイズを乗せてクリップ
            noisy = val + np.random.normal(loc=0.0, scale=noise_std)
            #noisy = float(np.clip(noisy, 0.0, 1.0))

            observations.append((np.array([i, j], dtype=float), noisy))

    return observations


def generate_ground_truth_map(grid_size=20):
    gt = np.zeros((grid_size, grid_size))
    gt[15:25, 3:13] = 1.0
    gt[10:25, 20:25] = 1.0
    gt[3:6, 13:16] = 1.0
    gt[5:10, 5:10] = 1.0
    mask = (gt == 0)
    #gt[mask] = np.random.uniform(0.0, 0.2, mask.sum())
    return gt

# ─── Field Limitation ──────────────────────────────────────────────
class FieldLimitation:
    def __init__(self, pos, alpha, grid_size: int):
        self.field_x_min=0
        self.field_x_max=grid_size-1
        self.field_y_min=0
        self.field_y_max=grid_size-1
        self.pos= pos
        self.alpha=alpha
        x_cen= (self.field_x_min + self.field_x_max) / 2
        y_cen= (self.field_y_min + self.field_y_max) / 2
        self.center = np.array([x_cen, y_cen], dtype=float)
        x_radius= (self.field_x_max - self.field_x_min) / 2
        y_radius= (self.field_y_max - self.field_y_min) / 2
        self.radius = np.array([x_radius, y_radius], dtype=float)

    def calc_cbf(self):
        L4_norm=np.power(np.sum(((self.center - self.pos) / self.radius)**4), 1/4)
        return self.alpha*(1-L4_norm)

    def calc_grad(self):
        grad=4*((self.center - self.pos)**3)/(self.radius**4)
        return grad

# ─── UGV Controller──────────────────────────────────────────────


class UGVController:
    def __init__(self, grid_size: int = 20, reward_type: int = 0, discount_factor: float = 0.95,
                 steps_per_cell: int = 1):
        self.grid_size = grid_size
        self.reward_type = reward_type
        self.position = np.array([grid_size // 2, grid_size // 2], dtype=int)
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited[self.position[0], self.position[1]] = True
        self.expectation_map = None
        self.variance_map = None
        self.discount_factor = discount_factor  # 割引率

        # ★ 追加：セル移動に必要なステップ数と進捗カウンタ
        self.steps_per_cell = max(1, int(steps_per_cell))
        self._move_progress = 0

    def set_maps(self, expectation_map: np.ndarray, variance_map: np.ndarray):
        self.expectation_map = expectation_map
        self.variance_map = variance_map

    def _calculate_reward(self, pos: np.ndarray, current_pos: np.ndarray,
                          expectation_map: np.ndarray, variance_map: np.ndarray,
                          k1, k2, k3, k4, k5, k6, epsilon,
                          reward_type: int, step: int) -> float:
        d = np.linalg.norm(pos - current_pos)
        E = expectation_map[pos[0], pos[1]]
        V = variance_map[pos[0], pos[1]]
        if reward_type == 0:
            return (k1 * E - V) / (d**2 + epsilon)
        elif reward_type == 1:
            return np.exp(-(d**2)/k4) * np.exp((k2*E - V)/k3)
        elif reward_type == 2:
            return (np.tanh(k5 * E) - k6*V) / (d**2 + epsilon)
        elif reward_type == 3:
            delta = 0.1
            beta = 2 * np.log((np.pi**2) * (step**2) / (6 * delta))
            return E - np.sqrt(beta * V)
        else:
            return 0.0

    def _recursive_search(self, pos: np.ndarray,
                          expectation_map: np.ndarray, variance_map: np.ndarray,
                          depth: int, visited: np.ndarray, step: int) -> Tuple[np.ndarray, float]:
        actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        k1, k2, k3, k4, k5, k6, epsilon = 2, 2, 0.1, 0.1, 1, 1, 1e-3
        best_reward = -np.inf
        best_move = pos.copy()
        for a in actions:
            nxt = pos + a
            if not (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size):
                continue
            new_visited = visited.copy()
            new_visited[nxt[0], nxt[1]] = True
            r = self._calculate_reward(
                nxt, pos, expectation_map, variance_map,
                k1, k2, k3, k4, k5, k6, epsilon,
                self.reward_type, step
            )
            if visited[nxt[0], nxt[1]]:
                r = -100
            total = r
            if depth > 1:
                _, fut = self._recursive_search(
                    nxt, expectation_map, variance_map,
                    depth-1, new_visited, step+1
                )
                total += self.discount_factor * fut
            if total > best_reward:
                best_reward = total
                best_move = nxt.copy()
        return best_move, best_reward

    def calc(self, expectation_map: np.ndarray, variance_map: np.ndarray,
                depth: int = 10, step: int = 1):
            """
            1 呼び出し = 1 ステップ。
            steps_per_cell ステップ溜まったら 1 マスだけ移動する。
            """
            # 計画だけは毎ステップ更新
            new_pos, _ = self._recursive_search(
                self.position, expectation_map, variance_map,
                depth, self.visited.copy(), step
            )

            self._move_progress += 1
            if self._move_progress >= self.steps_per_cell:
                if not np.array_equal(new_pos, self.position):
                    self.position = new_pos
                    self.visited[new_pos[0], new_pos[1]] = True
                self._move_progress = 0  # 進捗リセット
            # 規定数に達していない場合は位置を据え置く

    def get_planned_path(self, expectation_map: np.ndarray, variance_map: np.ndarray,
                         depth: int = 10) -> List[np.ndarray]:
        path = []
        pos = self.position.copy()
        visited = self.visited.copy()
        for t in range(depth):
            nxt, _ = self._recursive_search(
                pos, expectation_map, variance_map,
                depth-t, visited, t+1
            )
            path.append(nxt.copy())
            visited[nxt[0], nxt[1]] = True
            pos = nxt
        return path

# ─── UAV Controller ──────────────────────────────────────────


class UAVController:
    def __init__(self, train_data_x, train_data_y: np.ndarray, grid_size, ugv: UGVController, step_of_ugv_path_used=8, suenaga=False, map_publish_period: float = 0.5):
        self.r = 1.0
        self.alpha = 1.0
        self.gamma =0.5
        self.k = 1.0 # unomのゲイン
        self.k_pp = 1.0    # ← 追加：waypoint用の比例ゲイン (式(3.9) の k_pp)
        self.pos = train_data_x[0].copy()  # ← floatで初期化
        self.control_period = 0.1
        self.gp = SparseOnlineGP(sigma0=0.4, kernel=rbf_kernel, max_basis=27, delta=0.1)
        self.grid_size = grid_size
        self.v = np.zeros(2)
        self.ugv = ugv
        self.step_of_ugv_path_used = step_of_ugv_path_used
        self.suenaga=suenaga
        self.map_publish_period = map_publish_period
        self.peer_uavs: List[np.ndarray]=[]

        for x, y in zip(train_data_x, train_data_y):
            self.gp.update(x, y)

        self._publish_counter = 0.0
        self._cached_mean_map, self._cached_var_map = self.get_map_estimates()


    def update_maps_for_ugv(self):
        """0.5s ごとにだけ全域推定を再計算してキャッシュを更新"""
        self._publish_counter += self.control_period
        if (self._publish_counter >= self.map_publish_period or
            self._cached_mean_map is None or self._cached_var_map is None):
            self._cached_mean_map, self._cached_var_map = self.get_map_estimates()
            self._publish_counter = 0.0

    def get_maps_for_ugv(self) -> Tuple[np.ndarray, np.ndarray]:
        """UGV に配る最新（0.5s ごとに更新）のマップを返す"""
        return self._cached_mean_map, self._cached_var_map


    def update_map(self, environment_function: Callable[..., List[Tuple[np.ndarray, float]]]):
        obs_list = environment_function(self.pos)

        # GP に取り込む
        for p_i, y_i in obs_list:
            self.gp.update(p_i, y_i)

        # 中心点の観測値だけ取り出して表示
        # self.pos に最も近い p_i を探す
        center_y = None
        for p_i, y_i in obs_list:
            if np.allclose(p_i, self.pos, atol=1e-6):
                center_y = y_i
                break

        if center_y is not None:
            print(f"New batch observations, center noisy = {center_y:.3f}")
        else:
            # もし境界で中心点そのものの観測がないなら
            print(f"New batch observations, center did not exist in obs_list")


    def calc_objective_function(self):
        J = np.sum(self.sigma2)
        self.objective_function = J

    def calc_cbf_terms(self, u: np.ndarray, gamma=3.0, alpha=1.0) -> Tuple[float, np.ndarray, float]:
        """
        bJ = J - γ
        ξ_J1 = ∇_p J
        ξ_J2 = 厳密な式 (A.8) に基づく 2階項
        """
        r2 = 2.0**2  # rbfカーネルのパラメータと揃える
        ns = len(self.gp.X)
        xi_J1 = np.zeros(2)
        xi_J2 = 0.0
        p = self.pos.copy()

        for l in range(ns):
            x_l = self.gp.X[l]
            k_vec_l = np.array([self.gp.kernel(xj, x_l) for xj in self.gp.X])  # (ns,)
            z_l = self.gp.Q @ k_vec_l                                     # (ns,)
            z_l_ns = z_l[-1]  # 自分自身
            k_lp = self.gp.kernel(x_l, p)

            # --- ξ_J1項 ---
            grad_k_lp = (k_lp / r2) * (p - x_l)
            grad_k_sum = np.zeros(2)
            for j in range(ns - 1):
                x_j = self.gp.X[j]
                k_jp = self.gp.kernel(x_j, p)
                grad_k_jp = (k_jp / r2) * (p - x_j)
                grad_k_sum += z_l[j] * grad_k_jp
            xi_J1 += z_l_ns * (grad_k_lp + grad_k_sum)

            # --- ξ_J2項の第一括弧 ---
            norm_u2 = np.dot(u, u)
            delta_lp = (x_l - p) / r2
            inner_lp = np.dot(delta_lp, u)
            term1 = -norm_u2 / r2 + inner_lp**2
            term1 *= 2 * z_l_ns * k_lp

            # --- dot_k_l ∈ R^{ns} 各基底点 j に対する ∇_p k(x_j, p) ⋅ u ---
            dot_k_l = np.array([
                np.dot((self.gp.kernel(self.gp.X[j], p) / r2) * (p - self.gp.X[j]), u)
                for j in range(ns)
            ])  # shape = (ns,)

            # --- dot_K_ns ∈ R^{ns×ns}：各行 j に ∇_p k(x_j, p) ⋅ u を並べる ---
            dot_K = np.zeros((ns, ns))

            # 最下行（ロボット位置 → 各観測点）: ∂k(p, x_j)/∂p ⋅ u
            for k in range(ns - 1):
                grad = (self.gp.kernel(p, self.gp.X[k]) / r2) * (p - self.gp.X[k])
                dot_K[-1, k] = np.dot(grad, u)

            # 最右列（各観測点 → ロボット位置）: ∂k(x_j, p)/∂p ⋅ u
            for j in range(ns - 1):
                grad = (self.gp.kernel(self.gp.X[j], p) / r2) * (p - self.gp.X[j])
                dot_K[j, -1] = np.dot(grad, u)

            term2 = -4 * dot_k_l.T @ self.gp.Q @ dot_K @ z_l
            term3 = 2 * dot_k_l.T @ self.gp.Q @ dot_k_l
            term4 = 2 * z_l.T @ dot_K @ self.gp.Q @ dot_K @ z_l

            # --- ξ_J2項のクロス項（最後の和） ---
            cross_term = 0.0
            for j in range(ns - 1):
                x_j = self.gp.X[j]
                k_jp = self.gp.kernel(x_j, p)
                delta_jp = (x_j - p) / r2
                inner_jp = np.dot(delta_jp, u)
                scalar = -norm_u2 / r2 + inner_jp**2
                cross_term += z_l[j] * k_jp * scalar
            term5 = -2 * z_l_ns * cross_term

            xi_J2 += -(term1 + term2 + term3 + term4 + term5)

        # bJ = self.objective_function - gamma
        xi_J2 += alpha * (np.dot(xi_J1, u) + gamma)

        return xi_J1, xi_J2

    def get_map_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        グリッド全体に対して GP 予測を行い、平均値マップと分散マップを返す
        """
        H, W = self.grid_size, self.grid_size
        mean_map = np.zeros((H, W))
        var_map = np.zeros((H, W))

        for i in range(H):
            for j in range(W):
                x = np.array([i, j], dtype=float)
                mean, var, _, _ = self.gp.predict(x)
                mean_map[i, j] = mean
                var_map[i, j] = var

        return mean_map, var_map

    def set_peer_uavs(self, peers: List[np.ndarray]):
        self.peer_uavs = [np.asarray(p, dytpe=float) for p in (peers or [])]

    def _voronoi_mask_uavs_only(self) -> np.ndarray:
        """自身 vs 他UAV だけで Voronoi。UAVが自分1機なら全域 True。"""
        H = W = self.grid_size
        if not self.peer_uavs:  # 単機
            return np.ones((H, W), dtype=bool)

        mask = np.ones((H, W), dtype=bool)
        for i in range(H):
            for j in range(W):
                d_self = math.hypot(self.pos[0]-i, self.pos[1]-j)
                d_peer = min(math.hypot(p[0]-i, p[1]-j) for p in self.peer_uavs)
                if d_self > d_peer:
                    mask[i, j] = False
        return mask


    def path_generation_for_uav(
        self,
        var_map: np.ndarray,
        rho: float = 0.95,
        depth: int = 8,
        use_voronoi: bool = True,
        eta: float = 1.0,     # 報酬の距離べき（論文の ι）
        stride: int = 1       # 全結合の計算量削減用の間引き
    ) -> np.ndarray:
        H, W = var_map.shape
        vor_mask = self._voronoi_mask_uavs_only() if use_voronoi else np.ones((H, W), dtype=bool)

        # 候補セル（Voronoi 内）を全結合遷移で評価
        cells = [(i, j) for i in range(0, H, stride) for j in range(0, W, stride) if vor_mask[i, j]]
        if not cells:
            cells = [(int(round(self.pos[0])), int(round(self.pos[1])))]

        V = {c: 0.0 for c in cells}
        policy = {c: c for c in cells}
        eps_d = 1e-6

        for _ in range(depth):
            V_new = {}
            for c in cells:
                best, best_next = -1e9, c
                for c2 in cells:
                    if c2 == c:
                        continue
                    dist = math.hypot(c[0]-c2[0], c[1]-c2[1])
                    R = float(var_map[c2]) / ((dist**eta) + eps_d)  # 式(3.6)
                    val = R + rho * V[c2]
                    if val > best:
                        best, best_next = val, c2
                V_new[c] = best
                policy[c] = best_next
            V = V_new

        ci = (int(round(self.pos[0])), int(round(self.pos[1])))
        if ci not in policy:  # 現在セルが候補集合に無い場合は最近傍に写像
            ci = min(cells, key=lambda c: math.hypot(c[0]-self.pos[0], c[1]-self.pos[1]))
        next_cell = policy[ci]
        return np.array([next_cell[0], next_cell[1]], dtype=float)

    def _in_my_voronoi(self, cell: Tuple[int,int]) -> bool:
        """
        （オプション）自身を基準に前もって定義した Voronoi 領域内かを返す。
        もし UGV 等複数ロボット間の割り当てを使わないなら、常に True を返してください。
        """
        # 例：常に True
        return True

    def calc(self, environment_function: Callable[[np.ndarray], float], v_limit=1.0):
        # --- 1) 観測は毎ステップ（0.1s） ---
        self.update_map(environment_function)
        self.prob, self.sigma2, self.k_star, self.K = self.gp.predict(self.pos)
        print(f" predict → mean = {self.prob:.3f}, var = {self.sigma2:.3f}")
        self.calc_objective_function()

        # --- 2) UGV 用マップは 0.5s ごとにだけ再計算してキャッシュ更新 ---
        self.update_maps_for_ugv()
        E_map, V_map = self.get_maps_for_ugv()  # ★キャッシュ版を取得
        self.ugv.set_maps(E_map, V_map)         # ★typo修正：set_mps → set_maps

        # UGV の将来位置はキャッシュされたマップで計算
        planned = self.ugv.get_planned_path(E_map, V_map, depth=self.step_of_ugv_path_used)
        ugv_future_pos = np.array(planned[-1], dtype=float)

        # --- 3) nominal 速度 --
        if self.suenaga:
            waypoint = self.path_generation_for_uav(V_map, rho=0.95, depth=5, use_voronoi=True)
            self.current_waypoint=waypoint
            v_nom = -self.k_pp*(self.pos - waypoint)
            nu_nom = (v_nom - self.v) / self.control_period

        else:
            if self.k == 0:
                nu_nom = np.zeros(2)
            else:
                v_nom = - self.k * (self.pos - ugv_future_pos)
                nu_nom = (v_nom - self.v) / self.control_period

        # --- 4) CBF/QP ---
        xi_J1, xi_J2 = self.calc_cbf_terms(self.v, gamma=self.gamma, alpha=self.alpha)
        cbf_J = [-xi_J2, -xi_J1[0], -xi_J1[1], 1.0]
        self.solver = solver()
        self.solver.add_cbfs([cbf_J])

        # QPの直前
        dot_violate = float(np.dot(xi_J1, nu_nom))
        print(f"dot(xi_J1, u_nom) = {dot_violate:.3f},  rhs = {-xi_J2:.3f}")
        nu = self.solver.solve(nu_nom)
        # QPの直後
        print(f"dot(xi_J1, nu_sol) = {np.dot(xi_J1, nu):.3f}")

        # --- 5) 状態更新 ---
        self.v += nu * self.control_period
        if np.linalg.norm(self.v) >= v_limit:
            self.v = v_limit * self.v / np.linalg.norm(self.v)
        self.pos += self.v * self.control_period

        # --- 6) クランプ＆ログ ---
        H, W = self.grid_size, self.grid_size
        self.pos[0] = np.clip(self.pos[0], 0, H-1)
        self.pos[1] = np.clip(self.pos[1], 0, W-1)
        print(f"xi_J1 = {xi_J1}, norm = {np.linalg.norm(xi_J1)}")
        print(f"velocity = {np.linalg.norm(self.v)}")


# ─── main ─────────────────────────────────────────────────────


def main():
    grid_size = 30
    gt = generate_ground_truth_map(grid_size)
    noise_std=0.5

    # 初期観測点
    start = np.array([12.0, 15.0])

    # 1) ９点まとめて観測
    init_obs = environment_function(start, gt, noise_std=0.5)
    # 2) 観測点位置だけ取り出して配列に
    init_x = np.vstack([ p for p, _ in init_obs ])   # shape (9,2)
    # 3) 観測値だけ取り出して配列に
    init_y = np.array([ y for _, y in init_obs ])    # shape (9,)

    # コントローラ生成
    ugv = UGVController(grid_size, reward_type=3, discount_factor=0.95,steps_per_cell=10)
    uav = UAVController(init_x, init_y, grid_size,
                        ugv=ugv, step_of_ugv_path_used=6,suenaga=False, map_publish_period=1.0)

    ENABLE_LIVE_PLOT = True          # 逐次の2画面（Mean/Var）更新をするなら True
    ENABLE_FINAL_MAPS_PLOT = True     # 最後に True/Mean/Var の三面図を出す
    ENABLE_HISTORY_PLOT = True        # 最後に J の推移（理論線との比較）を出す
    DISPLAY_MAP_SOURCE = "fresh"   # "fresh" or "cached"



        # インタラクティブ表示設定（★フラグで制御）
    if ENABLE_LIVE_PLOT:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gt, origin='lower', cmap='gray', alpha=0.3)
        im0 = ax[0].imshow(np.zeros((grid_size, grid_size)), vmin=0, vmax=1, cmap='jet', origin='lower')
        im1 = ax[1].imshow(np.zeros((grid_size, grid_size)), vmin=0, vmax=1, cmap='jet', origin='lower')
        cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04); cb0.set_label('Estimated mean')
        cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04); cb1.set_label('Estimated variance')

        ax[0].imshow(gt, origin='lower', cmap='gray', alpha=0.3)
        ax[0].contour(gt, levels=[0.5], colors='white', linewidths=2, origin='lower')
        im0 = ax[0].imshow(np.zeros((grid_size, grid_size)), vmin=0, vmax=1, cmap='jet', origin='lower')

        traj_uav = [uav.pos.copy()]
        traj_ugv = [ugv.position.copy()]
        waypoint_dot, = ax[0].plot([], [], 'ms', markersize=6, label='UAV Waypoint', zorder=15)
        uav_line,   = ax[0].plot([], [], '-o', color='cyan',   markersize=3, label='UAV Path', zorder=10)
        ugv_line,   = ax[0].plot([], [], '-x', color='black',  markersize=3, label='UGV Path', zorder=11)
        plan_line,  = ax[0].plot([], [], '--*', color='magenta', markersize=4, label='UGV Planned', zorder=12)
        uav_dot,    = ax[0].plot([], [], 'ro', label='UAV', zorder=13)
        ugv_dot,    = ax[0].plot([], [], 'wo', label='UGV', zorder=14)
        ax[0].legend(loc='upper right')
        ax[0].set_xlim(0, grid_size-1); ax[0].set_ylim(0, grid_size-1); ax[0].autoscale(False)
    else:
        # ライブ描画しない場合でも、軌跡は最後の三面図で使えるので記録だけしておく
        traj_uav = [uav.pos.copy()]
        traj_ugv = [ugv.position.copy()]


    # J の履歴
    J_history = []
    true_sum_history = []

# ループの中（for step in range(600):）を一部変更
    for step in range(600):
        depth = 8
        uav.calc(lambda p: environment_function(p, gt, noise_std), v_limit=5.0)

        # --- UGVにはキャッシュ版を渡す ---
        m_map_ugv, v_map_ugv = uav.get_maps_for_ugv()
        ugv.calc(m_map_ugv, v_map_ugv, depth=depth, step=step+1)

        # --- 描画用と J 計算用のマップを切り替え ---
        if DISPLAY_MAP_SOURCE == "fresh":
            m_map_disp, v_map_disp = uav.get_map_estimates()   # 毎step新規
        else:
            m_map_disp, v_map_disp = uav.get_maps_for_ugv()    # キャッシュ

        # --- J履歴 ---
        J = np.sum(v_map_disp)
        J_history.append(J)

        total_crop = np.sum(gt[ugv.visited])
        true_sum_history.append(total_crop)

        # --- 軌跡の記録 ---
        traj_uav.append(uav.pos.copy())
        traj_ugv.append(ugv.position.copy())
        planned = ugv.get_planned_path(m_map_ugv, v_map_ugv, depth=depth)

        # --- プロット更新 ---
        if ENABLE_LIVE_PLOT:
            im0.set_data(m_map_disp)   # 切替えたマップを表示
            im1.set_data(v_map_disp)

            pu = np.array(traj_uav); pv = np.array(traj_ugv); pp = np.array(planned)
            if hasattr(uav, 'current_waypoint'):
                wp = uav.current_waypoint
                waypoint_dot.set_data(wp[1], wp[0])
            uav_line.set_data(pu[:, 1], pu[:, 0])
            ugv_line.set_data(pv[:, 1], pv[:, 0])
            plan_line.set_data(pp[:, 1], pp[:, 0])
            uav_dot.set_data([uav.pos[1]], [uav.pos[0]])
            ugv_dot.set_data([ugv.position[1]], [ugv.position[0]])

            sim_time = step * uav.control_period
            ax[0].set_title(f"t = {sim_time:.1f} s  (Step {step}) Mean")
            ax[1].set_title(f"t = {sim_time:.1f} s  (Step {step}) Var")
            fig.canvas.draw()
            plt.pause(uav.control_period)


    # インタラクティブ終了
    plt.ioff()
    plt.close(fig)

        # ライブ描画を使っていた場合は片付け
    if ENABLE_LIVE_PLOT:
        plt.ioff()
        plt.close(fig)

    # 収穫合計の出力・CSV保存は常に実施
    visited = ugv.visited
    total_crop = np.sum(gt[visited])
    print(f"UGV が訪問した作物の合計（真値）: {total_crop:.3f}")
    df = pd.DataFrame({'step': np.arange(len(J_history)),
                       'J': J_history,
                       'true_crop_sum': true_sum_history})
    df.to_csv('uav_ugv_results.csv', index=False)
    print("results saved to uav_ugv_results.csv")

    # ★ 最後の三面図（True/Mean/Var）
    if ENABLE_FINAL_MAPS_PLOT:
        final_mean, final_var = uav.get_map_estimates()
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        im_true = axes[0].imshow(gt, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[0].set_title("True Map"); plt.colorbar(im_true, ax=axes[0], fraction=0.046, pad=0.04)

        im_mean = axes[1].imshow(final_mean, origin='lower', cmap='jet')
        axes[1].set_title("Estimated Mean"); plt.colorbar(im_mean, ax=axes[1], fraction=0.046, pad=0.04)

        im_var = axes[2].imshow(final_var, origin='lower', cmap='jet')
        axes[2].set_title("Estimated Variance"); plt.colorbar(im_var, ax=axes[2], fraction=0.046, pad=0.04)

        pv = np.array(traj_ugv)
        for ax in axes:
            ax.plot(pv[:, 1], pv[:, 0], '-x', color='black', markersize=4, label='UGV Path')
        axes[0].legend(loc='upper right')

        plt.tight_layout()
        plt.show(block=True)

    # ★ J の推移（理論線との比較）
    if ENABLE_HISTORY_PLOT:
        t = np.arange(len(J_history))
        J0 = J_history[0] if len(J_history) > 0 else 0.0
        gamma = uav.gamma
        theory = J0 - gamma * t

        plt.figure(figsize=(6,4))
        plt.plot(t, J_history, '-o', markersize=3, label='実測 J')
        plt.plot(t, theory, '--', label=r'$J_0 - \gamma t$')
        plt.xlabel('Step'); plt.ylabel('J (総分散)')
        plt.title('Objective $J$ の推移と理論直線')
        plt.grid(True); plt.legend()
        plt.show(block=True)



if __name__ == "__main__":
    main()
