import math
import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import pandas as pd
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe

# ─── CBF QP Solver ─────────────────────────────────────────────


class solver():
    def __init__(self):
        self.alpha = 1.0
        self.cbf = None
        self.slack = 0.0
        self.cbf_list = []
        self.slack_list = []
        self.P_co = 1

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        self.cbf_list.append(np.array([bJ, dbJ_du_x, dbJ_du_y]))
        self.slack_list.append(slack)

    def add_cbfs(self, cbfs: List[Tuple[float, float, float]]):
        for cbf in cbfs:
            cbf_bJ = cbf[0]
            cbf_grad_x = cbf[1]
            cbf_grad_y = cbf[2]
            slack = cbf[3]
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
        # cbfなければ nominal を返す
        if m == 0:
            return nominal_input.copy()

        # 3) G, h の構築
        G = np.zeros((m, dim))
        h = np.zeros((m, 1))
        for i, ((bJ, gx, gy), slack_coef) in enumerate(zip(self.cbf_list, self.slack_list)):
            G[i, 0] = -gx
            G[i, 1] = -gy
            G[i, 2 + i] = slack_coef   # スラック変数 s_i の係数
            h[i,0] = bJ

        P = np.zeros((dim, dim))
        P[0, 0] = 2*self.P_co
        P[1, 1] = 2*self.P_co
        for i in range(m):
            P[2+i, 2+i] = 2*1  # self.gad_co

        q = np.zeros(dim)
        q[0:2] = -2 * nominal_input

        # 5) QP を解く
        sol = solve_qp(P, q, G, h, solver="quadprog")
        if sol is None:
            # 解が見つからなければ nominal
            print(f"ノミナルです")
            return nominal_input.copy()
        # 6) 先頭2要素(ν_x,ν_y) を返す
        return sol[:2]


# ─── Sparse Online Gaussian Process (論文準拠実装) ─────────────


def rbf_kernel(x, y, rbf_sigma=2.0):
    return np.exp(-np.linalg.norm(x-y)**2/(2*rbf_sigma**2))


class SparseOnlineGP:
    def __init__(self, sigma0: float, kernel=rbf_kernel, max_basis: int = None, delta: float = 0.1):
        self.sigma0 = sigma0         # 観測ノイズ分散
        self.kernel = kernel
        self.max_basis = max_basis      # 基底の最大数
        self.delta = delta              # ノベリティ閾値 ω
        self.count_case1 = 0
        self.count_case2 = 0
        self.count_case3 = 0
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

        # ―― 2) q_t, r_t の計算 (Eq.2.20–2.21) ――
        denom = var_star + self.sigma0**2
        q_t = (y - f_star) / denom
        r_t = -1.0 / denom

        # ―― 3) ノベリティ h_t の計算 (Eq.2.22) ――
        h_t = k_tt - k_vec.dot(self.Q.dot(k_vec))

        n = self.X.shape[0]

        if h_t < self.delta: # 基底数が上限到達 → h_t でさらに分岐
                # --- ケース２: discard branch (2.24) だけ更新 ---
                ehat = self.Q.dot(k_vec)
                s_short = self.C.dot(k_vec) + ehat     # Eq.(2.24)
                self.a += q_t * s_short
                self.C += r_t * np.outer(s_short, s_short)
                # self.Q, self.X はそのまま
                self.count_case2 += 1
                print(f"[SOGP] ケース2 更新(似た情報なので上書き！)：既存基底を更新のみ (h_t={h_t:.4f} < δ={self.delta})")

        # --- 【ケース１】基底数に余裕あり → 常に拡張 branch (2.17) ---
        else:
            if self.max_basis is None or n < self.max_basis:
                # 2.17 の a,C,Q 拡張更新
                s_t = np.concatenate([self.C.dot(k_vec), [1.0]])
                # a, C の拡張
                a_ext = np.concatenate([self.a, [0.0]])
                C_ext = np.pad(self.C, ((0, 1), (0, 1)), 'constant')
                self.a = a_ext + q_t * s_t
                self.C = C_ext + r_t * np.outer(s_t, s_t)
                # Q の拡張（2.23）
                ehat = self.Q.dot(k_vec)
                ehat_full = np.concatenate([ehat, [0.0]])
                efull = np.zeros(n+1)
                efull[-1] = 1.0
                Q_ext = np.pad(self.Q, ((0, 1), (0, 1)), 'constant')
                self.Q = Q_ext + (1.0/h_t)*np.outer(ehat_full-efull, ehat_full-efull)
                # X に追加
                self.X = np.vstack([self.X, x])
                self.count_case1 += 1
                print(f"[SOGP] ケース1 拡張(新しい情報がきた！)：新しい基底を追加 (現在 {self.X.shape[0]+1} 基底)")

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
                self.count_case3 += 1
                print(f"[SOGP] ケース3 削除(新しい情報を入れるため古いもの削除！)：拡張 + prune (h_t={h_t:.4f} ≥ δ, 基底上限 {self.max_basis})")

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
    pos に最も近い格子点 (i0,j0) を中心に、81点 (中心＋周囲8点) について
      1) その点を中心とした 9x9 の平均値 val
      2) val + N(0, noise_std^2) を [0,1] にクリップ
    を計算し [(位置ベクトル, noisy_value), ...] のリストで返す。
    """
    i0, j0 = int(round(pos[0])), int(round(pos[1]))
    H, W = true_map.shape
    observations = []

    for di in (-4, 0, 4):
        for dj in (-4, 0, 4):
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
            # noisy = float(np.clip(noisy, 0.0, 1.0))

            observations.append((np.array([i, j], dtype=float), noisy))

    return observations


def generate_ground_truth_map(grid_size=20):
    """
    grid_size に対して相対指定で矩形を配置し、必ず範囲内に収まる真値マップを生成
    """
    gt = np.zeros((grid_size, grid_size))
    H = W = grid_size

    def rect(y0, y1, x0, x1, val=1.0):
        # y,x は [0,1] の割合で与える
        i0 = int(max(0, min(H, round(y0 * H))))
        i1 = int(max(0, min(H, round(y1 * H))))
        j0 = int(max(0, min(W, round(x0 * W))))
        j1 = int(max(0, min(W, round(x1 * W))))
        if i1 > i0 and j1 > j0:
            gt[i0:i1, j0:j1] = val

    # 好きな形をいくつか（行=Y, 列=Xに注意）
    rect(0.30, 0.80, 0.06, 0.46, 1.0)  # 大きめの帯
    rect(0.20, 0.70, 0.60, 0.92, 1.0)  # 右側縦長
    rect(0.80, 0.95, 0.70, 0.90, 1.0)  # 右下
    rect(0.10, 0.40, 0.10, 0.40, 1.0)  # 左上

    return gt


def mask_to_rgba(mask: np.ndarray, color: str, alpha: float = 0.18) -> np.ndarray:
    """
    True の画素にだけ color(+alpha) を割り当てた RGBA 画像を返す
    """
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    r, g, b, _ = to_rgba(color, alpha)  # alphaは引数の値を使う
    rgba[mask] = [r, g, b, alpha]
    return rgba

def compute_voronoi_masks(positions: list[np.ndarray], H: int, W: int) -> list[np.ndarray]:
    """
    positions: 各UAVの現在位置 [ [y,x], ... ]（float可）
    H, W     : グリッドの縦横サイズ（行=Y, 列=X）
    return   : 各UAV担当のブールマスク (H, W) のリスト
    """
    K = len(positions)
    masks = [np.zeros((H, W), dtype=bool) for _ in range(K)]
    if K == 0:
        return masks

    # 格子座標 (H,W,2) を作る。[..., 0]=y, [...,1]=x
    grid_y, grid_x = np.mgrid[0:H, 0:W]
    grid = np.stack([grid_y, grid_x], axis=-1).astype(float)  # (H,W,2)

    # 各UAVまでの距離スタック (H,W,K)
    dists = []
    for p in positions:
        p2 = np.asarray(p, dtype=float).reshape(1, 1, 2)  # [y,x]
        d = np.linalg.norm(grid - p2, axis=-1)
        dists.append(d)
    dists = np.stack(dists, axis=-1)  # (H,W,K)

    # 最も近いUAVのインデックス
    owner = np.argmin(dists, axis=-1)  # (H,W)

    for k in range(K):
        masks[k] = (owner == k)
    return masks


# ─── Field Limitation ──────────────────────────────────────────────


class FieldLimitation:
    def __init__(self, pos, alpha, grid_size: int):
        self.field_x_min = 0
        self.field_x_max = grid_size-1
        self.field_y_min = 0
        self.field_y_max = grid_size-1
        self.pos = pos
        self.alpha = alpha
        x_cen = (self.field_x_min + self.field_x_max) / 2
        y_cen = (self.field_y_min + self.field_y_max) / 2
        self.center = np.array([x_cen, y_cen], dtype=float)
        x_radius = (self.field_x_max - self.field_x_min) / 2
        y_radius = (self.field_y_max - self.field_y_min) / 2
        self.radius = np.array([x_radius, y_radius], dtype=float)

    def calc_cbf(self):
        L4_norm = np.power(np.sum(((self.center - self.pos) / self.radius)**4), 1/4)
        return self.alpha*(1-L4_norm)

    def calc_grad(self):
        grad = 4*((self.center - self.pos)**3)/(self.radius**4)
        return grad

# ─── UGV Controller──────────────────────────────────────────────


class UGVController:
    def __init__(self, grid_size: int = 20, reward_type: int = 0, discount_factor: float = 0.95):
        self.grid_size = grid_size
        self.reward_type = reward_type
        self.position = np.array([grid_size // 2, grid_size // 2], dtype=int)
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited[self.position[0], self.position[1]] = True
        self.expectation_map = None
        self.variance_map = None
        self.discount_factor = discount_factor  # 割引率

    def set_maps(self, expectation_map: np.ndarray, variance_map: np.ndarray):
        """
        UAV側から渡されたマップを保存しておくためのメソッド
        """
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
                        depth: int, visited: np.ndarray, step: int,
                        allowed_mask: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
        actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        k1, k2, k3, k4, k5, k6, epsilon = 2, 2, 0.1, 0.1, 1, 1, 1e-3
        best_reward = -np.inf
        best_move = pos.copy()

        def is_allowed(cell):
            if allowed_mask is None:
                return True
            i, j = cell
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
                    depth-1, new_visited, step+1, allowed_mask=allowed_mask
                )
                total += self.discount_factor * fut

            if total > best_reward:
                best_reward = total
                best_move = nxt.copy()
        return best_move, best_reward

    # 置換: UGVController.calc に allowed_mask を追加
    def calc(self, expectation_map: np.ndarray, variance_map: np.ndarray,
            depth: int = 10, step: int = 1, allowed_mask: np.ndarray | None = None):
        new_pos, _ = self._recursive_search(
            self.position, expectation_map, variance_map,
            depth, self.visited.copy(), step, allowed_mask=allowed_mask
        )
        if not np.array_equal(new_pos, self.position):
            self.position = new_pos
            self.visited[new_pos[0], new_pos[1]] = True

    # 置換: UGVController.get_planned_path に allowed_mask を追加
    def get_planned_path(self, expectation_map: np.ndarray, variance_map: np.ndarray,
                        depth: int = 10, allowed_mask: np.ndarray | None = None) -> List[np.ndarray]:
        path = []
        pos = self.position.copy()
        visited = self.visited.copy()

        def is_allowed(cell):
            if allowed_mask is None:
                return True
            i, j = cell
            return bool(allowed_mask[i, j])

        for t in range(depth):
            # allowed_mask を使って1手先を探索
            nxt, _ = self._recursive_search(
                pos, expectation_map, variance_map,
                depth-t, visited, t+1, allowed_mask=allowed_mask
            )
            # 念のため許可外に出そうなら据え置き
            if not is_allowed(nxt):
                nxt = pos.copy()

            path.append(nxt.copy())
            visited[nxt[0], nxt[1]] = True
            pos = nxt
        return path

# 追加: UGVFleet クラス
class UGVFleet:
    def __init__(self, ugvs: list[UGVController]):
        self.ugvs = ugvs
        self.planned_paths: list[list[np.ndarray]] = []
        self.voronoi_masks: list[np.ndarray] = []

    def positions(self) -> list[np.ndarray]:
        return [u.position.copy() for u in self.ugvs]

    def compute_voronoi(self, H: int, W: int):
        pos = self.positions()
        self.voronoi_masks = compute_voronoi_masks(pos, H, W)

    def plan_all(self, mean_map: np.ndarray, var_map: np.ndarray, depth: int):
        H, W = mean_map.shape
        if not self.voronoi_masks or len(self.voronoi_masks) != len(self.ugvs):
            self.compute_voronoi(H, W)

        self.planned_paths = []
        for u, mask in zip(self.ugvs, self.voronoi_masks):
            path = u.get_planned_path(mean_map, var_map, depth=depth, allowed_mask=mask)
            self.planned_paths.append(path)

    def step_all(self, mean_map: np.ndarray, var_map: np.ndarray, depth: int, step: int):
        # 計画に基づいて各UGVを一歩進める
        for u, mask in zip(self.ugvs, self.voronoi_masks):
            u.calc(mean_map, var_map, depth=depth, step=step, allowed_mask=mask)

    def target_cell_for_uav(self, uav_pos: np.ndarray, step_offset: int) -> tuple[int, int]:
        """
        UAVの位置が属するUGV Voronoi を調べ、そのUGVの計画パスの step_offset の点を返す。
        無ければ最近傍UGVを使う。
        """
        # 所属判定
        for mask, path in zip(self.voronoi_masks, self.planned_paths):
            i, j = int(round(uav_pos[0])), int(round(uav_pos[1]))
            H, W = mask.shape
            if 0 <= i < H and 0 <= j < W and mask[i, j]:
                idx = min(max(step_offset - 1, 0), len(path) - 1) if path else 0
                cell = path[idx] if path else self.ugvs[0].position
                return int(cell[0]), int(cell[1])

        # 最近傍UGVにフォールバック
        dists = [np.linalg.norm(uav_pos - u.position) for u in self.ugvs]
        k = int(np.argmin(dists))
        path = self.planned_paths[k] if k < len(self.planned_paths) else []
        idx = min(max(step_offset - 1, 0), len(path) - 1) if path else 0
        cell = path[idx] if path else self.ugvs[k].position
        return int(cell[0]), int(cell[1])



# ─── UAV Controller ──────────────────────────────────────────


class UAVController:
    def __init__(self, train_data_x, train_data_y: np.ndarray, grid_size,
                 ugv_fleet: "UGVFleet", step_of_ugv_path_used=8, ugv_future_path_sigma=5.0, suenaga=False, use_j_gradient_cbf=True, use_voronoi=True,
                 map_publish_period: float = 0.5, d0=1, suenaga_discount_rate=0.95, suenaga_path_gene_depth=5, unom_gain=2.0, suenaga_gain=2.0, cbf_j_alpha=1.0, cbf_j_gamma=3.0, shared_gp: SparseOnlineGP | None = None, uav_id=0, gp_sensing_noise_sigma0=0.4, gp_max_basis=27, gp_threshold_delta=0.1, rbf_sigma=2.0):
        ...
        self.r = 1.0
        self.alpha = cbf_j_alpha
        self.gamma = cbf_j_gamma
        self.k = unom_gain  # unomのゲイン
        self.k_pp= suenaga_gain  # suenagaのゲイン
        self.pos = train_data_x[0].copy()  # ← floatで初期化
        self.control_period = 0.1
        self.gp = shared_gp if shared_gp is not None else SparseOnlineGP(sigma0=gp_sensing_noise_sigma0, kernel=lambda x,y,s=rbf_sigma: rbf_kernel(x,y,s), max_basis=gp_max_basis, delta=gp_threshold_delta)
        self.uav_id=uav_id
        self.grid_size = grid_size
        self.v = np.zeros(2)
        self.step_of_ugv_path_used = step_of_ugv_path_used
        self.suenaga = suenaga
        self.use_voronoi = use_voronoi
        self.map_publish_period = map_publish_period
        self.ugv_fleet = ugv_fleet
        self.rbf_sigma=rbf_sigma
        self.use_j_gradient_cbf=use_j_gradient_cbf
        self.d0=d0
        self.suenega_discount_rate=suenaga_discount_rate
        self.suenega_path_gene_depth=suenaga_path_gene_depth
        self.ugv_future_path_sigma=ugv_future_path_sigma

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
        r2 = self.rbf_sigma**2  # rbfカーネルのパラメータと揃える
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

    def path_generation_for_uav_suenaga(self, var_map, rho=0.95, depth=8, use_voronoi=False):
        H, W = var_map.shape

        if use_voronoi and hasattr(self, "_voronoi_mask") and self._voronoi_mask is not None:
            allowed = np.argwhere(self._voronoi_mask)
            cells = [tuple(x) for x in allowed]                 # ★ マスク内セルだけ
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
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    c2 = (ni, nj)
                    # ★ 遷移先も allowed に限定
                    if in_allowed(c2):
                        nbrs.append(c2)
            return nbrs

        # 価値反復は現状どおり…
        for _ in range(depth):
            V_cur = {}
            for c in cells:
                best, best_n = -1e9, c
                for c2 in neighbors(c):
                    dist = np.hypot(c[0]-c2[0], c[1]-c2[1])
                    if dist < 1e-6:
                        continue
                    R = var_map[c2] / dist
                    val = R + rho * V_prev[c2]
                    if val > best:
                        best, best_n = val, c2
                V_cur[c] = best
                policy[c] = best_n
            V_prev = V_cur

        # ★ 現在セルが allowed 外なら最寄り allowed へ射影してから方策適用
        ci = (int(round(self.pos[0])), int(round(self.pos[1])))
        if not in_allowed(ci):
            # 最寄り allowed cell を1つ探す
            if use_voronoi and len(cells) > 0:
                arr = np.array(cells)
                d = np.hypot(arr[:,0]-ci[0], arr[:,1]-ci[1])
                ci = tuple(arr[np.argmin(d)])
            else:
                # allowed の定義がない場合はそのまま
                pass

        next_cell = policy.get(ci, ci)
        return np.array([next_cell[0], next_cell[1]], dtype=float)

    def path_generation_for_uav(self, var_map, d0: float = 1.0, use_voronoi=False):
        """
        分散 var_map と現在位置 self.pos から、
        score = variance / distance を最大にするセルそのものを返す版。
        近傍ステップはせず、グリッド上の best cell をそのまま次の目標にする。
        """
        H, W = var_map.shape

        # ---- 現在セル（グリッドインデックス） ----
        ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=int)
        ci[0] = np.clip(ci[0], 0, H - 1)
        ci[1] = np.clip(ci[1], 0, W - 1)

        # ---- Voronoi マスクの処理（担当領域があれば現在位置をスナップ） ----
        allowed_mask = None
        if use_voronoi and hasattr(self, "_voronoi_mask") and self._voronoi_mask is not None:
            allowed_mask = self._voronoi_mask.astype(bool)

            # もし担当領域に1つもセルがない場合 → 動きようがないので今のセルを返す
            if not np.any(allowed_mask):
                return ci.astype(float)

            # 現在位置が担当外なら、一番近い allowed にスナップ
            if not allowed_mask[ci[0], ci[1]]:
                idxs = np.argwhere(allowed_mask)
                d = np.hypot(idxs[:, 0] - ci[0], idxs[:, 1] - ci[1])
                ci = idxs[np.argmin(d)]

        # ---- 全マスの距離とスコアを計算 ----
        I, J = np.indices((H, W))                # I[i,j] = i, J[i,j] = j
        dist = np.hypot(I - ci[0], J - ci[1])    # 各セルまでの距離


        score = var_map / (dist**2+d0**2)             # 近くて分散が高いほど高スコア

        # ---- Voronoi がある場合は担当領域外のスコアを無効化 ----
        if allowed_mask is not None:
            score[~allowed_mask] = -np.inf

        # ---- スコアが有限なセルがなければ、その場にとどまる ----
        if not np.isfinite(score).any():
            return ci.astype(float)

        # ---- score 最大のセルをそのまま次の目標にする ----
        ti, tj = np.unravel_index(np.nanargmax(score), score.shape)
        next_cell = np.array([ti, tj], dtype=float)

        return next_cell

    def path_generation_for_high_variance_point_including_effect_of_ugv(
        self,
        var_map: np.ndarray,
        ugv_future_point: np.ndarray,
        use_voronoi: bool = True,
        d0: float = 1.0,
        ell: float = 5.0,
    ) -> np.ndarray:
        """
        J_var_ugv(q) = (sigma(q)^2 / (||p_i - q||^2 + d0^2))
                        * exp( - ||q - ugv_future_point||^2 / (2 * ell^2) )
        を最大にするセル q ∈ Voronoi_i を waypoint として返す。

        ※ 数式では argmin と書いていたが、上式は「大きいほど好ましいセル」
           なので実装は argmax にしている。argmin にしたい場合は -J を使えばよい。
        """
        H, W = var_map.shape

        # ---- 現在セル（グリッドインデックス） ----
        ci = np.array([int(round(self.pos[0])), int(round(self.pos[1]))], dtype=int)
        ci[0] = np.clip(ci[0], 0, H - 1)
        ci[1] = np.clip(ci[1], 0, W - 1)

        # ---- Voronoi マスク（自分の担当領域） ----
        allowed_mask = None
        if use_voronoi and hasattr(self, "_voronoi_mask") and self._voronoi_mask is not None:
            allowed_mask = self._voronoi_mask.astype(bool)

            # 担当領域が 1 つもない場合 → その場に留まる
            if not np.any(allowed_mask):
                return ci.astype(float)

            # 現在位置が担当外なら、一番近い allowed にスナップ
            if not allowed_mask[ci[0], ci[1]]:
                idxs = np.argwhere(allowed_mask)
                d = np.hypot(idxs[:, 0] - ci[0], idxs[:, 1] - ci[1])
                ci = idxs[np.argmin(d)]

        # ---- 全マスの距離と J_var_ugv_path を計算 ----
        I, J = np.indices((H, W))  # I[i,j] = i (y), J[i,j] = j (x)

        # UAV から各セルまでの距離²
        dist2_uav = (I - ci[0])**2 + (J - ci[1])**2
        # 自分自身で 0 割りしないように
        # （実質、距離0 のときは d0 だけで決まる）
        # → denominator = dist2_uav + d0^2 なのでこのままでも OK

        # UGV 未来点から各セルまでの距離²
        ugv_i = float(ugv_future_point[0])
        ugv_j = float(ugv_future_point[1])
        dist2_ugv = (I - ugv_i)**2 + (J - ugv_j)**2

        # J_var_ugv(q) = (sigma^2 / (dist^2 + d0^2)) * exp(-dist_ugv^2 / (2 ell^2))
        denom = dist2_uav + d0**2
        sigma2 = var_map  # var_map が σ^2(q)
        J_var_ugv_path = (sigma2 / denom) * np.exp(-dist2_ugv / (2.0 * ell**2))

        # Voronoi がある場合は担当外を無効化
        if allowed_mask is not None:
            J_var_ugv_path[~allowed_mask] = -np.inf

        # 有効なセルがなければ、その場にとどまる
        if not np.isfinite(J_var_ugv_path).any():
            return ci.astype(float)

        # J_var_ugv 最大のセルを waypoint に
        ti, tj = np.unravel_index(np.nanargmax(J_var_ugv_path), J_var_ugv_path.shape)
        next_cell = np.array([ti, tj], dtype=float)

        return next_cell


    def set_voronoi_mask(self, mask: np.ndarray):
        self._voronoi_mask = mask

    def _in_my_voronoi(self, cell: Tuple[int,int]) -> bool:
            if hasattr(self, "_voronoi_mask") and self._voronoi_mask is not None:
                i, j = cell
                return bool(self._voronoi_mask[i, j])
            return True  # デフォルトはTrue

    def calc(self, environment_function: Callable[[np.ndarray], float], v_limit=1.0):
        # --- 1) 観測 ---
        use_voronoi = self.use_voronoi
        self.update_map(environment_function)
        self.prob, self.sigma2, self.k_star, self.K = self.gp.predict(self.pos)
        print(f" predict → mean = {self.prob:.3f}, var = {self.sigma2:.3f}")
        self.calc_objective_function()

        # --- 2) UGV 用マップ（キャッシュ） ---
        self.update_maps_for_ugv()
        E_map, V_map = self.get_maps_for_ugv()

        # ★ UGV “艦隊”に配布は main 側でやるのでここでは返すだけ
        #   （または必要なら self.ugv_fleet.plan_all の引数に使われる）

        # --- 3) 参照するUGVの未来ターゲットを艦隊から取得 ---
        ugv_target_cell = self.ugv_fleet.target_cell_for_uav(
            self.pos, step_offset=self.step_of_ugv_path_used
        )

        # 自Voronoiに属しているか（“UAVの”Voronoiを使う既存ロジックはそのまま）
        ugv_in_my_voronoi = self._in_my_voronoi(ugv_target_cell)

        if self.use_j_gradient_cbf is False:
            if self.k == 0:
                v_nom = np.zeros(2)
            else:
                ugv_future_pos = np.array([ugv_target_cell[0], ugv_target_cell[1]], dtype=float)
                d0=self.d0
                ell=self.ugv_future_path_sigma
                waypoint=self.path_generation_for_high_variance_point_including_effect_of_ugv(V_map, ugv_future_pos, d0, ell, use_voronoi)
                self.current_waypoint = waypoint
                v_nom = - self.k_pp * (self.pos - waypoint)
                #nu_nom = (v_nom - self.v) / self.control_period
                # --- 4) QP ---
                self.solver = solver()
                #self.solver.add_cbfs([cbf_feildlimiation])
                u = self.solver.solve(v_nom)

            # --- 5) 状態更新 ---
            self.v += u * self.control_period
            if np.linalg.norm(self.v) >= v_limit:
                self.v = v_limit * self.v / np.linalg.norm(self.v)
            self.pos += self.v * self.control_period

            # --- 6) クランプ＆ログ ---
            H, W = self.grid_size, self.grid_size
            self.pos[0] = np.clip(self.pos[0], 0, H-1)
            self.pos[1] = np.clip(self.pos[1], 0, W-1)
            print(f"velocity = {np.linalg.norm(self.v)}")

        else:
            if self.suenaga:
                # ⇒ suenaga（WayPoint追従）へスイッチ
                #    自Voronoi外は探索させたい想定なので use_voronoi=True を推奨
                d0=self.d0
                rho=self.suenega_discount_rate
                depth=self.suenega_path_gene_depth
                waypoint = self.path_generation_for_uav_suenaga(V_map, rho, depth, use_voronoi)
                v_nom = -self.k_pp * (self.pos - waypoint)
                nu_nom = (v_nom - self.v) / self.control_period
                self.current_waypoint = waypoint

            else:
                if ugv_in_my_voronoi:
                    if self.k == 0:
                        nu_nom = np.zeros(2)
                    else:
                        ugv_future_pos = np.array([ugv_target_cell[0], ugv_target_cell[1]], dtype=float)
                        v_nom = - self.k * (self.pos - ugv_future_pos)
                        nu_nom = (v_nom - self.v) / self.control_period
                    if hasattr(self, 'current_waypoint'):
                        delattr(self, 'current_waypoint')
                else:
                    # ← ここが抜けると未代入だった（探索のフォールバックを必ず入れる）
                    waypoint = self.path_generation_for_uav(V_map, use_voronoi=True)
                    v_nom = - self.k_pp * (self.pos - waypoint)
                    nu_nom = (v_nom - self.v) / self.control_period
                    self.current_waypoint = waypoint

            # --- 4) CBF/QP ---
            xi_J1, xi_J2 = self.calc_cbf_terms(self.v, gamma=self.gamma, alpha=self.alpha)
            cbf_J = [-xi_J2, -xi_J1[0], -xi_J1[1], 0.0001]
            self.solver = solver()
            self.solver.add_cbfs([cbf_J])
            nu = self.solver.solve(nu_nom)

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
def main(visualize: bool = True):
    # ===== パラメータ =====
    #全体
    grid_size  = 50
    noise_std  = 0.5
    num_uavs   = 3
    num_ugvs   = 2            # ★ 複数UGV
    steps      = 100
    map_publish_preriod = 0.5
    d0=1.0
    use_voronoi = True

    #UGV
    ugv_depth  = 8
    reward_type=3
    discount_factor=0.95

    #UAV
    v_limit    = 25.0
    step_of_ugv_path_used = 6
    suenaga_on = True
    suenaga_discount_rate=0.95
    suenaga_path_gene_rate=5
    use_j_gradient_cbf = False
    ugv_future_path_sigma=5.0
    unom_gain=2.0
    suenaga_gain=2.0
    cbf_j_alpha=1.0
    cbf_j_gamma=3.0

    # GP
    gp_sensing_noise_sigma0=0.4
    gp_max_basis=100
    gp_threshold_delta=0.05
    rbf_sigma=2.0

    #visualize
    visualize=True

    gt = generate_ground_truth_map(grid_size)

    # ===== UGV 複数生成 =====
    ugvs = []
    center = np.array([grid_size/2, grid_size/2], dtype=float)
    Rg = grid_size/6
    thetas_g = np.linspace(0, 2*np.pi, num_ugvs, endpoint=False)
    for k in range(num_ugvs):
        ugv = UGVController(grid_size, reward_type=reward_type, discount_factor=discount_factor)
        p0 = center + Rg * np.array([np.cos(thetas_g[k]), np.sin(thetas_g[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size-1)
        ugv.position = p0.astype(int)  # UGVはセル座標（int）
        ugv.visited[ugv.position[0], ugv.position[1]] = True
        ugvs.append(ugv)

    ugv_fleet = UGVFleet(ugvs)

    # ===== UAV 群 生成 =====
    Ru = grid_size/4
    thetas = np.linspace(0, 2*np.pi, num_uavs, endpoint=False)
    uavs = []
    for k in range(num_uavs):
        p0 = center + Ru * np.array([np.cos(thetas[k]), np.sin(thetas[k])], dtype=float)
        p0 = np.clip(p0, 0, grid_size-1)

        init_obs = environment_function(p0, gt, noise_std=noise_std)
        init_x = np.vstack([p for p, _ in init_obs])
        init_y = np.array([y for _, y in init_obs])

        uav = UAVController(
            train_data_x=init_x,
            train_data_y=init_y,
            grid_size=grid_size,
            ugv_fleet=ugv_fleet,
            step_of_ugv_path_used=step_of_ugv_path_used,
            ugv_future_path_sigma=ugv_future_path_sigma,
            suenaga=suenaga_on,
            use_j_gradient_cbf=use_j_gradient_cbf,
            use_voronoi=use_voronoi,
            map_publish_period=map_publish_preriod,
            unom_gain=unom_gain,
            d0=d0,
            suenaga_discount_rate=suenaga_discount_rate,
            suenaga_path_gene_depth=suenaga_path_gene_rate,
            suenaga_gain=suenaga_gain,
            cbf_j_alpha=cbf_j_alpha,
            cbf_j_gamma=cbf_j_gamma,
            gp_sensing_noise_sigma0=gp_sensing_noise_sigma0,
            gp_max_basis=gp_max_basis,
            gp_threshold_delta=gp_threshold_delta,
            rbf_sigma=rbf_sigma
        )
        uav.pos = p0.astype(float)
        uavs.append(uav)

    # ===== 可視化セットアップ =====
    colors = ['r','c','m','y','g','b']
    ugv_colors = ['k', '#444444', '#111111', '#888888']

    trajs_uav = [[u.pos.copy()] for u in uavs]
    trajs_ugv = [[u.position.copy()] for u in ugvs]

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].contour(gt, levels=[0.5], colors='white', linewidths=2, origin='lower', zorder=12)
        im_mean = ax[0].imshow(np.zeros((grid_size, grid_size)),
                               vmin=0, vmax=1, cmap='jet', origin='lower')
        im_var  = ax[1].imshow(np.zeros((grid_size, grid_size)),
                               vmin=0, vmax=1, cmap='jet', origin='lower')
        cb0 = fig.colorbar(im_mean, ax=ax[0], fraction=0.046, pad=0.04); cb0.set_label('Estimated mean (fused)')
        cb1 = fig.colorbar(im_var,  ax=ax[1], fraction=0.046, pad=0.04); cb1.set_label('Estimated variance (fused)')

        uav_dots  = [ax[0].plot([], [], 'o', color=colors[i % len(colors)],
                                 label=f'UAV{i}', zorder=12+i)[0] for i in range(num_uavs)]
        uav_lines = [ax[0].plot([], [], '-', color=colors[i % len(colors)],
                                 markersize=3, zorder=6+i)[0] for i in range(num_uavs)]

        ugv_lines = [ax[0].plot([], [], '-x', color=ugv_colors[i % len(ugv_colors)],
                                markersize=3, label=f'UGV{i} Path', zorder=20+i)[0]
                     for i in range(num_ugvs)]
        ugv_dots  = [ax[0].plot([], [], 'wo', markeredgecolor='k',
                                label=f'UGV{i}', zorder=21+i)[0]
                     for i in range(num_ugvs)]

        waypoint_dots = [ ax[0].plot([], [], 's', color=colors[i % len(colors)],
                                     markersize=6, label=f'UAV{i} WP', zorder=30+i)[0]
                          for i in range(num_uavs) ]

        ugv_plan_lines = []
        ugv_plan_targets = []
        for i in range(num_ugvs):
            line, = ax[0].plot([], [], '--', alpha=1.0, linewidth=2.5,
                               label=f'UGV{i} planned', zorder=25+i)
            line.set_color('#FFFF00')
            line.set_path_effects([pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()])
            ugv_plan_lines.append(line)

            tgt, = ax[0].plot([], [], 'o', mfc='#FFFF00', mec='black',
                              markersize=9, label=f'UGV{i} target', zorder=26+i)
            ugv_plan_targets.append(tgt)

        empty_mask = np.zeros((grid_size, grid_size), dtype=bool)
        vor_layers = [ax[0].imshow(mask_to_rgba(empty_mask, colors[i % len(colors)], alpha=0.18),
                                   origin='lower', zorder=9) for i in range(num_uavs)]
        vor_cnt_lines = [None for _ in range(num_uavs)]

        ax[0].legend(loc='upper right')
        ax[0].set_xlim(0, grid_size-1)
        ax[0].set_ylim(0, grid_size-1)
        ax[0].autoscale(False)
    else:
        # 可視化しないとき用のダミー（参照されない）
        fig = ax = im_mean = im_var = None
        uav_dots = uav_lines = ugv_lines = ugv_dots = waypoint_dots = []
        ugv_plan_lines = ugv_plan_targets = []
        vor_layers = []
        vor_cnt_lines = []

    # ===== ループ =====
    J_history = []
    true_sum_history = []

    for step in range(steps):
        print(f"=== Step {step} ===")
        # --- (A) UAVのVoronoi 更新（計算は常に行う） ---
        positions_now_uav = [u.pos.copy() for u in uavs]
        masks_uav = compute_voronoi_masks(positions_now_uav, grid_size, grid_size)

        for i, m in enumerate(masks_uav):
            uavs[i].set_voronoi_mask(m)

        # 可視化だけ別途
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

        # --- (B) UAV を更新（観測→キャッシュ更新まで） ---
        for uav in uavs:
            env_fn = (lambda p, _gt=gt: environment_function(p, _gt, noise_std))
            # 本当はここで観測させたいなら uav.update_map(env_fn) だけ呼ぶ、などに分離してもOK

        # --- (C) 融合マップ作成 ---
        mean_maps = []
        var_maps  = []
        for uav in uavs:
            m_map_i, v_map_i = uav.get_maps_for_ugv()
            mean_maps.append(m_map_i)
            var_maps.append(v_map_i)
        fused_mean = np.mean(np.stack(mean_maps, axis=0), axis=0)
        fused_var  = np.mean(np.stack(var_maps,  axis=0), axis=0)

        # --- (D) UGV Voronoi（UGV間）計算 → 全台プラン ---
        ugv_fleet.compute_voronoi(grid_size, grid_size)
        ugv_fleet.plan_all(fused_mean, fused_var, depth=ugv_depth)

        # UGV計画の可視化
        if visualize:
            for i, path in enumerate(ugv_fleet.planned_paths):
                if path:
                    path_arr = np.array(path)
                    plan_y = path_arr[:, 0]
                    plan_x = path_arr[:, 1]
                    ugv_plan_lines[i].set_data(plan_x, plan_y)

                    idx = min(len(path)-1, max(0, 0))
                    tgt = path[idx]
                    ugv_plan_targets[i].set_data([tgt[1]], [tgt[0]])
                else:
                    ugv_plan_lines[i].set_data([], [])
                    ugv_plan_targets[i].set_data([], [])

        # --- (E) UAV 制御 ---
        for uav in uavs:
            env_fn = (lambda p, _gt=gt: environment_function(p, _gt, noise_std))
            uav.calc(env_fn, v_limit=v_limit)

        # --- (F) UGV を一歩進める ---
        ugv_fleet.step_all(fused_mean, fused_var, depth=ugv_depth, step=step+1)

        # --- (G) ログ＆可視化更新 ---
        J = np.sum(fused_var)
        J_history.append(J)

        total_crop = 0.0
        for u in ugvs:
            total_crop += np.sum(gt[u.visited])
        true_sum_history.append(total_crop)

        if visualize:
            im_mean.set_data(fused_mean)
            im_var.set_data(fused_var)

            # UAVの軌跡
            for i, uav in enumerate(uavs):
                trajs_uav[i].append(uav.pos.copy())
                pu = np.array(trajs_uav[i])
                uav_lines[i].set_data(pu[:, 1], pu[:, 0])
                uav_dots[i].set_data([uav.pos[1]], [uav.pos[0]])

                if hasattr(uav, 'current_waypoint'):
                    wp = uav.current_waypoint
                    waypoint_dots[i].set_data(wp[1], wp[0])

            # UGVの軌跡
            for i, ugv in enumerate(ugvs):
                trajs_ugv[i].append(ugv.position.copy())
                pv = np.array(trajs_ugv[i])
                ugv_lines[i].set_data(pv[:, 1], pv[:, 0])
                ugv_dots[i].set_data([ugv.position[1]], [ugv.position[0]])

            ax[0].set_title(f"Step {step} Mean (fused)")
            ax[1].set_title(f"Step {step} Var (fused)")
            fig.canvas.draw()
            plt.pause(0.1)

            print(f"J={J:.3f}, True crop sum={J:.3f} at step {step}")
            print(f"C_matrix={uavs[0].gp.C}, #basis={len(uavs[0].gp.X)}")


    if visualize:
        plt.ioff()
        plt.close(fig)

    # ===== 結果保存など（可視化OFFでもやる） =====
    visited_union = np.zeros_like(gt, dtype=bool)
    for u in ugvs:
        visited_union |= u.visited
    total_crop = np.sum(gt[visited_union])
    print(f"UGVs が訪問した作物の合計（真値）: {total_crop:.3f}")

    # ステップごとの指標
    df = pd.DataFrame({
        'step': np.arange(len(J_history)),
        'J': J_history,
        'true_crop_sum': true_sum_history
    })
    df.to_csv('multi_uav_multi_ugv_results_suenaga.csv', index=False)

    # パラメータ記録
    gp0 = uavs[0].gp
    params_info = {
        # 全体
        'grid_size': grid_size,
        'noise_std': noise_std,
        'num_uavs': num_uavs,
        'num_ugvs': num_ugvs,
        'steps': steps,
        'map_publish_period': map_publish_preriod,

        # UGV
        'ugv_depth': ugv_depth,
        'reward_type': reward_type,
        'discount_factor': discount_factor,

        # UAV
        'v_limit': v_limit,
        'step_of_ugv_path_used': step_of_ugv_path_used,
        'suenaga_on': suenaga_on,
        'suenaga_discount_rate': suenaga_discount_rate,
        'suenaga_path_gene_depth': suenaga_path_gene_rate,
        'use_j_gradient_cbf': use_j_gradient_cbf,
        'use_voronoi': use_voronoi,
        'd0': d0,
        'ugv_future_path_sigma': ugv_future_path_sigma,
        'unom_gain': unom_gain,
        'suenaga_gain': suenaga_gain,
        'cbf_j_alpha': cbf_j_alpha,
        'cbf_j_gamma': cbf_j_gamma,

        # GP
        'gp_sigma0': gp0.sigma0,
        'gp_max_basis': gp0.max_basis,
        'gp_delta': gp0.delta,
        'rbf_sigma': uavs[0].rbf_sigma,

        # SOGP のケースカウント
        'case1_count': gp0.count_case1,
        'case2_count': gp0.count_case2,
        'case3_count': gp0.count_case3,

        # 結果系
        'final_total_crop': float(total_crop),
        'final_J': float(J_history[-1]) if len(J_history) > 0 else np.nan,
    }

    param_df = pd.DataFrame([params_info])
    param_df.to_csv('multi_uav_multi_ugv_params_suenaga.csv', index=False)

    print("\n=== 実験パラメータを記録しました ===")
    for k, v in params_info.items():
        print(f"{k:25s}: {v}")
    print("results saved to multi_uav_multi_ugv_results_suenaga.csv")

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
            for i in range(num_ugvs):
                pv = np.array(trajs_ugv[i])
                ax_.plot(pv[:, 1], pv[:, 0], '-x', color='black', markersize=4,
                         label=f'UGV{i} Path', zorder=5+i)
            for i in range(num_uavs):
                pu = np.array(trajs_uav[i])
                ax_.plot(pu[:, 1], pu[:, 0], '-', color=colors[i % len(colors)], linewidth=1,
                         label=f'UAV{i} Path', zorder=4+i)
        axes[0].legend(loc='upper right')

        plt.tight_layout()
        plt.show(block=True)

        # ===== Jの推移と理論直線 =====
        t = np.arange(len(J_history))
        J0 = J_history[0]
        gamma = uavs[0].gamma

        theory = J0 - gamma * t
        plt.figure(figsize=(6, 4))
        plt.plot(t, J_history, '-o', markersize=3, label='実測 J (fused)')
        plt.plot(t, theory, '--', label=r'$J_0 - \gamma t$')
        plt.xlabel('Step')
        plt.ylabel('J (総分散)')
        plt.title('Objective $J$ の推移（融合）と理論直線')
        plt.grid(True)
        plt.legend()
        plt.show(block=True)


if __name__ == "__main__":
    main()
