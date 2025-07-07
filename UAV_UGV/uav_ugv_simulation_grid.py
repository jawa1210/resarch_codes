import math
import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# ─── CBF QP Solver ─────────────────────────────────────────────
class solver():
    def __init__(self):
        self.alpha = 1.0
        self.cbf = None
        self.slack = 0.0

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        self.cbf = np.array([bJ, dbJ_du_x, dbJ_du_y])
        self.slack = slack

    def solve_qp(self,nominal=None):
        if self.cbf is not None:
            bJ, gx, gy = self.cbf
            G = np.array([[-gx, -gy, self.slack]])
            h = np.array([ bJ ])
        else:
            G = np.zeros((1,3))
            h = np.zeros((1,1))

        if nominal is not None:
            self.nominal = nominal
            q=np.array([-2*self.nominal[0],-2*self.nominal[1],0])
        else:
            q = np.zeros(3)
        
        P = 2*np.eye(3)

        sol = solve_qp(P, q, G, h, solver="cvxopt")
        if sol is None:
            print("QP solver failed to find a solution.")
            return np.zeros(2)
        else:
            return sol[:2]

# ─── Sparse Online Gaussian Process (SOGP) ────────────────────
def rbf_kernel(x, y, sigma=3.0):
    return np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

class SparseOnlineGP:
    def __init__(self, sigma0: float, kernel=rbf_kernel, max_basis: int=None, delta: float=1e-4):
        self.sigma0 = sigma0         # 観測ノイズ分散の逆数の √1/… に相当
        self.kernel = kernel
        self.max_basis = max_basis   # 基底の最大数
        self.delta = delta           # 新サンプル追加のノベリティ閾値
        # 初期化：データなし
        self.X = np.zeros((0,2))     # basis inputs
        self.a = np.zeros((0,))      # coefficients a_t
        self.C = np.zeros((0,0))     # C_t 行列
        self.Q = np.zeros((0,0))     # Gram の逆 Q_t

    def init_first(self, x, y):
        # t=1 のとき
        k00 = self.kernel(x,x)
        denom = k00 + self.sigma0**2
        self.X = x.reshape(1,2)
        self.a = np.array([y/denom])
        self.C = np.array([[-1.0/denom]])
        self.Q = np.array([[1.0/k00]])

    def update(self, x: np.ndarray, y: float):
        """
        新しい観測 (x,y) をオンラインで追加 or 更新
        """
        if self.X.shape[0]==0:
            return self.init_first(x,y)

        #―― 1) 事前予測 f*, var_*――
        k_vec = np.array([self.kernel(xi,x) for xi in self.X])  # (N,)
        k_tt  = self.kernel(x,x)
        f_star = float(self.a.dot(k_vec))
        var_star = float(k_tt + k_vec.dot(self.C.dot(k_vec)))

        #―― 2) q_t, r_t の計算――
        denom = var_star + self.sigma0**2
        q_t = (y - f_star) / denom
        r_t = 1.0 / denom

        #―― 3) ノベリティ h_t の計算―― (Eq 2.22)
        h_t = k_tt - k_vec.dot(self.Q.dot(k_vec))

        #―― 4) st ベクトル――
        c_k = self.C.dot(k_vec)             # (N,)
        s_t = np.concatenate([c_k, [1.0]])  # (N+1,)

        #―― 5) 基底追加 or 既存更新――
        if h_t > self.delta:
            # ---- (a) 新基底として追加 ----
            N = self.X.shape[0]
            # a, C を Ut, Tt で拡張
            a_ext = np.concatenate([self.a, [0.0]])               # (N+1,)
            C_ext = np.pad(self.C, ((0,1),(0,1)), mode='constant')  # (N+1,N+1)
            # 再帰更新 (Eq 2.17)
            self.a = a_ext + q_t * s_t
            self.C = C_ext + r_t * np.outer(s_t, s_t)

            # Q も再帰更新 (Eq 2.23)
            #   Ut(Q) + (1/h_t) (T(ehat)-e)(T(ehat)-e)^T
            #   ehat = Q k_vec, 拡張後の ehat_full, e_t_full を作る
            ehat = self.Q.dot(k_vec)  # (N,)
            ehat_full = np.concatenate([ehat, [0.0]])  # (N+1,)
            e_t_full = np.zeros(N+1);  e_t_full[-1]=1.0
            # Ut(Q)
            Q_ext = np.pad(self.Q, ((0,1),(0,1)), mode='constant')
            self.Q = Q_ext + (1.0/h_t)*np.outer(ehat_full-e_t_full, ehat_full-e_t_full)

            # X に追加
            self.X = np.vstack([self.X, x.reshape(1,2)])

        else:
            # ---- (b) 既存基底のみで更新 ----
            self.a = self.a + q_t * s_t[:-1]        # N 要素分だけ
            self.C = self.C + r_t * np.outer(s_t[:-1], s_t[:-1])
            # Q は変えない

        #―― 6) 基底数制限――
        if self.max_basis is not None and self.X.shape[0] > self.max_basis:
            self._prune_basis()

    def _prune_basis(self):
        """
        max_basis 超過時に φ_i = |a_i|/Q_{ii} が最小の基底 i を削除 (Eq 2.25–2.26)
        """
        N = self.X.shape[0]
        phi = np.abs(self.a) / np.diag(self.Q)
        j = np.argmin(phi)

        # j を除いたインデックス
        idx = [i for i in range(N) if i!=j]

        # 除去前のパラメータ
        a_old, C_old, Q_old = self.a.copy(), self.C.copy(), self.Q.copy()
        a_j, Q_jj = a_old[j], Q_old[j,j]
        Q_jcol = Q_old[idx, j]  # j列のうち diag 以外

        # 更新後
        self.a = a_old[idx] - (a_j/Q_jj)*Q_jcol
        self.C = C_old[np.ix_(idx,idx)] - np.outer(Q_jcol, Q_jcol)/Q_jj
        self.Q =  Q_old[np.ix_(idx,idx)]     - np.outer(Q_jcol, Q_jcol)/Q_jj
        self.X = self.X[idx]

    def predict(self, x: np.ndarray) -> Tuple[float,float]:
        """
        平均 μ(x), 分散 σ²(x) の予測
        """
        if self.X.shape[0]==0:
            return 0.0, self.kernel(x,x)
        k_vec = np.array([self.kernel(xi,x) for xi in self.X])
        mu = float(self.a.dot(k_vec))
        var = self.kernel(x,x) + float(k_vec.dot(self.C.dot(k_vec)))
        return mu, var, k_vec, self.C


# ─── Environment and GT Map ───────────────────────────────────
def environment_function(pos, true_map, noise_std=0.1):
    i, j = int(round(pos[0])), int(round(pos[1]))
    neighborhood = true_map[max(0,i-1):i+2, max(0,j-1):j+2]
    val = np.mean(neighborhood)
    return float(np.clip(val + np.random.normal(0, noise_std), 0, 1))

def generate_ground_truth_map(grid_size=20):
    gt = np.zeros((grid_size,grid_size))
    gt[0:5,13:20] = np.random.uniform(0.8,1.0,(5,7))
    gt[9:11,13:20] = np.random.uniform(0.8,1.0,(2,7))
    gt[15:20,0:7] = np.random.uniform(0.8,1.0,(5,7))
    mask = (gt==0)
    gt[mask] = np.random.uniform(0.0,0.2,mask.sum())
    return gt

# ─── UGV Controller──────────────────────────────────────────────
class UGVController:
    def __init__(self, grid_size: int = 20, reward_type: int = 0, discount_factor: float = 0.95):
        self.grid_size = grid_size
        self.reward_type = reward_type
        self.position = np.array([grid_size // 2, grid_size // 2], dtype=int)
        self.visited = np.zeros((grid_size, grid_size), dtype=bool)
        self.visited[self.position[0], self.position[1]] = True
        self.expectation_map = None
        self.variance_map  = None
        self.discount_factor = discount_factor  # 割引率
    
    def set_maps(self, expectation_map: np.ndarray, variance_map: np.ndarray):
        """
        UAV側から渡されたマップを保存しておくためのメソッド
        """
        self.expectation_map = expectation_map
        self.variance_map  = variance_map

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
            if visited[nxt[0], nxt[1]]:
                continue
            new_visited = visited.copy()
            new_visited[nxt[0], nxt[1]] = True
            r = self._calculate_reward(
                nxt, pos, expectation_map, variance_map,
                k1, k2, k3, k4, k5, k6, epsilon,
                self.reward_type, step
            )
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
        new_pos, _ = self._recursive_search(
            self.position, expectation_map, variance_map,
            depth, self.visited.copy(), step
        )
        if not np.array_equal(new_pos, self.position):
            self.position = new_pos
            self.visited[new_pos[0], new_pos[1]] = True

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
    def __init__(self, train_data_x, train_data_y: np.ndarray, grid_size, ugv: UGVController, step_of_ugv_path_used=8):
        self.r = 1.0
        self.alpha = 1.0
        self.gamma = 3.0
        self.k = 2.0 #unomのゲイン
        self.pos = train_data_x[0].copy()  # ← floatで初期化
        self.control_period = 0.1
        self.gp = SparseOnlineGP(sigma0=0.1, kernel=rbf_kernel,max_basis=30,delta=1e-2)
        self.grid_size=grid_size
        self.v = np.zeros(2) 
        self.ugv = ugv
        self.step_of_ugv_path_used = step_of_ugv_path_used
        for x, y in zip(train_data_x, train_data_y):
            self.gp.update(x, y)

    def update_map(self, environment_function: Callable[[np.ndarray], float]):
        """
        現在の位置での観測値を取得し、GPモデルを更新する。
        """
        # 1. 観測値の取得
        observed_value = environment_function(self.pos)

        # 2. SOGP モデルを更新
        self.gp.update(self.pos, observed_value)

        print(f"New observation added at {self.pos}, value = {observed_value:.3f}")

    def calc_objective_function(self):
        J = np.sum(self.sigma2)
        self.objective_function = J

    def calc_cbf_terms(self, u: np.ndarray, gamma=3.0, alpha=1.0) -> Tuple[float, np.ndarray, float]:
        """
        bJ = J - γ
        ξ_J1 = ∇_p J
        ξ_J2 = 厳密な式 (A.8) に基づく 2階項
        """
        r2 = 3.0**2  # rbfカーネルのパラメータと揃える
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
            term4 = 2 * z_l.T @ dot_K @self.gp.Q @ dot_K @ z_l

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

    def calc(self, environment_function: Callable[[np.ndarray], float],v_limit=1.0):
        # --- 1) UAV がマップ更新 ---
        self.update_map(environment_function)
        self.prob, self.sigma2, self.k_star, self.K = self.gp.predict(self.pos)
        # mean と var を出力
        print(f" predict → mean = {self.prob:.3f}, var = {self.sigma2:.3f}")
        self.calc_objective_function()

        # --- 2) UGV の将来 n ステップ先位置を取得 ---
        E_map, V_map = self.get_map_estimates()
        self.ugv.set_maps(E_map, V_map)   # UGVController にマップを渡すメソッド
        planned = self.ugv.get_planned_path(E_map, V_map,
                                            depth=self.step_of_ugv_path_used,
                                            )
        ugv_future_pos = np.array(planned[-1], dtype=float)

        # --- 3) nominal速度を「自分の位置 − UGV の未来位置」で定義 ---
        v_nom = - self.k * (self.pos - ugv_future_pos)
        nu_nom = (v_nom - self.v) / self.control_period

        # --- 4) CBF 項を計算 ← 先ほど修正したシグネチャを呼び出し ---
        xi_J1, xi_J2 = self.calc_cbf_terms(self.v, gamma=3.0, alpha=1.0)

        # --- 5) QP を解いて Δν を得る ---
        self.solver = solver()
        # CBF 制約として xi_J2 を使うなら
        self.solver.add_cbf(-xi_J2, -xi_J1[0], -xi_J1[1],slack=1.0)
        nu= self.solver.solve_qp(nu_nom)

        # --- 6) 状態更新 ---
        self.v += nu * self.control_period
        if np.linalg.norm(self.v) >= v_limit:
            self.v= v_limit * self.v / np.linalg.norm(self.v)  
        self.pos += self.v * self.control_period

        # --- 7) グリッドにクランプ & debug 出力 ---
        H, W = self.grid_size, self.grid_size
        self.pos[0] = np.clip(self.pos[0], 0, H-1)
        self.pos[1] = np.clip(self.pos[1], 0, W-1)
        print(f"xi_J1 = {xi_J1}, norm = {np.linalg.norm(xi_J1)}")
        print(f"velocity = {np.linalg.norm(self.v)}")

# ─── main ─────────────────────────────────────────────────────
def main():
    grid_size = 20
    gt = generate_ground_truth_map(grid_size)

    # 初期観測点の設定
    start = np.array([5.0, 5.0])
    offsets = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    init_x = start + offsets
    init_y = np.array([environment_function(p, gt) for p in init_x])

    # コントローラ初期化
    ugv = UGVController(grid_size, reward_type=3, discount_factor=0.95)
    uav = UAVController(init_x, init_y, grid_size, ugv=ugv, step_of_ugv_path_used=5)

    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # ── 真値マップを薄グレーで背景表示 ──
    ax[0].imshow(gt, origin='lower', cmap='gray', alpha=0.3)

    # ── 推定 Mean / Variance マップの初期化 ──
    im0 = ax[0].imshow(
        np.zeros((grid_size, grid_size)),
        vmin=0, vmax=1,
        cmap='jet',
        origin='lower'
    )
    im1 = ax[1].imshow(
        np.zeros((grid_size, grid_size)),
        vmin=0, vmax=1,
        cmap='jet',
        origin='lower'
    )

    # ── カラーバー（凡例）の追加 ──
    cb0 = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    cb0.set_label('Estimated mean')
    cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    cb1.set_label('Estimated variance')

    # ── 軌跡・現在位置・計画経路のラインオブジェクト ──
    traj_uav = [uav.pos.copy()]
    traj_ugv = [ugv.position.copy()]
    uav_line, = ax[0].plot([], [], '-o', color='cyan',   markersize=3, label='UAV Path')
    ugv_line, = ax[0].plot([], [], '-x', color='yellow', markersize=3, label='UGV Path')
    plan_line,= ax[0].plot([], [], '--*', color='magenta',markersize=4, label='UGV Planned')
    uav_dot,  = ax[0].plot([], [], 'ro', label='UAV')
    ugv_dot,  = ax[0].plot([], [], 'bo', label='UGV')
    ax[0].legend(loc='upper right')

    # ── メインループ ──
    for step in range(300):
        # UAV 更新
        uav.calc(lambda p: environment_function(p, gt), v_limit=5.0)
        # GP 推定マップ取得
        m_map, v_map = uav.get_map_estimates()
        # UGV 更新
        ugv.calc(m_map, v_map, depth=10, step=step+1)

        # 軌跡・計画経路を記録
        traj_uav.append(uav.pos.copy())
        traj_ugv.append(ugv.position.copy())
        planned = ugv.get_planned_path(m_map, v_map, depth=10)

        # ── 画像データ更新 ──
        im0.set_data(m_map)
        im1.set_data(v_map)
        # Variance の色スケールを動的に再設定
        im1.set_clim(v_map.min(), v_map.max())

        # ── ラインデータ更新 ──
        pu = np.array(traj_uav)
        pv = np.array(traj_ugv)
        pp = np.array(planned)
        uav_line.set_data(pu[:,1], pu[:,0])
        ugv_line.set_data(pv[:,1], pv[:,0])
        plan_line.set_data(pp[:,1], pp[:,0])
        uav_dot.set_data([uav.pos[1]], [uav.pos[0]])
        ugv_dot.set_data([ugv.position[1]], [ugv.position[0]])

        # タイトル更新＆描画
        ax[0].set_title(f"Step {step} Mean")
        ax[1].set_title(f"Step {step} Var")
        fig.canvas.draw()
        plt.pause(uav.control_period)

    plt.ioff()

    # ── 最終軌跡表示 ──
    traj_uav = np.array(traj_uav)
    traj_ugv = np.array(traj_ugv)
    plt.figure(figsize=(6,6))
    plt.imshow(gt, origin='lower', cmap='jet', vmin=0, vmax=1)
    plt.plot(traj_uav[:,1], traj_uav[:,0], '-o', color='cyan',   label='UAV Path')
    plt.plot(traj_ugv[:,1], traj_ugv[:,0], '-x', color='yellow', label='UGV Path')
    plt.legend()
    plt.title("Trajectories")
    plt.show()

if __name__ == "__main__":
    main()

