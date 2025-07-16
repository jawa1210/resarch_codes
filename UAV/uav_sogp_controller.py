import math
import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# ─── CBF QP Solver ─────────────────────────────────────────────
import numpy as np
from qpsolvers import solve_qp

class solver():
    def __init__(self):
        self.alpha = 1.0
        self.cbf = None
        self.slack = 0.0
        self.v_nom = np.zeros(2)

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        self.cbf = np.array([bJ, dbJ_du_x, dbJ_du_y])
        self.slack = slack

    def set_nominal_input(self, v_nom: np.ndarray):
        self.v_nom = v_nom

    def solve_qp(self):
        if self.cbf is not None:
            bJ, gx, gy = self.cbf
            G = np.array([[-gx, -gy, 1.0]])
            h = np.array([[ self.alpha * bJ ]])
        else:
            G = np.zeros((1,3))
            h = np.zeros((1,1))

        P = np.eye(3)
        P[0, 0] = P[1, 1] = 2
        q = np.zeros(3)
        q[0:2] = -2 * self.v_nom

        sol = solve_qp(P, q, G, h, solver="cvxopt")
        return sol[:2] if sol is not None else np.zeros(2)


def rbf_kernel(x, y, sigma=3.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))

class SparseOnlineGP:
    """
    SOGP 本体。max_basis=None なら全データ保持、数を制限したい場合は整数を入れる。
    """
    def __init__(self, sigma0: float, kernel: Callable, max_basis: int = None, omega: float = 1e-4):
        self.sigma0 = sigma0
        self.kernel = kernel
        self.max_basis = max_basis
        self.omega = omega

        # 初期状態：データ空
        self.X = np.zeros((0,2))    # basis inputs
        self.a = np.zeros((0,))     # coefficients
        self.C = np.zeros((0,0))    # matrix C_t
        self.Q = np.zeros((0,0))    # inverse gram Q_t

    def init_first(self, x: np.ndarray, y: float):
        # t=1 のとき
        k00 = self.kernel(x, x)
        denom = k00 + self.sigma0**2
        self.X = x.reshape(1,2)
        self.a = np.array([y/denom])
        self.C = np.array([[-1.0/denom]])
        self.Q = np.array([[1.0/k00]])

    def predict(self, x: np.ndarray) -> Tuple[float,float]:
        """
        SOGP の予測(mean, variance) を返す。
        """
        k_vec = np.array([self.kernel(xi, x) for xi in self.X])  # (N,)
        mean = float(self.a.dot(k_vec))
        var = self.kernel(x, x) + float(k_vec.dot(self.C.dot(k_vec)))
        return mean, var

    def update(self, x: np.ndarray, y: float):
        # 1) 初回追加
        if self.X.shape[0] == 0:
            return self.init_first(x, y)

        # 2) 既存基底数
        N = self.X.shape[0]

        # 3) 事前予測 f_*, var_*
        k_vec = np.array([self.kernel(xi, x) for xi in self.X])  # (N,)
        k_tt  = self.kernel(x, x)
        f_star   = float(self.a.dot(k_vec))
        var_star = float(k_tt + k_vec.dot(self.C.dot(k_vec)))

        # 4) q_t, r_t の計算 (Eq.2.20,2.21)
        denom = var_star + self.sigma0**2
        q_t = (y - f_star) / denom
        r_t = -1.0 / denom

        # 5) ノベリティ h_t の計算 (Eq.2.22)
        h_t = k_tt - k_vec.dot(self.Q.dot(k_vec))

        # 6) s_t の構築 (長さ N+1)
        #    - 既存Cに対する寄与を下に0を足して伸長
        s_base = self.C.dot(k_vec)                  # (N,)
        s_t     = np.concatenate([s_base, [0.0]])   # (N+1,) のうち下端は0
        e_t     = np.zeros(N+1); e_t[-1] = 1.0      # 単位ベクトル
        s_t    += e_t                               # Eq.2.17 の s_t

        # 7) ブランチ：新基底追加 or 既存更新
        if h_t > self.omega:
            # —— 新基底を追加する場合 ——
            # a_t と C_t の Ut/Tt 拡張＋更新 (Eq.2.17)
            self.a = np.r_[self.a, 0.0] + q_t * s_t
            C_ext     = np.pad(self.C, ((0,1),(0,1)), mode='constant')
            self.C   = C_ext        + r_t * np.outer(s_t, s_t)

            # Q_t の Ut 更新 + rank-1 update (Eq.2.23)
            ehat      = self.Q.dot(k_vec)           # (N,)
            ehat_full = np.concatenate([ehat, [0.0]])
            Q_ext     = np.pad(self.Q, ((0,1),(0,1)), mode='constant')
            self.Q    = Q_ext + (1.0/h_t) * np.outer(ehat_full - e_t,
                                                    ehat_full - e_t)

            # 新基底を X に追加
            self.X = np.vstack([self.X, x.reshape(1,2)])

        else:
            # —— 既存基底だけで更新する場合 ——
            # Eq.2.24 相当
            # s_t[:-1] が既存基底への寄与分
            s_short = s_t[:-1]
            self.a = self.a + q_t * s_short
            self.C = self.C + r_t * np.outer(s_short, s_short)
            # Q は変えない

        # 8) 基底数超過時の Prune (Eq.2.25–2.26)
        # 基底追加 or 既存更新 が済んだあと
        if self.max_basis is not None and len(self.X) > self.max_basis:
            self._prune_basis()

    def _prune_basis(self):
        """
        max_basis 超過時に φ_i = |a_i|/Q_{ii} が最小の基底 i を削除し、
        論文 Eq.(2.26) のリダクション更新を行う。
        """
        N = len(self.X)
        # φ_i = |a_i| / Q_{ii}
        phi = np.abs(self.a) / np.diag(self.Q)
        j = np.argmin(phi)  # 削除対象のインデックス

        # 中間パラメータをコピー
        a_mid = self.a.copy()
        C_mid = self.C.copy()
        Q_mid = self.Q.copy()

        # j 番目の値を取り出す
        a_j = a_mid[j]            # a^{(t+1)}_j
        Q_j = Q_mid[:, j].reshape(-1,1)   # Q^j (列ベクトル)
        q_jj = Q_mid[j, j]        # q^j

        # 1) a の更新
        #   a_hat = a_mid - a_j * (Q^j / q^j)
        a_new = a_mid - (a_j / q_jj) * Q_j.flatten()

        # 2) C の更新
        #   c^j は C_mid の j 列から j 行を除いたもの
        idx = [i for i in range(N) if i != j]
        Cj = C_mid[:, j]          # 全行の j 列
        cj = Cj[idx]              # 削除後の c^j (長さ N-1)
        Qj = Q_j[idx,:]           # Q^j のうち削除後の行 (形: (N-1,1))

        # 部分的な rank-1 更新項
        term1 = (cj[:,None] @ Qj.T) / (q_jj**2)   # c^j Q^j^T / (q^j)^2
        term2 = (Qj @ C_mid[j: j+1, :][:, idx]) / q_jj  # Q^j C^j^T / q^j とその転置
        term3 = (C_mid[idx, j:j+1] @ Qj.T) / q_jj       # C^j Q^j^T / q^j

        C_new = C_mid[np.ix_(idx, idx)] + term1 - (term2 + term3)

        # 3) Q の更新
        Q_new = Q_mid[np.ix_(idx, idx)] - (Qj @ Qj.T) / q_jj

        # 4) 基底 X, a, C, Q を置き換える
        self.X = self.X[idx]
        self.a = a_new[idx]
        self.C = C_new
        self.Q = Q_new

# ──────────────────────────────────────────────────────────────────────────


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

class Controller:
    def __init__(self, train_data_x, train_data_y: np.ndarray, grid_size):
        self.r = 1.0
        self.alpha = 0.1
        self.k = 2.0
        self.pos = train_data_x[0].copy()  # ← floatで初期化
        self.control_period = 0.1
        self.gp = SparseOnlineGP(sigma0=0.1, kernel=rbf_kernel)
        self.grid_size=grid_size
        self.v = np.zeros(2) 
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


    # def get_next_position(self, current_position, use_qp, grad: Tuple[float, float]) -> Tuple[float, float]:
    #     v_limit=0.5
    #     if use_qp is True:
    #         self.solver= solver()
    #         self.solver.add_cbf(self.objective_function-self.gamma, grad[0], grad[1])
    #         self.v = self.solver.solve_qp()[0:2]
    #     else:
    #         self.v=-self.k*grad*self.control_period
    #     if np.linalg.norm(self.v) > v_limit*self.control_period:
    #         self.v = self.v / np.linalg.norm(self.v) * v_limit
    #     # Placeholder for actual logic to determine the next position
    #     next_position = current_position+self.v
    #     # グリッド境界内にクランプ
    #     H, W = self.grid_size, self.grid_size
    #     next_position[0] = np.clip(next_position[0], 0, H-1)
    #     next_position[1] = np.clip(next_position[1], 0, W-1)

    #     self.pos = next_position

    def calc_objective_function(self):
        J = np.sum(self.sigma2)
        self.objective_function = J
    

    def calc_cbf_terms(self, u: np.ndarray, epsilon: float) -> Tuple[float, np.ndarray, float]:
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

        bJ = self.objective_function - self.gamma
        xi_J2 += epsilon * (np.dot(xi_J1, u) + self.gamma)

        return bJ, xi_J1, xi_J2
 
        
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

    def calc(self, environment_function: Callable[[np.ndarray], float]):
        self.update_map(environment_function)
        self.prob, self.sigma2, self.k_star, self.K = self.gp.predict(self.pos)
        self.calc_objective_function()
        self.gamma=3.0
        bJ, xi_J1, xi_J2 = self.calc_cbf_terms(self.v, epsilon=0.5)
        v_nom = -self.k * xi_J1  # 目的関数の調整
        nu_nom = (v_nom - self.v) / self.control_period
        
        # QPソルバの設定
        self.solver = solver()
        nu_star = self.solver.solve_qp()

        # 速度の更新: u(t+Δt) = u(t) + ν Δt
        self.v += nu_star * self.control_period
        self.pos += self.v * self.control_period
        self.solver.add_cbf(bJ, xi_J1[0], xi_J1[1])
        self.solver.set_nominal_input(nu_nom)
        v_delta = self.solver.solve_qp()
        if v_delta is None:
            print("QP solver failed, using nominal velocity")

        # 2. 状態更新
        self.v += v_delta * self.control_period
        self.pos += self.v * self.control_period

        # 3. グリッド境界にクランプ
        H, W = self.grid_size, self.grid_size
        self.pos[0] = np.clip(self.pos[0], 0, H-1)
        self.pos[1] = np.clip(self.pos[1], 0, W-1)
        print(f"xi_J1 = {xi_J1}, norm = {np.linalg.norm(xi_J1)}")
        print(f"velocity={np.linalg.norm(self.v)}")

def main():
    grid_size=20
    gt = generate_ground_truth_map(grid_size)
    start = np.array([5.0,5.0])
    # 初期観測点として近傍 5 点だけ使う
    offsets = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
    init_x = start + offsets
    init_y = np.array([environment_function(p,gt) for p in init_x])

    ctrl = Controller(init_x, init_y, grid_size)

    plt.ion()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    im0 = ax[0].imshow(np.zeros((grid_size,grid_size)),vmin=0,vmax=1,cmap='viridis')
    im1 = ax[1].imshow(np.zeros((grid_size,grid_size)),cmap='plasma')
    cb0 = fig.colorbar(im0, ax=ax[0])
    cb1 = fig.colorbar(im1, ax=ax[1])

    traj = [ctrl.pos.copy()]
    for step in range(30):
        ctrl.calc(lambda p: environment_function(p,gt))
        traj.append(ctrl.pos.copy())

        m_map, v_map = ctrl.get_map_estimates()
        im0.set_data(m_map)
        im1.set_data(v_map)
        
        # 赤い現在位置を描画（前回の点を消すために scatter をリセット）
        [p.remove() for p in ax[0].collections if isinstance(p, plt.Line2D)]
        ax[0].plot(ctrl.pos[1], ctrl.pos[0], 'ro')  # x=col, y=row 順
        
        ax[0].set_title(f"Step {step} Mean")
        ax[1].set_title(f"Step {step} Var")
        fig.canvas.draw()
        plt.pause(0.3)

    traj = np.array(traj)
    plt.ioff()
    plt.figure()
    plt.imshow(gt, origin='lower', cmap='hot')
    plt.plot(traj[:,1], traj[:,0], '-o', color='cyan')
    plt.title("True map and UAV trajectory")
    plt.show()

if __name__=="__main__":
    main()
