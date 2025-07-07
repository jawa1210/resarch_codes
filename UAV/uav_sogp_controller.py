import math
import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

# ─── CBF QP Solver ─────────────────────────────────────────────
class solver():
    def __init__(self):
        self.alpha = 1.0  # CBF の緩和ゲイン
        self.cbf = None
        self.slack = 0.0

    def add_cbf(self, bJ: float, dbJ_du_x: float, dbJ_du_y: float, slack: float = 0.0):
        """
        bJ + ∇_u bJ^T v >= 0
        ∇_u bJ = [dbJ_du_x, dbJ_du_y]
        実装では
          -dbJ_du_x * v_x - dbJ_du_y * v_y <= alpha * bJ
        として G, h にセット
        """
        self.cbf = np.array([bJ, dbJ_du_x, dbJ_du_y])
        self.slack = slack

    def solve_qp(self):
        # decision var [v_x, v_y, slack]
        if self.cbf is not None:
            bJ, gx, gy = self.cbf
            # -∇_u bJ^T v <= alpha * bJ
            G = np.array([[-gx, -gy, 1.0]])
            h = np.array([[ self.alpha * bJ ]])
        else:
            G = np.zeros((1,3))
            h = np.zeros((1,1))

        P = np.eye(3)
        q = np.zeros(3)

        sol = solve_qp(P, q, G, h, solver="cvxopt")
        return sol  # shape=(3,)

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
        """
        新しい観測 (x,y) をオンライン更新。
        """
        # 最初のデータなら初期化だけ
        if self.X.shape[0] == 0:
            self.init_first(x, y)
            return

        # 既存の basis size
        N = self.X.shape[0]

        # 1) 事前予測
        k_vec = np.array([self.kernel(xi, x) for xi in self.X])  # (N,)
        k_tt = self.kernel(x, x)
        f_star = float(self.a.dot(k_vec))
        var_star = float(k_tt + k_vec.dot(self.C.dot(k_vec)))

        # 2) likelihood の 1st/2nd 微分 → q_t, r_t
        denom = var_star + self.sigma0**2
        q = (y - f_star) / denom
        r = 1.0 / denom

        # 3) st ベクトルを作る (N+1,)
        c_k = self.C.dot(k_vec)            # (N,)
        st = np.concatenate([c_k, [1.0]])  # (N+1,)

        # 4) a, C を拡張＋更新
        a_ext = np.concatenate([self.a, [0.0]])                # (N+1,)
        C_ext = np.pad(self.C, ((0,1),(0,1)), mode='constant') # (N+1,N+1)

        self.a = a_ext + q * st
        self.C = C_ext + r * np.outer(st, st)

        # 5) basis データ列を拡張
        self.X = np.vstack([self.X, x.reshape(1,2)])

        # 6) Q_t の更新（次の novelty 評価に使う）
        #    Qt = Ut(Qt-1) + (1/h_t)(T(et~ - et) (T(et~ - et))^T)
        #    ここでは簡略化して毎回逆行列を保持
        K = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(N+1):
                K[i,j] = self.kernel(self.X[i], self.X[j])
        self.Q = np.linalg.inv(K)

        # 7) sparsify (max_basis 制限)
        if self.max_basis is not None and self.X.shape[0] > self.max_basis:
            self._sparsify()

    def _sparsify(self):
        """
        χ_i 基準で最も貢献の小さい basis を削除
        """
        # χ_i = |a_i| / Q_{ii}
        chi = np.abs(self.a) / np.diag(self.Q)
        j = np.argmin(chi)  # 削除対象
        # j 行 j 列を落とす
        mask = np.ones(self.X.shape[0], dtype=bool)
        mask[j] = False
        self.X = self.X[mask]
        self.a = self.a[mask]
        self.C = self.C[np.ix_(mask,mask)]
        self.Q = self.Q[np.ix_(mask,mask)]


# ──────────────────────────────────────────────────────────────────────────

def environment_function(pos: np.ndarray, true_map: np.ndarray, noise_std: float = 0.1) -> float:
    i, j = int(round(pos[0])), int(round(pos[1]))
    val = true_map[i,j]
    return float(np.clip(val + np.random.normal(0,noise_std), 0,1))

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
        self.gp = SparseOnlineGP(train_data_x, train_data_y, rbf_kernel)
        self.grid_size=grid_size

    def update_map(self, environment_function: Callable[[np.ndarray], float]):
        """
        現在の位置での観測値を取得し、訓練データに追加。
        Parameters:
        - environment_function: Callable[[np.ndarray], float]
            現在位置 `pos` を入力して環境から観測値 (0~1) を返す関数
        """
        # 1. 観測値の取得
        observed_value = environment_function(self.pos)

        # 2. データ追加
        self.gp.train_data_x = np.vstack([self.gp.train_data_x, self.pos])
        self.gp.train_data_y = np.append(self.gp.train_data_y, observed_value)

        # 3. GP再学習（K再計算されるため、次回の推論で反映される）
        # 明示的に何かする必要はありません（Kは毎回再計算）

        print(f"New observation added at {self.pos}, value = {observed_value:.3f}")

    def get_next_position(self, current_position, use_qp, grad: Tuple[float, float]) -> Tuple[float, float]:
        v_limit=0.5
        if use_qp is True:
            self.solver= solver()
            self.solver.add_cbf(self.objective_function-self.gamma, grad[0], grad[1])
            self.v = self.solver.solve_qp()[0:2]
        else:
            self.v=-self.k*grad*self.control_period
        if np.linalg.norm(self.v) > v_limit*self.control_period:
            self.v = self.v / np.linalg.norm(self.v) * v_limit
        # Placeholder for actual logic to determine the next position
        next_position = current_position+self.v
        # グリッド境界内にクランプ
        H, W = self.grid_size, self.grid_size
        next_position[0] = np.clip(next_position[0], 0, H-1)
        next_position[1] = np.clip(next_position[1], 0, W-1)

        self.pos = next_position

    def calc_objective_function(self):
        J = np.sum(self.sigma2)
        self.objective_function = J

    def calc_objective_function_grad(self) -> np.ndarray:
        H, W = self.grid_size, self.grid_size
        grad = np.zeros(2)          # ∇J は (2,) ベクトル
        K_inv = np.linalg.inv(self.K)  # もし K が変わらなければ１回だけ計算
        rbf_sigma = 3.0  # RBFカーネルのσ

        for i in range(H):
            for j in range(W):
                # セル (i,j) の位置 x = [i,j]
                x = np.array([i, j], dtype=float)
                # そのセルの k_* と ∇k_* を求める
                _, sigma2, k_star, K = self.gp.gaussian_process_estimation(x)
                # ∇_p k_*(x,p) の各要素を構築
                k_star_dot = np.zeros((k_star.shape[0], 2))
                for t in range(k_star.shape[0]):
                    xi = self.gp.train_data_x[t]
                    ki = k_star[t]
                    diff = xi - self.pos
                    k_star_dot[t, :] = (ki / rbf_sigma**2) * diff
                # σ²(x) の勾配
                grad += -2 * (k_star @ (K_inv @ k_star_dot))
        return grad

    def calc(self, environment_function: Callable[[np.ndarray], float]):
        self.update_map(environment_function)
        self.prob, self.sigma2, self.k_star, self.K = self.gp.gaussian_process_estimation(self.pos)
        grad = self.calc_objective_function_grad()
        self.calc_objective_function()
        use_qp=True
        self.gamma=3.0
        self.get_next_position(self.pos, use_qp, grad)
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
        ctrl.step(lambda p: environment_function(p,gt))
        traj.append(ctrl.pos.copy())

        m_map, v_map = ctrl.get_map_estimates()
        im0.set_data(m_map)
        im1.set_data(v_map)
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
