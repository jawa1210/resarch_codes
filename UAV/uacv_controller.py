import math
import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from qpsolvers import solve_qp

class solver():
    def __init__(self):
        self.alpha=1.0
        self.cbf=None

    def add_cbf(self, h,grad_x,grad_y,slack=0):
        self.cbf= np.array([h, grad_x, grad_y])
        self.slack=slack

    def solve_qp(self):
        if self.cbf is not None:
            G=np.array([[-self.cbf[1],-self.cbf[2],self.slack]])
            h=np.array([[self.alpha*self.cbf[0]]])
        else:
            G=np.array([[[0,0,0],[0,0,0],[0,0,0]]])
            h=np.array([[0]])
        P=np.eye(3)
        q=np.zeros(3)
        sol=solve_qp(P,q,G,h,solver="cvxopt")
        return sol


def rbf_kernel(x, y, sigma=3.0):  # ← self 削除
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def environment_function(pos: np.ndarray, true_map: np.ndarray, noise_std: float = 0.1) -> float:
    """真のマップから観測し、ノイズを加える"""
    i, j = int(np.round(pos[0])), int(np.round(pos[1]))
    true_value = true_map[i, j]  # 真の値（0～1）
    noise = np.random.normal(0, noise_std)
    return np.clip(true_value + noise, 0.0, 1.0)  # 観測値も0～1に制限


def generate_ground_truth_map(grid_size=20) -> np.ndarray:
    gt_map = np.zeros((grid_size, grid_size))

    # 右上（上から5行 × 右から7列）
    gt_map[0:5, 13:20] = np.random.uniform(0.8, 1.0, size=(5, 7))

    # 真ん中右（中央2行 × 右から7列）
    gt_map[9:11, 13:20] = np.random.uniform(0.8, 1.0, size=(2, 7))

    # 左下（下から5行 × 左から7列）
    gt_map[15:20, 0:7] = np.random.uniform(0.8, 1.0, size=(5, 7))

    # その他の位置にはノイズ（低い値）
    noise = np.random.uniform(0.0, 0.2, size=(grid_size, grid_size))
    mask = gt_map == 0.0
    gt_map[mask] = noise[mask]

    return gt_map


class GaussianProcess:
    def __init__(self, train_data_x: np.array, train_data_y: np.array, kernel: Callable[[np.ndarray, np.ndarray], float]):
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.kernel = kernel

    def gaussian_process_estimation(self, x: np.ndarray):
        N = self.train_data_x.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[i, j] = self.kernel(self.train_data_x[i], self.train_data_x[j])

        sigma_n2 = 1e-3
        K = K + sigma_n2 * np.eye(N)

        k_star = np.array([self.kernel(self.train_data_x[i], x) for i in range(N)])
        f_star = k_star.T @ np.linalg.solve(K, self.train_data_y)
        sigma2 = self.kernel(x, x) - k_star.T @ np.linalg.solve(K, k_star)

        z = f_star / np.sqrt(1 + (np.pi / 8) * sigma2)
        prob = 1 / (1 + np.exp(-z))

        return prob, sigma2, k_star, K


class Controller:
    def __init__(self, train_data_x, train_data_y: np.ndarray, grid_size):
        self.r = 1.0
        self.alpha = 0.1
        self.k = 2.0
        self.pos = train_data_x[0].copy()  # ← floatで初期化
        self.control_period = 0.1
        self.gp = GaussianProcess(train_data_x, train_data_y, rbf_kernel)
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
    grid_size = 20
    gt_map = generate_ground_truth_map(grid_size)
    start_pos = np.array([5.0, 5.0])
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    train_x = start_pos + offsets     # shape=(5,2)
    train_y = np.array([environment_function(p, gt_map) for p in train_x])
    # Controller はオフセット入りの train_x, train_y だけで作る
    controller = Controller(train_x, train_y, grid_size)

    def env_func(p): return environment_function(p, gt_map)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # 初期用ダミーマップ
    mean_map = np.zeros((grid_size, grid_size))
    var_map = np.zeros((grid_size, grid_size))

    # 一度だけ imshow と colorbar を作っておく
    im0 = axes[0].imshow(mean_map, cmap='viridis', origin='lower', vmin=0, vmax=1)
    im1 = axes[1].imshow(var_map,  cmap='plasma', origin='lower')
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for step in range(30):
        # UAV の１ステップ目標計算＋移動
        controller.calc(env_func)

        # GP マップの再計算
        for i in range(grid_size):
            for j in range(grid_size):
                mu, var, *_ = controller.gp.gaussian_process_estimation(np.array([i, j]))
                mean_map[i, j] = 1/(1+np.exp(-mu))
                var_map[i, j] = var

        # データだけ更新してタイトルも差し替え
        im0.set_data(mean_map)
        im1.set_data(var_map)
        axes[0].set_title(f"Step {step}: GP Mean")
        axes[1].set_title(f"Step {step}: GP Variance")

        # 再描画
        fig.canvas.draw()
        plt.pause(0.5)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
