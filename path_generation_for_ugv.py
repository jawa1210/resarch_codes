import math
import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def rbf_kernel(x, y, sigma=3.0):
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))


def gaussian_process_estimation(x: np.array, train_data_x: np.array, train_data_y: np.array, kernel: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float]:
    """
    ガウス過程分類器に基づいて、テスト点 x における期待値 (prob) と不確実性 (sigma) を返す

    Parameters:
    - x: 予測したい1点 (shape: [D,])
    - train_data_x: 訓練データの特徴量 (shape: [N, D])
    - train_data_y: 訓練データのラベル (shape: [N])
    - kernel: カーネル関数（例: RBF）

    Returns:
    - prob: クラス1の確率
    - sigma: 予測の分散
    """
    N = train_data_x.shape[0]

    # カーネル行列Kの計算
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(train_data_x[i], train_data_x[j])

    sigma_n2 = 1e-3
    K = K+sigma_n2*np.eye(N)

    # テスト点と訓練店のカーネルベクトルk_*
    k_star = np.array([kernel(train_data_x[i], x) for i in range(N)])

    # 平均予測値
    f_star = k_star.T @ np.linalg.solve(K, train_data_y)

    # 分散推定
    sigma2 = kernel(x, x) - k_star.T @ np.linalg.solve(K, k_star)

    # ロジスティック近似による確率化
    z = f_star / np.sqrt(1+(np.pi/8)*sigma2)
    prob = 1 / (1 + np.exp(-z))

    return prob, sigma2


def find_best_move(pos: np.array, reward_type: int, expectation_map: np.array, variance_map: np.array, actions, k1, k2, k3, k4, k5, k6, epsilon, grid_size, depth, discount_factor, visited, step):
    '''
    後で書く
    '''
    best_reward = -math.inf
    best_move = pos

    for action in actions:
        next_pos = pos + action

        # 範囲外チェック
        if not (0 <= next_pos[0] < expectation_map.shape[0] and 0 <= next_pos[1] < expectation_map.shape[1]):
            continue

        # すでに訪問済みのチェック
        if visited[next_pos[0], next_pos[1]]:
            continue

        # 報酬の計算
        new_visited = visited.copy()
        new_visited[next_pos[0], next_pos[1]] = True

        current_reward = calculate_reward(next_pos, pos, reward_type, expectation_map, variance_map, k1, k2, k3, k4, k5, k6, epsilon, step)
        total_reward = current_reward

        if depth > 1:
            _, future_reward = find_best_move(next_pos, reward_type, expectation_map, variance_map, actions, k1, k2, k3, k4, k5, k6, epsilon, grid_size, depth-1, discount_factor, new_visited, step+1)
            total_reward = total_reward+discount_factor * future_reward

        if total_reward > best_reward:
            best_reward = total_reward
            best_move = next_pos

    return best_move, best_reward


def calculate_reward(pos, current_pos, reward_type, expectation_map, variance_map, k1, k2, k3, k4, k5, k6, epsilon, step):
    d = np.linalg.norm(pos-current_pos)
    E = expectation_map[pos[0], pos[1]]
    V = variance_map[pos[0], pos[1]]

    if reward_type == 0:
        reward = (k1*E-V)/(d**2+epsilon)
    elif reward_type == 1:
        reward = np.exp(-(d**2)/k4)*np.exp((k2*E-V)/k3)
    elif reward_type == 2:
        reward = (np.tanh(k5*E)-k6*V)/(d**2+epsilon)
    elif reward_type == 3:
        d_ucb = 2
        delta = 0.1
        beta = d_ucb*np.log(((np.pi**2)*(step**2))/(6*delta))
        reward = E-np.sqrt(beta*V)
    else:
        reward = 0
    return reward


# --- Drawing a red rectangle at target_pos ---
def draw_red_rectangle(ax, target_pos):
    ax.add_patch(Rectangle((target_pos[0]-0.5, target_pos[1]-0.5), 1, 1,
                           linewidth=2, edgecolor='red', facecolor='none'))

def draw_blue_square(ax, point):
    """点 (x, y) に青枠を表示"""
    ax.add_patch(Rectangle((point[0]-0.5, point[1]-0.5), 1, 1,
                           linewidth=1.5, edgecolor='blue', facecolor='none'))


def main():
    np.random.seed(1)
    grid_size = 20
    num_steps = 50
    train_x = np.random.rand(20, 2) * grid_size
    train_y = np.random.randint(0, 2, 20)
    # ラベル1の点を抽出（注意: x, y の順番に注意）
    label_1_points = train_x[train_y == 1].astype(int)

    expectation_map = np.zeros((grid_size, grid_size))
    variance_map = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            prob, sigma2 = gaussian_process_estimation(np.array([i, j]), train_x, train_y, rbf_kernel)
            expectation_map[i, j] = prob
            variance_map[i, j] = sigma2

    high_E = expectation_map > 0.8
    low_V = variance_map < 0.2
    highE_lowV_mask = high_E & low_V
    masked_E = expectation_map.copy()
    masked_E[~highE_lowV_mask] = -np.inf
    target_idx = np.unravel_index(np.argmax(masked_E), masked_E.shape)
    target_pos = np.array(target_idx[::-1])  # x, y

    start = np.array([15, 15])
    actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    k1, k2, k3, k4, k5, k6, epsilon = 2, 2, 0.1, 0.1, 1, 1, 1e-3

    paths = []
    cumulative_rewards = []
    reward_logs = []
    E_logs = []
    V_logs = []
    distance_logs = []

    for reward_type in range(4):
        pos = start.copy()
        path = [pos.copy()]
        rewards = []
        E_log = []
        V_log = []
        distances = []
        visited = np.zeros((grid_size, grid_size), dtype=bool)  # 実際は訪れたところに行くのを禁止ではなく、なるべく行かないようにするがやむを得ない場合は通るとしないと動かなくなる
        visited[pos[0], pos[1]] = True

        for step in range(1, num_steps + 1):
            best_move, best_r = find_best_move(pos, reward_type, expectation_map, variance_map, actions,
                                               k1, k2, k3, k4, k5, k6, epsilon,
                                               grid_size, depth=10, discount_factor=0.95,
                                               visited=visited, step=step)
            pos = best_move
            path.append(pos.copy())
            rewards.append(best_r)
            E_log.append(expectation_map[pos[0], pos[1]])
            V_log.append(variance_map[pos[0], pos[1]])
            distances.append(np.linalg.norm(pos - target_pos))  # 最も果物が有ると推定されている場所までの距離
            visited[pos[0], pos[1]] = True

        paths.append(np.array(path))
        cumulative_rewards.append(np.cumsum(rewards))
        reward_logs.append(rewards)
        E_logs.append(E_log)
        V_logs.append(V_log)
        distance_logs.append(distances)

    titles = [
        'Reward 1: Linear',
        'Reward 2: Exponential',
        'Reward 3: Hyperbolic',
        'Reward 4: UCB'
    ]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axs[i].imshow(expectation_map, cmap='hot', origin='lower')
        axs[i].plot(paths[i][:, 1], paths[i][:, 0], '-o', linewidth=2)
        draw_red_rectangle(axs[i], target_pos)
        for pt in label_1_points:
            draw_blue_square(axs[i], pt[::-1])  # [x, y] → [row, col]draw_blue_square(axs[i], pt[::-1])
        axs[i].set_title(titles[i])
        axs[i].axis('equal')
    plt.suptitle('UGV paths under different reward functions')
    plt.show(block=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(expectation_map, cmap='hot', origin='lower')
    draw_red_rectangle(axs[0], target_pos)
    for pt in label_1_points:
        draw_blue_square(axs[0], pt[::-1])  # [x, y] → [row, col]
    axs[0].set_title('Expectation E')
    axs[1].imshow(variance_map, cmap='hot', origin='lower')
    draw_red_rectangle(axs[1], target_pos)
    for pt in label_1_points:
        draw_blue_square(axs[1], pt[::-1])  # [x, y] → [row, col]
    axs[1].set_title('Variance V')
    plt.suptitle('Expectation and Variance Map')
    plt.show(block=False)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(4):
        axs[i].plot(cumulative_rewards[i], linewidth=2)
        axs[i].set_title(f'{titles[i]} - Cumulative Reward')
        axs[i].set_xlabel("Step")
        axs[i].set_ylabel("Cumulative Reward")
        axs[i].tick_params(axis='both', which='both', direction='in', length=6)
        axs[i].grid(True)
    plt.suptitle('Cumulative Rewards')
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(4):
        axs[i].plot(reward_logs[i], linewidth=2)
        axs[i].set_title(f'{titles[i]} - Stepwise Reward')
        axs[i].set_xlabel("Step")
        axs[i].set_ylabel("Reward")
        axs[i].tick_params(axis='both', which='both', direction='in', length=6)
        axs[i].grid(True)
    plt.suptitle('Stepwise Rewards')
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(4):
        E_vals = E_logs[i]
        V_vals = V_logs[i]
        mean_E = np.mean(E_vals)
        mean_V = np.mean(V_vals)

        axs[i].plot(E_vals, label='E (Expectation)', color='red')
        axs[i].plot(V_vals, label='V (Variance)', color='blue')
        axs[i].hlines(mean_E, 0, len(E_vals) - 1, colors='red', linestyles='dashed', label=f'mean E = {mean_E:.3f}')
        axs[i].hlines(mean_V, 0, len(V_vals) - 1, colors='blue', linestyles='dashed', label=f'mean V = {mean_V:.3f}')

        axs[i].text(len(E_vals)*0.5, mean_E + 0.01, f'{mean_E:.3f}', color='red', fontsize=10)
        axs[i].text(len(V_vals)*0.5, mean_V - 0.01, f'{mean_V:.3f}', color='blue', fontsize=10)

        axs[i].legend()
        axs[i].set_ylabel('Value')
        axs[i].set_xlabel('Step')
        axs[i].set_title(f'{titles[i]} - E and V per Step')
        axs[i].tick_params(axis='both', which='both', direction='in', length=6)
        axs[i].grid(True)
    plt.suptitle('E and V Progression with Mean Value Labels')
    plt.tight_layout()
    plt.show(block=False)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12))
    for i in range(4):
        axs[i].plot(distance_logs[i], linewidth=2)
        axs[i].set_title(f'{titles[i]} - Distance to Max E')
        axs[i].set_xlabel("Step")
        axs[i].set_ylabel("Distance")
        axs[i].tick_params(axis='both', which='both', direction='in', length=6)
        axs[i].grid(True)
    plt.suptitle('Distance to Target')
    plt.tight_layout()
    plt.show(block=False)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(high_E & low_V, cmap='gray', origin='lower')
    draw_red_rectangle(plt.gca(), target_pos)
    for pt in label_1_points:
        draw_blue_square(plt.gca(), pt[::-1])  # [x, y] → [row, col]
    colors = ['cyan', 'magenta', 'yellow', 'green']
    for i in range(4):
        plt.plot(paths[i][:, 1], paths[i][:, 0], 'o-', color=colors[i], label=titles[i])
    plt.legend()
    plt.title('High-E Low-V Regions with Paths')
    plt.show(block=False)
    input("Press Enter to close all plots...")


if __name__ == '__main__':
    main()
