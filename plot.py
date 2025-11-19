import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib

# ── フォント／mathtext 設定 ────────────────────────────────
mpl.rcParams['font.family']        = 'IPAexGothic'
mpl.rcParams['text.usetex']        = False    # ← LaTeX エンジンはオフ
mpl.rcParams['font.size']          = 16
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['lines.linewidth']    = 4
mpl.rcParams['axes.titlesize']     = 20
mpl.rcParams['axes.labelsize']     = 20
mpl.rcParams['legend.fontsize']    = 16
mpl.rcParams['xtick.labelsize']    = 16
mpl.rcParams['ytick.labelsize']    = 16r

def plot_files(file_list, gamma=3.0):
    data = []
    for path in file_list:
        df = pd.read_csv(path)
        t        = df['step'].to_numpy()/10
        J        = df['J'].to_numpy()
        true_sum = df['true_crop_sum'].to_numpy()
        J0       = J[0]
        theory   = J0 - gamma*10 * t
        label    = path.rsplit('.', 1)[0]
        data.append((label, t, J, theory, true_sum))

    # 目的関数 J の推移
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, J, theory, _ in data:
        ax.plot(t, J,       label=f'{label} 実測 $J$')
        ax.plot(t, theory, '--', label=f'{label} 理論 $J_0 - \\gamma t$')
    ax.set_xlabel('時間 (Step)')
    ax.set_ylabel('目的関数 $J$')
    ax.set_title('Objective $J$ の推移と理論直線')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 累積真値の合計
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, t, _, _, true_sum in data:
        ax.plot(t, true_sum, label=f'{label} 累積真値の合計')
    ax.set_xlabel('時間 (Step)')
    ax.set_ylabel('真値の合計')
    ax.set_title('UGV 訪問セルの累積真値')
    ax.legend()
    ax.grid(True)
    plt.ylim(0,)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # 1つだけプロットしたいときは ['with_ugv_path.csv'] のようにリストを1要素にする
    files = ['multi_uav_multi_ugv_results_suenaga.csv']  # 実際のファイル名に置き換えてください
    plot_files(files)
