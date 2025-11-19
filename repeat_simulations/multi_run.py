# 別ファイル multi_run_suenaga.py でもいいし、
# 同じファイルの末尾に書いてもOK
import pandas as pd
from run_experiment import run_experiment


def multi_run(
    n_runs: int = 3,
    base_name: str = "test2",

    # 以下は run_experiment_suenaga の引数とほぼ同じ
    visualize: bool = False,
    grid_size: int = 50,
    noise_std: float = 0.5,
    num_uavs: int = 3,
    num_ugvs: int = 2,
    steps: int = 10,
    map_publish_period: float = 0.5,
    ugv_depth: int = 8,
    reward_type: int = 3,
    discount_factor: float = 0.95,
    v_limit: float = 25.0,
    step_of_ugv_path_used: int = 6,
    suenaga_on: bool = False,
    use_j_gradient_cbf: bool = True,
    use_voronoi: bool = True,
    unom_gain: float = 2.0,
    suenaga_gain: float = 2.0,
    cbf_j_alpha: float = 1.0,
    cbf_j_gamma: float = 3.0,
    d0: float = 1.0,
    ugv_future_path_sigma: float = 5.0,
    suenaga_discount_rate: float = 0.95,
    suenaga_path_gene_depth: int = 5,
    gp_sensing_noise_sigma0: float = 0.4,
    gp_max_basis: int = 100,
    gp_threshold_delta: float = 0.05,
    rbf_sigma: float = 2.0,
):
    all_results = []
    all_params = []

    for r in range(n_runs):
        print(f"\n===== RUN {r} / {n_runs} =====")
        df_res, df_param = run_experiment(
            visualize=visualize if r == 0 else False,  # 必要なら最初だけ表示など
            seed=r,
            result_csv=None,
            param_csv=None,
            grid_size=grid_size,
            noise_std=noise_std,
            num_uavs=num_uavs,
            num_ugvs=num_ugvs,
            steps=steps,
            map_publish_period=map_publish_period,
            ugv_depth=ugv_depth,
            reward_type=reward_type,
            discount_factor=discount_factor,
            v_limit=v_limit,
            step_of_ugv_path_used=step_of_ugv_path_used,
            suenaga_on=suenaga_on,
            use_j_gradient_cbf=use_j_gradient_cbf,
            use_voronoi=use_voronoi,
            unom_gain=unom_gain,
            suenaga_gain=suenaga_gain,
            cbf_j_alpha=cbf_j_alpha,
            cbf_j_gamma=cbf_j_gamma,
            d0=d0,
            ugv_future_path_sigma=ugv_future_path_sigma,
            suenaga_discount_rate=suenaga_discount_rate,
            suenaga_path_gene_depth=suenaga_path_gene_depth,
            gp_sensing_noise_sigma0=gp_sensing_noise_sigma0,
            gp_max_basis=gp_max_basis,
            gp_threshold_delta=gp_threshold_delta,
            rbf_sigma=rbf_sigma,
        )

        df_res = df_res.copy()
        df_res["run"] = r
        df_param = df_param.copy()
        df_param["run"] = r

        all_results.append(df_res)
        all_params.append(df_param)

    results_all = pd.concat(all_results, ignore_index=True)
    params_all = pd.concat(all_params, ignore_index=True)

    results_path = f"{base_name}_results_{n_runs}runs.csv"
    params_path = f"{base_name}_params_{n_runs}runs.csv"

    results_all.to_csv(results_path, index=False)
    params_all.to_csv(params_path, index=False)

    print("\n=== DONE ===")
    print(f"結果: {results_path}")
    print(f"パラメータ: {params_path}")


if __name__ == "__main__":
    multi_run()
