import matplotlib.pyplot as plt
import csv
import gymnasium as gym
import numpy as np
import torch
import src.metropolis_hastings as mh
import time
from sklearn.metrics import r2_score, root_mean_squared_error

from src.time_varying_vine_copula import tv_vinecop, obs_transform_pend2, action_transform_pendulum, SlidingWindowDataset
from src.copula import conditional_vine_copula
from rl_code.modelnp import pendulum
from rl_code.lspi import LSPI

from src.qbvine import qb_conditional_vine, splitdiscrete_qb_conditional_vines, pendulum_vines

n_episodes, sim_len, n_eval, n_stop_modelupdate = 100, 1000, 101, 25
true_param = np.array([9.8, 2.0, 8.0, 0.5,10.0, 0.1])
true_pendulum = pendulum(true_param)
my_lspi = LSPI(true_pendulum, sim_len, None)
old_policy = my_lspi.random_policy

#Iterate the procedure over 10 simulations to establish a mean and standard error for each of the rmse, r2, etc. for each length of training data.
#Store these in a csv file
results_path = "pend_discrete_rmse_results.csv"
with open(results_path, "w", newline="") as results_file:
    results_writer = csv.writer(results_file)
    results_writer.writerow([
        "iteration",
        "rmse_dimension_1",
        "rmse_dimension_2",
        "r2_dimension_1",
        "r2_dimension_2",
        "eval_times",
        "fit_times",
    ])

for j in range(10):
    print("---------------------------")
    print(f"iteration {j}")

    all_s = np.empty((0, 2))
    all_a = np.empty((0,1))
    episode_ends = np.zeros(n_eval, dtype = int)
    end_idx = 0
    for i in range(n_eval):
        init_state = np.random.uniform(low=-1e-10, high=1e-10, size=2)
        eval_s, eval_a, eval_s2, eval_r = true_pendulum.simulate_episode(first_state=init_state, policy=old_policy)
        eval_s_corrected = np.concatenate((eval_s, [eval_s2[-1]]))
        eval_a_corrected = np.concatenate((eval_a, [[0]]))
        all_s = np.concatenate((all_s, eval_s_corrected))
        all_a = np.concatenate((all_a, eval_a_corrected))

        end_idx += len(eval_a_corrected)
        episode_ends[i] = end_idx-1
    episode_ends = episode_ends[:-1]

    test_s = np.empty((0, 2))
    test_s2 = np.empty((0, 2))
    test_a = np.empty((0,1))

    for i in range(n_episodes):
        init_state = np.random.uniform(low=-1e-10, high=1e-10, size=2)
        eval_s, eval_a, eval_s2, eval_r = true_pendulum.simulate_episode(first_state=init_state, policy=old_policy)
        test_s =  np.concatenate((test_s, eval_s))
        test_s2 =  np.concatenate((test_s2, eval_s2))
        test_a = np.concatenate((test_a, eval_a))

    test_s2 = test_s2[:1000]

    n_epis = np.array([3, 9, 27, 60, 100])-1
    test_vine = pendulum_vines(max_rho=None)

    rmses = []
    r2s = []
    eval_times = []
    fit_times = []

    for eps in n_epis:
        predictions = np.zeros((1000, 2))
        model_stop = episode_ends[eps] + 1
        print("fit started")
        fit_start = time.time()

        test_vine.fit(all_s[:model_stop], all_a[:model_stop], episode_ends=episode_ends[:eps-1])
        fit_time = time.time() - fit_start
        fit_times.append(fit_time)
        print(f"Fit finished in {fit_time:.3f} seconds, starting evaluation")

        nan_idxs = []
        eval_start = time.time()
        for i in range(1000):
            predictions[i] = test_vine.predict_next_state(torch.from_numpy(np.expand_dims(test_s[i], 0)), test_a[i][0], num_samples = 100)
            if not np.isfinite(predictions[i]).all():
                print(f"Non-finite prediction at index {i}: {predictions[i]}")
                nan_idxs.append(i)
        eval_time = time.time() - eval_start
        eval_times.append(eval_time)
        print(f"Evaluation of predictions took {eval_time:.3f} seconds")

        valid_rows = np.isfinite(predictions).all(axis=1)
        if not valid_rows.all():
            print("number of non-finite prediction rows =", np.count_nonzero(~valid_rows))
        evaluation_targets = test_s2[valid_rows]
        evaluation_predictions = predictions[valid_rows]

        rm = root_mean_squared_error(
            evaluation_targets, evaluation_predictions, multioutput="raw_values"
        )
        r2 = r2_score(evaluation_targets, evaluation_predictions, multioutput="raw_values")
        rmses.append(rm)
        r2s.append(r2)
        print("Rounds complete:", rm, r2)

    with open(results_path, "a", newline="") as results_file:
        results_writer = csv.writer(results_file)
        results_writer.writerow([
            j,
            [rmse[0] for rmse in rmses],
            [rmse[1] for rmse in rmses],
            [r2[0] for r2 in r2s],
            [r2[1] for r2 in r2s],
            eval_times,
            fit_times,
        ])



