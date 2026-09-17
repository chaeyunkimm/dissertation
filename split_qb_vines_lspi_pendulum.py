import torch
import numpy as np
import time
from functorch import make_functional
import torchopt
import matplotlib.pyplot as plt

#from mpi4py import MPI

from rl_code.lspi import LSPI
from rl_code.modelnp import pendulum
from src.qbvine import pendulum_vines


def append_episode_terminals(states, actions, next_states, rewards):
    """Append each episode's terminal next state and a dummy terminal action."""
    states = np.asarray(states)
    actions = np.asarray(actions).reshape(-1, 1)
    next_states = np.asarray(next_states)
    rewards = np.asarray(rewards).reshape(-1)

    n_transitions = len(states)
    if not (len(actions) == len(next_states) == len(rewards) == n_transitions):
        raise ValueError("state, action, nstate, and reward arrays must have equal lengths")

    terminal_transitions = np.flatnonzero(rewards == 0)
    if len(terminal_transitions) == 0 or terminal_transitions[-1] != n_transitions - 1:
        raise ValueError("the final transition must have reward 0")

    augmented_states = []
    augmented_actions = []
    episode_ends = []
    start = 0

    for terminal_idx in terminal_transitions:
        stop = terminal_idx + 1
        augmented_states.append(np.concatenate((states[start:stop], next_states[terminal_idx:terminal_idx + 1])))
        augmented_actions.append(np.concatenate((actions[start:stop], np.zeros((1, 1)))))
        episode_ends.append(sum(len(episode) for episode in augmented_states) - 1)
        start = stop

    return np.concatenate(augmented_states), np.concatenate(augmented_actions), np.asarray(episode_ends, dtype=int)


def minsr_lspi_misspecified_pend(seed_env):

    n_episodes, sim_len, n_eval, n_stop_modelupdate = 10, 1000, 5, 25
    n_sim_sr, h_units, active, adamlr, n_epoch_adam = 10, [10, 10, 10], "swish", 0.001, 500
    print(f"seed {seed_env} pendulum starting")
    print("n_sim_sr, h_units, active, adamlr", n_sim_sr, h_units, active, adamlr)
    print("seed_env, n_episodes, sim_len, n_eval", seed_env, n_episodes, sim_len, n_eval)
    torch.manual_seed(1)
    np.random.seed(1)

    true_param = np.array([9.8, 2.0, 8.0, 0.5,10.0, 0.1])
    true_pendulum = pendulum(true_param)
    # action_space = true_pendulum.action_space
    my_lspi = LSPI(true_pendulum, sim_len, None)

    ### to keep a record of all the learnt w(policy), episodic return
    all_w, all_returns_epi, all_discreturns_epi = [], [], []
    old_policy = my_lspi.random_policy

    # Declaring the initial variables Need to make this work with my setup.
    training_data = np.load(f"pend_data/epi1000/epis_{seed_env}.npz")
    np_s, np_a, np_s1, np_r = training_data['state'], training_data['action'], training_data['nstate'], training_data['reward']


    all_s, all_a, episode_ends = append_episode_terminals(np_s, np_a, np_s1, np_r)
    raw_episode_ends = np.flatnonzero(np.asarray(np_r).reshape(-1) == 0)

    print("Data Loaded")

    # Initialize the qbvine model
    qbvines = pendulum_vines(max_rho=0.95)

    # The last appended row is a terminal state, not a transition to fit.
    epi_ends = episode_ends[:-1]
    print("Starting Run")
    for ind_episode in range(n_episodes):

        st = time.time()

        ### calculate episodic return
        all_returns, disc_returns = [], []
        init_state = np.random.uniform(low=-1e-10, high=1e-10, size=2)
        for i in range(n_eval):
            eval_s, eval_a, eval_s2, eval_r = true_pendulum.simulate_episode(first_state=init_state, policy=old_policy)
            discounts = np.array(my_lspi.gamma**np.arange(len(eval_r)))
            discounted_return = discounts.T @ np.array(eval_r)
            all_returns.append(np.sum(eval_r)), disc_returns.append(discounted_return[0])
        print(f"Episode {ind_episode} return: {np.mean(all_returns)}")
        all_returns_epi.append(np.mean(all_returns)), all_discreturns_epi.append(disc_returns)
        # np.savez(f'pend_misp/minsr_lspi/lspi{seed_env}ret_all1000.npz' ,all_returns_epi = all_returns_epi, n_episodes = n_episodes, all_discreturns_epi=all_discreturns_epi)

        if np.mean(all_returns) < 1000 and ind_episode < n_stop_modelupdate:
            #### model learning with the data collected up to the current episode
            model_stop = epi_ends[ind_episode] + 1
            qbvines.fit(
                all_s[:model_stop],
                all_a[:model_stop],
                episode_ends=epi_ends[:ind_episode],
            )
            #### model learning ends here.

            #### simualtions with the learnt model

            sim_s, sim_a, sim_s1, sim_r = qbvines.simulate_trajectory(horizon=sim_len)
            np_sim_s, np_sim_a, np_sim_s1, np_sim_r = sim_s.detach().numpy(), sim_a.detach().numpy(), sim_s1.detach().numpy(), sim_r.detach().numpy()

            my_lspi.simple_lspi(np_sim_s, np_sim_a, np_sim_s1, np_sim_r)
            new_policy = lambda x: my_lspi.calc_policy(x, my_lspi.w)

            old_policy = new_policy

        else:

            raw_stop = raw_episode_ends[ind_episode] + 1
            my_lspi.simple_lspi(np_s[:raw_stop], np_a[:raw_stop], np_s1[:raw_stop], np_r[:raw_stop])
            new_policy = lambda x: my_lspi.calc_policy(x, my_lspi.w)

            old_policy = new_policy

    plt.plot(all_returns_epi)
    plt.savefig("see_the_line")
    print(f'seed {seed_env} smc minsr lspi finished!!!')

    return seed_env

result = minsr_lspi_misspecified_pend(8)
