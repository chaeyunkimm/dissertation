#### this is the final code for min sr reinforce for hopper.
from __future__ import annotations

import random
import time
import numpy as np

import torch

from functorch import make_functional
import torchopt

import gymnasium as gym

#from mpi4py import MPI


from src.prob_mlphopper import probabilistic_MLP, calc_loss, make_simulations


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

    # Declaring the initial variables
    training_data = np.load(f"pend_data/epi1000/epis_{seed_env}.npz")
    np_s, np_a, np_s1, np_r = training_data['state'], training_data['action'], training_data['nstate'], training_data['reward']
    alltrain_s = torch.tensor(training_data['state'], dtype=torch.float32)
    alltrain_a = torch.tensor(training_data['action'], dtype=torch.float32)
    alltrain_s1 = torch.tensor(training_data['nstate'], dtype=torch.float32)
    alltrain_r = torch.tensor(training_data['reward'], dtype=torch.float32)

    ### for training the model network
    all_training_y, all_training_x = alltrain_s1 , torch.cat((alltrain_s, alltrain_a), 1)

    # Initialize the probabilistic MLP model
    my_mlp = probabilistic_MLP(in_units=(all_training_x.shape[1]+1), z_dim=1, hidden_units= h_units,\
                                        out_units=(all_training_y.shape[1]), activation=active, bias_last_layer=True)

    my_model, my_params = make_functional(my_mlp)

    ## list of where rewards are zero
    epi_ends = torch.where(alltrain_r == 0)[0].tolist()

    ################## to get an initial estimate of the parameters ##################
    learning_rate = adamlr
    optimizer = torchopt.adam(learning_rate)
    opt_state = optimizer.init(my_params)


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
            training_x, training_y = all_training_x[:epi_ends[ind_episode]+1], all_training_y[:epi_ends[ind_episode]+1]
            my_mlp.learn_std(training_x)
            loss_record = []    
            for i in range(n_epoch_adam):
                simulations = make_simulations(my_model, my_params, training_x, n_sim_es = n_sim_sr)
                loss = calc_loss(simulations, training_y)
                # print(f'Loss at epoch {i}:', loss.item()/training_x.shape[0])
                loss_record.append(loss.item()/training_x.shape[0])

                grads = torch.autograd.grad(loss, my_params)
                updates, opt_state = optimizer.update(grads, opt_state)

                # print('Parameters before update:', params)
                my_params = torchopt.apply_updates(my_params, updates)
            optimised_theta = torch.cat([g.view(-1) for g in [param for param in my_params]])
            #### model learning ends here.

            #### simualtions with the learnt model

            index = 0
            with torch.no_grad():
                for param in my_mlp._model.parameters():
                    if len(param.shape) > 1:
                        param.copy_(optimised_theta[index:index+param.numel()].view(param.shape))
                        index += param.numel()
                    else:
                        param.copy_(optimised_theta[index:index+param.numel()])
                        index += param.numel()

            my_model1, my_params = make_functional(my_mlp)
            # print("my_params: ", my_params)
            sim_s, sim_a, sim_s1, sim_r = random_traj_simulator(my_model, my_params, sim_len)
            np_sim_s, np_sim_a, np_sim_s1, np_sim_r = sim_s.detach().numpy(), sim_a.detach().numpy(), sim_s1.detach().numpy(), sim_r.detach().numpy()

            my_lspi.simple_lspi(np_sim_s, np_sim_a, np_sim_s1, np_sim_r)
            new_policy = lambda x: my_lspi.calc_policy(x, my_lspi.w)

            old_policy = new_policy

        else:

            my_lspi.simple_lspi(np_s[:epi_ends[ind_episode]+1], np_a[:epi_ends[ind_episode]+1], np_s1[:epi_ends[ind_episode]+1], np_r[:epi_ends[ind_episode]+1])
            new_policy = lambda x: my_lspi.calc_policy(x, my_lspi.w)

            old_policy = new_policy

    plt.plot(all_returns_epi)
    plt.savefig("see_the_line")
    print(f'seed {seed_env} smc minsr lspi finished!!!')

    return seed_env