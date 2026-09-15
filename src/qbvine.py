import numpy as np
import pyvinecopulib as pv
import torch
import matplotlib.pyplot as plt

from src.copula import conditional_vine_copula
from src.time_varying_vine_copula import SlidingWindowDataset, rbp_process, cauchy_prior, discrete_action_prior

class qb_conditional_vine:
    '''
    Initialises the rbp transform, a set of conditional vine copulas, and the time varying copula.

    Has various functions like pdf, cdf to evaluate for data and allow for metropolis hastings.
    '''
    def __init__(self, dimension = None, rl = True, observation_d = 4, action_d = 1, discrete_action = 0, action_set = None):
        if rl:
            self.conditioned_set = np.arange(1, observation_d+1) #correct
            self.conditioning_set  = np.arange(observation_d+1, 2*observation_d+action_d+1) #incorrect
            self.mask_set = np.concatenate((self.conditioned_set, np.arange(observation_d+action_d+discrete_action, 2*(observation_d+action_d)+discrete_action+1)))
            self.discrete_action = discrete_action
            print(self.conditioning_set, self.conditioned_set, self.mask_set)

        else:
            raise NotImplementedError("The conditioning set construction has not been implemented outside of reinforcement learning")
        self.cond_vines = []
        self.rbp = rbp_process()
        self.prior = cauchy_prior()
        self.observation_d = observation_d
        self.action_d = action_d
        if self.discrete_action > 0:
            if action_set is None:
                raise ValueError("An action set must be given for discrete actions")
            self.action_set = action_set
            self.action_prior = discrete_action_prior(action_set=self.action_set)

    def fit(self, data, check_vines = False, check_tv = False, episode_ends = None):
        '''
        Assumes we want to condition on the last state and action, nothing else.

        You must pass a numpy_array of episode ends to stop the model from learning a massive jump is normal!

        Assumes the discrete action is at the end, does not take any discrete observations.
        '''

        n, self.d = data.shape
        if episode_ends is None:
            self.ts_per_condvine = np.zeros((n-1, self.d-1))
        else:
            self.ts_per_condvine = np.zeros((n-len(episode_ends)-1, self.d-1))
        self.last_state = data[-1, :-1]
        if self.discrete_action>0:
            self.prior.fit(data[:, :-1])
            p, c = self.prior.eval(data[:,:-1])
            self.rbp.fit(p, c)
            p, c = self.rbp.eval(p, c)

            self.action_prior.fit(data[:,-1])
            c_acts = self.action_prior.cdf(data[:,-1])
            c_acts_minus = self.action_prior.cdf(data[:, -1], left_limit = True)

            c = np.column_stack((c, c_acts, c_acts_minus))

        else:
            self.prior.fit(data)
            p, c = self.prior.eval(data)
            self.rbp.fit(p, c)
            p, c = self.rbp.eval(p, c)

        self.last_state_and_action_postrbp = [c[-1,:]]

        time_series = torch.from_numpy(c)
        sliding_dataset = SlidingWindowDataset(time_series, window_size = 1)
        ts = torch.stack([torch.cat((target, history.flatten())) for history, target in sliding_dataset]).squeeze(1).numpy()
        if episode_ends is not None:
            ts = np.delete(ts, episode_ends, axis = 0)

        self.cond_vine = conditional_vine_copula(conditioning_set=self.conditioning_set, n_discrete=self.discrete_action)
        self.cond_vine.fit_from_data(ts[:, self.mask_set], check_vine=check_vines) #-1?

        # if seperate_prediction:
            # for i, cond in enumerate(self.conditioned_set):
            #     mask = np.concatenate(([cond - 1], self.conditioning_set - 1))
            #     print(mask)
            #     conditional_vine = conditional_vine_copula(self.vine_cond_set, n_discrete=self.discrete_action)
            #     conditional_vine.fit_from_data(ts[:, mask], check_vine = check_vines)
            #     self.cond_vines.append(conditional_vine)

# Below here there be dragons
    def step_forward(self, data, action):
        '''
        Takes a data point and moves the process to it. 

        data is in its raw form and thus is transformed before storing it in self.last_state_post_rbp

        This is ok when it is synchronous - we need to check how it is when we restart a chain. 

        It may be useful to then take 2 data points to start from as this will give a good estimate of the 
        conditional marginal copulas to start from. This would need to be a seperate function.
        '''
        self.last_state = data[0, :]
        cond_vine_cdfs = np.zeros((1, self.d-1))
        data_and_action = np.concatenate((data, np.array([[action]])), axis=1)

        if self.discrete_action>0:
            p, c = self.prior.eval(data_and_action[:,:-1])
            p, c = self.rbp.eval(p, c)

            c_acts = self.action_prior.cdf(data_and_action[:,-1])
            c_acts_minus = self.action_prior.cdf(data_and_action[:, -1], left_limit = True)

            c = np.column_stack((c, c_acts, c_acts_minus))
        else:
            p, c = self.prior.eval(data_and_action)
            _, c = self.rbp.eval(p, c)

        self.last_state_and_action_postrbp = c

    def check_rbp_pdfs_real_line(self, x_min=-10, x_max=10, n_points=500):
        """
        Plot each fitted RBP marginal PDF over the real line against its prior KDE.
        Useful for checking whether each marginal transformation looks sensible.
        """ 

        if not hasattr(self, 'rbp') or not hasattr(self.rbp, 'rhos'):
            raise ValueError("The RBP model has not been fitted yet.")
        print(self.rbp.rhos)
        x_grid = np.linspace(x_min, x_max, n_points)
        if self.discrete_action ==0:
            iter_range = self.observation_d+self.action_d
            fig, axes = plt.subplots(1, iter_range, figsize=(4 * self.observation_d, 4), squeeze=False)
            
        else:
            iter_range = self.observation_d
            fig, axes = plt.subplots(1, iter_range, figsize=(4 * self.observation_d, 4), squeeze=False)

        prior_p, prior_c = self.prior.eval(np.repeat([x_grid], iter_range, axis=0).T)
        rbp_p, rbp_c = self.rbp.eval(prior_p, prior_c) 

        for j in range(iter_range):

            axes[0, j].plot(x_grid, prior_p[:,j], linestyle='--', alpha=0.7, label='Prior')
            axes[0, j].plot(x_grid, rbp_p[:,j], linewidth=2, label='RBP PDF')
            axes[0, j].set_title(f'Marginal {j + 1}')
            axes[0, j].set_xlabel('x')
            axes[0, j].set_ylabel('density')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def pdf_predictive(self, data):
        '''
        The probability of a state given the previous state and action stored in the object by fit or step_forward
        '''
        if data.ndim != 2:
            data = np.expand_dims(data, axis=0)
        assert data.shape[0] == 1, "Currently we can only check one state at a time."

        data_and_action = np.concatenate((data, np.array([[0]])), axis=1)

        cond_vine_pdfs = np.zeros((1, self.d-1))
        cond_vine_cdfs = np.zeros((1, self.d-1))

        if self.discrete_action>0:
            p, c = self.prior.eval(data_and_action[:,:-1])
            rbp_pdfs, c = self.rbp.eval(p, c)

            c_acts = self.action_prior.cdf(data_and_action[:,-1])
            c_acts_minus = self.action_prior.cdf(data_and_action[:, -1], left_limit = True)

            c = np.column_stack((c, c_acts, c_acts_minus))
        else:
            p, c = self.prior.eval(data_and_action)
            rbp_pdfs, c = self.rbp.eval(p, c)

        ts = np.expand_dims(np.concatenate((c, self.last_state_and_action_postrbp), axis = None), 0)

        cond_vine_pdf = self.cond_vine.pdf(ts[:, self.mask_set])
        
        #This is not required for this model yet
        # for i, cond in enumerate(self.conditioned_set):
        #     mask = np.concatenate(([cond - 1], self.conditioning_set - 1))
        #     cond_vine_pdfs[:,i] = self.cond_vines[i].pdf(ts[:, mask])
        #     cond_vine_cdfs[:,i] = self.cond_vines[i].cdf_implicit(ts[:, mask])

        #Need to check if this is multiplying by the action rbp as well
        if self.discrete_action>0:
            return np.prod(np.concatenate((rbp_pdfs, cond_vine_pdf), axis = 1), axis = 1)
        else:
            return np.prod(np.concatenate((rbp_pdfs[:, :-1], cond_vine_pdf), axis = 1), axis = 1)

    def predict_next_state(self, state_and_action = None):
        '''
        Predicts the next state given the previous state and an action using monte carlo estimation via conditional sampling
        and inversion of the rbp process.
        '''

        u_samples = self.cond_vine.conditional_sample(self.last_state_and_action_postrbp)
        print("samples: ", u_samples[0:3])
        u_mean = np.mean(u_samples[:, :self.observation_d], axis=0)
        u_std = np.std(u_samples[:, :self.observation_d], axis=0)
        us = np.vstack((u_mean, u_mean-u_std, u_mean+u_std))
        print(u_mean)
        print(u_std)
        prior_us = self.rbp.inverse_cdf(us)
        print(prior_us)
        xs = self.prior.ppf(prior_us)
        return xs



class splitdiscrete_qb_conditional_vines:
    '''
    Initialises the rbp transform, a set of conditional vine copulas, and the time varying copula.

    Has various functions like pdf, cdf to evaluate for data and allow for metropolis hastings.
    '''
    def __init__(self, dimension = None, rl = True, observation_d = 4, action_d = 1, discrete_action = 1, action_set = None):
        if rl:
            self.conditioned_set = np.arange(1, observation_d+1)
            self.conditioning_set  = np.arange(observation_d++1, 2*(observation_d)+1) 
            self.mask_set = np.concatenate((self.conditioned_set, np.arange(observation_d+action_d+1, 2*(observation_d)+action_d+1)))
            self.discrete_action = discrete_action
            print(self.conditioning_set, self.conditioned_set, self.mask_set)

        else:
            raise NotImplementedError("The conditioning set construction has not been implemented outside of reinforcement learning")
        self.cond_vines = []
        self.rbp = rbp_process()
        self.prior = cauchy_prior()
        self.observation_d = observation_d
        self.action_d = action_d
        if self.discrete_action > 0:
            if action_set is None:
                raise ValueError("An action set must be given for discrete actions")
            self.action_set = action_set

    def fit(self, data, check_vines = False, check_tv = False, episode_ends = None):
        '''
        Assumes we want to condition on the last state and action, nothing else.

        You must pass a numpy_array of episode ends to stop the model from learning a massive jump is normal!

        Assumes the discrete action is at the end, does not take any discrete observations.
        '''

        n, self.d = data.shape
        if episode_ends is None:
            self.ts_per_condvine = np.zeros((n-1, self.d-1))
        else:
            self.ts_per_condvine = np.zeros((n-len(episode_ends)-1, self.d-1))
        self.last_state = data[-1, :-1]

        self.prior.fit(data[:, :-1])
        p, c = self.prior.eval(data[:,:-1])
        self.rbp.fit(p, c)
        p, c = self.rbp.eval(p, c)

        self.last_state_and_action_postrbp = [c[-1,:]]
        c = np.hstack((c, np.expand_dims(data[:,-1], axis=1)))
        print("c after hstack", c.shape)

        time_series = torch.from_numpy(c)
        sliding_dataset = SlidingWindowDataset(time_series, window_size = 1)
        ts = torch.stack([torch.cat((target, history.flatten())) for history, target in sliding_dataset]).squeeze(1).numpy()
        if episode_ends is not None:
            ts = np.delete(ts, episode_ends, axis = 0)

        for act in self.action_set:
            tstemp = ts[ts[:, -1]==act]
            print("tstemp",tstemp.shape)
            print(self.mask_set)

            cond_vine = conditional_vine_copula(conditioning_set=self.conditioning_set, n_discrete=0)
            cond_vine.fit_from_data(tstemp[:, self.mask_set-1], check_vine=check_vines)
            self.cond_vines.append(cond_vine)


# Below here be dragons
    def step_forward(self, data, action):
        '''
        Takes a data point and moves the process to it. 

        data is in its raw form and thus is transformed before storing it in self.last_state_post_rbp

        This is ok when it is synchronous - we need to check how it is when we restart a chain. 

        It may be useful to then take 2 data points to start from as this will give a good estimate of the 
        conditional marginal copulas to start from. This would need to be a seperate function.
        '''
        self.last_state = data[0, :]
        cond_vine_cdfs = np.zeros((1, self.d-1))
        data_and_action = np.concatenate((data, np.array([[action]])), axis=1)


        p, c = self.prior.eval(data)
        p, c = self.rbp.eval(p, c)
        c = np.concatenate((c, np.array([[action]])), axis=1)

        self.last_state_and_action_postrbp = c

    def check_rbp_pdfs_real_line(self, x_min=-10, x_max=10, n_points=500):
        """
        Plot each fitted RBP marginal PDF over the real line against its prior KDE.
        Useful for checking whether each marginal transformation looks sensible.
        """ 

        if not hasattr(self, 'rbp') or not hasattr(self.rbp, 'rhos'):
            raise ValueError("The RBP model has not been fitted yet.")
        print(self.rbp.rhos)
        x_grid = np.linspace(x_min, x_max, n_points)
        if self.discrete_action ==0:
            iter_range = self.observation_d+self.action_d
            fig, axes = plt.subplots(1, iter_range, figsize=(4 * self.observation_d, 4), squeeze=False)
            
        else:
            iter_range = self.observation_d
            fig, axes = plt.subplots(1, iter_range, figsize=(4 * self.observation_d, 4), squeeze=False)

        prior_p, prior_c = self.prior.eval(np.repeat([x_grid], iter_range, axis=0).T)
        rbp_p, rbp_c = self.rbp.eval(prior_p, prior_c) 

        for j in range(iter_range):

            axes[0, j].plot(x_grid, prior_p[:,j], linestyle='--', alpha=0.7, label='Prior')
            axes[0, j].plot(x_grid, rbp_p[:,j], linewidth=2, label='RBP PDF')
            axes[0, j].set_title(f'Marginal {j + 1}')
            axes[0, j].set_xlabel('x')
            axes[0, j].set_ylabel('density')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def pdf_predictive(self, data):
        '''
        The probability of a state given the previous state and action stored in the object by fit or step_forward
        '''
        if data.ndim != 2:
            data = np.expand_dims(data, axis=0)
        assert data.shape[0] == 1, "Currently we can only check one state at a time."



        cond_vine_pdfs = np.zeros((1, self.d-1))
        cond_vine_cdfs = np.zeros((1, self.d-1))

        p, c = self.prior.eval(data)
        rbp_pdfs, c = self.rbp.eval(p, c)
        c = np.concatenate((c, np.array([[0]])), axis=1)

        ts = np.expand_dims(np.concatenate((c, self.last_state_and_action_postrbp), axis = None), 0)

        cond_vine_pdf = self.cond_vine.pdf(ts[:, self.mask_set])

        idx = np.where(self.action_set==ts[:, -1])
        self.cond_vines[idx].pdf(ts[:self.mask_set])

        #Need to check if this is multiplying by the action rbp as well
        if self.discrete_action>0:
            return np.prod(np.concatenate((rbp_pdfs, cond_vine_pdf), axis = 1), axis = 1)
        else:
            return np.prod(np.concatenate((rbp_pdfs[:, :-1], cond_vine_pdf), axis = 1), axis = 1)

    def sample_next_state(self, state_and_action = None, n_samples = 20):
        '''
        Predicts the next state given the previous state and an action using monte carlo estimation via conditional sampling
        and inversion of the rbp process.
        '''
        action = self.last_state_and_action_postrbp[:, -1]
        idx = int(np.where(self.action_set == action)[0][0])

        u_samples = self.cond_vines[idx].conditional_sample(self.last_state_and_action_postrbp[:,:-1], n_samples = n_samples)

        prior_us = self.rbp.inverse_cdf(u_samples[:, :self.observation_d])

        xs = self.prior.ppf(prior_us)

        return xs