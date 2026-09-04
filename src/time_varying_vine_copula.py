import numpy as np
import pyvinecopulib as pv
import torch
import gymnasium as gym
import src.metropolis_hastings as mh
import matplotlib.pyplot as plt

from src.copula import conditional_vine_copula, time_varying_copula
from src.R_BP_james import rbp_process
from scipy.stats import gaussian_kde, cauchy
from scipy.special import ndtr

from torch.utils.data import Dataset
'''
We want to transform our data through some prior distributions then through the rbp.

Then fit a set of vine copulas over each of the conditioned set, conditioned on the conditioning set.

These are then used to transform the training data.

Then fit a time varying copula on this transformed data.

These parameters are all stored such that we can call for evaluations of the pdf from
proposals of the environment.

    {priors + rbps}      {CondVineCops}
X          --->      U        --->       V      ===>    Time Varying Copula

'''

#reference this
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size=10, steps_ahead=1):
        self.data = data
        self.steps_ahead = steps_ahead
        self.window_size = window_size
    def __len__(self):
        return len(self.data) - self.window_size - self.steps_ahead + 1
    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx
        history = self.data[idx : idx + self.window_size]
        target = self.data[idx + self.window_size + self.steps_ahead - 1]
        return history, target

def model():
    #fit priors
    #transform data through priors
    #fit rbp process rhos and pivots
    #transform data through rbp
    #fit conditional vinecops
    #transform data through condcops
    #fit time varying copula
    #evaluate the time varying copula
    pass

'''
We need four classes: (So we can fit and contain the fit in one instance while being contained)
Prior
RBP
Conditional_Vinecop
Time_varying Copula
'''

class kde_prior:
    def __init__(self):
        self.kdes = []

    def fit(self, observations):
        n, self.d = observations.shape

        # p0_grid = np.zeros((n, self.d))
        # P0_grid = np.zeros((n, self.d))

        for i in range(self.d):
            # Fit prior distributions and transform data for rbp
            kde = gaussian_kde(observations.T[i])
            self.kdes.append(kde)
            # p0_grid[:,i] = kde(observations.T[i])
            # P0_grid[:,i] = np.mean(ndtr((observations.T[i] - kde.dataset.T) / np.sqrt(kde.covariance[0, 0])), axis=0)

    def eval(self, data):
        n, _ = data.shape
        p0_grid = np.zeros((n, self.d))
        P0_grid = np.zeros((n, self.d))

        for i in range(self.d):
            # Fit prior distributions and transform data for rbp
            kde = self.kdes[i]

            p0_grid[:,i] = kde(data.T[i])
            P0_grid[:,i] = np.mean(ndtr((data.T[i] - kde.dataset.T) / np.sqrt(kde.covariance[0, 0])), axis=0)

        return p0_grid, P0_grid


class cauchy_prior:
    def __init__(self):
        self.cauchies = []

    def fit(self, observations):
        n, self.d = observations.shape
        for i in range(self.d):
            dist = cauchy().fit(observations[:, i])
            self.cauchies.append(dist)

    def eval(self, data):
        n, _ = data.shape
        p0_grid = np.zeros((n, self.d))
        P0_grid = np.zeros((n, self.d))
        for dist in self.cauchies:
            p0_grid = dist.pdf(data)
            P0_grid = dist.cdf(data)

        return p0_grid, P0_grid

def interval_to_R(x:torch.tensor, low:float, high:float) -> torch.tensor: 
    '''
    Tranforms from an interval to the real line via a logit tranformation.

    Low and high are the interval bounds.
    ''' 
    frac = (x - low)/(high - low ) 
    y = torch.special.logit(frac, eps=1e-6)     
    return y

def R_to_interval(r:torch.tensor, low:float, high:float) -> torch.tensor:
    '''
    Transforms from the real line to an interval.

    Low and high are the interval bounds.
    '''
    y = torch.special.expit(r)
    x = y*(high-low)+low
    return x

def angle_to_R2(x:torch.tensor) -> torch.tensor:
    '''
    Transforms from an angle to a pair of values on the real line.
    '''
    sin_angles = torch.sin(x)
    cos_angles = torch.cos(x)

    ### transform to R
    sin_angles_r = interval_to_R(sin_angles, -1.0, 1.0)
    cos_angles_r = interval_to_R(cos_angles, -1.0, 1.0)

    return sin_angles_r, cos_angles_r

def R2_to_angle(r2:torch.tensor) -> torch.tensor:
    '''
    Transforms from a pair of values on the real line to an angle.
    '''
    sin_angles_r = r2[:, 0]
    cos_angles_r = r2[:, 1]

    sin_angles = R_to_interval(sin_angles_r, -1.0, 1.0)
    cos_angles = R_to_interval(cos_angles_r, -1.0, 1.0)

    theta = torch.atan2(sin_angles, cos_angles)

    return theta

class tv_vinecop:
    '''
    Initialises the rbp transform, a set of conditional vine copulas, and the time varying copula.

    Has various functions like pdf, cdf to evaluate for data and allow for metropolis hastings.
    '''
    def __init__(self, dimension = None, rl = True, observation_d = 4, action_d = 1):
        if rl:
            self.conditioned_set = np.arange(1, observation_d+1)
            self.conditioning_set  = np.arange(1+observation_d+action_d, 2*(observation_d+action_d)+1)
            self.vine_cond_set = np.arange(2, 2+observation_d+action_d)

        else:
            raise NotImplementedError("The conditioning set construction has not been implemented outside of reinforcement learning")
        self.cond_vines = []
        self.rbp = rbp_process()
        self.kde_prior = kde_prior()
        self.tv_cop = time_varying_copula()
        self.observation_d = observation_d

    def fit(self, data, check_vines = False, check_tv = False):
        '''
        Assumes we want to condition on the last state and action, nothing else.
        '''

        n, self.d = data.shape
        self.ts_per_condvine = np.zeros((n-1, self.d-1))

        self.kde_prior.fit(data)
        p, c = self.kde_prior.eval(data)

        self.rbp.fit(p, c)
        p, c = self.rbp.eval(p, c)
        self.last_state_and_action_postrbp = [c[-1,:]]
        time_series = torch.from_numpy(c)
        sliding_dataset = SlidingWindowDataset(time_series, window_size = 1)
        ts = torch.stack([torch.cat((target, history.flatten())) for history, target in sliding_dataset]).squeeze(1).numpy()

        for i, cond in enumerate(self.conditioned_set):
            mask = np.concatenate(([cond - 1], self.conditioning_set - 1))

            conditional_vine = conditional_vine_copula(self.vine_cond_set)
            conditional_vine.fit_from_data(ts[:, mask], check_vine = check_vines)
            self.cond_vines.append(conditional_vine)

            self.ts_per_condvine[:, i] = conditional_vine.cdf_implicit(ts[:, mask]) # Is this the correct number of dimensions to put in the tv_cop

        self.tv_cop.fit(self.ts_per_condvine, printout=check_tv)


    def step_forward(self, data, action):
        '''
        Takes a data point and moves the process to it. 

        data is in its raw form and thus is transformed before storing it in self.last_state_post_rbp

        This is ok when it is synchronous - we need to check how it is when we restart a chain. 

        It may be useful to then take 2 data points to start from as this will give a good estimate of the 
        conditional marginal copulas to start from. This would need to be a seperate function.
        '''
        cond_vine_cdfs = np.zeros((1, self.d-1))
        data_and_action = np.concatenate((data, np.array([[action]])), axis=1)

        p, c = self.kde_prior.eval(data_and_action)
        _, c = self.rbp.eval(p, c)
# Is it valid to propose the jump from the previous last state to the new last state - This could be very large and thus be unstable.
# I think we need to initialise with a pair of observations, not just the initial state to keep this valid but need ot check whether this works
# with the process as a whole.
        #time_series = torch.from_numpy(np.concatenate((self.last_state_and_action_postrbp, c), axis=0))
        ts = np.expand_dims(np.concatenate((c, self.last_state_and_action_postrbp), axis = None), 0)

        self.last_state_and_action_postrbp = c

        #Probably don't need to do this, just concatenate them?
        # sliding_dataset = SlidingWindowDataset(time_series, window_size = 1)
        # ts = torch.stack([torch.cat((target, history.flatten())) for history, target in sliding_dataset]).squeeze(1).numpy()

        for i, cond in enumerate(self.conditioned_set):
            mask = np.concatenate(([cond - 1], self.conditioning_set - 1))
            cond_vine_cdfs[:,i] = self.cond_vines[i].cdf_implicit(ts[:, mask])

        self.tv_cop.step_forward(cond_vine_cdfs)

    def jump_back(self, two_data, two_actions):
        cond_vine_cdfs = np.zeros((1, self.d-1))
        data_and_actions = np.concatenate((two_data, two_actions), axis=1)

        p, c = self.kde_prior.eval(data_and_actions)
        _, c = self.rbp.eval(p, c)

        time_series = torch.from_numpy(c)

        self.last_state_and_action_postrbp = [c[1,:]]

        # Porbably don't need to do this, just concatenate them? test first and then edit.
        sliding_dataset = SlidingWindowDataset(time_series, window_size = 1)
        ts = torch.stack([torch.cat((target, history.flatten())) for history, target in sliding_dataset]).squeeze(1).numpy()

        print(time_series.flatten())
        print(ts)

        for i, cond in enumerate(self.conditioned_set):
            mask = np.concatenate(([cond - 1], self.conditioning_set - 1))
            cond_vine_cdfs[:,i] = self.cond_vines[i].cdf_implicit(ts[:, mask])

        self.tv_cop.step_forward(cond_vine_cdfs)

    def check_rbp_pdfs_real_line(self, x_min=-10, x_max=10, n_points=500):
        """
        Plot each fitted RBP marginal PDF over the real line against its prior KDE.
        Useful for checking whether each marginal transformation looks sensible.
        """
        from R_BP_james import R_BP_density_U

        if not hasattr(self, 'rbp') or not hasattr(self.rbp, 'rhos'):
            raise ValueError("The RBP model has not been fitted yet.")

        x_grid = np.linspace(x_min, x_max, n_points)
        fig, axes = plt.subplots(1, self.observation_d, figsize=(4 * self.observation_d, 4), squeeze=False)

        for j in range(self.observation_d):
            kde = self.kde_prior.kdes[j]
            prior_p = kde(x_grid)
            prior_c = np.mean(ndtr((x_grid - kde.dataset.T) / np.sqrt(kde.covariance[0, 0])), axis=0)
            rbp_p, _ = R_BP_density_U(self.rbp.rhos[j], prior_p, prior_c, self.rbp.pivots[:, j])

            axes[0, j].plot(x_grid, prior_p, linestyle='--', alpha=0.7, label='prior KDE')
            axes[0, j].plot(x_grid, rbp_p, linewidth=2, label='RBP PDF')
            axes[0, j].set_title(f'Marginal {j + 1}')
            axes[0, j].set_xlabel('x')
            axes[0, j].set_ylabel('density')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def check_conditional_vine_pdfs(self, n_points=200):
        """
        Inspect each conditional vine copula pdf while conditioning on the last
        observed time-series point and varying only the conditioned coordinate.
        """
        if not hasattr(self, 'cond_vines') or len(self.cond_vines) == 0:
            raise ValueError("No conditional vine copulas have been fitted yet.")
        if not hasattr(self, 'last_state_postrbp'):
            raise ValueError("The fitted model has no stored last state for conditioning.")

        u_grid = np.linspace(1e-3, 1 - 1e-3, n_points)
        n_vines = len(self.cond_vines)
        fig, axes = plt.subplots(1, n_vines, figsize=(4 * n_vines, 4), squeeze=False)

        for i, vine in enumerate(self.cond_vines):
            # The conditional vine is fit on [conditioned, conditioning] blocks,
            # so fix the conditioning variables at the last observed post-RBP state
            # and pad any missing slots to match the model dimension.
            d = len(vine.conditioning_set) + 1
            cond_values = np.asarray(self.last_state_postrbp, dtype=float)
            cond_values = cond_values[: max(0, d - 1)]
            if cond_values.size < d - 1:
                cond_values = np.pad(cond_values, (0, d - 1 - cond_values.size), constant_values=0.5)

            x = np.full((n_points, d), 0.5)
            x[:, 0] = u_grid
            x[:, 1:] = np.tile(cond_values, (n_points, 1))
            vals = vine.pdf(x)
            axes[0, i].plot(u_grid, vals, linewidth=2)
            axes[0, i].set_title(f'Conditional vine {i + 1}')
            axes[0, i].set_xlabel('u')
            axes[0, i].set_ylabel('pdf')
            axes[0, i].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()
        return fig, axes

    def check_rbp_times_conditional_vine(self, x_grid=None, n_points=200):
        """
        Plot the product of each fitted marginal RBP density and its paired
        conditional vine copula pdf on the same grid of values after passing the
        grid through the fitted prior KDE.
        The RBP and conditional vine at the same array index are paired together.
        """
        if not hasattr(self, 'cond_vines') or len(self.cond_vines) == 0:
            raise ValueError("No conditional vine copulas have been fitted yet.")
        if not hasattr(self, 'rbp') or not hasattr(self.rbp, 'rhos'):
            raise ValueError("The RBP model has not been fitted yet.")
        if not hasattr(self, 'last_state_and_action_postrbp'):
            raise ValueError("The fitted model has no stored last state for conditioning.")

        if x_grid is None:
            x_grid = np.linspace(-5, 5, n_points)
        x_grid = np.asarray(x_grid, dtype=float)
        if x_grid.ndim != 1:
            raise ValueError("x_grid must be a 1D array of real-line values.")

        n_pairs = min(len(self.cond_vines), len(self.rbp.rhos))
        fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), squeeze=False)

        for i in range(n_pairs):
            prior_grid = np.tile(x_grid, (self.d, 1)).T
            prior_p, prior_c = self.kde_prior.eval(prior_grid)
            rbp_pdf, rbp_cdf = self.rbp.eval(prior_p, prior_c)
            rbp_pdf = rbp_pdf[:, i] if rbp_pdf.ndim > 1 else rbp_pdf
            rbp_cdf = rbp_cdf[:, i] if rbp_cdf.ndim > 1 else rbp_cdf

            vine = self.cond_vines[i]
            d = len(vine.conditioning_set) + 1
            cond_values = np.asarray(self.last_state_and_action_postrbp, dtype=float)
            cond_values = cond_values[: max(0, d - 1)]
            if cond_values.size < d - 1:
                cond_values = np.pad(cond_values, (0, d - 1 - cond_values.size), constant_values=0.5)

            vine_grid = np.full((x_grid.size, d), 0.5)
            vine_grid[:, 0] = rbp_cdf
            vine_grid[:, 1:] = np.tile(cond_values, (x_grid.size, 1))
            vine_pdf = vine.pdf(vine_grid)

            product = rbp_pdf * vine_pdf
            axes[0, i].plot(x_grid, product, linewidth=2)
            axes[0, i].set_title(f'RBP {i + 1} × conditional vine {i + 1}')
            axes[0, i].set_xlabel('x')
            axes[0, i].set_ylabel('product')
            axes[0, i].grid(True, alpha=0.3)

        fig.tight_layout()

        return fig, axes
    
    def pdf_predictive(self, data, action=0):
        if data.ndim != 2:
            data = np.expand_dims(data, axis=0)
        assert data.shape[0] == 1, "Currently we can only check one state at a time."

        data_and_action = np.concatenate((data, np.array([[action]])), axis=1)

        cond_vine_pdfs = np.zeros((1, self.d-1))
        cond_vine_cdfs = np.zeros((1, self.d-1))
        
        p, c = self.kde_prior.eval(data_and_action)
        rbp_pdfs, c = self.rbp.eval(p, c)

        #last_state_and_action_postrbp = np.concatenate(([self.last_state_postrbp], [c[:, -1]]), axis = 1)

        ts = np.expand_dims(np.concatenate((c, self.last_state_and_action_postrbp), axis = None), 0)

        for i, cond in enumerate(self.conditioned_set):
            mask = np.concatenate(([cond - 1], self.conditioning_set - 1))
            cond_vine_pdfs[:,i] = self.cond_vines[i].pdf(ts[:, mask])
            cond_vine_cdfs[:,i] = self.cond_vines[i].cdf_implicit(ts[:, mask])

        tv_p = self.tv_cop.pdf(torch.from_numpy(cond_vine_cdfs).squeeze()).numpy()
        #print(tv_p, rbp_pdfs, cond_vine_pdfs)
        return np.prod(np.concatenate((rbp_pdfs, cond_vine_pdfs), axis = 1), axis = 1)*tv_p


    def predict_next_state(self, action):
        pass

if __name__ == "__main__":

    env = gym.make("InvertedPendulum-v5")  
    # Reset environment to start a new episode
    observation, info = env.reset(seed=123) 
    # observation: what the agent can "see" - cart position, velocity
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")

    n = 200 #truncation limit
    obs_d = env.observation_space.shape[0]
    act_d = env.action_space.shape[0]
    print(obs_d, act_d)

    episode_over = False
    total_reward = 0

    print(obs_d, act_d)
    observations = np.zeros((n+1, obs_d))
    actions = np.zeros((n+1, act_d))
    set_actions = np.genfromtxt('C:/Users/woodg/Documents/Vine_dissertation_chaeyun/actions.csv', delimiter=',')
    set_a_l = len(set_actions)
    actions[:set_a_l,0] = set_actions
    observations[0] = observation
    i=1
    while not episode_over:
        # Choose an action: 0 = push cart left, 1 = push cart right
        if i>set_a_l:
            actions[i-1] = env.action_space.sample()  # Random action for now - real agents will be smarter!
        # Take the action and see what happens
        observations[i], reward, terminated, truncated, info = env.step(actions[i-1])

        total_reward += reward
        episode_over = terminated or truncated
        i+=1
    if i <= n+1:
        observations = observations[:i]
        actions = actions[:i]
    # if i>20:
    #     np.savetxt("actions.csv", actions, delimiter=",")
    #     exit = True
    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

    data = np.concatenate((observations, actions), axis=1)

    print("Starting Fit")
    tv = tv_vinecop()
    tv.fit(data, check_vines=False)
    print("Fit finished")
    def target_func(copula:tv_vinecop, action, log = False):
        if log:
            def log_pdf(x):
                return np.log(copula.pdf_predictive(x, action))

            return log_pdf

        def pdf(x):
            return (copula.pdf_predictive(x, action))
              
        return pdf
    

    #print(tv.pdf_predictive(np.array([observations[-2]]),0))
    # fig, ax = tv.check_rbp_times_conditional_vine()

    # plt.savefig("C:/Users/woodg/Documents/Vine_dissertation_chaeyun/plots_rl/conditionals")
    # start = np.repeat([observations[0]], repeats=2, axis = 0)
    # start_act = np.repeat([actions[0]], repeats = 2, axis = 0)
    # tv.jump_back(start, start_act)

    tv.step_forward(np.array([observations[0]]), actions[0][0])

    print("Starting Metropolis Hastings")
    var = .0001
    prop = mh.mvn_def(covariance=var)
    sample = mh.sample_mvn(covariance=var)
    predicted_next = np.zeros_like(observations)
    predicted_next[0] = observations[0]

    for i, act in enumerate(actions[0:-1]):
        target = target_func(tv, act[0], log=True)
        #chain, ar = mh.metropolis_hastings(target, prop, sample, np.array(observations[i+2]), symmetric_proposal=True, chain_length=800, burn_in=500)
        chain = mh.adaptive_mh(target, torch.tensor(predicted_next[i], dtype = torch.double), torch.eye(4, dtype=torch.double)*var, nmoves = 700, return_entire_chain=True, adapt_no=100, burn_in=400)

        # fig, ax = plt.subplots(1,4)
        # for j in range(4):
        #     ax[j].plot(chain[:,j])
        # plt.savefig(f"C:/Users/woodg/Documents/Vine_dissertation_chaeyun/plots_rl/chain{i}")
        # plt.close()

        predicted_next[i+1] = torch.mean(chain, axis = 0)
        print(i, act[0])
        tv.step_forward(np.array([predicted_next[i+1]]), actions[i+1][0])

    fig, ax = plt.subplots(2,4)

    for i in range(4):
        ax[0, i].plot(observations[1:,i])
        ax[0, i].plot(predicted_next[1:,i])
        ax[1, i].plot(observations[1:,i], predicted_next[1:,i], '.')

    #print(observations[-11], observations[-10], np.mean(chain, axis=0), observations[-9])
    plt.savefig("C:/Users/woodg/Documents/Vine_dissertation_chaeyun/plots_rl/roll_out_no_observations")
    plt.close()


    

