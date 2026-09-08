import numpy as np  

class pendulum:
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        self.action_space = np.array([-50.0, 0, 50.0])
        self.n_action = len(self.action_space)

    def dynamic_acc(self, state, action):
        g = self.model_parameters[0]
        m = self.model_parameters[1]
        M = self.model_parameters[2]
        l = self.model_parameters[3]
        noise = self.model_parameters[4]    

        alpha = 1.0 / (m + M)
        u = action + np.random.uniform(-noise, noise, 1)[0]
        acc_num = (g * np.sin(state[0])) - (
                alpha * m * l * (state[1] ** 2) * (np.sin(2 * state[0])) / 2.0) - (alpha * np.cos(state[0]) * u)
        acc_den = (4 * l / 3) - (alpha * m * l * (np.cos(state[0]) ** 2))
        

        return (acc_num / acc_den)

    def next_state_reward(self, n_simulation, state, action):
        t = self.model_parameters[5]

        next_state = np.zeros((n_simulation, 2))
        next_reward = np.zeros((n_simulation, 1))

        acc = np.zeros(n_simulation)
        for ind_sim in range(n_simulation):
            acc[ind_sim] = self.dynamic_acc(state, action)

            next_state[ind_sim, 1] = state[1] + t * acc[ind_sim]
            next_state[ind_sim, 0] = state[0] + t * state[1]

            if np.abs(next_state[ind_sim, 0]) <= np.pi / 2:
                next_reward[ind_sim] = 1.0
        return next_state, next_reward

    def simulator(self, horizon, policy):
        s_history = np.random.uniform(low=-1e-10, high=1e-10, size=(horizon, 2))
        s_dash_history = np.zeros((horizon, 2))
        a_history = np.zeros((horizon, 1))
        r_history = np.zeros((horizon, 1))

        for time in range(horizon):
            a_history[time] = policy(s_history[time, :])
            next_state, next_reward = self.next_state_reward(n_simulation=1, state=s_history[time, :],
                                                             action=a_history[time])
            s_dash_history[time, :] = np.squeeze(next_state) 
            r_history[time] = np.squeeze(next_reward)      

            if np.abs(s_dash_history[time, 0]) <= np.pi / 2:
                if time < horizon - 1:
                    s_history[time + 1, :] = s_dash_history[time, :]
        
        return s_history, a_history, s_dash_history, r_history
    
    def simulate_episode(self, first_state, policy, max_length=1000):
        ###first_state should be a numpy array of shape (2,)
        State_epi, Action_epi, Reward_epi, State_epinext = [], [], [], []
        state = first_state
        
        done = False
        while not done:
            State_epi.append(state)
            action = policy(state)
            ns, r = self.next_state_reward(n_simulation=1, state=state, action=action)
            nstate, reward = np.squeeze(ns), np.squeeze(r)

            State_epinext.append(nstate)
            Reward_epi.append(reward)
            Action_epi.append(action)
            
            if reward==0 or len(State_epi) >= max_length:
                done = True
            else:
                state = nstate
           
        np_state_epi = np.array(State_epi)  
        np_action_epi = np.array(Action_epi).reshape((len(Action_epi), 1))
        np_state_epinext = np.array(State_epinext)
        np_reward_epi = np.array(Reward_epi).reshape((len(Reward_epi), 1))

        return np_state_epi, np_action_epi, np_state_epinext, np_reward_epi