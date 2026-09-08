import numpy as np
import copy
from abc import ABCMeta, abstractmethod, abstractproperty

class Method(metaclass = ABCMeta):
    """
        This abstract base class represents an inference method.

    """

    def __getstate__(self):
        """Cloudpickle is used with the MPIBackend. This function ensures that the backend itself
        is not pickled
        """
        state = self.__dict__.copy()
        del state['backend']
        return state


class LSPI(Method):
    def __init__(self, mdp, horizon, backend):   
        self.mdp = mdp
        self.horizon = horizon
        self.action_space = mdp.action_space
        self.n_action = len(self.action_space)
        self.n_grid_basis = 4
        self.len_basis = int((1 + self.n_grid_basis**2)*self.n_action)
        self.gamma = 0.99
        self.backend = backend
        self.w = np.zeros(self.len_basis)   

    # Defining the basis function for state-action pairs
    def phi(self, state, action):
        len_basis_for_each_action = int(1 + self.n_grid_basis**2)
        basis = np.zeros(self.len_basis)    
        # Defining the mu vector    
        mu1 = np.linspace(-np.pi/4, np.pi/4, self.n_grid_basis) 
        mu2 = np.linspace(-1.0, 1.0, self.n_grid_basis)    
        mu = [[angle, velocity] for angle in mu1 for velocity in mu2]
        sigma = 1
        # defining the radial basis function
        rbf = np.zeros(len(mu))
        for i in range(len(rbf)):
            rbf[i] = np.exp(-(np.linalg.norm(state - mu[i])) / (2 * (sigma**2)))
        action_index = np.where(self.action_space == action)[0][0]
        # print("action_index",action_index)
        #defining the basis function for the specific action
        # print("basis",basis[int(action_index * len_basis_for_each_action)])
        basis[int(action_index * len_basis_for_each_action)] = 1
        for ind in range(len(rbf)):
            basis[int(action_index * len_basis_for_each_action + ind + 1)] = rbf[ind]
        return basis.flatten()
    
    def random_argmax(self, values):
        max_indices = np.where(values == np.max(values))[0]
        if len(max_indices) == 0:
            print("No maxima found")
            return np.random.choice(range(len(values)))

        elif len(max_indices) == 1:
            # If there is only one maximum, return it
            return max_indices[0]
        else:
            # If there are multiple maxima, randomly choose one
            return np.random.choice(max_indices)    
    
    
    def lstdq(self,states, actions, nstates, rewards, policy):
        # print("lstdq called and states, actions, nstates, rewards ", states, actions, nstates, rewards)
        delta = 1e-3
        B = np.eye(self.len_basis)*(1/delta)
        b = np.zeros((self.len_basis, 1))

        for ind_t in range(states.shape[0]):   
            # if rewards[ind_t] == 1.0: 
            action_s_dash = policy(nstates[ind_t]) 
            phi_sa = self.phi(states[ind_t], actions[ind_t]).reshape((self.len_basis, 1))
            phi_sa_dash = self.phi(nstates[ind_t], action_s_dash).reshape((self.len_basis, 1))
            loss = phi_sa - self.gamma * phi_sa_dash
            B_num = B @ phi_sa @ (loss.T) @ B
            B_den = 1 + ((loss.T) @ B @ phi_sa)
            B = B - B_num/B_den
            b = b + phi_sa * rewards[ind_t]
            
        w = B @ b
        # print("w_shape",w.shape)
        return w.flatten()
    
    
    def calc_policy1(self, state, w):
        q_values = []
        for action in self.action_space:
            q = w.T @ self.phi(state, action)   
            q_values.append(q)
        opt_action_index = self.random_argmax(q_values)
        # print("q_values",q_values, "opt_action_index",opt_action_index)   
        return self.action_space[opt_action_index]
    
    def calc_policy(self, state, w):
        q_values = np.zeros(self.n_action)
        for ind_action in range(self.n_action):
            action = self.action_space[ind_action]
            q_values[ind_action]= w.T @ self.phi(state, action)   
            
        opt_action_index = self.random_argmax(q_values)
        # print("q_values",q_values, "opt_action_index",opt_action_index)   
        return self.action_space[opt_action_index]
    
    def random_policy(self, state):
        return np.random.choice(self.action_space)

    def sequential_lspi(self, states, actions, nstates, rewards, split_len):
        for i in range(0, len(states), split_len):
            self.simple_lspi(states[:i+split_len], actions[:i+split_len], nstates[:i+split_len], rewards[:i+split_len])

    
    def simple_lspi(self, states, actions, nstates, rewards):
        n_teration = 30
        epsilon = 1e-3

        # w_old = np.zeros(self.len_basis)
        # w_old = self.w
        for i in range(n_teration):
            policy_old = lambda x: self.calc_policy(x, self.w)
            w_new = self.lstdq(states, actions, nstates, rewards, policy_old)
            diff_w = np.linalg.norm(self.w-w_new)/self.len_basis
            # print("w_norm at iter",i, ":::", np.linalg.norm(w_new))
            # print("diff_w at iter",i, ":::", diff_w)
            self.w = w_new
            if diff_w < epsilon:
                # print("converged at iteration: ",i)   
                break
        # print("simple lspi w shape", w_new.flatten().shape)
        return self.w.flatten()
        # self.w = w_new.flatten()

    def weighted_lspi(self, theta, IS_weight):
        np.random.seed(1)
        if np.all(IS_weight == 0) :
            IS_weight = np.ones(len(IS_weight))/ len(IS_weight)
        else:
            IS_weight = IS_weight/sum(IS_weight)

        epsilon = 10e-4 
        n_iteration = 30

        ## Simulate data from the random policy for each theta  and this same data will be used for training thereafter
        all_s, all_a, all_s1, all_r = [], [], [], []

        for ind_theta in range(theta.shape[0]):
            self.mdp.model_parameters = theta[ind_theta, :]
            s1, a1, s2, r1= self.mdp.simulator(horizon=self.horizon, policy=self.random_policy)
            all_s.append(s1)
            all_a.append(a1)
            all_s1.append(s2)
            all_r.append(r1)

        # self.backend.broadcast(all_s, all_a, all_s1, all_r)
        # policy_old = lambda x: 

        self.all_s_bds = self.backend.broadcast(all_s)
        self.all_a_bds = self.backend.broadcast(all_a)
        self.all_s1_bds = self.backend.broadcast(all_s1)
        self.all_r_bds = self.backend.broadcast(all_r)
        # w_old = np.zeros(self.len_basis)
        w_old = self.w

        for ind_iter in range(n_iteration):
            # w_new = []

            policy_old = lambda x: self.calc_policy(x, w_old)
            parallel_lstdq = lambda x: self.lstdq(self.all_s_bds.value()[x], self.all_a_bds.value()[x],
                                                self.all_s1_bds.value()[x], self.all_r_bds.value()[x], policy_old)
            # w_new = self.lstdq(states, actions, nstates, rewards, policy_old)

            sample = [x for x in range(len(all_s))]
            sample_pds = self.backend.parallelize(sample)
            ##define map function
            # new_theta = lambda x: self.mh_kernel_grad(x, sigma, target_multiplier, epsilon, mu, b, if_sgld, if_mala)
            new_sample_pds = self.backend.map(parallel_lstdq, sample_pds)
            w_new = np.array(self.backend.collect(new_sample_pds))
            # print("w_new", w_new)
            # print("w shape", np.array(w_new).shape)
            w_new_weighted = np.average(np.array(w_new), axis = 0, weights = IS_weight) 
            diff_w = np.linalg.norm(w_old - w_new_weighted) / self.len_basis
            print("diff_w at iter",ind_iter, ":::", diff_w)
            self.w = w_new_weighted
            # w_old = copy.deepcopy(w_new_weighted)
            if diff_w < epsilon:
                print("converged at iteration: ",ind_iter)   
                break
        return self.w.flatten()

    def weighted_lspi_data(self, all_s, all_a, all_s1, all_r, IS_weight):
        if np.all(IS_weight == 0) :
            IS_weight = np.ones(len(IS_weight))/ len(IS_weight)
        else:
            IS_weight = IS_weight/sum(IS_weight)

        epsilon = 10e-4 
        n_iteration = 50

        all_s_bds = self.backend.broadcast(all_s)
        all_a_bds = self.backend.broadcast(all_a)
        all_s1_bds = self.backend.broadcast(all_s1)
        all_r_bds = self.backend.broadcast(all_r)

        w_old = self.w

        for ind_iter in range(n_iteration):
            # w_new = []

            policy_old = lambda x: self.calc_policy(x, w_old)
            parallel_lstdq = lambda x: self.lstdq(all_s_bds.value()[x], all_a_bds.value()[x],
                                                all_s1_bds.value()[x], all_r_bds.value()[x], policy_old)
            # w_new = self.lstdq(states, actions, nstates, rewards, policy_old)
            # print("states bds", all_s_bds.value()[1])
            sample = [x for x in range(len(all_s))]
            sample_pds = self.backend.parallelize(sample)
            ##define map function
            # new_theta = lambda x: self.mh_kernel_grad(x, sigma, target_multiplier, epsilon, mu, b, if_sgld, if_mala)
            new_sample_pds = self.backend.map(parallel_lstdq, sample_pds)
            w_new = np.array(self.backend.collect(new_sample_pds))
            # print("w_new", w_new)
            # print("w shape", np.array(w_new).shape)
            w_new_weighted = np.average(np.array(w_new), axis = 0, weights = IS_weight) 
            diff_w = np.linalg.norm(w_old - w_new_weighted) / self.len_basis
            # print("diff_w at iter",ind_iter, ":::", diff_w)
            self.w = w_new_weighted
            # w_old = copy.deepcopy(w_new_weighted)
            if diff_w < epsilon:
                print("converged at iteration: ",ind_iter)   
                break
        return self.w.flatten()





























        
        # for ind_sample_theta in range(theta.shape[0]):
        #     self.mdp.model_parameters = theta[ind_sample_theta, :]
        #     s1, a1, s2, r1= self.mdp.simulator(horizon=self.horizon, policy=self.random_policy)
        #     all_s.append(s1)
        #     all_a.append(a1)
        #     all_s1.append(s2)
        #     all_r.append(r1)

        
        # ### LSPI starts here  
        # weighted_w = np.zeros(self.len_basis)
        # for ind_iter in range(n_iteration):
        #     new_w = []
        #     new_policy = lambda x:  self.calc_policy(x, weighted_w)
        #     for ind_sample_theta in range(theta.shape[0]):
        #         w = self.lstdq(all_s[ind_sample_theta], all_a[ind_sample_theta], all_s1[ind_sample_theta], all_r[ind_sample_theta], policy=new_policy)
        #         print("w_norm for sample:", ind_sample_theta, "is: ", np.linalg.norm(w))
        #         new_w.append(w) 
        #         # value = self.estimated_v(np.array([0.0, 0.0]), new_w[ind_sample_theta]) 
        #         # print("Value at iteration: ", ind_iter, "theta: ", ind_sample_theta, "Value: ", value)
        #     # print(np.array(new_w).shape)    
        #     new_weighted_w = np.average(np.array(new_w), axis = 0, weights = IS_weight)
        #     # value = self.estimated_v(np.array([0.0, 0.0]), new_weighted_w) 
        #     # print("Value(avg w used) at iteration: ", ind_iter, "Value: ", value)
        #     diff_w = np.linalg.norm(weighted_w - new_weighted_w) / self.len_basis
        #     print(ind_iter, "Diff: ", diff_w)     
        #     # print("Weighted W: ", new_weighted_w)    
        #     if diff_w < epsilon:  # break if converged
        #         print("Converged at iteration: ", ind_iter)
        #         break
        #     weighted_w = copy.deepcopy(new_weighted_w)

        # return weighted_w
            
            



