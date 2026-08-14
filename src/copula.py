import numpy as np
from scipy.stats import norm, multivariate_normal
import pyvinecopulib as pv
from scipy.optimize import minimize
from scipy.special import expit
import torch
import math


#####################
#  Gaussian Copula  #
#####################


# Gaussian Copula Density : c_rho
def gaussian_copula_density(u, v, rho):

    # u, v in [ 10^{-6}, 1-10^{-6} ]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    numerator = np.exp( - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2)))
    denominator = np.sqrt(1 - rho**2)

    return numerator / denominator


# Gaussian Copula CDF : C_rho
def gaussian_copula_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    return multivariate_normal.cdf([z_u, z_v], mean=mean, cov=cov)

# Conditional Gaussian Copula CDF : H_rho
def gaussian_conditional_copula_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    rho = np.clip(rho, -1 + 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    return norm.cdf((z_u - rho * z_v) / np.sqrt(1 - rho**2))

# Multivariate
def gaussian_copula_density_multivariate(u, R):

    eps = 1e-6

    u = np.asarray(u)
    u = np.clip(u, eps, 1 - eps)

    z = norm.ppf(u)

    d = len(z)

    R = np.asarray(R)

    sign, logdet = np.linalg.slogdet(R)

    if sign <= 0:
        return 1e-12

    R_inv_z = np.linalg.solve(R, z)

    exponent = -0.5 * (z @ R_inv_z - z @ z)

    log_density = -0.5 * logdet + exponent

    density = np.exp(log_density)

    density = max(density, 1e-12)

    return density



#####################
#    Vine Copula    #
#####################

def expand_mat(mat):
    n = mat.shape[0]
    u = np.zeros(n, dtype=int)
    v = np.zeros((n+1,1), dtype=int)
    y_mat = np.vstack((mat, u))
    y_mat = np.hstack((v, y_mat))
    return y_mat

def show_bvcops_frommat(mat):
    d = mat.shape[0]
    for t in range(d-1):
        for e in range(d-t-1):
            if t > 0:
                print(t,e, "(",mat[d-1-e, e],",",mat[t,e],"|", end=" ")
                for i in range(t):
                    print(mat[t-i-1,e], end=" ")
                print(")",d-1-e, e)     
            else:
                print(t,e, "(",mat[d-1-e, e],",",mat[t,e],")")

def get_copulas_fm(mat, t=None, e = None):
    d = mat.shape[0]
    n = int(d*(d-1)/2)
    cops = np.zeros((n, d), dtype = int)
    #t=0S
    for e in range(d-1):
            c = d-2-e
            cops[c, 0] = mat[d-1-e, e]
            cops[c, 1] = mat[0,e]
    #t>0
    jump = d-1
    for t in range(1, d-1):
        for e in range(d-t-1):
            c = d-t-2-e
            cops[jump+c, 0] = mat[d-1-e, e]
            cops[jump+c, 1] = mat[t,e]
            # idx | a...
            for i in range(t):
                cops[jump+c, i+2] = mat[t-i-1,e]
        jump += d-1-t
    return cops


def set_matrix_fc(cops):
    d = cops.shape[1]
    n = int(d*(d-1)/2)
    mat = np.zeros((d,d), dtype = int)
    jump = 0
    options = np.arange(1,d)

    for e in range(d-1):
        c = d-2-e
        mat[d-1-e, e] = cops[c, 0]
        mat[0,e] = cops[c, 1]
        options = options[options!=cops[c,0]]
    mat[0,d-1] = options[0]

    #t>0
    jump = d-1
    for t in range(1, d-1):
        for e in range(d-t-1):
            c = d-t-2-e
            mat[d-1-e, e] = cops[jump+c, 0]
            mat[t,e] = cops[jump+c, 1]
            # idx | a...
            for i in range(t):
                mat[t-i-1,e] = cops[jump+c, i+2] 
        jump += d-1-t

    return mat

def find_proxims(cops, mother):
    mask = np.isin(cops, mother)
    n = len(mother)
    proxims = cops[np.sum(mask, axis=1)==n]
    return proxims

def add_leaf_copulas(cops, leafidx = 1):
    '''
    Adds a leaf node to a pre-established R vine and calculates the copulas associated with it.

    Currently chooses the first correct node available, but may be expanded to give all combinations for selection,
    or select the best one (not sure that this is available)

    Inputs:
        cops: numpy array, (d(d-1)/2, d), a copula array
        leafidx: int, the index of the node in cops to attach the leaf to. Must be less than d.

    Returns:
        expanded: numpy array, ((d+1)d/2, d+1), a copula array.
    '''
    n, d = cops.shape
    assert leafidx <= d, "Leaf index should be contained in the original vine"

    expanded = cops
    zs = np.zeros(d, dtype = int)
    zs[0] = d+1
    jump = d-1

    for i in range(d-1):
        expanded = np.insert(expanded, jump, zs, axis=0)
        #np.insert(conds, [1], [2,3,4], axis=0)
        jump += d-1-i
    expanded = np.append(expanded, [zs], axis=0)
    expanded = np.append(expanded, np.zeros((n+d, 1), dtype = int), axis = 1)

    jump = 0
    jump2 = d
    expanded[d-1,1] = leafidx

    for t in range(1,d):
        # Satisfy the proximity condition: 
        # Which copulas come from the same node as the leaf indexed copula in the previous tree
        proxims = find_proxims(expanded[jump:jump2-1], expanded[jump2-1, 1:t+1])
        print(proxims)
        a = proxims[0, 0:t+1] #where connected
        b = expanded[jump2-1, 0:t+1]

        conditioning = np.setxor1d(b, a)
        conditioned = np.intersect1d(a, b, return_indices=True)
        idxs = np.sort(conditioned[2])
        #Get the start and end of the new tree of copulas in the array.
        jump += d-t+1
        jump2 += d-t

        #Edit the copula involving the leaf in the new tree
        expanded[jump2-1, 1] = conditioning[0]
        expanded[jump2-1, 2:t+2] = b[idxs]

    return expanded

#This is obsolete unless you want to fit all (awful in high dimensions)
#Also, it doesn't work... (right now)
def add_leaf_copulas_variations(cops, leafidx = 1):
    '''
    Adds a leaf node to a pre-established R vine and calculates the copulas associated with it.

    Currently chooses the first correct node available, but may be expanded to give all combinations for selection,
    or select the best one (not sure that this is available)

    Inputs:
        cops: numpy array, (d(d-1)/2, d), a copula array
        leafidx: int, the index of the node in cops to attach the leaf to. Must be less than d.

    Returns:
        expanded: numpy array, ((d+1)d/2, d+1), a copula array.
    '''
    n, d = cops.shape
    assert leafidx <= d, "Leaf index should be contained in the original vine"

    expanded = cops
    zs = np.zeros(d, dtype = int)
    zs[0] = d+1
    jump = d-1

    for i in range(d-1):
        expanded = np.insert(expanded, jump, zs, axis=0)
        #np.insert(conds, [1], [2,3,4], axis=0)
        jump += d-1-i
    expanded = np.append(expanded, [zs], axis=0)
    expanded = np.append(expanded, np.zeros((n+d, 1), dtype = int), axis = 1)

    jump = 0
    jump2 = d
    expanded[d-1,1] = leafidx

    variations = np.array([expanded])

    for t in range(1,d):
        i = 0
        while i < 1:
            print(t, i, variations.shape[0])
            # Satisfy the proximity condition: 
            # Which copulas come from the same node as the leaf indexed copula in the previous tree
            proxims = find_proxims(variations[i, jump:jump2-1], variations[i, jump2-1, 1:t+1])
            print("proxims:",proxims)
            nproxims = len(proxims)

            for _ in range(nproxims-1):
                #vars = np.vstack((vars, np.copy([vars[i]])))
                variations = np.insert(variations, i+1, np.copy(variations[i]), axis = 0) # use np.insert?
            print(variations.shape[0])
            for k in range(nproxims):
                a = proxims[k, 0:t+1] #where connected
                b = variations[i+k, jump2-1, 0:t+1]
                print("i,k = ",i,k, "this is a,b",a,b)

                conditioning = np.setxor1d(b, a)
                #print("the conditioning set:",conditioning)
                conditioned = np.intersect1d(a, b, return_indices=True)
                idxs = np.sort(conditioned[2])
                #Get the start and end of the new tree of copulas in the array.
                jump += d-t+1
                jump2 += d-t

                #Edit the copula involving the leaf in the new tree
                variations[i+k, jump2-1, 1] = conditioning[0]
                variations[i+k, jump2-1, 2:t+2] = b[idxs]
                print("This is the new row:",variations[i+k, jump2-1, :], "\n ------")
                jump -= d-t+1
                jump2 -= d-t

            i += 1
            #print(variations, "\n ------")
        jump += d-t+1
        jump2 += d-t

    return variations


def add_leaf_to_mat(mat, leafidx = 1):
    cops = get_copulas_fm(mat)
    zcops = add_leaf_copulas(cops, leafidx)
    z = set_matrix_fc(zcops)
    return z

def add_leaf_to_mat_variations(mat, leafidx = 1):
    cops = get_copulas_fm(mat)
    zcops = add_leaf_copulas_variations(cops, leafidx)
    z = []
    for zc in zcops:
        zmat = set_matrix_fc(zc)
        z.append(zmat)
    return z

def fit_copula(u_y, u_x, k):
    '''
    Regression should go something like this:

    Fit model of order (1), store AIC
    Fit models of order (1, x), storing their AIC
    Compute AIC differences
    Pick model with minimum AIC difference
    Repeat with orders greater until AIC difference >0

    Try differently:
        Fit vine on X,
        Attach node Y based on Max log-likelihood. Perhaps MBIC? 
        Model should have roughly the same parameters though

        Now you have the conditional if you decontruct and cancel terms

        c(1|2,3,4) = c(1,2,3,4) 
                     ----------
                      c(1,2,3)
        This decontructs to c(1,2)c(1,3|2)c(1,4|3,2) and other 
        decompositions depending on the vine.
    
    '''

    xvine = pv.Vinecop.from_data(u_x)
    xmat = xvine.matrix
    d = xmat.shape[0]
    u_all = np.hstack((u_y, u_x))
    taus = np.zeros((d+1))

    for i in range(d-1):
        taus[i] = pv.wdm(u_x[i], u_y, method="ktau")
    #Find the maximal tau, then fit this copula.
    #Generate pseudo-observations for all copulas in T=1, then repeat the process for T>1.

    return(taus)
    

    # Do we need the bicop families chosen from before, if so how can we pass them and not choose the others?
    # Yes we need the previous ones, but only for pseudo-observations. The rest of the vine will be
    # instantiated by taking all the bicops, and the structure and putting it all together.

class conditional_vine_copula:
    '''
    Class to mimic some of pv.Vinecop for cleaner code.

    Can calculate the pdf of a conditional vine copula when passed a pv.Vinecop object constructed with a conditional set.
    '''
    def __init__(self, conditioning_set, vine:pv.Vinecop):
        if len(conditioning_set) == 1:
            print("WARNING: This wil not produce the true conditional pdf, you must not multiply by the marginal of the conditioning variable")
        if len(conditioning_set) == 2:
            print("WARNING: If the copula of the conditioning is not contained in the Vine, this will not give the correct value.")
        
        self.vine = vine
        self.trees_np = np.fromiter(chain.from_iterable(vine.get_trees()), dtype=object)

        self.subtree_mask = self.__find_subtree_mask(conditioning_set)
        self.conditioning_set = np.array(conditioning_set)
        self.ed_set = np.setdiff1d(vine.order, conditioning_set)
        print(self.ed_set)
        self.subtrees_np = self.trees_np[self.subtree_mask]

    def __find_subtree_mask(self, conditioning_set:tuple)->np.array:
        '''
        Finds the complement of the sub tree in the list of trees given. Returns a boolean array with True corresponding to the copulas which 
        should be evaluated.

        By definition in the conditioning-set control knob, there wil be a sub-tree containing the conditioning set, fitted first.
        Hence we only need to look for a tree of smaller depth!

        There are specific cases that do not require this function, such as if there are only two conditioned variables, positioned at either end of the order.
        '''
        mask = np.zeros_like(self.trees_np, dtype=bool)

        for i, cop in enumerate(self.trees_np):
            allc = cop['conditioned']+tuple(cop['conditioning'])
            # Is there a number that isn't in the conditioning set? Set to True
            mask[i] = False in np.isin(allc, conditioning_set)
        return mask
    
    @staticmethod
    def __t_idx(t:int, dim:int)->int:
        '''
        Finds the starting index of a tree along a flattened array of copulas.
        '''
        return int(t*(dim+1-(t+1)/2))

    def __eval_h_functions(self, u: np.array)-> tuple: #Should this be put into torch (we need gradients)?
        '''
        Evaluates all h functions required for the vine structure, therefore allowing conditional copulas to be evaluated
        from these values (From a structure containing the tree subset to condition on).
        '''
        cops = self.trees_np #Indexing: [tree][edges][other]. Order of edges is not necessarily increasing in ['conditioning'][0].
        n = u.shape[0]
        d = u.shape[1]
        n_cs = len(cops)
        h_evals = np.zeros((d+n_cs,2,n)) #Flatten over trees and edges so we have
                                                    # a nice data structure to deal with (i.e. numpy/torch)
                                                    # n_cops, 2, n
        h_evals[:d, 0] = h_evals[:d, 1] = u.T
        h_pointers = np.zeros((n_cs, 2, 2), dtype=int)
        end_first_tree = self.__t_idx(1, d-1)
        # Go through the first tree, this works!
        for j, cop in enumerate(cops[0:end_first_tree]):
            c1, c2 = cops[j]['conditioned']
            c1 -= 1
            c2 -= 1
            h_evals[j+d,0] = cop['pair_copula'].hfunc1(u[:, [c1,c2]]) # c2|c1
            h_evals[j+d,1] = cop['pair_copula'].hfunc2(u[:, [c1,c2]]) # c1|c2

            mask = ((c1, c2),(0,0))
            h_pointers[j] = mask
        t=1
        #Go through trees > 1. This does not work with indexing!
        for j, cop in enumerate(cops[end_first_tree:]):

                t += (j+self.__t_idx(t, d-1))//self.__t_idx(t+1, d-1)
                point_start = self.__t_idx(t-1, d-1)
                new_point_start = self.__t_idx(t, d-1)
                old_eval_start = self.__t_idx(t, d)
                #I want to find the h_eval required to compute the next one correctly according to the vine.
                c1, c2 = cop['conditioned']
                cing = cop['conditioning']
                up1 = np.append(cing, c1)
                up2 = np.append(cing, c2)
                nup = t+1

                count = 0
                #Can I vectorise this - numpy?
                #Find the h_evals in the vine above corresponding to the current copula.
                for k, upcop in enumerate(cops[point_start:new_point_start]):
                    allc = upcop['conditioned']+tuple(upcop['conditioning'])

                    if np.sum(np.isin(up1, allc))==nup: # Use np.all?
                        idx1 = k
                        count += 1
                        if count == 2:
                            break
                    elif np.sum(np.isin(up2, allc))==nup:
                        idx2 = k
                        count += 1
                        if count == 2:
                            break
                #Which h-function should I use?
                if cops[point_start+idx1]['conditioned'][0]==c1:
                    funcidx1 = 1
                else:
                    funcidx1 = 0
                if cops[point_start+idx2]['conditioned'][0]==c2:
                    funcidx2 = 1
                else:
                    funcidx2 = 0

                mask = ((old_eval_start+ idx1, old_eval_start+ idx2),(funcidx1,funcidx2))# Points to correct h_evals
                h_pointers[end_first_tree+j] = mask

                #Assign h function evaluations # if statement has not fixed it.
                h_evals[d+end_first_tree+j,0] = cop['pair_copula'].hfunc1(h_evals[mask].T)
                h_evals[d+end_first_tree+j,1] = cop['pair_copula'].hfunc2(h_evals[mask].T)

        return h_evals, (h_pointers[:,0], h_pointers[:,1])
    
    def pdf(self, data: np.array)->np.array:
        density = 1 # This expands out to nd automatically if data is nd
        h_evals, h_points = self.__eval_h_functions(data)

        for i, mask in enumerate(self.subtree_mask):
            if mask:
                density *= self.trees_np[i]['pair_copula'].pdf(h_evals[h_points][i].T)

        return density
    
    def cdf_numerical(self, data, n_points = 6):
        '''
        Numerical integration over all conditioned variables. This is more complex than it seems and infeasable in high dimensions,
        try Monte-Carlo Methods using Vinecop.simulate_conditional.
        '''
        
        ed_idxs = self.ed_set-1
        ing_idxs = self.conditioning_set-1
        print(ing_idxs)
        if len(ing_idxs) == 0:
            print("try cdf_mc")
            raise NotImplementedError
        elif len(ed_idxs) == 1:
            data_ing = data[:, ing_idxs]
            ed_idx = ed_idxs[0]
            data_ed = data[:, ed_idx]

            zero_to_data = np.vstack([
                np.linspace(1e-10, value, n_points)
                for value in data_ed
            ])
            print("zero to data :",zero_to_data)
            numer_inputs = np.repeat(data[:, None, :], n_points, axis=1)
            print(numer_inputs)
            numer_inputs[:, :, ed_idx] = zero_to_data
            print(numer_inputs)
            numer_inputs = numer_inputs.reshape(-1, data.shape[1])


            print(numer_inputs)
            y = self.pdf(numer_inputs).reshape(data.shape[0], n_points)
            print(y)

            return np.trapezoid(y, x=zero_to_data, axis=1)

        else:
            print("Not yet implemented, try cdf_mc")
            raise NotImplementedError
 
    def cdf_mc(self, data, n_samples = 100000):
        #Need to change
        if len(self.conditioning_set)>0:
            print(self.conditioning_set)
            data_ing = data[:, self.conditioning_set-1]
            data_ed = data[:, self.ed_set-1]
            conds = np.zeros((data_ing.shape[0]*n_samples, data_ing.shape[1]))
            # How to parallelise/optimise this?
            for i, point in enumerate(data_ing):
                conds[i*n_samples: (i+1)*n_samples] = np.repeat([point], n_samples, axis=0)

            samples = self.vine.simulate_conditional(conds, True, num_threads=4)
            samples = samples.reshape((data.shape[0], n_samples, data.shape[1]))

            p = np.mean(np.all(np.less(samples[:, :, self.ed_set-1], data_ed[:, None, :]), axis=2), axis = 1)
            return p
        else:
            samples = self.vine.simulate(n_samples)
            p = np.mean(np.all(np.less(samples, data[:, None, :]), axis=2), axis = 1)
            return p
import pyvinecopulib as pv

def fit_conditional_vines(U_condition, U_target):
    
    U_condition = np.asarray(U_condition)
    U_target = np.asarray(U_target)

    d = U_target.shape[1]

    vines = []
    for j in range(d):

        data_j = np.column_stack([U_target[:, j], U_condition])
        
        vine_j = pv.Vinecop.from_data(data_j)
        vines.append(vine_j)

    return vines

def conditional_vine_copula_pdf(vine, u, u_condition, n_grid=300):

    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)
    u_condition = np.asarray(u_condition)

    s_grid = np.linspace(eps, 1 - eps, n_grid)

    numerator_input = np.concatenate([[u], u_condition]).reshape(1, -1)
    numerator = vine.pdf(numerator_input)[0]

    denom_inputs = np.column_stack([s_grid, np.tile(u_condition, (n_grid, 1))])

    denom_values = vine.pdf(denom_inputs)
    denominator = np.trapz(denom_values, s_grid)

    cond_pdf = numerator / denominator
    cond_pdf = max(cond_pdf, 1e-12)

    return cond_pdf


def conditional_vine_copula_cdf(vine, u, u_condition, n_grid=300):
    
    eps = 1e-6
    u = np.clip(u, eps, 1 - eps)
    u_condition = np.asarray(u_condition)

    s_full = np.linspace(eps, 1 - eps, n_grid)

    denom_inputs = np.column_stack([s_full, np.tile(u_condition, (n_grid, 1))])

    denom_values = vine.pdf(denom_inputs)
    denominator = np.trapz(denom_values, s_full)

    s_part = np.linspace(eps, u, n_grid)

    numer_inputs = np.column_stack([s_part, np.tile(u_condition, (n_grid, 1))])

    numer_values = vine.pdf(numer_inputs)
    numerator = np.trapz(numer_values, s_part)

    cond_cdf = numerator / denominator
    cond_cdf = np.clip(cond_cdf, 0.0, 1.0)

    return cond_cdf



#########################
#  time-varying Copula  #
#########################
GAS_INFORMATION_RIDGE = 1e-4
GAS_INFORMATION_DECAY = 0.95

# gaussian 

def gaussian_parameter(f_G_i):

    eps = 1e-6

    m = f_G_i.numel()
    d = int((1 + math.sqrt(1 + 8 * m)) / 2)

    if d * (d - 1) // 2 != m:
        raise ValueError(f"Invalid Gaussian state length: {m}")

    # partial correlations between -1 and 1
    partial_rho = (1.0 - eps) * torch.tanh(f_G_i / 2.0)

    L_i = f_G_i.new_zeros((d, d))

    index = 0

    for row in range(d):

        remaining_scale = f_G_i.new_tensor(1.0)

        for col in range(row):

            rho = partial_rho[index]

            L_i[row, col] = rho * remaining_scale
            remaining_scale = remaining_scale * torch.sqrt(1.0 - rho**2)

            index += 1

        L_i[row, row] = remaining_scale

    R_i = L_i @ L_i.T

    return R_i

def gaussian_copula_log_density(z_i, R_i):
    
    eps = 1e-6
    identity = torch.eye(R_i.shape[0], dtype=R_i.dtype, device=R_i.device)

    # Cholesky decomposition of the correlation matrix
    L_i = torch.linalg.cholesky(R_i + eps * identity)

    log_det_R_i = 2.0 * torch.log(torch.diagonal(L_i)).sum()

    # R_i^{-1} * z_i 
    R_inv_z_i = torch.cholesky_solve(z_i.unsqueeze(1), L_i).squeeze(1)

    log_copula_density_i = -0.5 * (log_det_R_i + (torch.dot(z_i, R_inv_z_i) - torch.dot(z_i, z_i)))

    return log_copula_density_i


# Clayton 

def clayton_parameter(f_C_i):
    theta_i = torch.exp(f_C_i)
    return theta_i

def clayton_copula_log_density(u_tilde_i, theta_i):
  
    d = u_tilde_i.shape[0]

    a_i = -theta_i * torch.log(u_tilde_i)

    log_sum_exp_i = torch.logsumexp(a_i, dim=0)

    log_clayton_sum_i = (
        log_sum_exp_i
        + torch.log1p(
            -(d - 1.0) * torch.exp(-log_sum_exp_i)
        )
    )

    coefficient_index = torch.arange(
        1, d, dtype=theta_i.dtype, device=theta_i.device
    )

    log_coefficient_i = torch.sum(
        torch.log(1.0 + coefficient_index * theta_i)
    )

    log_marginal_term_i = (
        -(theta_i + 1.0) * torch.sum(torch.log(u_tilde_i))
    )

    log_generator_term_i = (
        -(d + 1.0 / theta_i) * log_clayton_sum_i
    )

    log_copula_density_i = (
        log_coefficient_i
        + log_marginal_term_i
        + log_generator_term_i
    )
    
    return log_copula_density_i


# Mixture

def gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight):
   
    log_weight = torch.nn.functional.logsigmoid(weight)
    log_one_minus_weight = torch.nn.functional.logsigmoid(-weight)

    log_c_mix_i = torch.logsumexp(torch.stack([ log_weight + log_c_G_i, log_one_minus_weight + log_c_C_i]), dim=0)

    return log_c_mix_i

def gaussian_clayton_mixture_raw_score(log_c_mix_i, f_G_i, f_C_i):
    
    raw_score_G_i, raw_score_C_i = torch.autograd.grad(log_c_mix_i, (f_G_i, f_C_i), create_graph=True)
    
    return raw_score_G_i, raw_score_C_i

def identity_scaling(raw_score_G_i, raw_score_C_i):
    
    scaled_score_G_i = raw_score_G_i
    scaled_score_C_i = raw_score_C_i
    
    return scaled_score_G_i, scaled_score_C_i

# square-root scaling
##################################
def square_root_information_scaling(
    raw_score_G_i,
    information_G_i,
    raw_score_C_i,
    information_C_i,
    ridge=GAS_INFORMATION_RIDGE
):

    identity = torch.eye(
        information_G_i.shape[0],
        dtype=information_G_i.dtype,
        device=information_G_i.device
    )

    information_cholesky = torch.linalg.cholesky(
        information_G_i + ridge * identity
    )

    scaled_score_G_i = torch.linalg.solve_triangular(
        information_cholesky,
        raw_score_G_i.unsqueeze(1),
        upper=False
    ).squeeze(1)

    scaled_score_C_i = (
        raw_score_C_i
        / torch.sqrt(information_C_i + ridge)
    )

    return scaled_score_G_i, scaled_score_C_i


def update_conditional_information(
    raw_score_G_i,
    information_G_i,
    raw_score_C_i,
    information_C_i,
    decay=GAS_INFORMATION_DECAY
):

    score_G = raw_score_G_i.detach()
    score_C = raw_score_C_i.detach()

    information_G_next = (
        decay * information_G_i
        + (1.0 - decay) * torch.outer(score_G, score_G)
    )

    information_C_next = (
        decay * information_C_i
        + (1.0 - decay) * score_C.square()
    )

    return information_G_next, information_C_next
#############################################
    
def gaussian_clayton_mixture_score_scaling(raw_score_G_i, S_G_i, raw_score_C_i, S_C_i):
    
    scaled_score_G_i = S_G_i @ raw_score_G_i
    scaled_score_C_i = S_C_i * raw_score_C_i
   
    return scaled_score_G_i, scaled_score_C_i

def gaussian_clayton_gas_update(f_G_i, scaled_score_G_i, omega_G, A_G, B_G, 
                                f_C_i, scaled_score_C_i, omega_C, A_C, B_C):
    
    f_G_next = omega_G + A_G * scaled_score_G_i + B_G * f_G_i
    f_C_next = omega_C + A_C * scaled_score_C_i + B_C * f_C_i
    
    return f_G_next, f_C_next


# Estismate

def gaussian_clayton_mixture_log_likelihood(u_tilde, z, weight,
                                             f_G_0, omega_G, A_G, B_G,
                                             f_C_0, omega_C, A_C, B_C,
                                             epoch=None):

    f_G_i = f_G_0
    f_C_i = f_C_0
    
    m = f_G_0.numel()

    information_G_i = torch.eye(m,dtype=u_tilde.dtype,device=u_tilde.device)
    information_C_i = u_tilde.new_tensor(1.0)
    
    log_likelihood = u_tilde.new_tensor(0.0)

    for i in range(u_tilde.shape[0]):
        R_i = gaussian_parameter(f_G_i)
        theta_i = clayton_parameter(f_C_i)

        f_G_i_finite = torch.isfinite(f_G_i)
        f_C_i_finite = torch.isfinite(f_C_i)
        R_i_finite = torch.isfinite(R_i)

        # copula log density
        try:
            log_c_G_i = gaussian_copula_log_density(z[i], R_i)

        except RuntimeError as error:

            if "cholesky" not in str(error).lower():
                raise

            print("epoch:", epoch)
            print("i:", i)
            print("f_G_i:", f_G_i.detach())
            print("f_C_i:", f_C_i.detach())
            print("R_i:", R_i.detach())
            print("torch.isfinite(f_G_i):", f_G_i_finite)
            print("torch.isfinite(f_C_i):", f_C_i_finite)
            print("torch.isfinite(R_i):", R_i_finite)

            try:
                print(
                    "torch.linalg.eigvalsh(R_i):",
                    torch.linalg.eigvalsh(R_i.detach())
                )
            except RuntimeError as eigenvalue_error:
                print(
                    "torch.linalg.eigvalsh(R_i): unavailable",
                    eigenvalue_error
                )

            raise

        log_c_C_i = clayton_copula_log_density(u_tilde[i], theta_i)

        # mixture
        log_c_mix_i = gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight)

        # score
        raw_score_G_i, raw_score_C_i = gaussian_clayton_mixture_raw_score(log_c_mix_i, f_G_i, f_C_i)
        scaled_score_G_i, scaled_score_C_i = square_root_information_scaling( raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)

        information_G_i, information_C_i = update_conditional_information( raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)
        
        log_likelihood = log_likelihood + log_c_mix_i

        f_G_next, f_C_next = gaussian_clayton_gas_update(
            f_G_i, scaled_score_G_i, omega_G, A_G, B_G,
            f_C_i, scaled_score_C_i, omega_C, A_C, B_C
        )

        f_G_next_finite = torch.isfinite(f_G_next).all()
        f_C_next_finite = torch.isfinite(f_C_next).all()

        if (
            not f_G_next_finite.item()
            or not f_C_next_finite.item()
        ):

            print("epoch:", epoch)
            print("i:", i)

            diagnostic_values = [
                ("weight", weight),
                ("omega_G", omega_G),
                ("A_G", A_G),
                ("B_G", B_G),
                ("omega_C", omega_C),
                ("A_C", A_C),
                ("B_C", B_C),

                ("f_G_i", f_G_i),
                ("f_C_i", f_C_i),
                ("R_i", R_i),
                ("theta_i", theta_i),

                ("log_c_G_i", log_c_G_i),
                ("log_c_C_i", log_c_C_i),
                ("log_c_mix_i", log_c_mix_i),

                ("raw_score_G_i", raw_score_G_i),
                ("raw_score_C_i", raw_score_C_i),

                ("scaled_score_G_i", scaled_score_G_i),
                ("scaled_score_C_i", scaled_score_C_i),

                ("f_G_next", f_G_next),
                ("f_C_next", f_C_next)
            ]

            for name, value in diagnostic_values:
                value_detached = value.detach()
                print(f"{name}:", value_detached)
                print(
                    f"torch.isfinite({name}):",
                    torch.isfinite(value_detached)
                )

            raise RuntimeError(
                f"First non-finite GAS update detected "
                f"at epoch={epoch}, i={i}"
            )

        f_G_i = f_G_next
        f_C_i = f_C_next

    return log_likelihood

def estimate_gaussian_clayton_gas(u_tilde, epochs=500, learning_rate=0.01,
                                  loss_history=None):

    u_tilde = torch.as_tensor(u_tilde, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    
    normal = torch.distributions.Normal(u_tilde.new_tensor(0.0), u_tilde.new_tensor(1.0))
    z = normal.icdf(u_tilde)

    d = u_tilde.shape[1]
    m = d * (d - 1) // 2

    # static parameters

    # mixture weight 
    weight = torch.nn.Parameter(u_tilde.new_tensor(0.0))

    # gaussian 
    omega_G = torch.nn.Parameter(u_tilde.new_zeros(m))
    A_G = torch.nn.Parameter(u_tilde.new_full((m,), 0.001))

    B_G_initial = u_tilde.new_full((m,), 0.90)
    B_G_raw = torch.nn.Parameter(torch.logit(B_G_initial))

    # clayton 
    omega_C = torch.nn.Parameter(u_tilde.new_tensor(0.0))
    A_C = torch.nn.Parameter(u_tilde.new_tensor(0.001))

    B_C_initial = u_tilde.new_tensor(0.90)
    B_C_raw = torch.nn.Parameter(torch.logit(B_C_initial))

    parameters = [weight,
                  omega_G, A_G, B_G_raw,
                  omega_C, A_C, B_C_raw]

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    f_G_0 = u_tilde.new_zeros(m).requires_grad_(True)
    f_C_0 = u_tilde.new_tensor(0.0).requires_grad_(True)

    for epoch in range(epochs):

        B_G = torch.sigmoid(B_G_raw)
        B_C = torch.sigmoid(B_C_raw)

        parameter_values = [
            ("weight", weight),
            ("omega_G", omega_G),
            ("A_G", A_G),
            ("B_G_raw", B_G_raw),
            ("B_G", B_G),
            ("omega_C", omega_C),
            ("A_C", A_C),
            ("B_C_raw", B_C_raw),
            ("B_C", B_C)
        ]

        non_finite_parameter_names = [
            name for name, value in parameter_values
            if not torch.isfinite(value).all().item()
        ]

        if non_finite_parameter_names:

            print("epoch:", epoch + 1)

            for name, value in parameter_values:
                value_detached = value.detach()
                print(f"{name}:", value_detached)
                print(
                    f"torch.isfinite({name}):",
                    torch.isfinite(value_detached)
                )

            raise RuntimeError(
                f"Non-finite GAS parameter detected: "
                f"{non_finite_parameter_names}"
            )
        
        optimizer.zero_grad(set_to_none=True)
        
        nll =  - gaussian_clayton_mixture_log_likelihood(u_tilde, z, weight,
                                                          f_G_0, omega_G, A_G, B_G,
                                                          f_C_0, omega_C, A_C, B_C,
                                                          epoch=epoch + 1 )
        mean_nll = nll / u_tilde.shape[0]

        if loss_history is not None:
            loss_history.append(mean_nll.detach().item())
        
        mean_nll.backward()
        optimizer.step()

    B_G = torch.sigmoid(B_G_raw)
    B_C = torch.sigmoid(B_C_raw)

    return { "weight": weight.detach(),
             "omega_G": omega_G.detach(),
             "A_G": A_G.detach(),
             "B_G": B_G.detach(),
             "omega_C": omega_C.detach(),
             "A_C": A_C.detach(),
             "B_C": B_C.detach() }



def compute_time_varying_copula_paths(u_tilde, estimated_parameters):
   
    u_tilde = torch.as_tensor( u_tilde, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)

    normal = torch.distributions.Normal(u_tilde.new_tensor(0.0), u_tilde.new_tensor(1.0))
    z = normal.icdf(u_tilde)

    d = u_tilde.shape[1]
    m = d * (d - 1) // 2

    # mixture weight
    weight = estimated_parameters["weight"].to( dtype=u_tilde.dtype, device=u_tilde.device)

    # gaussian
    omega_G = estimated_parameters["omega_G"].to(u_tilde)
    A_G = estimated_parameters["A_G"].to(u_tilde)
    B_G = estimated_parameters["B_G"].to(u_tilde)

    # clayton
    omega_C = estimated_parameters["omega_C"].to(u_tilde)
    A_C = estimated_parameters["A_C"].to(u_tilde)
    B_C = estimated_parameters["B_C"].to(u_tilde)

    f_G_i = u_tilde.new_zeros(m).requires_grad_(True)
    f_C_i = u_tilde.new_tensor(0.0).requires_grad_(True)

    information_G_i = torch.eye(m, dtype=u_tilde.dtype, device=u_tilde.device)
    information_C_i = u_tilde.new_tensor(1.0)

    R_path = []
    theta_path = []
    log_c_mix_path = []
    
    for i in range(u_tilde.shape[0]):
        
        R_i = gaussian_parameter(f_G_i)
        theta_i = clayton_parameter(f_C_i)

        log_c_G_i = gaussian_copula_log_density(z[i], R_i)
        log_c_C_i = clayton_copula_log_density( u_tilde[i], theta_i)
        # log{ w * c_G + (1-w) * c_C }
        log_c_mix_i = gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight)
        
        score_G_i, score_C_i = torch.autograd.grad(log_c_mix_i, (f_G_i, f_C_i))
        scaled_score_G_i, scaled_score_C_i = square_root_information_scaling(score_G_i, information_G_i, score_C_i, information_C_i)
        information_G_next, information_C_next = update_conditional_information( score_G_i, information_G_i, score_C_i, information_C_i)

        f_G_next, f_C_next = gaussian_clayton_gas_update(f_G_i, scaled_score_G_i, omega_G, A_G, B_G,
                                                   f_C_i, scaled_score_C_i, omega_C, A_C, B_C)

        f_G_i = f_G_next.detach().requires_grad_(True)
        f_C_i = f_C_next.detach().requires_grad_(True)

        information_G_i = information_G_next
        information_C_i = information_C_next

        R_path.append(R_i.detach())
        theta_path.append(theta_i.detach())
        log_c_mix_path.append(log_c_mix_i.detach())

    R_path = torch.stack(R_path)
    theta_path = torch.stack(theta_path)
    log_c_mix_path = torch.stack(log_c_mix_path)

    R_next = gaussian_parameter(f_G_i)
    theta_next = clayton_parameter(f_C_i)

    return {
        "weight": torch.sigmoid(weight.detach()),
        "R_path": R_path,
        "theta_path": theta_path,
        "log_c_mix_path": log_c_mix_path,
        "copula_density_path": torch.exp(log_c_mix_path),
        "f_G_next": f_G_i.detach(),
        "f_C_next": f_C_i.detach(),
        "R_next": R_next.detach(),
        "theta_next": theta_next.detach()}

def fit_gaussian_clayton_mixture(u_tilde, epochs=300, learning_rate=0.0001):
    
    u_tilde = np.asarray(u_tilde, dtype=float)
    u_tilde = np.clip(u_tilde, 1e-6, 1 - 1e-6)


    loss_history = []

    estimated_parameters =  estimate_gaussian_clayton_gas(u_tilde=u_tilde, epochs=epochs, learning_rate=learning_rate, loss_history=loss_history)

    copula_results = compute_time_varying_copula_paths( u_tilde=u_tilde, estimated_parameters=estimated_parameters)

    R_next = copula_results["R_next"]
    theta_next = copula_results["theta_next"]
    weight = copula_results["weight"]
    
    return (estimated_parameters,
            loss_history,
            R_next,
            theta_next,
            weight)
