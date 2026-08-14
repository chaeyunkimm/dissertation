import numpy as np
from scipy.stats import norm, multivariate_normal
import pyvinecopulib as pv


#####################
#  Gaussian Copula  #
#####################


# Gaussian Copula Density : c_rho
def gaussian_copula_density(u, v, rho):

    # u, v in [ 10^{-6}, 1-10^{-6} ]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    numerator = np.exp( - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2)))
    denominator = np.sqrt(1 - rho**2)

    return numerator / denominator


# Gaussian Copula CDF : C_rho
def gaussian_copula_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    return multivariate_normal.cdf([z_u, z_v], mean=mean, cov=cov)

# Conditional Gaussian Copula CDF : H_rho
def gaussian_conditional_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    return norm.cdf((z_u - rho * z_v) / np.sqrt(1 - rho**2))



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
