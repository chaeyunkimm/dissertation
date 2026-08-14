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

y = np.array([[3, 1, 1, 1, 1],
              [1, 2, 2, 2, 0],
              [2, 3, 3, 0, 0],
              [4, 4, 0, 0, 0],
              [5, 0, 0, 0, 0]])

y = np.array([[1, 4, 4, 4],
              [4, 3, 3, 0],
              [3, 1, 0, 0],
              [2, 0, 0, 0]])

z = add_leaf_to_mat(y,4)
p = pv.RVineStructure.from_matrix(z)

# y = add_leaf_to_mat(y,5)
# y = add_leaf_to_mat(y, 6)
# y = add_leaf_to_mat(y, 3)

# for i in range(1,10):
#     for j in range (1,11):
#         matt = add_leaf_to_mat(y, i)
#         print("--------------")
#         print("i =", i, "j =", j)

#         try:
#             mstruct = pv.RVineStructure.from_matrix(matt)
#         except Exception as e:
#             print("Not a valid r-vine array. Leaf index:", i)

#         matt = add_leaf_to_mat(matt, j)
#         try:
#             mstruct = pv.RVineStructure.from_matrix(matt)
#         except Exception as e:
#             print("Not a valid r-vine array. Leaf index:", i, j)

# vine = pv.Vinecop.from_structure(mstruct) 
# vine.plot(tree=[1])

