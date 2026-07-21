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

def add_leaf(mat, leafidx = 1):
    '''
    x_mat: 2D numpy array to be expanded
    leaf_idx: int, the index which the leaf should be attached to.
    '''
    xd = mat.shape[0]
    yd = xd+1
    y_mat = np.zeros((yd,yd), dtype = int)
    for i in range(xd):
        idx = i + leafidx - 1
        if idx >= xd:
            idx -= xd
        vec = mat[idx]>0
        num = mat[idx][vec][0]
        u = np.zeros(yd, dtype=int)+num
        for j in range(xd-i+1, yd):
            u[j] = 0
        y_mat[i] = u
    y_mat[xd, 0] = yd
    return y_mat

def add_leaf2(mat, leafidx = 1):
    '''
    x_mat: 2D numpy array to be expanded
    leaf_idx: int, the index which the leaf should be attached to.
    '''
    xd = mat.shape[0]

    assert leafidx <= xd, "Leaf index should be contained in the original vine"

    yd = xd+1
    y_mat = expand_mat(mat)
    y_mat[xd, 0] = yd
    y_mat[0,0] = leafidx
    nums = np.linspace(1, xd, xd, dtype=int)
    nums = nums[np.where(nums != leafidx)]
    for j in range(1,xd):
            y_mat[j,0] = nums[j-1]

    return y_mat

def add_leaf3(mat, leafidx = 1):
    '''
    x_mat: 2D numpy array to be expanded
    leaf_idx: int, the index which the leaf should be attached to.
    '''
    xd = mat.shape[0]

    assert leafidx <= xd, "Leaf index should be contained in the original vine"

    yd = xd+1
    y_mat = expand_mat(mat)
    y_mat[xd, 0] = yd
    y_mat[0,0] = leafidx
    conds = mat[:, 0]
    conds = conds[np.where(conds != leafidx)]
    print(conds)
    y_mat[1:xd, 0] = conds
    return y_mat

def add_leaf4(mat, leafidx = 1):
    '''
    mat: 2D numpy array to be expanded
    leaf_idx: int, the index which the leaf should be attached to.
    '''
    def find_attachment(t):
        for k in range(0, xd-2-t):
            print(mat[xd-1-t-k,:], mat[xd-1-t-k, k], "opposite with:", mat[t+k,:], mat[t+k,k])
            if mat[xd-1-t-k, k] == leafidx:
                return mat[t+k,k]
        print("No attachment found, use any")
        return 0
    
    xd = mat.shape[0]

    assert leafidx <= xd, "Leaf index should be contained in the original vine"

    yd = xd+1
    y_mat = expand_mat(mat)
    y_mat[xd, 0] = yd
    y_mat[0,0] = leafidx
    conds = np.linspace(1,xd, xd, dtype=int)
    used_ids = np.array([leafidx-1])
    #assume natural order
    y_mat[1,0] = mat[0, xd-leafidx]
    used_ids = np.append(used_ids, mat[0, xd-leafidx]-1)
    print("pre-array:", used_ids)

    for i in range(1, xd-2):
        att = find_attachment(i)
        print(att)
        if att == 0:
            att = np.delete(conds, used_ids)[0]
        y_mat[i+1,0] = att
        used_ids = np.append(used_ids, att-1)
        print("mid iteration:", used_ids)
    print(np.delete(conds, used_ids)[0])
    y_mat[xd-1, 0] = np.delete(conds, used_ids)[0]

    return y_mat

def check_proximity(M, t=0, e=0):
    '''
    The proximity condition must hold: For all t = 1, …, d - 2 and e = 0, …, d - t - 1 
    there must exist an index j > d, such that (M[t, e], {M[0, e], ..., M[t-1, e]}) 
    equals either (M[d-j-1, j], {M[0, j], ..., M[t-1, j]}) 
    or (M[t-1, j], {M[d-j-1, j], M[0, j], ..., M[t-2, j]})

    check only t=0 for starters?
    '''
    d = M.shape[0]
    for j in range(d):
        if M[t,e] == M[d-j-1,j]:
            count = 0
            for p in range(t+1):
                if M[p, e] == M[p, j]:
                    count += 1
            if count == t+1:
                return True
        if t>0:
            if M[t,e] == M[t-1, j]:
                count = 0
                if M[0,e] == M[d-j-1,j]:
                    count+=1
                for p in range(t):
                    if M[p, e] == M[p, j]:
                        count += 1
                if count == t+1:
                    return True

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
    '''
    Assumes cops given in natural (first index ascending) order
    and already obeys the proximity condition. 
    i.e. fine if you have taken it from pyvinecoplib, but not if you do the copulas without 
    thinking about the order of conditioning.
    '''
    d = cops.shape[1]
    n = int(d*(d-1)/2)
    mat = np.zeros((d,d), dtype = int)
    mat[0,d-1] = 1
    jump = 0
    for i in range(d-1):
        mat[0:i+2, d-2-i] = np.flip(cops[jump,0:i+2])
        jump += d-1-i

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

def add_leaf_to_mat(mat, leafidx = 1):
    cops = get_copulas_fm(mat)
    zcops = add_leaf_copulas(cops, leafidx)
    z = set_matrix_fc(zcops)
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
        decomps depending on the vine.
    
    '''

    vcx = pv.Vinecop.from_data(u_x)
    structx = vcx.structure
    return 0

y = np.array([[3, 1, 1, 1, 1],
              [1, 2, 2, 2, 0],
              [2, 3, 3, 0, 0],
              [4, 4, 0, 0, 0],
              [5, 0, 0, 0, 0]])

y = add_leaf_to_mat(y, 2)
y = add_leaf_to_mat(y,5)
y = add_leaf_to_mat(y, 6)
y = add_leaf_to_mat(y, 3)


for i in range(1,10):
    for j in range (1,11):
        matt = add_leaf_to_mat(y, i)
        print("--------------")
        print("i =", i, "j =", j)

        try:
            mstruct = pv.RVineStructure.from_matrix(matt)
        except Exception as e:
            print("Not a valid r-vine array. Leaf index:", i)

        matt = add_leaf_to_mat(matt, j)
        try:
            mstruct = pv.RVineStructure.from_matrix(matt)
        except Exception as e:
            print("Not a valid r-vine array. Leaf index:", i, j)

vine = pv.Vinecop.from_structure(mstruct) 
vine.plot(tree=[1])

