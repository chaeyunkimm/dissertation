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
    y_mat = np.vstack((u, mat))
    y_mat = np.hstack((y_mat, v))
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
    xd = x_mat.shape[0]
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
    structx = vcx.structur
    return 0

x_mat = np.array([[1,1,1], [2,2,0], [3,0,0]])
z_mat = np.array([[1,1,1,1], [2,2,2,0], [3,3,0,0], [4,0,0,0]])
print(x_mat)
show_bvcops_frommat(x_mat)
'''
y_mat = expand_mat(x_mat)
print(y_mat)
show_bvcops_frommat(y_mat)
'''
#y_mat = add_leaf(x_mat)
#print(y_mat)
#show_bvcops_frommat(y_mat)
xstruct = pv.RVineStructure.from_matrix(x_mat)
print(xstruct)
xvine = pv.Vinecop.from_structure(xstruct)
z_mat = add_leaf2(x_mat, leafidx=4)
print(z_mat)
zstruct = pv.RVineStructure.from_matrix(z_mat)
print(zstruct)
zvine = pv.Vinecop.from_structure(zstruct)
zvine.plot()





