import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import torch

def gaussian_copula_density(u, v, rho):

    # u, v in [ 10^{-6}, 1-10^{-6} ]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    numerator = np.exp( - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2)))
    denominator = np.sqrt(1 - rho**2)

    return numerator / denominator

def gaussian_copula_density_torch(u, v, rho):
    eps = 1e-6

    u = torch.clamp(u, eps, 1 - eps)
    v = torch.clamp(v, eps, 1 - eps)

    z_u = normal_ppf(u)
    z_v = normal_ppf(v)

    numerator = torch.exp(
        - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v)
        / (2 * (1 - rho**2))
    )

    denominator = torch.sqrt(1 - rho**2)

    return numerator / denominator


def gaussian_conditional_cdf_torch(u, v, rho):
    eps = 1e-6

    u = torch.clamp(u, eps, 1 - eps)
    v = torch.clamp(v, eps, 1 - eps)

    z_u = normal_ppf(u)
    z_v = normal_ppf(v)

    return normal_cdf((z_u - rho * z_v) / torch.sqrt(1 - rho**2))

def gaussian_conditional_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    return norm.cdf((z_u - rho * z_v) / np.sqrt(1 - rho**2))

def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def normal_ppf(u):
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.icdf(u)


def R_BP_torch_edit(n, rho, p0_grid, P0_grid):
    p = p0_grid 
    P = P0_grid 

    for i in range(n):
        alpha = 1 / (i + 2)

        u = P # P_{i-1}(x_grid) : grid 전체에 대한 cdf값들 
        u_i = P[i] #P_{i-1}(x_n)

        c_rho = gaussian_copula_density_torch(u, u_i, rho)
        H_rho = gaussian_conditional_cdf_torch(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P


# R-BP
def R_BP_density(n, rho, p0_grid, P0_grid, U = None):
    # p_0, P_0
    p = p0_grid
    P = P0_grid
    if U is None:
        # update
        for i in range(n):
            # weight
            alpha = 1 / (i + 2)

            u = P # P_{i-1}(x_grid) 
            u_i = P[i] #P_{i-1}(x_n)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

        return p, P
    else:
        for i in range(len(U)):

            # weight
            alpha = 1 / (i + 2)

            u = P # P_{i-1}(x_grid) 
            u_i = U[i] #P_{i-1}(x_i)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

        return p, P
    
def R_BP_density_U(rho, p0, P0, U):
    p = p0
    P = P0
    for i, ui in enumerate(U):

            # weight
            alpha = 1 / (i + 2)

            u = P # P_{i-1}(x_grid) 
            u_i = ui #P_{i-1}(x_i)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

    return p, P

def R_BP_coefs(rho, p0_obs, P0_obs):
    p = p0_obs
    P = P0_obs
    n = len(p0_obs)
    U = np.zeros(n)
    for i in range(n):

        # weight
        alpha = 1 / (i + 2)

        u = P # P_{i-1}(x_grid) 
        U[i] = P[i] #P_{i-1}(x_n)
        

        c_rho = gaussian_copula_density(u, U[i], rho)
        H_rho = gaussian_conditional_cdf(u, U[i], rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return U

def R_BP_coefs_torch(rho, p0_obs, P0_obs):
    p = p0_obs
    P = P0_obs
    n = len(p0_obs)
    U = torch.zeros(n)
    for i in range(n):

        # weight
        alpha = 1 / (i + 2)

        u = P # P_{i-1}(x_grid) 
        U[i] = P[i] #P_{i-1}(x_n)
        

        c_rho = gaussian_copula_density_torch(u, U[i], rho)
        H_rho = gaussian_conditional_cdf_torch(u, U[i], rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return U

def np_sigmoid(x):
    return 1/(1+np.exp(-x))

def neg_log_like(theta, n, p0, P0):
    '''
    Calculates the negative log likelihood for an R-BP given observed values
    and a prior distribution.

    These observed values must be passed as their probability density 
    under the prior distribution. (PDF and CDF)

    Faster using torch for calculations
    '''

    p = torch.from_numpy(p0)
    P = torch.from_numpy(P0)

    rho = 0.999 * torch.sigmoid(torch.from_numpy(theta))

    p, _ = R_BP_torch_edit(n=n, rho=rho, p0_grid=p, P0_grid=P)
    p_x = torch.clip(p, 1e-12, None).numpy() 

    return float(np.sum(-np.log(p_x))) #Faster than using torch



def estimate_rho_optim(n, p0, P0, max_iter=100, rho_0 = 0.6):
    '''
    Estimates the shape parameter rho in the R-BP algorithm via
    minimising the negative log-likelihood for a set observed values.

    These observed values must be passed as their probability density 
    under the prior distribution. (PDF and CDF)
    --------------------------------------------------------------------
    n: int ; the number of observed values in the prior grid
    p0: float array ; the prior PDF evaluated at the observed values
    P0: float array ; the prior CDF evaluated at the observed values

    max_iter: int ; the maximum number of iterations to run .minimize for.
    rho_0: float ; the initial guess for rho. Must be in (0, 1).

    '''
    p0 = np.asarray(p0, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)
    theta = minimize(
        neg_log_like,
        x0=np.log(rho_0 / (1.0 - rho_0)),
        args=(n, p0, P0),
        method="BFGS",
        options={"maxiter": max_iter},
    ).x[0]
    return float(0.999 * np_sigmoid(theta))

def fit_R_BP_marginals(p0_obs, P0_obs):
    n, d = p0_obs.shape
    rhos = np.zeros(d)
    Us = np.zeros((n,d))
    for i in range(d):
        rhos[i] = estimate_rho_optim(n, p0_obs[:,i], P0_obs[:,i])
        Us[:, i] = R_BP_coefs(rhos[i], p0_obs[:,i], P0_obs[:,i])
    return rhos, Us

def fit_R_BP_marginals_torch(p0_obs, P0_obs):
    n, d = p0_obs.shape
    p = torch.from_numpy(p0_obs)
    P = torch.from_numpy(P0_obs)
    rhos = torch.zeros(d)
    Us = torch.zeros((n,d))
    for i in range(d):
        rhos[i] = estimate_rho_optim(n, p0_obs[:,i], P0_obs[:,i])
        Us[:, i] = R_BP_coefs_torch(rhos[i], p[:,i], P[:,i])
    return rhos, Us
