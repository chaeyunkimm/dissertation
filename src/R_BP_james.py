import numpy as np
from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.optimize import minimize
from scipy.special import ndtr
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

def np_sigmoid(x):
    return 1/(1+np.exp(-x))

def R_BP_torch_edit(n, rho, p0_grid, P0_grid):
    p = p0_grid #.clone() removing as may cause a memory leak?
    P = P0_grid #.clone() This hopefully won't cause any autodif problems - let's test it!

    for i in range(n):
        alpha = 1 / (i + 2)

        u = P # P_{i-1}(x_grid) : grid 전체에 대한 cdf값들 
        u_i = P[i] #P_{i-1}(x_n)

        c_rho = gaussian_copula_density_torch(u, u_i, rho)
        H_rho = gaussian_conditional_cdf_torch(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P

def neg_log_like(theta, n, p0, P0):
    '''
    Calculates the negative log likelihood for an R-BP given observed values
    and a prior distribution.

    These observed values must be passed as their probability density 
    under the prior distribution. (PDF and CDF)

    '''

    p = torch.from_numpy(p0)
    P = torch.from_numpy(P0)

    rho = 0.999 * torch.sigmoid(torch.from_numpy(theta))

    p, _ = R_BP_torch_edit(n=n, rho=rho, p0_grid=p, P0_grid=P)
    p_x = torch.clip(p, 1e-12, None).numpy() 

    return float(np.sum(-np.log(p_x))) #Faster than torch for some reason...

# R-BP
def R_BP_density(n, rho, p0_grid, P0_grid, U = None):
    # p_0, P_0
    p = p0_grid
    P = P0_grid
    if U == None:
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
    Us = np.zeros_like(p0_obs)
    for i in range(d):
        rhos[i] = estimate_rho_optim(n, p0_obs[:,i], P0_obs[:,i])
        Us[:, i] = R_BP_coefs(rhos[i], p0_obs[:,i], P0_obs[:,i])
    return rhos, Us



####################################################################

def fit_R_BP_marginals_observed(X_train, n_grid=1000,):
    """
    James 원본 방식으로 R-BP marginal을 적합하고,
    1번 파이프라인과 동일한 7개 값을 반환한다.

    x_grids:
        변수별 x grid

    p0_grids:
        변수별 초기 KDE PDF grid

    P0_grids:
        변수별 초기 KDE CDF grid

    rhos:
        변수별 추정 rho

    p_grids:
        변수별 최종 R-BP PDF grid

    P_grids:
        변수별 최종 R-BP CDF grid

    Us:
        transform된
    """

    x = np.asarray(X_train, dtype=float)

    # 관측 개수와 변수 개수
    n = x.shape[0]
    d = x.shape[1]

    # 변수별 KDE 객체를 저장
    kdes = []

    # 관측 위치 초기 PDF
    p0_grid = np.zeros((n, d))
    # 관측 위치 초기 CDF
    P0_grid = np.zeros((n, d))

    #  grid
    #
    # 기본 shape: (1000,)
    x_grid = np.linspace(-10, 15,num=n_grid)

    # ========================================================
    # James 원본:
    # 관측 위치에서 KDE PDF/CDF 계산
    # ========================================================

   

    for i in range(x.shape[1]):
        # Fit prior distributions and transform data for rbp
        kde = gaussian_kde(x.T[i])
        kdes.append(kde)
        p0_grid[:,i] = kde(x.T[i])
        P0_grid[:,i] = np.mean(ndtr((x.T[i] - kde.dataset.T) / kde.factor), axis=0)



    rhos, Us = fit_R_BP_marginals(p0_grid, P0_grid)
    

    x_grids = []

    p0_grids = []
    P0_grids = []

    p_grids = []
    P_grids = []

    for i in range(d):

        kde = kdes[i]

        # grid에서 초기 KDE PDF
        p0_grid_i = kde(x_grid)

        # grid에서 초기 KDE CDF
        P0_grid_i = np.mean(ndtr((x_grid - kde.dataset.T) / kde.factor), axis=0)

        # 저장된 Us를 grid 전체에 적용
        p_grid_i, P_grid_i = R_BP_density_U(rhos[i], p0_grid_i, P0_grid_i, Us[:, i])


        x_grids.append(x_grid)
        p0_grids.append(p0_grid_i)
        P0_grids.append(P0_grid_i)
        p_grids.append(p_grid_i)
        P_grids.append(P_grid_i)

    # ========================================================
    # 1번 파이프라인과 동일한 7개 값 반환
    # ========================================================

    return (x_grids,
            p0_grids,
            P0_grids,
            rhos,
            p_grids,
            P_grids,
            Us)