import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.special import expit

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
