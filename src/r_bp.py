import numpy as np
from scipy.stats import norm, multivariate_normal
from .copula import gaussian_copula_density, gaussian_copula_cdf, gaussian_conditional_cdf

# R-BP
def R_BP_density(x, x_grid, rho, p0_grid, P0_grid):
    # p_0, P_0
    p = p0_grid.copy()
    P = P0_grid.copy()

    # update
    for i in range(len(x)):

        # weight
        alpha = 1 / (i + 2)

        u = P
        u_i = np.interp(x[i], x_grid, P)

        c_rho = gaussian_copula_density(u, u_i, rho)
        H_rho = gaussian_conditional_cdf(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P

#Check the git push