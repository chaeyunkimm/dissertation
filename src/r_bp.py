import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.stats import gaussian_kde
from .copula import gaussian_copula_density, gaussian_copula_cdf, gaussian_conditional_copula_cdf
from .data_generators import generate_x_grid


# kernel density esitmation
def kde_initial(x, x_grid):

    kde = gaussian_kde(x)

    p0_grid = kde(x_grid)

    dx = x_grid[1] - x_grid[0]

    P0_grid = np.cumsum(p0_grid) * dx
    P0_grid = P0_grid / P0_grid[-1]

    return p0_grid, P0_grid

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
        H_rho = gaussian_conditional_copula_cdf(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P

def fit_R_BP_marginals_until_t(X_train, rho_grid=None):

    from .rho_estimation import estimate_rho_grid
    
    d = X_train.shape[1]

    x_grids = []
    p0_grids = []
    P0_grids = []

    rho_hats = []
    p_grids = []
    P_grids = []

    if rho_grid is None:
        rho_grid = np.linspace(0.01, 0.99, 50)

    for j in range(d):

        x_j = X_train[:, j]

        x_grid_j = generate_x_grid(x_j)

        p0_grid_j, P0_grid_j = kde_initial(x_j, x_grid_j)

        rho_hat_j, log_lik_j = estimate_rho_grid(x=x_j, x_grid=x_grid_j, p0_grid=p0_grid_j, P0_grid=P0_grid_j, rho_grid=rho_grid)

        p_est_j, P_est_j = R_BP_density( x=x_j, x_grid=x_grid_j, rho=rho_hat_j, p0_grid=p0_grid_j, P0_grid=P0_grid_j)

        x_grids.append(x_grid_j)
        p0_grids.append(p0_grid_j)
        P0_grids.append(P0_grid_j)

        rho_hats.append(rho_hat_j)
        p_grids.append(p_est_j)
        P_grids.append(P_est_j)

    return x_grids, p0_grids, P0_grids, rho_hats, p_grids, P_grids