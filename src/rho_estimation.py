import numpy as np

from src.r_bp import R_BP_density


def R_BP_log_likelihood(x, x_grid, rho, p0_grid, P0_grid):

    p_est, P_est = R_BP_density(x=x, x_grid=x_grid, rho=rho, p0_grid=p0_grid, P0_grid=P0_grid)

    # p(x)
    p_x = np.interp(x, x_grid, p_est)

    # avoid log(0)
    p_x = np.clip(p_x, 1e-12, None)

    log_likelihood = np.sum(np.log(p_x))

    return log_likelihood


def estimate_rho_grid(x, x_grid, p0_grid, P0_grid, rho_grid):


    log_lik_list = []

    for rho in rho_grid:
        log_lik = R_BP_log_likelihood(x=x, x_grid=x_grid, rho=rho, p0_grid=p0_grid, P0_grid=P0_grid)

        log_lik_list.append(log_lik)

    log_lik_list = np.array(log_lik_list)

    best_index = np.argmax(log_lik_list)
    best_rho = rho_grid[best_index]

    return best_rho, log_lik_list