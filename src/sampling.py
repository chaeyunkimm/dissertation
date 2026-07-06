import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from src.copula import gaussian_copula_density, gaussian_copula_cdf, gaussian_conditional_copula_cdf
from src.r_bp import R_BP_density

# Sampling from the estimated CDF
def sample_from_cdf(x_grid, F, n, seed=None):

    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    
    samples = np.interp(u, F, x_grid)

    return samples


# Sampling from the R-BP distribution 
def R_BP_sample(x, x_grid, rho, n, P0_grid, P0_inv, seed=None):
    rng = np.random.default_rng(seed)

    # P_0
    P = P0_grid.copy()

    v_list = []
    alpha_list = []

    for i in range(len(x)):

        alpha = 1 / (i + 2)
        v_i = np.interp(x[i], x_grid, P)
        H_rho = gaussian_conditional_copula_cdf(P, v_i, rho)

        P = (1 - alpha) * P + alpha * H_rho

        v_list.append(v_i)
        alpha_list.append(alpha)

    samples = []

    for _ in range(n):

        # U_n
        U = rng.uniform(0, 1)

        # backward recursion
        for i in reversed(range(len(x))):

            alpha = alpha_list[i]
            v_i = v_list[i]

            def eq(U_prev):

                f = ((1 - alpha) * U_prev + alpha * gaussian_conditional_copula_cdf(U_prev, v_i, rho) - U)
                
                return f

            # finding root : f(U_pre) = 0 
            U_prev = brentq(eq, 1e-10, 1 - 1e-10)

            U = U_prev

        # Y = P_0^{-1}(U_0)
        y = P0_inv(U)

        samples.append(y)


    return np.array(samples)
