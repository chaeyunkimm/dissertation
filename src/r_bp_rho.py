import numpy as np
from src.copula import gaussian_copula_density, gaussian_copula_cdf, gaussian_conditional_cdf

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

# For fixed rho, what can we do to speed up the evaluation of the 