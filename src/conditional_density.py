import numpy as np

from src.transform import x_to_u
from src.copula import conditional_vine_copula_pdf, conditional_vine_copula_cdf


def conditional_marginal_pdf(y_j, vine_j, u_condition, x_grid_j, p_grid_j, P_grid_j):
    
    u_y_j = x_to_u(x=y_j, x_grid=x_grid_j, P_grid=P_grid_j)

    marginal_pdf_j = np.interp(y_j, x_grid_j, p_grid_j)
    copula_pdf_j = conditional_vine_copula_pdf(vine=vine_j,u=u_y_j, u_condition=u_condition)

    p_cond_j = marginal_pdf_j * copula_pdf_j

    return p_cond_j

def conditional_marginal_cdf(y_j, vine_j, u_condition, x_grid_j, P_grid_j):
    
    u_y_j = x_to_u(x=y_j, x_grid=x_grid_j, P_grid=P_grid_j)

    u_tilde_j = conditional_vine_copula_cdf(vine=vine_j, u=u_y_j, u_condition=u_condition)

    return u_tilde_j

def conditional_marginal_all(y, vines, u_condition, x_grids, p_grids, P_grids):
    
    y = np.asarray(y)
    d = len(y)

    p_cond = np.zeros(d)
    u_tilde = np.zeros(d)

    for j in range(d):

        p_cond[j] = conditional_marginal_pdf(y_j=y[j], vine_j=vines[j], u_condition=u_condition, 
                                             x_grid_j=x_grids[j], p_grid_j=p_grids[j], P_grid_j=P_grids[j])
        u_tilde[j] = conditional_marginal_cdf(y_j=y[j], vine_j=vines[j], u_condition=u_condition, 
                                              x_grid_j=x_grids[j], P_grid_j=P_grids[j])

    return p_cond, u_tilde

def build_u_tilde_data(X_target, U_condition, vines, x_grids, p_grids, P_grids):
    
    n, d = X_target.shape

    p_cond_all = np.zeros((n, d))
    U_tilde = np.zeros((n, d))

    for i in range(n):

        p_cond_i, u_tilde_i = conditional_marginal_all(y=X_target[i], vines=vines, u_condition=U_condition[i],
                                                       x_grids=x_grids, p_grids=p_grids, P_grids=P_grids)

        p_cond_all[i, :] = p_cond_i
        U_tilde[i, :] = u_tilde_i

    return p_cond_all, U_tilde