import numpy as np
from scipy.stats import norm


def true_var3_3d_conditional_mean(x_t, x_t_minus_1, x_t_minus_2, A1, A2, A3):

    x_t = np.asarray(x_t)
    x_t_minus_1 = np.asarray(x_t_minus_1)
    x_t_minus_2 = np.asarray(x_t_minus_2)

    mean_cond = ( A1 @ x_t + A2 @ x_t_minus_1 + A3 @ x_t_minus_2 )

    return mean_cond

def true_var3_3d_conditional_marginal_pdf(y_j, j, mean_cond, Sigma):

    mean_cond = np.asarray(mean_cond)
    Sigma = np.asarray(Sigma)

    sd_j = np.sqrt(Sigma[j, j])

    p_cond_j = norm.pdf(y_j, loc=mean_cond[j], scale=sd_j)

    return p_cond_j

def true_var3_3d_conditional_marginal_cdf(y_j, j, mean_cond, Sigma):

    mean_cond = np.asarray(mean_cond)
    Sigma = np.asarray(Sigma)

    sd_j = np.sqrt(Sigma[j, j])

    P_cond_j = norm.cdf(y_j, loc=mean_cond[j], scale=sd_j)

    return P_cond_j

def true_var3_3d_conditional_pit(y, mean_cond, Sigma):

    y = np.asarray(y)
    d = len(y)

    U_true = np.zeros(d)

    for j in range(d):

        U_true[j] = true_var3_3d_conditional_marginal_cdf(
            y_j=y[j], j=j, mean_cond=mean_cond, Sigma=Sigma
        )

    return U_true

def true_gaussian_copula_correlation(Sigma):

    Sigma = np.asarray(Sigma)

    sd = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(sd, sd)

    return R
