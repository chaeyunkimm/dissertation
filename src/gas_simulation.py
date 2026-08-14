import numpy as np
from scipy.stats import norm
import torch

from src.copula import gaussian_parameter, clayton_parameter
from src.copula import gaussian_copula_log_density, clayton_copula_log_density
from src.copula import gaussian_clayton_mixture_log_density
from src.copula import gaussian_clayton_mixture_raw_score
from src.copula import square_root_information_scaling, update_conditional_information
from src.copula import gaussian_clayton_gas_update


def simulate_gaussian_clayton_gas(T, weight,
                                  omega_G, A_G, B_G,
                                  omega_C, A_C, B_C,
                                  seed=123):

    rng = np.random.default_rng(seed)

    weight = torch.as_tensor(weight, dtype=torch.float64)

    omega_G = torch.as_tensor(omega_G, dtype=torch.float64)
    A_G = torch.as_tensor(A_G, dtype=torch.float64)
    B_G = torch.as_tensor(B_G, dtype=torch.float64)

    omega_C = torch.as_tensor(omega_C, dtype=torch.float64)
    A_C = torch.as_tensor(A_C, dtype=torch.float64)
    B_C = torch.as_tensor(B_C, dtype=torch.float64)

    mixture_weight = torch.sigmoid(weight)

    normal = torch.distributions.Normal(
        weight.new_tensor(0.0),
        weight.new_tensor(1.0)
    )

    m = omega_G.numel()

    f_G_i = weight.new_zeros(m).requires_grad_(True)
    f_C_i = weight.new_tensor(0.0).requires_grad_(True)

    information_G_i = torch.eye(m, dtype=weight.dtype, device=weight.device)
    information_C_i = weight.new_tensor(1.0)
    
    U_sim = []
    f_G_true_path = []
    f_C_true_path = []
    R_true_path = []
    theta_true_path = []

    for i in range(T):

        R_i = gaussian_parameter(f_G_i)
        theta_i = clayton_parameter(f_C_i)

        d = R_i.shape[0]

        if rng.uniform() < mixture_weight.item():

            z_sim_i = rng.multivariate_normal(mean=np.zeros(d), cov=R_i.detach().numpy())

            u_tilde_i = norm.cdf(z_sim_i)

        else:

            theta_value_i = theta_i.detach().item()

            v_i = rng.gamma(shape=1.0 / theta_value_i, scale=1.0)

            e_i = rng.exponential(scale=1.0, size=d)

            u_tilde_i = (1.0 + e_i / v_i) ** (-1.0 / theta_value_i)

        u_tilde_i = torch.as_tensor(u_tilde_i, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)

        z_i = normal.icdf(u_tilde_i)

        log_c_G_i = gaussian_copula_log_density(z_i, R_i)
        log_c_C_i = clayton_copula_log_density(u_tilde_i, theta_i)

        log_c_mix_i = gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight)

        raw_score_G_i, raw_score_C_i = gaussian_clayton_mixture_raw_score(log_c_mix_i, f_G_i, f_C_i)

        scaled_score_G_i, scaled_score_C_i = square_root_information_scaling(raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)

        information_G_next, information_C_next = update_conditional_information(raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)

        f_G_true_path.append(f_G_i.detach())
        f_C_true_path.append(f_C_i.detach())
        R_true_path.append(R_i.detach())
        theta_true_path.append(theta_i.detach())
        U_sim.append(u_tilde_i.detach())

        f_G_next, f_C_next = gaussian_clayton_gas_update(f_G_i, scaled_score_G_i, omega_G, A_G, B_G,
                                                         f_C_i, scaled_score_C_i, omega_C, A_C, B_C)

        f_G_i = f_G_next.detach().requires_grad_(True)
        f_C_i = f_C_next.detach().requires_grad_(True)

        information_G_i = information_G_next
        information_C_i = information_C_next

    U_sim = torch.stack(U_sim)
    f_G_true_path = torch.stack(f_G_true_path)
    f_C_true_path = torch.stack(f_C_true_path)
    R_true_path = torch.stack(R_true_path)
    theta_true_path = torch.stack(theta_true_path)

    known_parameters = {
        "weight": weight.detach(),
        "omega_G": omega_G.detach(),
        "A_G": A_G.detach(),
        "B_G": B_G.detach(),
        "omega_C": omega_C.detach(),
        "A_C": A_C.detach(),
        "B_C": B_C.detach() }

    return {
        "U_sim": U_sim,
        "f_G_true_path": f_G_true_path,
        "f_C_true_path": f_C_true_path,
        "R_true_path": R_true_path,
        "theta_true_path": theta_true_path,
        "known_parameters": known_parameters,
        "mixture_weight": mixture_weight.detach() }
