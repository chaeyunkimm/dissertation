import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.special import expit
import torch
import math


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
GAS_INFORMATION_RIDGE = 1e-4
GAS_INFORMATION_DECAY = 0.95

# gaussian 

def gaussian_parameter(f_G_i):

    eps = 1e-6

    m = f_G_i.numel()
    d = int((1 + math.sqrt(1 + 8 * m)) / 2)

    if d * (d - 1) // 2 != m:
        raise ValueError(f"Invalid Gaussian state length: {m}")

    # partial correlations between -1 and 1
    partial_rho = (1.0 - eps) * torch.tanh(f_G_i / 2.0)

    L_i = f_G_i.new_zeros((d, d))

    index = 0

    for row in range(d):

        remaining_scale = f_G_i.new_tensor(1.0)

        for col in range(row):

            rho = partial_rho[index]

            L_i[row, col] = rho * remaining_scale
            remaining_scale = remaining_scale * torch.sqrt(1.0 - rho**2)

            index += 1

        L_i[row, row] = remaining_scale

    R_i = L_i @ L_i.T

    return R_i

def gaussian_copula_log_density(z_i, R_i):
    
    eps = 1e-6
    identity = torch.eye(R_i.shape[0], dtype=R_i.dtype, device=R_i.device)

    # Cholesky decomposition of the correlation matrix
    L_i = torch.linalg.cholesky(R_i + eps * identity)

    log_det_R_i = 2.0 * torch.log(torch.diagonal(L_i)).sum()

    # R_i^{-1} * z_i 
    R_inv_z_i = torch.cholesky_solve(z_i.unsqueeze(1), L_i).squeeze(1)

    log_copula_density_i = -0.5 * (log_det_R_i + (torch.dot(z_i, R_inv_z_i) - torch.dot(z_i, z_i)))

    return log_copula_density_i


# Clayton 

def clayton_parameter(f_C_i):
    theta_i = torch.exp(f_C_i)
    return theta_i

def clayton_copula_log_density(u_tilde_i, theta_i):
  
    d = u_tilde_i.shape[0]

    a_i = -theta_i * torch.log(u_tilde_i)

    log_sum_exp_i = torch.logsumexp(a_i, dim=0)

    log_clayton_sum_i = (
        log_sum_exp_i
        + torch.log1p(
            -(d - 1.0) * torch.exp(-log_sum_exp_i)
        )
    )

    coefficient_index = torch.arange(
        1, d, dtype=theta_i.dtype, device=theta_i.device
    )

    log_coefficient_i = torch.sum(
        torch.log(1.0 + coefficient_index * theta_i)
    )

    log_marginal_term_i = (
        -(theta_i + 1.0) * torch.sum(torch.log(u_tilde_i))
    )

    log_generator_term_i = (
        -(d + 1.0 / theta_i) * log_clayton_sum_i
    )

    log_copula_density_i = (
        log_coefficient_i
        + log_marginal_term_i
        + log_generator_term_i
    )
    
    return log_copula_density_i


# Mixture

def gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight):
   
    log_weight = torch.nn.functional.logsigmoid(weight)
    log_one_minus_weight = torch.nn.functional.logsigmoid(-weight)

    log_c_mix_i = torch.logsumexp(torch.stack([ log_weight + log_c_G_i, log_one_minus_weight + log_c_C_i]), dim=0)

    return log_c_mix_i

def gaussian_clayton_mixture_raw_score(log_c_mix_i, f_G_i, f_C_i):
    
    raw_score_G_i, raw_score_C_i = torch.autograd.grad(log_c_mix_i, (f_G_i, f_C_i), create_graph=True)
    
    return raw_score_G_i, raw_score_C_i

def identity_scaling(raw_score_G_i, raw_score_C_i):
    
    scaled_score_G_i = raw_score_G_i
    scaled_score_C_i = raw_score_C_i
    
    return scaled_score_G_i, scaled_score_C_i

# square-root scaling
##################################
def square_root_information_scaling(
    raw_score_G_i,
    information_G_i,
    raw_score_C_i,
    information_C_i,
    ridge=GAS_INFORMATION_RIDGE
):

    identity = torch.eye(
        information_G_i.shape[0],
        dtype=information_G_i.dtype,
        device=information_G_i.device
    )

    information_cholesky = torch.linalg.cholesky(
        information_G_i + ridge * identity
    )

    scaled_score_G_i = torch.linalg.solve_triangular(
        information_cholesky,
        raw_score_G_i.unsqueeze(1),
        upper=False
    ).squeeze(1)

    scaled_score_C_i = (
        raw_score_C_i
        / torch.sqrt(information_C_i + ridge)
    )

    return scaled_score_G_i, scaled_score_C_i


def update_conditional_information(
    raw_score_G_i,
    information_G_i,
    raw_score_C_i,
    information_C_i,
    decay=GAS_INFORMATION_DECAY
):

    score_G = raw_score_G_i.detach()
    score_C = raw_score_C_i.detach()

    information_G_next = (
        decay * information_G_i
        + (1.0 - decay) * torch.outer(score_G, score_G)
    )

    information_C_next = (
        decay * information_C_i
        + (1.0 - decay) * score_C.square()
    )

    return information_G_next, information_C_next
#############################################
    
def gaussian_clayton_mixture_score_scaling(raw_score_G_i, S_G_i, raw_score_C_i, S_C_i):
    
    scaled_score_G_i = S_G_i @ raw_score_G_i
    scaled_score_C_i = S_C_i * raw_score_C_i
   
    return scaled_score_G_i, scaled_score_C_i

def gaussian_clayton_gas_update(f_G_i, scaled_score_G_i, omega_G, A_G, B_G, 
                                f_C_i, scaled_score_C_i, omega_C, A_C, B_C):
    
    f_G_next = omega_G + A_G * scaled_score_G_i + B_G * f_G_i
    f_C_next = omega_C + A_C * scaled_score_C_i + B_C * f_C_i
    
    return f_G_next, f_C_next


# Estismate

def gaussian_clayton_mixture_log_likelihood(u_tilde, z, weight,
                                             f_G_0, omega_G, A_G, B_G,
                                             f_C_0, omega_C, A_C, B_C,
                                             epoch=None):

    f_G_i = f_G_0
    f_C_i = f_C_0
    
    m = f_G_0.numel()

    information_G_i = torch.eye(m,dtype=u_tilde.dtype,device=u_tilde.device)
    information_C_i = u_tilde.new_tensor(1.0)
    
    log_likelihood = u_tilde.new_tensor(0.0)

    for i in range(u_tilde.shape[0]):
        R_i = gaussian_parameter(f_G_i)
        theta_i = clayton_parameter(f_C_i)

        f_G_i_finite = torch.isfinite(f_G_i)
        f_C_i_finite = torch.isfinite(f_C_i)
        R_i_finite = torch.isfinite(R_i)

        # copula log density
        try:
            log_c_G_i = gaussian_copula_log_density(z[i], R_i)

        except RuntimeError as error:

            if "cholesky" not in str(error).lower():
                raise

            print("epoch:", epoch)
            print("i:", i)
            print("f_G_i:", f_G_i.detach())
            print("f_C_i:", f_C_i.detach())
            print("R_i:", R_i.detach())
            print("torch.isfinite(f_G_i):", f_G_i_finite)
            print("torch.isfinite(f_C_i):", f_C_i_finite)
            print("torch.isfinite(R_i):", R_i_finite)

            try:
                print(
                    "torch.linalg.eigvalsh(R_i):",
                    torch.linalg.eigvalsh(R_i.detach())
                )
            except RuntimeError as eigenvalue_error:
                print(
                    "torch.linalg.eigvalsh(R_i): unavailable",
                    eigenvalue_error
                )

            raise

        log_c_C_i = clayton_copula_log_density(u_tilde[i], theta_i)

        # mixture
        log_c_mix_i = gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight)

        # score
        raw_score_G_i, raw_score_C_i = gaussian_clayton_mixture_raw_score(log_c_mix_i, f_G_i, f_C_i)
        scaled_score_G_i, scaled_score_C_i = square_root_information_scaling( raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)

        information_G_i, information_C_i = update_conditional_information( raw_score_G_i, information_G_i, raw_score_C_i, information_C_i)
        
        log_likelihood = log_likelihood + log_c_mix_i

        f_G_next, f_C_next = gaussian_clayton_gas_update(
            f_G_i, scaled_score_G_i, omega_G, A_G, B_G,
            f_C_i, scaled_score_C_i, omega_C, A_C, B_C
        )

        f_G_next_finite = torch.isfinite(f_G_next).all()
        f_C_next_finite = torch.isfinite(f_C_next).all()

        if (
            not f_G_next_finite.item()
            or not f_C_next_finite.item()
        ):

            print("epoch:", epoch)
            print("i:", i)

            diagnostic_values = [
                ("weight", weight),
                ("omega_G", omega_G),
                ("A_G", A_G),
                ("B_G", B_G),
                ("omega_C", omega_C),
                ("A_C", A_C),
                ("B_C", B_C),

                ("f_G_i", f_G_i),
                ("f_C_i", f_C_i),
                ("R_i", R_i),
                ("theta_i", theta_i),

                ("log_c_G_i", log_c_G_i),
                ("log_c_C_i", log_c_C_i),
                ("log_c_mix_i", log_c_mix_i),

                ("raw_score_G_i", raw_score_G_i),
                ("raw_score_C_i", raw_score_C_i),

                ("scaled_score_G_i", scaled_score_G_i),
                ("scaled_score_C_i", scaled_score_C_i),

                ("f_G_next", f_G_next),
                ("f_C_next", f_C_next)
            ]

            for name, value in diagnostic_values:
                value_detached = value.detach()
                print(f"{name}:", value_detached)
                print(
                    f"torch.isfinite({name}):",
                    torch.isfinite(value_detached)
                )

            raise RuntimeError(
                f"First non-finite GAS update detected "
                f"at epoch={epoch}, i={i}"
            )

        f_G_i = f_G_next
        f_C_i = f_C_next

    return log_likelihood

def estimate_gaussian_clayton_gas(u_tilde, epochs=500, learning_rate=0.01,
                                  loss_history=None):

    u_tilde = torch.as_tensor(u_tilde, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    
    normal = torch.distributions.Normal(u_tilde.new_tensor(0.0), u_tilde.new_tensor(1.0))
    z = normal.icdf(u_tilde)

    d = u_tilde.shape[1]
    m = d * (d - 1) // 2

    # static parameters

    # mixture weight 
    weight = torch.nn.Parameter(u_tilde.new_tensor(0.0))

    # gaussian 
    omega_G = torch.nn.Parameter(u_tilde.new_zeros(m))
    A_G = torch.nn.Parameter(u_tilde.new_full((m,), 0.001))

    B_G_initial = u_tilde.new_full((m,), 0.90)
    B_G_raw = torch.nn.Parameter(torch.logit(B_G_initial))

    # clayton 
    omega_C = torch.nn.Parameter(u_tilde.new_tensor(0.0))
    A_C = torch.nn.Parameter(u_tilde.new_tensor(0.001))

    B_C_initial = u_tilde.new_tensor(0.90)
    B_C_raw = torch.nn.Parameter(torch.logit(B_C_initial))

    parameters = [weight,
                  omega_G, A_G, B_G_raw,
                  omega_C, A_C, B_C_raw]

    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    f_G_0 = u_tilde.new_zeros(m).requires_grad_(True)
    f_C_0 = u_tilde.new_tensor(0.0).requires_grad_(True)

    for epoch in range(epochs):

        B_G = torch.sigmoid(B_G_raw)
        B_C = torch.sigmoid(B_C_raw)

        parameter_values = [
            ("weight", weight),
            ("omega_G", omega_G),
            ("A_G", A_G),
            ("B_G_raw", B_G_raw),
            ("B_G", B_G),
            ("omega_C", omega_C),
            ("A_C", A_C),
            ("B_C_raw", B_C_raw),
            ("B_C", B_C)
        ]

        non_finite_parameter_names = [
            name for name, value in parameter_values
            if not torch.isfinite(value).all().item()
        ]

        if non_finite_parameter_names:

            print("epoch:", epoch + 1)

            for name, value in parameter_values:
                value_detached = value.detach()
                print(f"{name}:", value_detached)
                print(
                    f"torch.isfinite({name}):",
                    torch.isfinite(value_detached)
                )

            raise RuntimeError(
                f"Non-finite GAS parameter detected: "
                f"{non_finite_parameter_names}"
            )
        
        optimizer.zero_grad(set_to_none=True)
        
        nll =  - gaussian_clayton_mixture_log_likelihood(u_tilde, z, weight,
                                                          f_G_0, omega_G, A_G, B_G,
                                                          f_C_0, omega_C, A_C, B_C,
                                                          epoch=epoch + 1 )
        mean_nll = nll / u_tilde.shape[0]

        if loss_history is not None:
            loss_history.append(mean_nll.detach().item())
        
        mean_nll.backward()
        optimizer.step()

    B_G = torch.sigmoid(B_G_raw)
    B_C = torch.sigmoid(B_C_raw)

    return { "weight": weight.detach(),
             "omega_G": omega_G.detach(),
             "A_G": A_G.detach(),
             "B_G": B_G.detach(),
             "omega_C": omega_C.detach(),
             "A_C": A_C.detach(),
             "B_C": B_C.detach() }



def compute_time_varying_copula_paths(u_tilde, estimated_parameters):
   
    u_tilde = torch.as_tensor( u_tilde, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)

    normal = torch.distributions.Normal(u_tilde.new_tensor(0.0), u_tilde.new_tensor(1.0))
    z = normal.icdf(u_tilde)

    d = u_tilde.shape[1]
    m = d * (d - 1) // 2

    # mixture weight
    weight = estimated_parameters["weight"].to( dtype=u_tilde.dtype, device=u_tilde.device)

    # gaussian
    omega_G = estimated_parameters["omega_G"].to(u_tilde)
    A_G = estimated_parameters["A_G"].to(u_tilde)
    B_G = estimated_parameters["B_G"].to(u_tilde)

    # clayton
    omega_C = estimated_parameters["omega_C"].to(u_tilde)
    A_C = estimated_parameters["A_C"].to(u_tilde)
    B_C = estimated_parameters["B_C"].to(u_tilde)

    f_G_i = u_tilde.new_zeros(m).requires_grad_(True)
    f_C_i = u_tilde.new_tensor(0.0).requires_grad_(True)

    information_G_i = torch.eye(m, dtype=u_tilde.dtype, device=u_tilde.device)
    information_C_i = u_tilde.new_tensor(1.0)

    R_path = []
    theta_path = []
    log_c_mix_path = []
    
    for i in range(u_tilde.shape[0]):
        
        R_i = gaussian_parameter(f_G_i)
        theta_i = clayton_parameter(f_C_i)

        log_c_G_i = gaussian_copula_log_density(z[i], R_i)
        log_c_C_i = clayton_copula_log_density( u_tilde[i], theta_i)
        # log{ w * c_G + (1-w) * c_C }
        log_c_mix_i = gaussian_clayton_mixture_log_density(log_c_G_i, log_c_C_i, weight)
        
        score_G_i, score_C_i = torch.autograd.grad(log_c_mix_i, (f_G_i, f_C_i))
        scaled_score_G_i, scaled_score_C_i = square_root_information_scaling(score_G_i, information_G_i, score_C_i, information_C_i)
        information_G_next, information_C_next = update_conditional_information( score_G_i, information_G_i, score_C_i, information_C_i)

        f_G_next, f_C_next = gaussian_clayton_gas_update(f_G_i, scaled_score_G_i, omega_G, A_G, B_G,
                                                   f_C_i, scaled_score_C_i, omega_C, A_C, B_C)

        f_G_i = f_G_next.detach().requires_grad_(True)
        f_C_i = f_C_next.detach().requires_grad_(True)

        information_G_i = information_G_next
        information_C_i = information_C_next

        R_path.append(R_i.detach())
        theta_path.append(theta_i.detach())
        log_c_mix_path.append(log_c_mix_i.detach())

    R_path = torch.stack(R_path)
    theta_path = torch.stack(theta_path)
    log_c_mix_path = torch.stack(log_c_mix_path)

    R_next = gaussian_parameter(f_G_i)
    theta_next = clayton_parameter(f_C_i)

    return {
        "weight": torch.sigmoid(weight.detach()),
        "R_path": R_path,
        "theta_path": theta_path,
        "log_c_mix_path": log_c_mix_path,
        "copula_density_path": torch.exp(log_c_mix_path),
        "f_G_next": f_G_i.detach(),
        "f_C_next": f_C_i.detach(),
        "R_next": R_next.detach(),
        "theta_next": theta_next.detach()}

def fit_gaussian_clayton_mixture(u_tilde, epochs=300, learning_rate=0.0001):
    
    u_tilde = np.asarray(u_tilde, dtype=float)
    u_tilde = np.clip(u_tilde, 1e-6, 1 - 1e-6)


    loss_history = []

    estimated_parameters =  estimate_gaussian_clayton_gas(u_tilde=u_tilde, epochs=epochs, learning_rate=learning_rate, loss_history=loss_history)

    copula_results = compute_time_varying_copula_paths( u_tilde=u_tilde, estimated_parameters=estimated_parameters)

    R_next = copula_results["R_next"]
    theta_next = copula_results["theta_next"]
    weight = copula_results["weight"]
    
    return (estimated_parameters,
            loss_history,
            R_next,
            theta_next,
            weight)