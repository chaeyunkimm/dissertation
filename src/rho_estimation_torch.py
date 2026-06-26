import numpy as np
import torch


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def normal_ppf(u):
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.icdf(u)


def gaussian_copula_density_torch(u, v, rho):
    eps = 1e-6

    u = torch.clamp(u, eps, 1 - eps)
    v = torch.clamp(v, eps, 1 - eps)

    z_u = normal_ppf(u)
    z_v = normal_ppf(v)

    numerator = torch.exp(
        - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v)
        / (2 * (1 - rho**2))
    )

    denominator = torch.sqrt(1 - rho**2)

    return numerator / denominator


def gaussian_conditional_cdf_torch(u, v, rho):
    eps = 1e-6

    u = torch.clamp(u, eps, 1 - eps)
    v = torch.clamp(v, eps, 1 - eps)

    z_u = normal_ppf(u)
    z_v = normal_ppf(v)

    return normal_cdf((z_u - rho * z_v) / torch.sqrt(1 - rho**2))


def interp_fixed_x_torch(x_values, x_grid, y_grid):
    idx = torch.searchsorted(x_grid, x_values) - 1
    idx = torch.clamp(idx, 0, len(x_grid) - 2)

    x_left = x_grid[idx]
    x_right = x_grid[idx + 1]

    y_left = y_grid[idx]
    y_right = y_grid[idx + 1]

    weight = (x_values - x_left) / (x_right - x_left)

    return (1 - weight) * y_left + weight * y_right


def R_BP_density_torch(x, x_grid, rho, p0_grid, P0_grid):
    p = p0_grid.clone()
    P = P0_grid.clone()

    for i in range(len(x)):
        alpha = 1 / (i + 2)

        u = P
        u_i = interp_fixed_x_torch(x[i:i+1], x_grid, P)[0]

        c_rho = gaussian_copula_density_torch(u, u_i, rho)
        H_rho = gaussian_conditional_cdf_torch(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P


def estimate_rho_adam(x, x_grid, p0_grid, P0_grid, lr=0.05, n_iter=300):
    x_torch = torch.tensor(x, dtype=torch.float64)
    x_grid_torch = torch.tensor(x_grid, dtype=torch.float64)
    p0_grid_torch = torch.tensor(p0_grid, dtype=torch.float64)
    P0_grid_torch = torch.tensor(P0_grid, dtype=torch.float64)

    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    loss_list = []
    rho_list = []

    for _ in range(n_iter):
        optimizer.zero_grad()

        rho = 0.99 * torch.sigmoid(theta)

        p_est, P_est = R_BP_density_torch(x=x_torch, x_grid=x_grid_torch, rho=rho, p0_grid=p0_grid_torch, P0_grid=P0_grid_torch)

        p_x = interp_fixed_x_torch(x_torch, x_grid_torch, p_est)
        p_x = torch.clamp(p_x, 1e-12, None)

        log_lik = torch.sum(torch.log(p_x))

        loss = -log_lik

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        rho_list.append(rho.item())

    rho_hat = rho_list[-1]

    return rho_hat, np.array(rho_list), np.array(loss_list)