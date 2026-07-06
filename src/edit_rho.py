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

# Potential edits: these will compute the p(x_n) explicitly. 
# For large n it may become infeasable, but this should run fast on a gpu until memory runs out!
# If memory becomes a problem, or an in-line version is required,
# the recursion will need to be done explicitly at each new x.

def R_BP_torch_edit(n, rho, p0_grid, P0_grid):
    p = p0_grid #.clone() removing as may cause a memory leak?
    P = P0_grid #.clone() This hopefully won't cause any autodif problems - let's test it!

    for i in range(n):
        alpha = 1 / (i + 2)

        u = P # P_{i-1}(x_grid) : grid 전체에 대한 cdf값들 
        u_i = P[i] #P_{i-1}(x_n)

        c_rho = gaussian_copula_density_torch(u, u_i, rho)
        H_rho = gaussian_conditional_cdf_torch(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P

def estimate_rho_adam_edit(n, p0_grid, P0_grid, lr=0.05, n_iter=300):

    p0_grid_torch = torch.tensor(p0_grid, dtype=torch.float64)
    P0_grid_torch = torch.tensor(P0_grid, dtype=torch.float64)

    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    loss_list = []
    rho_list = []
    
    for _ in range(n_iter):
        optimizer.zero_grad() # 이전 계산 지우기

        eps = 1e-6
        rho = (1 - eps) * torch.sigmoid(theta)

        p_est, _ = R_BP_torch_edit(n = n, rho=rho, p0_grid=p0_grid_torch, P0_grid=P0_grid_torch)

        p_x = p_est[:n]
        p_x = torch.clamp(p_x, 1e-6, None)

        log_lik = torch.sum(torch.log(p_x))

        loss = -log_lik # loss 계산

        loss.backward() # loss 줄이는 theta 방향 계산하기
        optimizer.step() # theta를 한번 계산하기 

        loss_list.append(loss.item())  #tensor에서 숫자만 꺼내서 list에 추가
        rho_list.append(rho.item())

    rho_hat = rho_list[-1]
    
    return rho_hat, np.array(rho_list), np.array(loss_list)
