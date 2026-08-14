import numpy as np
import torch

from src.conditional_density import conditional_marginal_all
from src.copula import gaussian_copula_log_density, clayton_copula_log_density


def predictive_density(
    y,
    vines,
    u_condition,
    x_grids,
    p_grids,
    P_grids,
    R_next,
    theta_next,
    weight):

    y = np.asarray(y)

    # conditional marginal densities and PITs
    p_cond, u_tilde = conditional_marginal_all(
        y=y,
        vines=vines,
        u_condition=u_condition,
        x_grids=x_grids,
        p_grids=p_grids,
        P_grids=P_grids
    )

    p_cond = np.asarray(p_cond, dtype=float)

    if not np.all(np.isfinite(p_cond)):
        raise ValueError(
            f"p_cond contains non-finite values: {p_cond}"
        )

    if np.any(p_cond < 0.0):
        raise ValueError(
            f"p_cond contains negative densities: {p_cond}"
        )

    # conditional PITs
    u_tilde = torch.as_tensor(
        u_tilde,
        dtype=torch.float64
    ).clamp(1e-6, 1.0 - 1e-6)

    R_next = torch.as_tensor(R_next, dtype=torch.float64)
    theta_next = torch.as_tensor(theta_next, dtype=torch.float64)
    weight = torch.as_tensor(weight, dtype=torch.float64)

    if weight.numel() != 1:
        raise ValueError("weight must be a scalar mixture probability.")

    weight = weight.reshape(())

    if not torch.isfinite(weight).item():
        raise ValueError("weight must be finite.")

    if not 0.0 <= weight.item() <= 1.0:
        raise ValueError(
            f"weight must be between 0 and 1, but received {weight.item()}."
        )

    # Gaussian copula
    normal = torch.distributions.Normal(
        u_tilde.new_tensor(0.0),
        u_tilde.new_tensor(1.0)
    )
    z = normal.icdf(u_tilde)

    log_c_G = gaussian_copula_log_density(z, R_next)
    c_G = torch.exp(log_c_G)

    # Clayton copula
    log_c_C = clayton_copula_log_density(
        u_tilde,
        theta_next
    )
    c_C = torch.exp(log_c_C)

    # Gaussian-Clayton mixture
    c_mix = weight * c_G + (1.0 - weight) * c_C

    # final predictive density
    density = float(c_mix.item() * np.prod(p_cond))

    if not np.isfinite(density):
        raise FloatingPointError(
            "Final predictive density is non-finite."
        )

    return density
