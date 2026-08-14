import numpy as np

def x_to_u(x, x_grid, P_grid):
    
    x = np.asarray(x)
    u = np.interp(x, x_grid, P_grid)
    
    eps=1e-6
    u = np.clip(u, eps, 1 - eps)

    return u


def X_to_U(X, x_grids, P_grids):
    
    X = np.asarray(X)
    T, d = X.shape

    U = np.zeros_like(X, dtype=float)

    for j in range(d):
        U[:, j] = x_to_u(x=X[:, j], x_grid=x_grids[j], P_grid=P_grids[j])
        
    return U