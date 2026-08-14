import numpy as np

def lagged_k_U(U, k):
    
    U = np.asarray(U)
    T, d = U.shape

    U_condition = []
    U_target = []

    for t in range(k - 1, T - 1):
        cond = U[t-k+1:t+1, :].T.reshape(-1)
        target = U[t + 1, :]

        U_condition.append(cond)
        U_target.append(target)

    U_condition = np.array(U_condition)
    U_target = np.array(U_target)

    return U_condition, U_target