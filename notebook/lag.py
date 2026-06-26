import numpy as np

# lag k
def lag_k_matrix(x, k):
    
    rows = []
    for t in range(len(x) - k + 1):
        rows.append(x[t:t+k])

    X_lag = np.array(rows)

    x_condition = X_lag[:, :-1]
    x_target = X_lag[:, -1]
    
    return X_lag, x_condition, x_target

