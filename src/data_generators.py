import numpy as np
from scipy.stats import norm, multivariate_normal
 
##################
#      grid      #
##################

def generate_x_grid(x, n_grid=1000):
    
    margin = 0.1 * (np.max(x) - np.min(x))
    x_grid = np.linspace(np.min(x) - margin, np.max(x) + margin, n_grid)
    
    return x_grid

##################
#    iid data    #
##################

# Gaussian distribution
def generate_normal (n, mean=0.0, sd=1.0, seed=None):
   
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=mean, scale=sd, size=n)
    
    return x

# Gaussian mixture distribution
def generate_gaussian_mixture(n, weights=(0.5, 0.5), means=(-2.0, 2.0), sds=(0.5, 1.0), seed=None ):
    
    rng = np.random.default_rng(seed)

    weights = np.array(weights)
    means = np.array(means)
    sds = np.array(sds)
    
    components = rng.choice(len(weights), size=n, p=weights)     

    x = rng.normal(loc=means[components], scale=sds[components])

    return x


#################################
#   Dependent time series data  #
#################################

# AR(1)
def generate_ar1(n, phi=0.7, sigma=1.0, x0=0.0, seed=None):

    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    epsilon = rng.normal(loc=0.0, scale=sigma, size=n)
    
    for t in range(1, n):
        x[0] = x0
        x[t] = phi * x[t-1] + epsilon[t]

    return x

# GARCH(1,1)
def generate_garch11(n, omega=0.1, alpha=0.1, beta=0.8, seed=None):
  
    rng = np.random.default_rng(seed)

    x = np.zeros(n)
    sigma2 = np.zeros(n)
    epsilon = rng.normal(loc=0.0, scale=1.0, size=n)

    # unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)
    x[0] = np.sqrt(sigma2[0]) * epsilon[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * x[t - 1] ** 2 + beta * sigma2[t - 1]
        x[t] = np.sqrt(sigma2[t]) * epsilon[t]

    return x, sigma2


####################
#   multivariate   #
####################


def generate_var1(T=1000, burn_in=200):

    rng = np.random.default_rng(seed=123)

    A = np.array([
        [0.6, 0.2],
        [0.1, 0.5] 
    ])

    Sigma = np.array([
        [1.0, 0.6],
        [0.6, 1.0]
    ])

    total_T = T + burn_in

    X = np.zeros((total_T, 2))

    eps = rng.multivariate_normal(mean=np.zeros(2), cov=Sigma, size=total_T)

    for t in range(total_T - 1):
        X[t + 1] = A @ X[t] + eps[t + 1]

    X = X[burn_in:]

    return X, A, Sigma

def true_var1_conditional_density(y, x_t, A, Sigma):
 
    y = np.asarray(y)
    x_t = np.asarray(x_t)

    mean_cond = A @ x_t

    density = multivariate_normal.pdf(y, mean=mean_cond, cov=Sigma)

    return density





def generate_var3_3d(T=1000, burn_in=300):

    
    rng = np.random.default_rng(seed=123)

    A1 = np.array([
        [0.4, 0.1, 0.1],
        [0.1, 0.3, 0.1],
        [0.1, 0.1, 0.3]
    ])

    A2 = np.array([
        [0.2, 0.1, 0.0],
        [0.1, 0.2, 0.1],
        [0.0, 0.1, 0.2]
    ])

    A3 = np.array([
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1]
    ])
    Sigma = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.5, 0.4],
        [0.3, 0.4, 0.7]
    ])

    total_T = T + burn_in

    X = np.zeros((total_T, 3))

    eps = rng.multivariate_normal(mean=np.zeros(3), cov=Sigma, size=total_T)

    for t in range(2, total_T - 1):

        X[t + 1] = ( A1 @ X[t] + A2 @ X[t - 1] + A3 @ X[t - 2] + eps[t + 1] )

    X = X[burn_in:]

    return X, A1, A2, A3, Sigma

def true_var3_3d_conditional_density(y, x_t, x_t_minus_1, x_t_minus_2, A1, A2, A3, Sigma):

    y = np.asarray(y)
    x_t = np.asarray(x_t)
    x_t_minus_1 = np.asarray(x_t_minus_1)
    x_t_minus_2 = np.asarray(x_t_minus_2)

    mean_cond = ( A1 @ x_t + A2 @ x_t_minus_1 + A3 @ x_t_minus_2 )

    density = multivariate_normal.pdf(y, mean=mean_cond, cov=Sigma)

    return density