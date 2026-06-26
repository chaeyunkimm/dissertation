import numpy as np
from scipy.stats import norm

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