import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def metropolis_hastings(target, proposal, random_proposal, initial_position: np.array, chain_length = 100, burn_in = None) -> np.array:
    '''
    Runs a MH algorithm.
    '''
    def get_alpha(x, y):
        alpha = target(x)*proposal(y, x)/(target(y)*proposal(x, y))
        return min(1, alpha)
    
    gen = np.random.default_rng()
    chain = np.zeros((chain_length, len(initial_position)))
    chain[0] = initial_position

    for i in range(1, chain_length):
        x_current = chain[i-1]
        x_dash = random_proposal(x_current) #Change this?
        a = get_alpha(x_dash, x_current)
        #print(x_current, x_dash, a)
        chain[i] = gen.choice((x_dash, x_current), p = (a, 1-a))

    if burn_in is not None:
        chain = chain[burn_in:]

    return chain

def mvn_def(covariance = None):
    if covariance is None:
        def mvn(x, y):
            return multivariate_normal.pdf(x, mean=y)
            
    else:
        def mvn(x, y):
            return multivariate_normal.pdf(x, mean=y, cov=covariance)
    return mvn

def sample_mvn(covariance = None):
    if covariance is None:
        def smvn(y):
            return multivariate_normal.rvs(mean=y)
    else:
        def smvn(y):
            return multivariate_normal.rvs(mean=y, cov=covariance)
    return smvn


def multimodal_normal_3d(x):
    """
    Evaluate a fixed 3D multimodal Gaussian density at x.

    Parameters
    ----------
    x : np.ndarray
        A 3D point, e.g. np.array([x1, x2, x3]).

    Returns
    -------
    float
        Probability density at x.
    """

    # Arbitrary mixture weights
    weights = np.array([0.3, 0.4, 0.3])

    # Arbitrary 3D means
    means = np.array([
        [0, 0, 0],
        [3, 3, 3],
        [-3, 2, 1]
    ])

    # Arbitrary 3D covariance matrices
    covariances = np.array([
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [
            [1.0, 0.0, 0.3],
            [0.0, 0.8, 0.0],
            [0.3, 0.0, 1.0]
        ]
    ])

    # Make sure x is a NumPy array
    x = np.asarray(x, dtype=float)

    density = 0.0

    # Sum the Gaussian components
    for w, mu, Sigma in zip(weights, means, covariances):

        diff = x - mu

        # Multivariate normal normalization constant
        normalization = 1 / np.sqrt(
            (2 * np.pi) ** 3 * np.linalg.det(Sigma)
        )

        # Mahalanobis distance
        exponent = -0.5 * diff @ np.linalg.inv(Sigma) @ diff

        # Add weighted Gaussian
        density += w * normalization * np.exp(exponent)

    return density

if __name__ == "__main__":
    prop = mvn_def()
    sample = sample_mvn()

    c = metropolis_hastings(multimodal_normal_3d, prop, sample, np.array([10,10,10]), 3000)

    plt.plot(c)
    plt.show()