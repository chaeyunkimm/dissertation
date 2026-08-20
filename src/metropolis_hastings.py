import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions.multivariate_normal as MVN
import torch.distributions.normal as normal

from scipy.stats import uniform, multivariate_normal

def adaptive_mh(logdensityfunc, x0, sigma, nmoves=5, return_entire_chain=False, adapt=True, adapt_no = 100):
    acceptance = 0
    d = x0.shape[0]
    #print(sigma.dim())

    x0chainnumpy = [x0.detach().numpy()]
    if return_entire_chain:
        x0chain = [x0.view(-1)]

    for iter in range(nmoves):
        print('Fraction of steps:',iter/nmoves,'(Total:',nmoves,')')
        if np.remainder(iter, adapt_no)==0 and iter>0 and sigma.dim()>0 and adapt:
            sigma = torch.tensor((5.66/d)*(np.cov(np.array(x0chainnumpy)[-100:,:].T)+1e-10*np.eye(d)))
            #print('Time to update to sigma:', sigma)

        ### Log-Exp transformation used to return on real line as the parameter is positive valued ###
        if sigma.dim() == 0:
            x_new = x0 + normal.Normal(torch.tensor([0.0]), sigma * torch.tensor([1.0])).sample(sample_shape=torch.Size([1]))[0].to(dtype=torch.float32)
        else:
            x_new = x0 + MVN.MultivariateNormal(torch.zeros(d, dtype=torch.double), sigma).sample(sample_shape=torch.Size([1]))[0].to(dtype=torch.float32)
        alpha = np.log(uniform.rvs(0, 1, 1)[0])
        if alpha < min(logdensityfunc(x_new)-logdensityfunc(x0),0):
            xt = x_new
            #print('Accepted')
            #print(min(np.exp(logdensityfunc(x_new)-logdensityfunc(x0)),1), 'Accepted')
            acceptance = acceptance + 1
        else:
            xt = x0
            #print('Rejected')
        if torch.isnan(xt).any():
            print("nan values in the MCMC")
            break
        else:
            x0 = xt
        #print('Updated value', x0)
        #if return_entire_chain:

        print('Acceptance rate:', acceptance / (iter+1))

        x0chainnumpy.append(x0.detach().numpy())
        if return_entire_chain:
            x0chain.append(x0.view(-1))

    if return_entire_chain:
        return torch.stack(x0chain)
    else:
        return xt.view(-1)

def metropolis_hastings(target: function, 
                        proposal: function, 
                        random_proposal: function, 
                        initial_position: np.array, 
                        chain_length = 100, 
                        burn_in = None, 
                        gen = np.random.default_rng()) -> np.array:
    '''
    Runs a MH algorithm.
    '''
    def get_alpha(x, y):
        alpha = target(x)*proposal(y, x)/(target(y)*proposal(x, y))
        return min(1, alpha)

    chain = np.zeros((chain_length, len(initial_position)))
    chain[0] = initial_position
    accepted = 0

    for i in range(1, chain_length):
        u = gen.uniform()
        x_current = chain[i-1]
        x_dash = random_proposal(x_current) 
        a = get_alpha(x_dash, x_current)
        
        if u<a:
            chain[i] = x_dash
            accepted += 1
        else:
            chain[i] = chain[i-1]


    if burn_in is not None:
        chain = chain[burn_in:]

    return chain, accepted/chain_length

def adaptive_mh():
    ...

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