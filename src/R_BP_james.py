import numpy as np
import torch
import matplotlib.pyplot as plt


from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.optimize import minimize, brentq
from scipy.special import ndtr

def get_alpha(k):
    '''
    Calculates the alpha for the rbp process, change here to change for all code
    '''
    return (2-1/k)/(k+1)

def gaussian_copula_density(u, v, rho):

    # u, v in [ 10^{-6}, 1-10^{-6} ]
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    numerator = np.exp( - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2)))
    denominator = np.sqrt(1 - rho**2)

    return numerator / denominator

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

def gaussian_conditional_cdf(u, v, rho):

    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    return norm.cdf((z_u - rho * z_v) / np.sqrt(1 - np.square(rho)))

def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype))))


def normal_ppf(u):
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.icdf(u)

def np_sigmoid(x):
    return 1/(1+np.exp(-x))

def R_BP_torch_edit(n, rho, p0_grid, P0_grid):
    p = p0_grid #.clone() removing as may cause a memory leak?
    P = P0_grid #.clone() This hopefully won't cause any autodif problems - let's test it!

    for i in range(n):
        alpha = get_alpha(i+1)

        u = P # P_{i-1}(x_grid) : grid 전체에 대한 cdf값들 
        u_i = P[i] #P_{i-1}(x_n)

        c_rho = gaussian_copula_density_torch(u, u_i, rho)
        H_rho = gaussian_conditional_cdf_torch(u, u_i, rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return p, P

def neg_log_like(theta, n, p0, P0):
    '''
    Calculates the negative log likelihood for an R-BP given observed values
    and a prior distribution.

    These observed values must be passed as their probability density 
    under the prior distribution. (PDF and CDF)

    '''

    p = torch.from_numpy(p0)
    P = torch.from_numpy(P0)

    rho = 0.999 * torch.sigmoid(torch.from_numpy(theta))

    p, _ = R_BP_torch_edit(n=n, rho=rho, p0_grid=p, P0_grid=P)
    p_x = torch.clip(p, 1e-12, None).numpy() 

    return float(np.sum(-np.log(p_x))) #Faster than torch for some reason...

# R-BP
def R_BP_density(n, rho, p0_grid, P0_grid, U = None):
    # p_0, P_0
    p = p0_grid
    P = P0_grid
    if U == None:
        # update
        for i in range(n):

            # weight
            alpha = get_alpha(i+1)

            u = P # P_{i-1}(x_grid) 
            u_i = P[i] #P_{i-1}(x_n)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

        return p, P
    else:
        for i in range(len(U)):

            # weight
            alpha = get_alpha(i+1)

            u = P # P_{i-1}(x_grid) 
            u_i = U[i] #P_{i-1}(x_i)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

        return p, P
    
def R_BP_density_U(rho, p0, P0, U):
    p = p0
    P = P0
    for i, ui in enumerate(U):

            # weight
            alpha = get_alpha(i+1)

            u = P # P_{i-1}(x_grid) 
            u_i = ui #P_{i-1}(x_i)

            c_rho = gaussian_copula_density(u, u_i, rho)
            H_rho = gaussian_conditional_cdf(u, u_i, rho)

            p = p * ((1 - alpha) + alpha * c_rho)
            P = (1 - alpha) * P + alpha * H_rho

    return p, P

def R_BP_coefs(rho, p0_obs, P0_obs):
    p = p0_obs
    P = P0_obs
    n = len(p0_obs)
    U = np.zeros(n)
    for i in range(n):

        # weight
        alpha = get_alpha(i+1)

        u = P # P_{i-1}(x_grid) 
        U[i] = P[i] #P_{i-1}(x_n)
        

        c_rho = gaussian_copula_density(u, U[i], rho)
        H_rho = gaussian_conditional_cdf(u, U[i], rho)

        p = p * ((1 - alpha) + alpha * c_rho)
        P = (1 - alpha) * P + alpha * H_rho

    return U

def estimate_rho_optim(n, p0, P0, max_iter=100, rho_0 = 0.6):
    '''
    Estimates the shape parameter rho in the R-BP algorithm via
    minimising the negative log-likelihood for a set observed values.

    These observed values must be passed as their probability density 
    under the prior distribution. (PDF and CDF)

    --------------------------------------------------------------------

    n: int ; the number of observed values in the prior grid
    p0: float array ; the prior PDF evaluated at the observed values
    P0: float array ; the prior CDF evaluated at the observed values

    max_iter: int ; the maximum number of iterations to run .minimize for.
    rho_0: float ; the initial guess for rho. Must be in (0, 1).

    '''
    p0 = np.asarray(p0, dtype=np.float64)
    P0 = np.asarray(P0, dtype=np.float64)
    theta = minimize(
        neg_log_like,
        x0=np.log(rho_0 / (1.0 - rho_0)),
        args=(n, p0, P0),
        method="BFGS",
        options={"maxiter": max_iter},
    ).x[0]
    return float(0.999 * np_sigmoid(theta))

def fit_R_BP_marginals(p0_obs, P0_obs):
    n, d = p0_obs.shape
    rhos = np.zeros(d)
    Us = np.zeros_like(p0_obs)
    for i in range(d):
        rhos[i] = estimate_rho_optim(n, p0_obs[:,i], P0_obs[:,i])
        Us[:, i] = R_BP_coefs(rhos[i], p0_obs[:,i], P0_obs[:,i])
    return rhos, Us

def invert_rbp_cdf(data:np.ndarray, rhos:np.ndarray, Us:np.ndarray, P0_inv:function)->np.ndarray:
    '''
    Go from an evaluation of the RBP CDF back to a value in X.

    Parameters:
    -----------
    data : Evaluations to be inverted
    rhos : Bandwidth parameter
    Us : Trained intermediary RBP CDF evaluations
    '''
    n = Us.shape[0]
    d = Us.shape[1]

    def eq(alpha, v, U):
        def f(U_prev):
            return ((1 - alpha) * U_prev + alpha * gaussian_conditional_cdf(U_prev, v, rho) - U)
        return f

    for j in range(d):
        rho = rhos[j]
        current_U = data[j]

        # backward recursion
        for i in reversed(range(n)):

            alpha = get_alpha(i+1)

            func = eq(alpha, Us[i], current_U) 

            # finding root : f(U_pre) = 0 
            U_prev = brentq(func, 1e-10, 1 - 1e-10)

            current_U = U_prev

        # Y = P_0^{-1}(U_0)
        y = P0_inv(current_U)
    return 0




class rbp_process:
    def __init__(self, grid_size = 800, grid_max = 50, grid_type = 'cosine'):
        '''
        Initialises the rbp process and grid to evaluate on.

        Assumes input will be standardised.
        '''

        self.grid_size = grid_size
        self.grid_max = grid_max

        if grid_type == 'cosine':
            t = np.linspace(0, 1, num=self.grid_size)
            self.grid = (1 - np.cos(np.pi * t)) / 2 # Uses cosine decay to place more points near the boundary
        elif grid_type == 'linear':
            self.grid = np.linspace(0, 1, num=self.grid_size)

    def fit(self, prior_p_data:np.ndarray, prior_c_data:np.ndarray, ):
        self.prior_p_data, self.prior_c_data = prior_p_data, prior_c_data
        self.rhos, self.pivots = fit_R_BP_marginals(prior_p_data, prior_c_data)

        full_grid = np.repeat(np.expand_dims(self.grid, axis = 1), repeats = len(self.rhos), axis=1)
        self.cdf_grid = self.cdf(full_grid) # size (len(self.grid), len(self.rhos))

    def update_fit(self, new_p_data, new_c_data):
        self.prior_p_data = np.concatenate((self.prior_p_data, new_p_data))
        self.prior_c_data = np.concatenate((self.prior_c_data, new_c_data))
        self.rhos, self.pivots = fit_R_BP_marginals(self.prior_p_data, self.prior_c_data)

    def pdf(self, p_data, c_data):
        p, _ = R_BP_density_U(self.rhos, p_data, c_data, self.pivots)
        return p

    def cdf__(self, p_data, c_data):
        '''
        Deprecated, evaluates the full recursion, requiring the p_data also.
        '''
        _, c = R_BP_density_U(self.rhos, p_data, c_data, self.pivots)
        return c

    def cdf(self, c_data):
        '''
        Evaluates only the cdf of the rbp process.

        c_data: Prior cdf evaluations on [0,1]
        '''
        c = c_data
        for i, u_i in enumerate(self.pivots):
                # weight
                alpha = get_alpha(i+1)
                u = c

                H_rho = gaussian_conditional_cdf(u, u_i, self.rhos)

                c = (1 - alpha) * c + alpha * H_rho
        return c 

    def eval(self, p_data, c_data):
        '''
        Evaluates the full recursion on both the pdf and cdf.
        '''
        return R_BP_density_U(self.rhos, p_data, c_data, self.pivots)

    def inverse_cdf(self, rbp_c_data:np.ndarray)->np.ndarray:
        '''
        Go from 1 evaluation of the RBP CDF back to a value in X.

        Uses Brentq recursively.
    
        Parameters:
        -----------
        rbp_c_data : Evaluations to be inverted
        '''
        n, _ = self.pivots.shape
        inverted_data = np.zeros_like(rbp_c_data)

        def eq(alpha, v, U):
            def f(U_prev):
                return ((1 - alpha) * U_prev + alpha * gaussian_conditional_cdf(U_prev, v, rho) - U)
            return f

        for i, c_datum in enumerate(rbp_c_data):

            inv_datum = np.zeros_like(c_datum)
            for j, (rho, current_U) in enumerate(zip(self.rhos, c_datum)):
                # backward recursion
                for k, pivot in enumerate(reversed(self.pivots)):
                    idx = n-1-k

                    alpha = get_alpha(idx+1)
        
                    func = eq(alpha, pivot[j], current_U) 
        
                    # finding root : f(U_pre) = 0 
                    U_prev = brentq(func, 1e-6, 1 - 1e-6)
        
                    current_U = U_prev

                inv_datum[j] = current_U

            inverted_data[i] = inv_datum

        return inverted_data

    def estimate_inverse_cdf(self, rbp_c_data:np.ndarray)->np.ndarray:
        '''
        Estimates the inverse cdf of the rbp process using the grid of evaluations
        '''
        inverted_c_data = np.zeros_like(rbp_c_data)
        for i, c_col in enumerate(rbp_c_data.T):
            inverted_c_data[:, i] = np.interp(c_col, self.cdf_grid[:, i], self.grid)

        return inverted_c_data

    def plot_real_line_pdfs(self, x_grid=None, n_points=500):
        """
        Inspect the fitted RBP marginal densities on the unit interval [0, 1].

        Parameters
        ----------
        x_grid : np.ndarray, optional
            Grid over [0, 1]. If not provided, a uniform grid is created.
        n_points : int
            Number of points if x_grid is not supplied.
        """
        if not hasattr(self, 'rhos') or not hasattr(self, 'pivots'):
            raise ValueError("The RBP process has not been fitted yet.")

        if x_grid is None:
            x_grid = np.linspace(0, 1, n_points)
        x_grid = np.asarray(x_grid, dtype=float)
        if x_grid.ndim != 1:
            raise ValueError("x_grid must be a 1D array of points in [0, 1].")
        if np.any((x_grid < 0) | (x_grid > 1)):
            raise ValueError("x_grid must lie within [0, 1].")

        d = self.prior_p_data.shape[1]
        fig, axes = plt.subplots(1, d, figsize=(4 * d, 4), squeeze=False)

        for j in range(d):
            prior_p = np.ones_like(x_grid)
            prior_c = x_grid.copy()
            rbp_p, _ = R_BP_density_U(self.rhos[j], prior_p, prior_c, self.pivots[:, j])

            axes[0, j].plot(x_grid, rbp_p, linewidth=2, label='RBP PDF')
            axes[0, j].set_title(f'Marginal {j + 1}')
            axes[0, j].set_xlabel('u')
            axes[0, j].set_ylabel('density')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()




####################################################################

def fit_R_BP_marginals_observed(X_train, n_grid=1000,):
    """
    James 원본 방식으로 R-BP marginal을 적합하고,
    1번 파이프라인과 동일한 7개 값을 반환한다.

    x_grids:
        변수별 x grid

    p0_grids:
        변수별 초기 KDE PDF grid

    P0_grids:
        변수별 초기 KDE CDF grid

    rhos:
        변수별 추정 rho

    p_grids:
        변수별 최종 R-BP PDF grid

    P_grids:
        변수별 최종 R-BP CDF grid

    Us:
        transform된
    """

    x = np.asarray(X_train, dtype=float)

    # 관측 개수와 변수 개수
    n = x.shape[0]
    d = x.shape[1]

    # 변수별 KDE 객체를 저장
    kdes = []

    # 관측 위치 초기 PDF
    p0_grid = np.zeros((n, d))
    # 관측 위치 초기 CDF
    P0_grid = np.zeros((n, d))

    #  grid
    #
    # 기본 shape: (1000,)
    x_grid = np.linspace(-10, 15,num=n_grid)

    # ========================================================
    # James 원본:
    # 관측 위치에서 KDE PDF/CDF 계산
    # ========================================================

   

    for i in range(x.shape[1]):
        # Fit prior distributions and transform data for rbp
        kde = gaussian_kde(x.T[i])
        kdes.append(kde)
        p0_grid[:,i] = kde(x.T[i])
        P0_grid[:,i] = np.mean(ndtr((x.T[i] - kde.dataset.T) / kde.factor), axis=0)



    rhos, Us = fit_R_BP_marginals(p0_grid, P0_grid)
    

    x_grids = []

    p0_grids = []
    P0_grids = []

    p_grids = []
    P_grids = []

    for i in range(d):

        kde = kdes[i]

        # grid에서 초기 KDE PDF
        p0_grid_i = kde(x_grid)

        # grid에서 초기 KDE CDF
        P0_grid_i = np.mean(ndtr((x_grid - kde.dataset.T) / kde.factor), axis=0)

        # 저장된 Us를 grid 전체에 적용
        p_grid_i, P_grid_i = R_BP_density_U(rhos[i], p0_grid_i, P0_grid_i, Us[:, i])


        x_grids.append(x_grid)
        p0_grids.append(p0_grid_i)
        P0_grids.append(P0_grid_i)
        p_grids.append(p_grid_i)
        P_grids.append(P_grid_i)

    # ========================================================
    # 1번 파이프라인과 동일한 7개 값 반환
    # ========================================================

    return (x_grids,
            p0_grids,
            P0_grids,
            rhos,
            p_grids,
            P_grids,
            Us)