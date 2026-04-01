import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(pi, lam, k, N):
    Cat = scipy.stats.multinomial.rvs(1, pi, N)
    X = np.zeros((N, k))
    for i in range(k):
        X[:,i] = scipy.stats.poisson.rvs(lam[i], size=N)
    x = np.zeros(N)
    for i in range(N):
        x[i] = Cat[i] @ X[i]
    return x

def E_lambda(a,b):
    return a / b

def E_lambda_log(a,b):
    return scipy.special.digamma(a) - np.log(b)

def E_pi_log(alpha):
    return scipy.special.digamma(alpha) - scipy.special.digamma(sum(alpha))

def calculate_posterior_parameters(a, b, alpha, X, N, maxiter):
    a_hat = np.array([2, 3])
    b_hat = np.array([2, 3])
    for i in range(maxiter):
        # S
        eta_list = []
        for n in range(N):
            eta_n = np.exp(X[n]*E_lambda_log(a_hat, b_hat) - E_lambda(a_hat, b_hat) + E_pi_log(alpha))
            eta_n = eta_n / sum(eta_n)
            eta_list.append(eta_n)
        eta = np.stack(eta_list)
        # lambda
        a_hat = eta.T @ X + a
        b_hat = np.sum(eta, axis=0) + b
        # pi
        alpha_hat = np.sum(eta, axis=0) + alpha
    return eta, a_hat, b_hat, alpha_hat

def plot_sampled_data(ax, x):
    ax.hist(x, bins=20, color="cornflowerblue")
    ax.set_ylabel('number of data', fontsize=16)

def fitted_function(x, a_hat, b_hat, alpha_hat, k):
    pi = alpha_hat / sum(alpha_hat)
    lam = a_hat / b_hat
    y = 0
    for i in range(k):
        y += pi[i]*scipy.stats.poisson.pmf(x, lam[i])
    return y

def plot_fitted_distribution(a_hat, b_hat, alpha_hat, k, ax):
    x = np.arange(start=0, stop=50+1, step=1)
    y = fitted_function(x, a_hat, b_hat, alpha_hat, k)
    ax.bar(x, y, color='gray', alpha=0.5, width=0.7)
    ax.grid(True)
    ax.set_ylabel('probability', fontsize=16)

def show_all(ax1, ax2):
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right')
    plt.show()
    
if __name__  == "__main__":
    k = 2
    # parameters for generating data
    pi_gene = np.array((0.7, 0.3))
    lam_gene = np.array((10, 30))
    # # hyper parameters
    a = 1
    b = 1
    alpha = np.ones(k)*1
    # number of epoch
    maxiter = 30
    # number of data
    N=100
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    x = generate_data(pi_gene, lam_gene, k, N)
    eta, a_hat, b_hat, alpha_hat = calculate_posterior_parameters(a, b, alpha, x, k, N, maxiter)
    plot_sampled_data(ax1, x)
    plot_fitted_distribution(a_hat, b_hat, alpha_hat, k, ax2)
    show_all(ax1, ax2)