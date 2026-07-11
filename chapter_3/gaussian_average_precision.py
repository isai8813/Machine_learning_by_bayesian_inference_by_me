import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(loc, pre, N):
    scale = 1 / np.sqrt(pre)
    return scipy.stats.norm.rvs(loc, scale, N)

def calculate_posterior_parameters(m, beta, a, b, X, N):
    sum1 = 0
    sum2 = 0
    for i in range(N):
        sum1 = sum1 + X[i]
        sum2 = sum2 + X[i]**2
    beta_hat = N + beta
    m_hat = (sum1 + beta*m) / beta_hat
    a_hat = N/2 + a
    b_hat = (sum2 + beta*m**2 - beta_hat*m_hat**2) / 2 + b
    return beta_hat, m_hat, a_hat, b_hat

def plot_a_posterior_distribution_mu(m_hat, beta_hat, a_hat, b_hat, xmin, xmax, N, ax):
    lam_ave = a_hat / b_hat
    scale = 1 / np.sqrt(beta_hat*lam_ave)
    x = np.linspace(xmin, xmax, 100000)
    y = scipy.stats.norm.pdf(x, m_hat, scale)
    ax.plot(x, y, label=f"n={N}")

def plot_a_posterior_distribution_lam(a_hat, b_hat, xmin, xmax, N, ax):
    scale = 1 / b_hat
    x = np.linspace(xmin, xmax, 100000)
    y = scipy.stats.gamma.pdf(x, a_hat, scale=scale)
    ax.plot(x, y, label=f"n={N}")

def show(ax, xmin, xmax, title):
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x", fontsize=16, fontweight='bold')
    ax.set_ylabel("y", fontsize=16, fontweight='bold')
    ax.set_xlim([xmin, xmax])
    ax.grid(True)

if __name__  == "__main__":
    # parameters for generating data
    mu = 2
    lam = 4
    # hyper parameters
    m = 0
    beta = 1
    a = 2
    b = 2
    # settings for plotting distributions
    xmin_mu = -2
    xmax_mu = 4
    lam_xmin = 0
    lam_xmax = 10
    title_mu = "average"
    title_lam = "precision"
    fig_1, ax_1 = plt.subplots()
    fig_2, ax_2 = plt.subplots()
    for N in [i for i in [0, 5,10,50,100,500]]:
        X = generate_data(mu, lam, N)
        beta_hat, m_hat, a_hat, b_hat = calculate_posterior_parameters(m, beta, a, b, X, N)
        plot_a_posterior_distribution_mu(m_hat, beta_hat, a_hat, b_hat, xmin_mu, xmax_mu, N, ax_1)
        plot_a_posterior_distribution_lam(a_hat, b_hat, lam_xmin, lam_xmax, N, ax_2)
    show(ax_1, xmin_mu, xmax_mu, title_mu)
    show(ax_2, lam_xmin, lam_xmax, title_lam)
    plt.show()