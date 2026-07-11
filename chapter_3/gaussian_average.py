import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(loc, pre, N):
    scale = 1 / np.sqrt(pre)
    return scipy.stats.norm.rvs(loc, scale, N)

def calculate_posterior_parameters(lam, m, lam_mu, X, N):
    sum1 = 0
    for i in range(N):
        sum1 = sum1 + X[i]
    lam_mu_hat = N*lam + lam_mu
    m_hat = (lam*sum1 + lam_mu*m) / lam_mu_hat
    return m_hat, lam_mu_hat

def plot_a_posterior_distribution(m_hat, lam_mu_hat, xmin, xmax, N):
    scale = 1 / lam_mu_hat
    x = np.linspace(xmin, xmax, 100000)
    y = scipy.stats.norm.pdf(x, m_hat, scale)
    plt.plot(x, y, label=f"n={N}")

def show(xmin, xmax):
    plt.legend()
    plt.title("Posterior distribution")
    plt.xlabel("x",fontsize=16,fontweight='bold')
    plt.ylabel("y",fontsize=16,fontweight='bold')
    plt.xlim([xmin, xmax])
    plt.grid(True)
    plt.show()

if __name__  == "__main__":
    # parameters for generating data
    loc = 0
    lam = 1
    # hyper parameters
    m = 1
    lam_mu = 10
    # settings for plotting distributions
    xmin = -0.5
    xmax = 2
    for N in [i for i in range(0,100,10)]:
        X = generate_data(loc, lam, N)
        m_hat, lam_mu_hat = calculate_posterior_parameters(lam, m, lam_mu, X, N)
        print(f"n:{N}, average:{m_hat:.3f}, precision:{lam_mu_hat}")
        plot_a_posterior_distribution(m_hat, lam_mu_hat, xmin, xmax, N)
    show(xmin, xmax)