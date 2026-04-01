import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(loc, pre, N):
    scale = np.linalg.inv(pre)
    return scipy.stats.multivariate_normal.rvs(loc, scale, N)

def calculate_posterior_parameters(lam, m, lam_mu, X, k, N):
    sum1 = np.zeros(k)
    for i in range(N):
        sum1 = sum1 + X[i]
    lam_mu_hat = N * lam + lam_mu
    m_hat = np.linalg.inv(lam_mu_hat) @ (lam @ np.c_[sum1] + lam_mu @ np.c_[m]) 
    return m_hat, lam_mu_hat

if __name__  == "__main__":
    # parameters for generating data
    k = 2
    loc = np.zeros(k)
    lam = np.eye(k)
    # hyper parameters
    m = np.zeros(k) + 2
    lam_mu = np.eye(k) * 10
    # calculate parameters for a posterior distribution
    for N in [i for i in range(0,1000,200)]:
        X = generate_data(loc, lam, N)
        m_hat, lam_mu_hat = calculate_posterior_parameters(lam, m, lam_mu, X, k, N)
        print(f"n:{N}\n average:\n{m_hat}, \nprecision:\n{lam_mu_hat}")