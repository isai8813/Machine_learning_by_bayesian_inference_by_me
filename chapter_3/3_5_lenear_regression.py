import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

def generate_data(w, lam, N, xmin, xmax):
    k = w.shape[0]
    x = np.array([random.uniform(xmin, xmax) for i in range(N)])
    X = np.zeros((N, k))
    Y = np.zeros(N)
    scale = 1 / lam
    for i in range(N):
        x_power = np.zeros(k)
        for j in range(k):
            x_power[j] = x[i]**j
        ave = w @ x_power
        y = scipy.stats.norm.rvs(ave, scale)
        X[i] = x_power
        Y[i] = y
    return X, Y

def calculate_posterior_parameters(lam, m, lam_w, X, Y, N):
    k = m.shape[0]
    sum1 = 0
    sum2 = 0
    for i in range(N):
        sum1 = sum1 + X[i].reshape(k, 1) @ X[i].reshape(1, k)
        sum2 = sum2 + Y[i] * X[i].reshape(k, 1)
    lam_w_hat = lam*sum1 + lam_w
    m_hat = np.linalg.inv(lam_w_hat) @ (lam*sum2 + lam_w @ m.reshape(k, 1))
    return m_hat, lam_w_hat

def plot_a_sampled_regression_model(m_hat, lam, xmin, xmax, N, ax):
    k = m_hat.shape[0]
    n = 10000
    x = np.linspace(xmin, xmax, n)
    y = np.zeros(n)
    for i in range(n):
        x_power = np.zeros(k)
        for j in range(k):
            x_power[j] = x[i]**j
        ave = m_hat.reshape(k) @ x_power
        y[i] = ave
    ax.plot(x, y, label=f"n={N}")

def show(ax, xmin, xmax):
    ax.legend()
    ax.set_xlabel("x", fontsize=16, fontweight='bold')
    ax.set_ylabel("y", fontsize=16, fontweight='bold')
    ax.set_xlim([xmin, xmax])
    ax.grid(True)

if __name__  == "__main__":
    # parameters for generating data
    w = np.array([-1, -1, 1, 1])
    lam = 10
    # hyper parameters
    m = np.zeros(4)
    lam_w = np.eye(4)
    # calculate parameters for a posterior distribution
    xmin = -1
    xmax = 1
    fig, ax = plt.subplots()
    for N in [i for i in range(0,30,5)]:
        X, Y = generate_data(w, lam, N, xmin, xmax)
        m_hat, lam_w_hat = calculate_posterior_parameters(lam, m, lam_w, X, Y, N)
        plot_a_sampled_regression_model(m_hat, lam, xmin, xmax, N, ax)
    show(ax, xmin, xmax)
    plt.show()