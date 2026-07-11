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
    return x, X, Y

def calculate_posterior_parameters(lam, m, lam_w, X, Y, N):
    k = m.shape[0]
    sum1 = 0
    sum2 = 0
    for i in range(X.shape[0]):
        sum1 = sum1 + X[i].reshape(k, 1) @ X[i].reshape(1, k)
        sum2 = sum2 + Y[i] * X[i].reshape(k, 1)
    lam_w_hat = lam*sum1 + lam_w
    m_hat = np.linalg.inv(lam_w_hat) @ (lam*sum2 + lam_w @ m.reshape(k, 1))
    return m_hat, lam_w_hat

def plot_a_sampled_regression_model(m_hat, xmin, xmax, N, ax, no_n=False):
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
    if no_n:
        ax.plot(x, y, label=f"{N}")
    else:
        ax.plot(x, y, label=f"n={N}")

def plot_sampled_data(ax, x_plot, y_plot):
    ax.scatter(x_plot, y_plot)

def show(ax, xmin, xmax):
    ax.legend(bbox_to_anchor=(0, 0), loc='lower left', fontsize = 10)
    ax.set_xlabel("x", fontsize=16, fontweight='bold')
    ax.set_ylabel("y", fontsize=16, fontweight='bold')
    ax.set_xlim([xmin, xmax])
    ax.grid(True)

if __name__  == "__main__":
    k = 4
    # parameters for generating data
    w = np.array([-1, -1, 1, 1])
    lam = 10
    # hyper parameters
    m = np.array([0,0,0,0])
    lam_w = np.eye(k)*1
    # calculate parameters for a posterior distribution
    xmin = -1
    xmax = 1
    fig, ax = plt.subplots()
    stop = 10
    step = 2
    X = []
    Y = []
    for i, N in enumerate(range(0, stop+1, step)):
        if N == 0:
            x_only = np.array([])
            y = np.array([])
            plot_sampled_data(ax, x_only, y)
            plot_a_sampled_regression_model(m, xmin, xmax, N, ax)
        if N != 0:
            x_only, x, y = generate_data(w, lam, step, xmin, xmax)
            X.append(x)
            Y.append(y)
            X_stacked = np.stack(X).reshape(-1, k)
            Y_stacked = np.stack(Y).reshape(-1)
            m_hat, lam_w_hat = calculate_posterior_parameters(lam, m, lam_w, X_stacked, Y_stacked, N)
            plot_sampled_data(ax, x_only, y)
            plot_a_sampled_regression_model(m_hat, xmin, xmax, N, ax)
    plot_a_sampled_regression_model(w, xmin, xmax, "true expectation", ax, no_n=True)
    show(ax, xmin, xmax)
    plt.show()