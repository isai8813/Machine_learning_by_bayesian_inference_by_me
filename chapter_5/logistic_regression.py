import numpy as np
import scipy
import matplotlib.pyplot as plt

def true_func(w1, w2, x1, x2):
    return 1 / (1 + np.exp(w1*x1 + w2*x2))

def generate_data(n, w1, w2, rng):
    x1 = rng.random(n)*2 - 1
    x2 = rng.random(n)*2 - 1
    y_e = true_func(w1, w2, x1, x2)
    y = np.zeros((n, 2))
    for i in range(n):
        b = scipy.stats.binom.rvs(1, y_e[i], random_state=123)
        y[i, b] = 1
    x = np.stack([x1, x2], 1)
    return x, y

def softmax(W, x, d):
    return np.exp(W @ x)[d] / np.exp(W @ x).sum()

def optimize(x, y, lam, maxiter, rng, lr):
    mu = rng.standard_normal((2, 2))
    rho = rng.standard_normal((2, 2))
    for _ in range(maxiter):
        eps_h = rng.standard_normal((2, 2))
        W_h = mu + rho * eps_h
        for d in range(W_h.shape[0]):
            for m in range(W_h.shape[1]):
                sum1 = 0
                for n in range(y.shape[0]):
                    sum1 += (softmax(W_h, x[n], d) - y[n, d]) * x[n, m]
                delta_mu = lam * W_h[d, m] + sum1
                delta_rho = (- 1 / np.log(1 + np.exp(rho[d, m])) + lam * W_h[d, m] * eps_h[d, m] + sum1 * eps_h[d, m]) * (1 / (1 + np.exp(-rho[d, m])))
                mu[d, m] -= lr * delta_mu
                rho[d, m] -= lr * delta_rho
    return mu, rho

def sample_params(mu, rho, n):
    W = np.zeros((n,) + mu.shape)
    for i in range(W.shape[1]):
        for j in range(W.shape[2]):
            W[:, i, j] = scipy.stats.norm.rvs(loc=mu[i, j], scale=rho[i, j], size=n, random_state=123)
    return W

def predict(W, x):
    sum1 = []
    for i in range(W.shape[0]):
        sum_sm = np.zeros(W[i].shape[0])
        for d in range(W[i].shape[0]):
            sum_sm[d] += softmax(W[i], x, d)
        sum1.append(sum_sm)
    return np.stack(sum1).mean(axis=0)

def plot_data(x, y):
    fig = plt.figure()
    ax = fig.add_subplot()
    x_1 = []
    x_2 = []
    for i in range(x.shape[0]):
        if y[i, 0] == 1:
            x_1.append(x[i])
        elif y[i, 0] == 0:
            x_2.append(x[i])
    x_1 = np.stack(x_1)
    x_2 = np.stack(x_2)
    ax.scatter(x_1[:, 0], x_1[:, 1], c="blue")
    ax.scatter(x_2[:, 0], x_2[:, 1], c="red")
    return fig, ax

def plot_sampled_model(W_all, fig, ax):
    for W in W_all:
        w1 = W[1, 0] - W[0, 0]
        w2 = W[1, 1] - W[0, 1]
        ax.axline((0, 0), (1, - w1 / w2), c="black")
    return fig, ax

def plot_probability(W, fig, ax):
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = predict(W, np.array((X[i, j], Y[i, j])))[0] 
    ax.pcolormesh(X, Y, Z)
    ax.contour(X, Y, Z, 5, vmin=-1, vmax=1, colors=['black'])
    return fig, ax

if __name__  == "__main__":
    n = 30
    w1 = -1
    w2 = 2
    lam = 1
    rng = np.random.default_rng(123)
    x, y = generate_data(n, w1, w2, rng)
    maxiter = 10
    lr = 0.1
    mu, rho = optimize(x, y, lam, maxiter, rng, lr)
    n_params = 10
    var = np.log(1 + np.exp(rho))
    W = sample_params(mu, var, n_params)
    fig, ax = plot_data(x, y)
    fig_p, ax_p = plot_data(x, y)
    fig_line, ax_line = plot_sampled_model(W, fig, ax)
    fig_p, ax_p = plot_probability(W, fig_p, ax_p)
    plt.show()