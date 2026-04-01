import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(loc, pre, N):
    scale = 1 / np.sqrt(pre)
    return scipy.stats.norm.rvs(loc, scale, N)

def calculate_posterior_parameters(mu, a, b, X, N):
    sum1 = 0
    for i in range(N):
        sum1 = sum1 + (X[i] - mu)**2
    a_hat = N / 2 + a
    b_hat = sum1 / 2 + b
    return a_hat, b_hat

def plot_a_posterior_distribution(a_hat, b_hat, xmin, xmax, N):
    scale = 1 / b_hat
    x = np.linspace(xmin, xmax, 100000)
    y = scipy.stats.gamma.pdf(x, a_hat, scale=scale)
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
    mu = 0
    pre = 4
    # hyper parameters
    a = 2
    b = 2
    # settings for plotting distributions
    xmin = 0
    xmax = 10
    for N in [i for i in range(0,500,50)]:
        X = generate_data(mu, pre, N)
        a_hat, b_hat = calculate_posterior_parameters(mu, a, b, X, N)
        print(f"n:{N}, a:{a_hat}, b:{b_hat:.3f}, average:{a_hat/b_hat}")
        plot_a_posterior_distribution(a_hat, b_hat, xmin, xmax, N)
    show(xmin, xmax)