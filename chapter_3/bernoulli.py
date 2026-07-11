import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(p, number_of_data):
    return np.random.binomial(1, p, number_of_data)

def calculate_posterior_parameters(a, b, X, number_of_data):
    sum1 = 0
    for i in range(number_of_data):
        sum1 = sum1 + X[i]
    a_hat = sum1 + a
    b_hat = number_of_data - sum1 + b
    return a_hat, b_hat

def plot_a_posterior_distribution(a_hat, b_hat, xmin, xmax, number_of_data):
    x = np.linspace(xmin, xmax, 1000)
    y = scipy.stats.beta.pdf(x, a_hat, b_hat)
    plt.plot(x, y, label=f"n={number_of_data}")

def show(xmin, xmax):
    plt.legend()
    plt.title("Posterior distribution")
    plt.xlabel("x",fontsize=16,fontweight='bold')
    plt.ylabel("y",fontsize=16,fontweight='bold')
    plt.xlim([xmin, xmax])
    plt.grid(True)
    plt.show()

if __name__  == "__main__":
    p = 0.2
    a = 10
    b = 10
    xmin = 0
    xmax = 1
    for number_of_data in [5,10,50,100,500]:
        X = generate_data(p, number_of_data)
        a_hat, b_hat = calculate_posterior_parameters(a, b, X, number_of_data)
        plot_a_posterior_distribution(a_hat, b_hat, xmin, xmax, number_of_data)
    show(xmin, xmax)