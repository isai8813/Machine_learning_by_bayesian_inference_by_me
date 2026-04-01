import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(p, number_of_data):
    return scipy.stats.multinomial.rvs(1, p, number_of_data)

def calculate_posterior_parameters(alpha, X, number_of_data):
    alpha_hat = np.zeros(3)
    for i in range(len(alpha)):
        sum1 = 0
        for j in range(number_of_data):
            sum1 = sum1 + X[j, i]
        alpha_hat[i] = sum1 + alpha[i]
    return alpha_hat

if __name__  == "__main__":
    p = [0.2, 0.3, 0.5]
    alpha = [10, 10, 10]
    for number_of_data in [5,10,50,100,500]:
        X = generate_data(p, number_of_data)
        alpha_hat = calculate_posterior_parameters(alpha, X, number_of_data)
        print(f"number_of_data:{number_of_data} alpha_hat:{alpha_hat}")