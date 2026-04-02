import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_data(pi, A, lam, N):
    k = A.shape[0]
    S = np.zeros((N, k))
    s = scipy.stats.multinomial.rvs(1, pi)
    S[0] = s
    for i in range(1,N):
        A_selected = A @ s
        s = scipy.stats.multinomial.rvs(1, A_selected)
        S[i] = s
    X = np.zeros(N)
    for i in range(N):
        x = 0
        for j in range(k):
            x += S[i, j] * scipy.stats.poisson.rvs(lam[j])
        X[i] = x
    return X

def E_lambda(a, b):
    return a / b

def E_lambda_log(a, b):
    return scipy.special.digamma(a) - np.log(b)

def E_pi_log(alpha):
    return scipy.special.digamma(alpha) - scipy.special.digamma(sum(alpha))

def p_tilde_s_1(s, alpha):
    return np.exp(s @ E_pi_log(alpha))

def p_tilde_s_n(s_n, s_n_1, beta):
    beta_row = beta @ s_n_1
    return np.exp(s_n @ E_pi_log(beta_row))

def p_tilde_x_n(s_n, x, a, b):
    k = a.shape[0]
    ind = np.argwhere(s_n == 1)[0, 0]
    return np.exp(-E_lambda(a[ind], b[ind]) + x * E_lambda_log(a[ind], b[ind]) - np.log(scipy.special.factorial(x)))

def calculate_posterior_parameters(a, b, alpha, beta, x, maxiter):
    N = x.shape[0]
    k = alpha.shape[0]
    a_hat = np.full(k, a)
    b_hat = np.full(k, b)
    alpha_hat = alpha
    beta_hat = beta
    q_1 = np.zeros((N, k))
    q_2 = np.zeros((N-1, k, k))
    F = np.zeros((N, k))
    B = np.zeros((N, k))
    for _ in range(maxiter):
        # forward
        for i in range(N):
            if i == 0:
                for j in range(k):
                    s_1 = np.zeros(k)
                    s_1[j] = 1
                    F[i, j] = p_tilde_x_n(s_1, x[i], a_hat, b_hat) * p_tilde_s_1(s_1, alpha_hat)
                F[i] = F[i] / F[i].sum()
            else:
                for j in range(k):
                    s_n = np.zeros(k)
                    s_n[j] = 1
                    sum1 = 0
                    for l in range(k):
                        s_n_m1 = np.zeros(k)
                        s_n_m1[l] = 1
                        sum1 += p_tilde_s_n(s_n, s_n_m1, beta_hat) * F[i-1, l]
                    F[i, j] = p_tilde_x_n(s_n, x[i], a_hat, b_hat) * sum1
                F[i] = F[i] / F[i].sum()
        # backward
        for i in range(N-1,-1,-1):
            if i == N-1:
                B[i] = 1/k
            else:
                for j in range(k):
                    s_n = np.zeros(k)
                    s_n[j] = 1
                    sum1 = 0
                    for l in range(k):
                        s_n_p1 = np.zeros(k)
                        s_n_p1[l] = 1
                        sum1 += p_tilde_x_n(s_n_p1, x[i+1], a_hat, b_hat) * p_tilde_s_n(s_n_p1, s_n, beta_hat) * B[i+1, l]
                    B[i, j] = sum1
                B[i] = B[i] / B[i].sum()
        # update parameters of S
        q_1 = F * B
        q_1_sum = np.sum(q_1, axis=1)
        for i in range(k):
            q_1[:, i] = q_1[:, i] / q_1_sum
        for i in range(1, N):
            for j in range(k):
                s_n_m1 = np.zeros(k)
                s_n_m1[j] = 1
                for l in range(k):
                    s_n = np.zeros(k)
                    s_n[l] = 1
                    q_2[i-1, j, l] = p_tilde_x_n(s_n, x[i], a_hat, b_hat) * p_tilde_s_n(s_n, s_n_m1, beta_hat) * F[i-1, j] * B[i, l]
            q_2[i-1] = q_2[i-1] / q_2[i-1].sum()
        # lambda
        a_hat = q_1.T @ x + a
        b_hat = np.sum(q_1, axis=0) + b
        # pi
        alpha_hat = q_1[0] + alpha
        # beta
        beta_hat = np.sum(q_2, axis=0) + beta
    return q_1, a_hat, b_hat, alpha_hat, beta_hat

def plot_sampled_data(ax, x, N):
    ax.plot(x)
    ax.set_ylabel("data", fontsize=16)
    ax.set_xlim([0, N])

def plot_parameters_of_s(ax, s_params, N):
    x = np.arange(N)
    ax.plot(x, s_params[:, 1])
    ax.fill_between(x, s_params[:, 1], facecolor='blue', alpha=0.5)
    ax.set_ylabel("s", fontsize=16)
    ax.set_xlim([0, N])

if __name__  == "__main__":
    k = 2
    # parameters for generating data
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    lam = np.array([10, 20])
    N = 100
    # hyper parameters
    a = 1
    b = 1
    alpha = np.arange(1,k+1)
    beta = np.array([[1, 2],[4, 5]])
    # number of epoch
    maxiter = 30
    x = generate_data(pi, A, lam, N)
    s_params, a_hat, b_hat, alpha_hat, beta_hat = calculate_posterior_parameters(a, b, alpha, beta, x, maxiter)
    A_ave = np.zeros((k, k))
    for i in range(k):
        A_ave[:, i] = beta_hat[:, i] / beta_hat[:, i].sum()
    print(f"lambda true:{lam}, fitted value:{a_hat / b_hat}")
    print(f"A \ntrue:\n{A}\nfitted value:\n{A_ave}")
    print(f"")
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plot_sampled_data(ax1, x, N)
    plot_parameters_of_s(ax2, s_params, N)
    plt.show()