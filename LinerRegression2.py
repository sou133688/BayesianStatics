import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

def bivariate_normal(x_seq, y_seq, mu, S):
    return mlab.bivariate_normal(x_seq, y_seq, S[0,0] ** (0.5), S[1,1] ** (0.5), mu[0], mu[1], S[0, 1])

def plot_data(ax, X, T):
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)
    ax.scatter(X, T, color = 'b')

def plot_heatmap(ax, Z, x_seq, y_seq):
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)
    ax.pcolor(x_seq, y_seq, Z, cmap=plt.cm.jet)

def plot_line(ax, W, seq):
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1)
    for _w in W:
        ax.plot(seq, _w.dot(np.vstack((np.ones(seq.size), seq))), color = 'r')


if __name__ == "__main__":
    # init
    w = np.array([-0.3, 0.5])
    sigma = 0.2
    alpha, beta = 2.0, 1.0 / (sigma ** 2)
    mu = np.zeros(2)
    S = np.identity(2) / alpha
    N = 15

    # plot
    fig = plt.figure(figsize = (15, 5 * N))
    seq = np.linspace(-1.0, 1.0, 51)
    x_seq, y_seq = np.meshgrid(seq, seq)
    Z = bivariate_normal(x_seq, y_seq, mu, S)
    W = np.random.multivariate_normal(mu, S, 6)
    axes = [fig.add_subplot(N + 1, 3, j) for j in range(1, 4)]
    plot_heatmap(axes[1], Z, x_seq, y_seq)
    plot_line(axes[2], W, seq)

    # generate data
    X = 2 * (np.random.rand(N) - 0.5)
    T = w.dot(np.vstack((np.ones(N), X))) + np.random.normal(0, sigma, N)

    # fit
    for n in range(N):
        x_n, t_n = X[n], T[n]
        Phi = np.array([1.0, x_n])

        # estimate parameters
        S_inv = np.linalg.inv(S)
        S = np.linalg.inv(S_inv + beta * Phi.reshape(-1, 1) * Phi) # eq(12)
        mu = S.dot(S_inv.dot(mu) + beta *  Phi * t_n) # eq(11)

        # plot
        Z = bivariate_normal(x_seq, y_seq, mu, S)
        W = np.random.multivariate_normal(mu, S, 6)
        axes = [fig.add_subplot(N + 1, 3, (n + 1) * 3 + j) for j in range(1, 4)]
        plot_data(axes[0], X[:n+1], T[:n+1])
        plot_heatmap(axes[1], Z, x_seq, y_seq)
        plot_line(axes[2], W, seq)

    plt.savefig('bayes.png')
    plt.clf()