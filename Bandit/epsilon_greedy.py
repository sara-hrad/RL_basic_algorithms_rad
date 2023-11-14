# @sara-hrad.github.io
# Simple bandit algorithm in page 32 of the book " Introduction to RL"

import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, k):
        self.k = k          # action
        self.Q = 0          # action value estimate
        self.N = 0          # numbers that action is chosen

    # Random reward of action k
    def reward(self):
        return np.random.randn() + self.k

    # Update the action-value estimate
    def update(self, x):
        self.N += 1
        self.Q = self.Q + (1.0/self.N)*(x - self.Q)


def run_experiment(k, eps, N):
    actions = np.empty(len(k), dtype=object)
    for i in range(len(k)):
        actions[i] = Bandit(k[i])
    data = np.empty(N)
    for n_run in range(N):
        # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(k))
        else:
            j = np.argmax([a.Q for a in actions])
        x = actions[j].reward()
        actions[j].update(x)
        # for the plot
        data[n_run] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average)
    for n_k in range(len(k)):
        plt.plot(np.ones(N) * k[n_k])
    plt.xscale('log')
    plt.show()

    for a in actions:
        print(a.Q)
    print(data[-1])
    return cumulative_average


if __name__ == '__main__':
    k_actions = np.array([1.0, 2.0, 3.0, 4.0])
    c_1 = run_experiment(k_actions, 0.1, 100)
