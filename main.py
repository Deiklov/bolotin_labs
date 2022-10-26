import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from math import cos, sin, pi
import numpy as np
from scipy.optimize import minimize


def main():
    r = 7
    y10 = 2
    y20 = 3
    N = 2 * pi
    count = 1000
    phi = np.linspace(0, N, count)
    y1Noise = np.random.normal(0, 0.2, count)
    y2Noise = np.random.normal(0, 0.2, count)

    y1 = np.roll(np.array([r * cos(t) + y10 for t in phi]), 70)
    y1 += y1Noise
    y2 = np.roll(np.array([r * sin(t) + y20 for t in phi]), 70)
    y2 += y2Noise
    plt.figure(figsize=(5, 5))
    plt.plot(y1, y2)
    #
    # plt.show()
    Y = np.array([y1, y2])
    A = np.array([[np.cos(phi), np.ones(len(phi)), np.zeros(len(phi))],
                  [np.sin(phi), np.zeros(len(phi)), np.ones(len(phi))]]).T

    # y10,y20,r0
    def J(x):
        b = np.zeros((2, count))
        for i in range(count):
            b[:, i] = A[i].T.dot(x).reshape(2, )
        data = np.sum(np.linalg.norm(Y - b, axis=0))
        return data

    x0 = np.array([0, 0, 0]).reshape(3, 1)
    res = minimize(J, [7, 2, 3], method='Nelder-Mead', tol=1e-6)
    print('r=', res.x[0], 'y10=', res.x[1], 'y20=', res.x[2])
    plt.plot(res.x[1], res.x[2], marker='o')
    plt.show()


if __name__ == '__main__':
    main()
