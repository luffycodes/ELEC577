# minimize f(x) = 1/2 x^T Q x - b^T x using gradient descent
# example to show that what happens if the stepsize is to large/just right/too small

import matplotlib.pyplot as plt
from numpy import *

niter = 100
stepsize = 1 / 8
plot_f = []
xopt = 0
best_alpha = 1
best_beta = 1
best_closeness = 1


def getQ_BasedOnX(xk1):
    if xk1 > 0 or xk1 < -3:
        return 16
    else:
        return 1


def getB_BasedOnX(xk1):
    if xk1 < -3:
        return 45
    else:
        return 0


def getF(xk1):
    if xk1 < -3:
        return 8 * xk1 * xk1 + 45 * xk1 + 67.5
    elif xk1 < 0:
        return 0.5 * xk1 * xk1
    else:
        return 8 * xk1 * xk1


def get_gradient(xk1):
    return getQ_BasedOnX(xk1) * xk1 + getB_BasedOnX(xk1)


def compute_convergence(alpha1, beta1, plot=False):
    global best_closeness, best_alpha, best_beta, plot_f
    xk = 2
    xk_last = 2
    for k in range(niter):
        if plot:
            plot_f.append(getF(xk))

        diff_last_two = beta1 * (xk - xk_last)
        xk_last = xk
        xk = xk - alpha1 * get_gradient(xk + diff_last_two) + diff_last_two
        distance = linalg.norm(xk - xopt)
        if distance < 0.0001:
            if k < 10 and not plot:
                print("solutions: ", best_closeness, best_alpha, best_beta, k)
                break

        if distance < best_closeness:
            best_closeness = distance
            best_alpha = alpha1
            best_beta = beta1


for alpha in arange(0.0, 1.0, 0.01):
    for beta in arange(0.0, 1.0, 0.01):
        compute_convergence(alpha, beta)

print(best_closeness, best_alpha, best_beta)

compute_convergence(1 / 16, 3/5, True)
plt.plot(log(plot_f))
plt.xlabel("iterations")
plt.ylabel("log(f(x))")
plt.show()
