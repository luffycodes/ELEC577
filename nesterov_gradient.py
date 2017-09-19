# minimize f(x) = 1/2 x^T Q x - b^T x using gradient descent
# example to show that what happens if the stepsize is to large/just right/too small

import matplotlib.pyplot as plt
from numpy import *

niter = 100
stepsize = 1 / 8
distance = []
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


def get_gradient(xk1):
    return getQ_BasedOnX(xk1) + getB_BasedOnX(xk1)


for alpha in arange(0.0, 1.0, 0.01):
    for beta in arange(0.0, 1.0, 0.01):
        xk = 2
        xk_last = 2
        for k in range(niter):
            diff_last_two = beta * (xk - xk_last)
            xk_last = xk
            xk = xk - alpha * get_gradient(xk + diff_last_two) + diff_last_two
            distance = log(linalg.norm(xk - xopt))
            if distance < best_closeness:
                best_closeness = distance
                best_alpha = alpha
                best_beta = beta

print(best_closeness, best_beta, best_alpha)