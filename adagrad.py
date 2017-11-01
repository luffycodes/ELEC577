import numpy as np
import random as random
import matplotlib.pyplot as plt

sparse = True
m = 5000
n = 1000
A = np.zeros((m, n))
x_opt = np.zeros(n)
x_rand = np.zeros(n)

# Code to generate A - tune sparse to generate different A types
for j in range(1, n + 1):
    if sparse:
        prob_sign = 1 / j
    else:
        prob_sign = 1
    for iterCount in range(1, m + 1):
        p = np.random.binomial(1, p=prob_sign)
        if p == 1:
            sign = np.random.binomial(1, p=0.5)
            if sign == 0:
                A[iterCount - 1][j - 1] = 1
            else:
                A[iterCount - 1][j - 1] = -1

# Code to generate random X
for j in range(1, n + 1):
    p = np.random.binomial(1, p=0.5)
    if p == 1:
        x_opt[j - 1] = 1
    else:
        x_opt[j - 1] = -1

b = A.dot(x_opt)
# Code to generate random B
for iterCount in range(1, m + 1):
    p = np.random.binomial(1, p=0.95)
    if p == 0:
        b[iterCount - 1] = - 1 * b[iterCount - 1]

# SGD
alpha = 0.001
distance = []
for iterCount in range(1, 200000):
    i = random.randint(0, m-1)
    cond = 1 - b[i] * np.dot(A[i], x_rand)
    if cond > 1:
        x_rand = x_rand + alpha/np.sqrt(iterCount + 1) * b[i] * A[i]
    distance.append(np.linalg.norm(x_rand - x_opt))

plt.plot(np.log(distance), label="SGD")
plt.xlabel("iterations")
plt.ylabel("log(norm(x-x_optimal))")

plt.legend()
plt.show()
