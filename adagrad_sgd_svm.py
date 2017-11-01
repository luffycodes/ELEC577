import numpy as np
import random as random
import matplotlib.pyplot as plt

sparse = True
m = 5000
n = 1000
A = np.zeros((m, n))
x_opt = np.zeros(n)
x_rand = np.ones(n)

# Code to generate A - tune sparse to generate different A types
for j in range(1, n + 1):
    if sparse:
        prob_sign = 1 / j
    else:
        prob_sign = 1
    for i in range(1, m + 1):
        p = np.random.binomial(1, p=prob_sign)
        if p == 1:
            sign = np.random.binomial(1, p=0.5)
            if sign == 0:
                A[i - 1][j - 1] = 1
            else:
                A[i - 1][j - 1] = -1

# Code to generate random X_opt
for j in range(1, n + 1):
    p = np.random.binomial(1, p=0.5)
    if p == 1:
        x_opt[j - 1] = 1
    else:
        x_opt[j - 1] = -1

# Code to generate random X
for j in range(1, n + 1):
    p = np.random.binomial(1, p=0.5)
    if p == 1:
        x_rand[j - 1] = 1
    else:
        x_rand[j - 1] = -1

x_rand_copy = np.copy(x_rand)

b = A.dot(x_opt)
# Code to generate random B
for i in range(1, m + 1):
    if b[i - 1] > 0:
        b[i - 1] = 1
    else:
        b[i - 1] = -1
    p = np.random.binomial(1, p=0.95)
    if p == 0:
        b[i - 1] = - 1 * b[i - 1]

fx = 0
for i in range(1, m + 1):
    fx = fx + np.maximum(0, 1 - b[i - 1] * np.dot(A[i - 1], x_opt))
fx = fx / m

# Adagrad
alpha = 1
delta = 1e-8
distance = []
H = np.zeros(n)
H_2 = np.zeros(n)
for i in range(1, 100000):
    i_rand = random.randint(0, m - 1)
    cond = 1 - b[i_rand] * np.dot(A[i_rand], x_rand)
    g = np.zeros(n)
    if cond > 0:
        g = -1 * b[i_rand] * A[i_rand]
        update = np.zeros(n)
        for j in range(0, n):
            H_2[j] = H_2[j] + g[j] * g[j]
            H[j] = np.sqrt(H_2[j] + delta)
            update[j] = (1 / H[j]) * g[j]
        x_rand = x_rand - alpha * update

    if i % 100 == 0:
        fx_k = 0
        for j in range(1, m + 1):
            fx_k = fx_k + np.maximum(0, 1 - b[j - 1] * np.dot(A[j - 1], x_rand))
        fx_k = fx_k / m
        distance.append(fx_k - fx)
        print("ada:step:", i, " & error:", fx_k - fx)

plt.plot(distance, label="Adagrad")

# SGD
alpha = 10
distance = []
x_rand = np.copy(x_rand_copy)

for i in range(1, 100000):
    i_rand = random.randint(0, m - 1)
    cond = 1 - b[i_rand] * np.dot(A[i_rand], x_rand)
    if cond > 0:
        x_rand = x_rand + alpha / np.sqrt(i + 1) * b[i_rand] * A[i_rand]

    if i % 100 == 0:
        fx_k = 0
        for j in range(1, m + 1):
            fx_k = fx_k + np.maximum(0, 1 - b[j - 1] * np.dot(A[j - 1], x_rand))
        fx_k = fx_k / m
        distance.append(fx_k - fx)
        print("sgd:step:", i, " & error:", fx_k - fx)

plt.plot(distance, label="SGD")

plt.xlabel("iterations")
plt.ylabel("log(norm(fx-fx_optimal))")

plt.legend()
plt.show()
