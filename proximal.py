import matplotlib.pyplot as plt
import numpy as np
import random as random

lambda_param = 0.01


def getProx(x1):
    prox_x = np.zeros(len(x1))
    for i in range(len(x1)):
        if x1[i] > lambda_param:
            prox_x[i] = x1[i] - lambda_param
        elif x1[i] < -1 * lambda_param:
            prox_x[i] = x1[i] + lambda_param
        else:
            prox_x[i] = 0
    return prox_x


# Generating random data
A = np.random.randn(2000, 1000)
e = [random.normalvariate(0, 0.1) for i in range(2000)]
x = np.zeros(1000)
positions = np.random.choice(np.arange(1000), 100, replace=False)
x[positions] = np.random.normal(0, 1, 100)
y = np.dot(A, x) + e
x_k = np.random.randn(1000)

# Subgradient descent
ata = np.dot(A.transpose(), A)
aty = np.dot(A.transpose(), y)
c = 0.0001
distance = []
print(np.shape(ata), np.shape(aty))
for i in np.arange(1, 1000, 1):
    a = [lambda_param if x_ki > 0 else -1 * lambda_param for x_ki in x_k]
    subg = np.dot(ata, x_k) - aty + a
    x_k = x_k - (c / i) * subg
    distance.append(np.linalg.norm(x_k - x))

# plt.plot(np.log(distance), label="subg")

# ISTA
alpha = 0.0001
distance = []
x_k = np.random.randn(1000)
for i in np.arange(1, 1000, 1):
    df = np.dot(ata, x_k) - aty
    x_k = x_k - alpha * df
    x_k = getProx(x_k)
    distance.append(np.linalg.norm(x_k - x))

plt.plot(np.log(distance), label="ISTA")

# ISTA
alpha = 0.0001
gamma = -0.4
distance = []
x_k = np.random.randn(1000)
x_k_prev = getProx(np.random.randn(1000))
for i in np.arange(1, 1000, 1):
    df = np.dot(ata, x_k) - aty
    x_k = x_k - alpha * df
    prox = getProx(x_k)
    x_k = (1 - gamma) * prox + gamma * x_k_prev
    x_k_prev = prox
    distance.append(np.linalg.norm(x_k - x))

plt.plot(np.log(distance), label="FISTA")

plt.legend()
plt.show()
