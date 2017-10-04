import numpy as np
import matplotlib.pyplot as plt

n = 500
d = 50
A = np.random.randn(n, d)
x_optimal = np.random.randn(d)
b = np.dot(A, x_optimal)
stepsize = 1

niter = 1000
distance = []
xk = np.zeros(d)
for k in range(niter):
    randomA = A[np.random.randint(0, n)]
    randomA.shape = (d, 1)
    lhs = np.identity(d) - stepsize * np.dot(randomA, randomA.transpose())
    rhs = xk - x_optimal
    xk = np.dot(lhs, rhs) + stepsize * x_optimal
    distance.append(np.linalg.norm(xk - x_optimal))
    print(np.linalg.norm(xk - x_optimal))

plt.plot(np.log(distance))
plt.show()
