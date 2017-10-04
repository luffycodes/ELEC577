import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

n = 500
d = 50
A = np.random.randn(n, d)
A = normalize(A, axis=1, norm='l2')

x_optimal = np.random.randn(d)
b = np.dot(A, x_optimal)
for stepsize in np.arange(0.2, 2, 0.4):
    niter = 1000
    distance = []
    xk = np.random.randn(d)
    for k in range(niter):
        randomA = A[np.random.randint(0, n)]
        randomA = randomA.reshape(d, 1)

        lhs = np.identity(d) - stepsize * np.dot(randomA, randomA.transpose())
        rhs = xk - x_optimal
        xk = np.dot(lhs, rhs) + x_optimal
        distance.append(np.linalg.norm(xk - x_optimal))
        print(np.linalg.norm(xk - x_optimal))

    plt.plot(np.log(distance), label=("step:", "%.2f" % round(stepsize, 2)))

plt.legend()
plt.xlabel("iterations")
plt.ylabel("error")
plt.show()
