
# minimize f(x) = 1/2 x^T Q x - b^T x using gradient descent
# example to show that what happens if the stepsize is to large/just right/too small

import matplotlib.pyplot as plt
from numpy import *

# generate a random problem instance 
n = 5
A = random.randn(n,n)
Q = dot(A.T,A)
b = random.randn(n)

xopt = dot(linalg.inv(Q),b)

eigenvalues = linalg.eigvals(Q)
M = max(eigenvalues)
m = min(eigenvalues)


# gradient descent

niter = 100
stepsize = 2/(M+m)
distance = []
xk = zeros(n) #random.randn(n) # random initializer
for k in range(niter):
	xk = xk - stepsize*(dot(Q,xk) - b)
	distance.append( linalg.norm(xk-xopt) )

plt.plot(log(distance))
plt.show()
