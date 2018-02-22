import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

noise = np.random.laplace(scale=1, size=1024)
sines = []
n = 100

for i in range(1024):
    sines.append(np.sin(2 * np.pi * i * (n / 1024)))

mle = []
i = 0
s = 0.95
e = 1.05
gap = 0.001
for j in np.arange(s, e, gap):
    mle.append(0)
    for k in range(1024):
        mle[i] = mle[i] + np.sign(sines[k] + noise[k] - j * sines[k]) * sines[k]
    print(str(j) + " " + str(mle[i]))
    i = i + 1

plt.plot(np.arange(s, e, gap), mle)
plt.show()
