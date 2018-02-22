import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

n = 100
sines = []
for i in range(1024):
    sines.append(np.cos(2 * np.pi * i * (n / 1024) - np.pi/4))

phase_estimate = []
i = 0
s = 1
e = 1000
gap = 1
for trial in np.arange(s, e, gap):
    noise = np.random.normal(scale=1, size=1024)

    phase_estimate.append(0)
    for k in range(1024):
        num = (sines[k] + noise[k]) * np.sin(2 * np.pi * k * (n / 1024))
        den = (sines[k] + noise[k]) * np.cos(2 * np.pi * k * (n / 1024))
        phase_estimate[i] = phase_estimate[i] + num/den

    phase_estimate[i] = np.arctan(phase_estimate[i])
    print(str(trial) + " " + str(phase_estimate[i]))
    i = i + 1
