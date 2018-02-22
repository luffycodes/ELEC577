import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

n = 100
sines = []
noise_std = 1
for i in range(1024):
    sines.append(np.cos(2 * np.pi * i * (n / 1024) - np.pi/4))

phase_estimate = []
amplitude_estimate = []
i = 0
s = 1
e = 1000
gap = 1
for trial in np.arange(s, e, gap):
    noise = np.random.normal(scale=noise_std, size=1024)

    phase_estimate.append(0)
    num_phase_estimate = 0
    den_phase_estimate = 0

    amplitude_estimate.append(0)
    num_amplitude_estimate = 0

    for k in range(1024):
        num_phase_estimate = num_phase_estimate + (sines[k] + noise[k]) * np.sin(2 * np.pi * k * (n / 1024))
        den_phase_estimate = den_phase_estimate + (sines[k] + noise[k]) * np.cos(2 * np.pi * k * (n / 1024))
    phase_estimate[i] = np.arctan(num_phase_estimate / den_phase_estimate)

    for k in range(1024):
        num_amplitude_estimate = num_amplitude_estimate + 2 * (sines[k] + noise[k]) * (np.cos(2 * np.pi * k * (n / 1024) - phase_estimate[i]))
    amplitude_estimate[i] = num_amplitude_estimate/1024

    print(str(trial) + " " + str(phase_estimate[i]) + " " + str(amplitude_estimate[i]))
    i = i + 1

print(np.mean(phase_estimate))
print(np.std(phase_estimate))
print(np.mean(amplitude_estimate))
print(np.std(amplitude_estimate))
