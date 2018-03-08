import cmath
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.fftpack as fft

fft_freq_seq = np.fft.fftfreq(1024, 1/1024)

H1f = [1/(1+0.8 * cmath.exp(-2*np.pi*(fft_freq_seq[i])*1j)) for i in range(1024)]
H2f = [1/(1-0.6 * cmath.exp(-2*np.pi*(fft_freq_seq[i])*1j)) for i in range(1024)]
W = np.random.randn(1024).tolist()

N1l = np.real(np.fft.ifft(np.fft.fft(np.array(W))*np.array(H1f)))
N2l = np.real(np.fft.ifft(np.fft.fft(np.array(W))*np.array(H2f)))

W = np.zeros(10)
pred = []
noise_cancelled = []
mu = 0.005
signal_without_noise = []
signal_with_noise = []
for i in range(1014):
    N2l_hat = 0
    for j in range(10):
        N2l_hat = N2l_hat + N1l[i+10-j] * W[j]
    pred.append(N2l_hat)
    for j in range(10):
        W[j] = W[j] + 2 * mu * (N2l[i+10] - N2l_hat) * N1l[i+10-j]
    noise_cancelled.append(N2l[i+10] - N2l_hat)
    signal_with_noise.append(np.sin(2 * np.pi * 0.25 * (i+10)) + N2l[i+10])
    signal_without_noise.append(np.sin(2 * np.pi * 0.25 * (i + 10)) + N2l[i + 10] - N2l_hat)

    print(N2l[i+10] - N2l_hat)

plt.plot(signal_with_noise, label='noisy signal')
plt.plot(signal_without_noise, label='noise cancelled signal')
# plt.plot(noise_cancelled, alpha=0.4, label='real - pred')
plt.xlabel("mu = 0.005")
plt.legend()
plt.savefig('noiseCancel.png')
plt.show()