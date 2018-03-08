import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.image as mplimg
from numpy.fft import fftshift, fft2, ifftshift, ifft2


G = loadmat('G.mat')['G']
s = loadmat('spine.mat')['x']
S = fftshift(fft2(s))
S_s = np.abs(S)**2
# S_s =

G_cap = G.conj() / (np.abs(G)**2 + 2*10**5/ S_s)
Y = G_cap * S

y = ifft2(ifftshift(Y))
plt.imshow(np.abs(y), cmap=plt.get_cmap('gray'))
mplimg.imsave('spine_deblurred.png', np.abs(y), cmap=plt.get_cmap('gray'))
mplimg.imsave('spine.png', s, cmap=plt.get_cmap('gray'))
plt.show()