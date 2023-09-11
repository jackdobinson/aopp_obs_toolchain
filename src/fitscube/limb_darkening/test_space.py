#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.color
import scipy as sp
import scipy.signal
import scipy.fft

plt.ion()
#%%
b = skimage.color.rgb2gray(skimage.data.astronaut())
plt.imshow(b)
plt.show()
#%%
a = np.ones(b.shape)
dd_a = np.zeros_like(a)
dd_a[a.shape[0]//2+1, a.shape[1]//2+1] = 1

#center_slice = (slice(252,260), slice(252,260))
center_slice = (slice(12,500), slice(12,500))
xx,yy = np.mgrid[range(a.shape[0]), range(a.shape[1])]
r = 350
center_slice = (xx - a.shape[0]//2)**2 + (yy - a.shape[1]//2)**2 < r**2
a[center_slice] = 0
plt.imshow(a)
plt.show()
#%%
dd_a_fft = np.fft.fft2(dd_a)
low_pass = np.fft.ifft2(dd_a_fft*a).real
plt.imshow(np.log(np.abs(low_pass)))
plt.show()
#%%
#b_conv_a_ifft = sp.signal.convolve2d(b, low_pass, mode='same')
b_conv_a = sp.fft.ifft2(sp.fft.fft2(b)*(a))
plt.imshow(b_conv_a.real)
#plt.imshow(sp.fft.fft2(b).real)
plt.show()