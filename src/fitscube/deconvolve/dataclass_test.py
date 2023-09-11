#!/usr/bin/env python3

import numba
import numpy as np
import matplotlib.pyplot as plt

@numba.jit(nopython=True, parallel=True)
def fft_cooley_turkey(a : np.ndarray, b : np.ndarray):
	if a.size==1:
		b[0] = a[0]
	else:
		fft_cooley_turkey(a[::2], b[:a.size//2])
		fft_cooley_turkey(a[1::2], b[a.size//2:])
		for i in range(a.size//2):
			q = np.exp(-2*np.pi*1j)*b[i+a.size//2]
			b[i+a.size//2] = b[i] - q
			b[i] += q
	return

@numba.jit
def fft(a : np.ndarray):
	n = a.size
	n2 = int(2**np.ceil(np.log2(n)))
	a_extended = np.zeros((n2,))
	a_extended[:n] = a
	b = np.zeros_like(a_extended, dtype=np.complex128)
	fft_cooley_turkey(a_extended, b)
	return(b[:n])

@numba.jit
def npfft(a: np.ndarray):
	with numba.objmode(out='complex128[:]'):
		out = np.fft.fft(a)
	return(out)

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(10*x)
#y_fft = np.zeros_like(y, dtype=complex)
y_fft = fft(y)
print(f'{np.max(y_fft)=}')
plt.plot(np.abs(y_fft))
plt.plot(np.real(y_fft))
plt.plot(np.imag(y_fft))
plt.show()
