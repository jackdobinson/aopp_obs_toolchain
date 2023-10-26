import numpy as np
import matplotlib.pyplot as plt

import decorators

decorators.skip(True)
def test_fft_conjugate_variable_scaling():
	n = 101*20
	x_min, x_max = 0, 1*20
	f_sin = (1,3,5)
	phase_sin = (0, (1/3)*2*np.pi, (2/3)*2*np.pi)
	amp_sin = (1,1,1)

	amp_noise = 0.1

	x = np.linspace(x_min, x_max, n)

	a = amp_noise*np.random.random(n)
	for f, phase, amp in zip(f_sin, phase_sin, amp_sin):
		a += amp*np.sin((2*np.pi*f)*(x - x_min) + phase)


	k = np.fft.fftshift(np.fft.fftfreq(n, (x_max-x_min)/n))
	A = np.fft.fftshift(np.fft.fft(a))

	plt.subplot(121).plot(x, a)
	plt.subplot(122).plot(k,A)
	plt.show()









if __name__=='__main__':
	test_fft_conjugate_variable_scaling()
