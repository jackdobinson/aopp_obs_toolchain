
import numpy as np
import matplotlib.pyplot as plt

from observation_system_model import ObservationSystemModel

from optical_model.instrument.vlt import VLT_MUSE
from optical_model.atmosphere import KolmogorovTurbulence2, KolmogorovTurbulence
from optical_model.adaptive_optics import AdaptiveOpticsModel




def test_atmosphere_model():
	turbulence_model = KolmogorovTurbulence2(0.1, 2)
	f = np.linspace(-10,10,101)
	f_mag = np.abs(f)
	psd = turbulence_model.get_phase_psd(f_mag)

	plt.figure()
	plt.plot(f, psd)

	ao_model = AdaptiveOpticsModel(
		KolmogorovTurbulence,
		VLT_MUSE,
		shape=(101,101),
		supersample_factor=11
	)


	moffat_params = dict(alpha=1, beta=2, A=1, C=1)
	ao_otf_full = ao_model.get_otf_full(5E-7, 0.1, **moffat_params)

	print(f'{ao_otf_full=}')

	ao_psf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ao_otf_full)))

	plt.figure()
	plt.imshow(np.abs(ao_otf_full))

	plt.figure()
	plt.imshow(np.abs(ao_psf))

	plt.show()











if __name__=='__main__':
	test_atmosphere_model()


