import os
from types import SimpleNamespace
import numpy as np

from astropy.io import fits
import astropy_helper as aph
import astropy_helper.fits.header
from astropy_helper.fits.specifier import FitsSpecifier
import numpy_helper as nph
import numpy_helper.axes
import numpy_helper.slice
import numpy_helper.array
from algorithm.deconv.clean_modified import CleanModified
import algorithm.bad_pixels.simple

import test_data
import decorators

algo_basic_test_data = SimpleNamespace(
	n_iter = 10,
	test_obs = np.ones((7,7)),
	test_psf = np.ones((7,7))
)

def test_clean_modified_runs_for_basic_data():
	deconv = CleanModified(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf)
	
	assert result[2] == deconv.n_iter, f"Expect {deconv.n_iter} iterations of CleanModified, have {result[2]} instead."


def test_clean_modified_call_altered_instantiated_parameters():
	n_iter_overwrite=15
	assert n_iter_overwrite != algo_basic_test_data.n_iter, "Must have an overwrite value that is not the same as the original value"
	
	deconv = CleanModified(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf, n_iter=n_iter_overwrite)
	
	assert result[2] == n_iter_overwrite, f"Expect {n_iter_overwrite} iterations of CleanModified, have {result[2]} instead."


@decorators.skip(True)
def test_clean_modified_on_example_data():
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	

	deconvolver = CleanModified()

	with fits.open(obs.path) as obs_hdul, fits.open(psf.path) as psf_hdul:
		deconv_components_raw = np.full_like(obs_hdul[obs.ext].data, fill_value=np.nan)
		deconv_residual_raw = np.full_like(obs_hdul[obs.ext].data, fill_value=np.nan)

		with (	nph.axes.to_end(obs_hdul[obs.ext].data, obs.axes['CELESTIAL']) as obs_data,
				nph.axes.to_end(psf_hdul[psf.ext].data, psf.axes['CELESTIAL']) as psf_data,
				nph.axes.to_end(deconv_components_raw, obs.axes['CELESTIAL']) as deconv_components,
				nph.axes.to_end(deconv_residual_raw, obs.axes['CELESTIAL']) as deconv_residual,
			):

			psf_data = nph.array.ensure_odd_shape(psf_data)
			psf_data_offset = nph.array.get_center_offset_brightest_pixel(psf_data)
			psf_data = nph.array.apply_offset(psf_data, psf_data_offset)

			obs_data_bp_mask = algorithm.bad_pixels.simple.get_map(obs_data)
			obs_data = algorithm.bad_pixels.simple.fix(obs_data, obs_data_bp_mask)

			psf_data_bp_mask = algorithm.bad_pixels.simple.get_map(psf_data)
			psf_data = algorithm.bad_pixels.simple.fix(psf_data, psf_data_bp_mask)

			#print(f'{obs_data.shape=} {psf_data.shape=}')
			#print(f'{obs.slices=} {psf.slices=}')

			for k, (i,j) in enumerate(
					zip(nph.slice.iter_indices(
							obs_data, 
							obs.slices, 
							group=(x for x in range(len(obs.axes['CELESTIAL'])-1,obs_data.ndim))
						), 
						nph.slice.iter_indices(
							psf_data, 
							psf.slices, 
							group=(x for x in range(len(psf.axes['CELESTIAL'])-1,psf_data.ndim))
						)
					)
				):
				print(f'{k=}')
				print(f'{i=}')
				print(f'{obs_data[i].shape=}')
				deconvolver(obs_data[i], psf_data[j])
				deconv_components[i] = deconvolver.get_components()
				deconv_residual[i] = deconvolver.get_residual()

		print('Finished deconvolution, copying and updating header data')
		hdr = obs_hdul[obs.ext].header
		hdr.update(aph.fits.header.DictReader(deconvolver.get_parameters()))
	
	print('Constructing output HDUs')
	hdu_components = fits.PrimaryHDU(
		header = hdr,
		data = deconv_components_raw
	)
	hdu_residual = fits.ImageHDU(
		header=hdr,
		data=deconv_residual_raw,
		name='RESIDUAL'
	)
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual
	])

	output_dir = os.path.join(test_data.test_dir, 'output')
	os.makedirs(output_dir, exist_ok=True)
	output_fname = os.path.join(output_dir, f"{__name__}_test_clean_modified_on_example_data_output.fits")
	print(f'Outputting test result to {output_fname}')
	hdul_output.writeto(output_fname, overwrite=True)


@decorators.skip(False)
def test_clean_modified_on_example_data_with_plotting_hooks():
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	
	class Histogram:
		def __init__(self, ax, nbins, data : np.ndarray | None = None):
			self.ax = ax
			self.nbins = nbins
			self.n_updates = 0
			self._hist = None
			self._bins = None
			self._lines = self.ax.step([],[])
			self.set_data(data)
		
		def set_data(self, data : np.ndarray | None = None):
			print(f'{data.shape if data is not None else None=}')
			self.data = data
			if self.data is not None:
				self._hist, self._bins = np.histogram(self.data, bins=self.nbins)
				print(f'{self._hist.shape=}')
			
		def __call__(self):
			"""
			Update the axes
			"""
			self.n_updates += 1
			if self.data is not None:
				self._lines[0].set_data(self._bins[1:], self._hist)
				self.ax.set_xlim(np.min(self._bins), np.max(self._bins))
				self.ax.set_ylim(np.min(self._hist), np.max(self._hist))
		
	fig = plt.gcf()
	ax = plt.gca()

	plots = {
		'histogram' : Histogram(ax, 100)
	}
	
	def update_plots(self, obs, psf):
		plots['histogram'].set_data(self._residual)
		plots['histogram']()
		plt.pause(0.001)
		


	deconvolver = CleanModified(
		n_iter = 10,
		rms_frac_threshold=1E-1,
		fabs_frac_threshold=1E-1,
		pre_init_hook = lambda self, obs, psf: None,
		post_init_hook = lambda *a, **k: print('POST_INIT'),
		pre_iter_hook = lambda *a, **k: print('PRE_ITER'),
		post_iter_hook = update_plots,
		final_hook = lambda *a, **k: plt.close(fig),
	)

	with fits.open(obs.path) as obs_hdul, fits.open(psf.path) as psf_hdul:
		deconv_components_raw = np.full_like(obs_hdul[obs.ext].data, fill_value=np.nan)
		deconv_residual_raw = np.full_like(obs_hdul[obs.ext].data, fill_value=np.nan)

		with (	nph.axes.to_end(obs_hdul[obs.ext].data, obs.axes['CELESTIAL']) as obs_data,
				nph.axes.to_end(psf_hdul[psf.ext].data, psf.axes['CELESTIAL']) as psf_data,
				nph.axes.to_end(deconv_components_raw, obs.axes['CELESTIAL']) as deconv_components,
				nph.axes.to_end(deconv_residual_raw, obs.axes['CELESTIAL']) as deconv_residual,
			):

			psf_data = nph.array.ensure_odd_shape(psf_data)
			psf_data_offset = nph.array.get_center_offset_brightest_pixel(psf_data)
			psf_data = nph.array.apply_offset(psf_data, psf_data_offset)

			obs_data_bp_mask = algorithm.bad_pixels.simple.get_map(obs_data)
			obs_data = algorithm.bad_pixels.simple.fix(obs_data, obs_data_bp_mask)

			psf_data_bp_mask = algorithm.bad_pixels.simple.get_map(psf_data)
			psf_data = algorithm.bad_pixels.simple.fix(psf_data, psf_data_bp_mask)

			#print(f'{obs_data.shape=} {psf_data.shape=}')
			#print(f'{obs.slices=} {psf.slices=}')

			for k, (i,j) in enumerate(
					zip(nph.slice.iter_indices(
							obs_data, 
							obs.slices, 
							group=(x for x in range(len(obs.axes['CELESTIAL'])-1,obs_data.ndim))
						), 
						nph.slice.iter_indices(
							psf_data, 
							psf.slices, 
							group=(x for x in range(len(psf.axes['CELESTIAL'])-1,psf_data.ndim))
						)
					)
				):
				print(f'{k=}')
				print(f'{i=}')
				print(f'{obs_data[i].shape=}')
				deconvolver(obs_data[i], psf_data[j])
				deconv_components[i] = deconvolver.get_components()
				deconv_residual[i] = deconvolver.get_residual()

		print('Finished deconvolution, copying and updating header data')
		hdr = obs_hdul[obs.ext].header
		hdr.update(aph.fits.header.DictReader(deconvolver.get_parameters()))
	
	print('Constructing output HDUs')
	hdu_components = fits.PrimaryHDU(
		header = hdr,
		data = deconv_components_raw
	)
	hdu_residual = fits.ImageHDU(
		header=hdr,
		data=deconv_residual_raw,
		name='RESIDUAL'
	)
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual
	])

	output_dir = os.path.join(test_data.test_dir, 'output')
	os.makedirs(output_dir, exist_ok=True)
	output_fname = os.path.join(output_dir, f"{__name__}_test_clean_modified_on_example_data_output.fits")
	print(f'Outputting test result to {output_fname}')
	hdul_output.writeto(output_fname, overwrite=True)
	
	assert plots['histogram'].n_updates == deconvolver._i, "histogram should be called once for each frame"


