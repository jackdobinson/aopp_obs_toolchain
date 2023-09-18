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


def test_clean_modified_on_example_data():
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(200,201),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(200,201),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	

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

			# TODO: Have another look at how this works,
			#       want interface to work more naturally, right now it fails.
			for k, (i,j) in enumerate(zip(nph.slice.get_indices(obs_data, obs.slices, group=(0,)), nph.slice.get_indices(psf_data, psf.slices, group=(0,)))):
				#print(f'{k=}')
				#print(f'{i=}')
				#print(f'{obs_data[i].shape=}')
				deconvolver(obs_data[i], psf_data[j])
				deconv_components[i,...] = deconvolver.get_components()
				deconv_residual[i,...] = deconvolver.get_residual()

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
	os.mkdir(output_dir)
	output_fname = os.path.join(output_dir, f"{__name__}_test_clean_modified_on_example_data_output.fits")
	print(f'Outputting test result to {output_fname}')
	hdul_output.writeto(output_fname, overwrite=True)


	



