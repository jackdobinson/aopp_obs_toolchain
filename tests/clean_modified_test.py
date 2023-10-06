import os
from types import SimpleNamespace
import copy

import numpy as np
from astropy.io import fits
import matplotlib as mpl
mpl.use('TKagg')
import matplotlib.pyplot as plt



import astropy_helper as aph
import astropy_helper.fits.header
from astropy_helper.fits.specifier import FitsSpecifier
import numpy_helper as nph
import numpy_helper.axes
import numpy_helper.slice
import numpy_helper.array
from algorithm.deconv.clean_modified import CleanModified
import algorithm.bad_pixels
import plot_helper
from plot_helper import figure_n_subplots, LimSymAroundValue
from plot_helper.plotters import PlotSet, Histogram, Image, VerticalLine, IterativeLineGraph, HorizontalLine
from plot_helper.base import AxisDataMapping


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


@decorators.skip(False)
def test_clean_modified_on_example_data(n_iter=200):
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	

	deconvolver = CleanModified(n_iter=n_iter)

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

			obs_data_bp_mask = algorithm.bad_pixels.get_map(obs_data)
			obs_data = algorithm.bad_pixels.fix(obs_data, obs_data_bp_mask, 'simple')

			psf_data_bp_mask = algorithm.bad_pixels.get_map(psf_data)
			psf_data = algorithm.bad_pixels.fix(psf_data, psf_data_bp_mask, 'simple')

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


@decorators.skip(True)
def test_clean_modified_on_example_data_with_plotting_hooks(n_iter=200):
	
	
	print(f'{mpl.is_interactive()=}')
	
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	deconvolver = CleanModified(
		n_iter = n_iter,
		rms_frac_threshold=1E-2,
		fabs_frac_threshold=1E-2,
		threshold=-1
	)
	
	
	fig, axes = figure_n_subplots(8)
	axes_iter = iter(axes)
	a7_2 = axes[7].twinx()
	
	cmap = copy.copy(mpl.cm.get_cmap('bwr'))
	cmap.set_over('magenta')
	cmap.set_under('green')
	cmap.set_bad('black')
	mpl.cm.register_cmap(name='bwr_oob', cmap=cmap)
	#mpl.rcParams['image.cmap'] = 'user_cmap'
	
	plot_set = PlotSet(
		fig,
		'clean modified step={self.n_frames}',
		cadence=10,
		plots = [	
			Histogram(
				'residual', 
				static_frame=False,
				axis_data_mappings = (AxisDataMapping('value','bins',limit_getter=plot_helper.lim), AxisDataMapping('count','_hist',limit_getter=plot_helper.LimRememberExtremes()))
			).attach(next(axes_iter), deconvolver, lambda x: x._residual),
		 	
			VerticalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red'}
			).attach(axes[0], deconvolver, lambda x: x._pixel_threshold),
			
			Image(
		 		'residual'
		 	).attach(next(axes_iter), deconvolver, lambda x: x._residual),
			
			Image(
		 		'current cleaned'
			).attach(next(axes_iter), deconvolver, lambda x: x._current_cleaned),
			
			Image(
		 		'components'
			).attach(next(axes_iter), deconvolver, lambda x: x._components),
			
			Image(
		 		'selected pixels'
			).attach(next(axes_iter), deconvolver, lambda x: x._selected_px),
			
			Image(
		 		'pixel choice metric',
		 		axis_data_mappings = (AxisDataMapping('x',None), AxisDataMapping('y',None), AxisDataMapping('brightness', '_z_data', LimSymAroundValue(0))),
		 		plt_kwargs={'cmap':'bwr_oob'}
			).attach(next(axes_iter), deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			Histogram(
				'pixel choice metric', 
				static_frame=False,
			).attach(next(axes_iter), deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			IterativeLineGraph(
				'metrics',
				datasource_name='fabs',
				axis_labels = (None, 'fabs value (blue)'),
				static_frame=False,
				plt_kwargs = {},
				ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(next(axes_iter), deconvolver, lambda x: np.fabs(np.nanmax(x._residual))),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'linestyle':'--'}
			).attach(axes[7], deconvolver, lambda x: x._fabs_threshold),
			
			IterativeLineGraph(
				'metrics',
				datasource_name='rms',
				axis_labels = (None,'rms value (red)'),
				static_frame=False,
				plt_kwargs={'color':'red'},
				ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(a7_2, deconvolver, lambda x: np.sqrt(np.nansum(x._residual**2)/x._residual.size)),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red', 'linestyle':'--'}
			).attach(a7_2, deconvolver, lambda x: x._rms_threshold),
		]
	)
		 
	
	deconvolver.post_iter_hooks.append(lambda *a, **k: plot_set.update())
	plot_set.show()

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

			obs_data_bp_mask = algorithm.bad_pixels.get_map(obs_data)
			obs_data = algorithm.bad_pixels.fix(obs_data, obs_data_bp_mask, 'mean')

			psf_data_bp_mask = algorithm.bad_pixels.get_map(psf_data)
			psf_data = algorithm.bad_pixels.fix(psf_data, psf_data_bp_mask, 'mean')

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
	
	for p in plot_set.plots:
		assert p.n_updates == deconvolver._i, "plot updates should be called once for each frame"
	
	plt.close()


