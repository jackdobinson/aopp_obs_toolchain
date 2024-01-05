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
from algorithm.deconv.lucy_richardson import LucyRichardson
import algorithm.bad_pixels
import plot_helper
from plot_helper import figure_n_subplots, LimSymAroundValue, DiffClass, LimSymAroundCurrent
from plot_helper.plotters import PlotSet, Histogram, Image, VerticalLine, IterativeLineGraph, HorizontalLine
from plot_helper.base import AxisDataMapping


import test_data
import scientest.decorators
import scientest.cfg.settings

algo_basic_test_data = SimpleNamespace(
	n_iter = 10,
	test_obs = np.ones((7,7)),
	test_psf = np.ones((7,7))
)


def test_runs_for_basic_data():
	deconv = LucyRichardson(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf)
	
	assert result[2] == deconv.n_iter, f"Expect {deconv.n_iter} iterations of LucyRichardson, have {result[2]} instead."


def test_call_altered_instantiated_parameters():
	n_iter_overwrite=15
	assert n_iter_overwrite != algo_basic_test_data.n_iter, "Must have an overwrite value that is not the same as the original value"
	
	deconv = LucyRichardson(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf, n_iter=n_iter_overwrite)
	
	assert result[2] == n_iter_overwrite, f"Expect {n_iter_overwrite} iterations of LucyRichardson, have {result[2]} instead."
 


@scientest.decorators.skip(True)
def test_on_example_data(n_iter=200):
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	

	deconvolver = LucyRichardson(
		n_iter=n_iter,
		nudge_factor=0,#-1E-2,
		strength=1,
		offset_obs=True,
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
	output_fname = os.path.join(output_dir, f"{__name__}_test_lucy_richardson_on_example_data_output.fits")
	print(f'Outputting test result to {output_fname}')
	hdul_output.writeto(output_fname, overwrite=True)



@scientest.decorators.skip(True)
def test_on_example_data_with_plotting_hooks(n_iter=200):
	# get example data
	obs = FitsSpecifier(test_data.example_fits_file, 'DATA', (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	psf = FitsSpecifier(test_data.example_fits_psf_file, 0, (slice(229,230),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	

	deconvolver = LucyRichardson(
		n_iter=n_iter,
		nudge_factor=0,#-1E-2,
		strength=1,
		offset_obs=True,
	)
	
	fig, axes = figure_n_subplots(8)
	axes_iter = iter(axes)
	a7_2 = axes[7].twinx()
	a7_3 = axes[7].twinx()
	a7_4 = axes[7].twinx()
	a7_5 = axes[7].twinx()
	
	cmap = copy.copy(mpl.cm.get_cmap('bwr'))
	cmap.set_over('magenta')
	cmap.set_under('green')
	cmap.set_bad('black')
	mpl.cm.register_cmap(name='bwr_oob', cmap=cmap)
	#mpl.rcParams['image.cmap'] = 'user_cmap'
	
	diff_instance_1 = DiffClass()
	diff_instance_2 = DiffClass()
	
	plot_set = PlotSet(
		fig,
		'clean modified step={self.n_frames}',
		cadence=10,
		plots = [
			Histogram(
				'residual', 
				static_frame=False,
				axis_data_mappings = (
					AxisDataMapping('value','bins',limit_getter=plot_helper.lim), AxisDataMapping('count','_hist',limit_getter=plot_helper.LimRememberExtremes())
				),
				ax_funcs=[lambda ax: ax.set_yscale('log')],
			).attach(next(axes_iter), deconvolver, lambda x: x._residual),
			
			
			Image(
				'residual'
			).attach(next(axes_iter), deconvolver, lambda x: x._residual),
			
			Image(
				'obs_per_est'
			).attach(next(axes_iter), deconvolver, lambda x: x._obs_per_est),
			
			Image(
				'components'
			).attach(next(axes_iter), deconvolver, lambda x: x._components),
			
			Image(
		 		'blurred_est'
			).attach(next(axes_iter), deconvolver, lambda x: x._blurred_est),
			
			Image(
				'correction factors',
				show_limits_in_title = {2:True},
				static_frame=False,
				axis_data_mappings = (
					AxisDataMapping('x',None), 
					AxisDataMapping('y',None), 
					AxisDataMapping('brightness', '_z_data', LimSymAroundValue(1))
				),
				plt_kwargs={'cmap':'bwr_oob'}
			).attach(next(axes_iter), deconvolver, lambda x: x._cf),
			
			Histogram(
				'correction_factors', 
				static_frame=False,
				axis_data_mappings = (
					AxisDataMapping('value','bins',limit_getter=plot_helper.lim),
					AxisDataMapping('count','_hist',limit_getter=plot_helper.lim)
				),
			).attach(next(axes_iter), deconvolver, lambda x: x._cf),
			
			VerticalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red'}
			).attach(axes[6], deconvolver, lambda x: x.cf_limit),
			
			IterativeLineGraph(
				'metrics',
				datasource_name='max(|correction factors|)',
				axis_labels = (None, 'correction factor fabs value (blue)'),
				static_frame=False,
				plt_kwargs = {'alpha':0.3},
				#ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(next(axes_iter), deconvolver, lambda x: np.nanmax(np.fabs(x._cf))),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'linestyle':'--'}
			).attach(axes[7], deconvolver, lambda x: x.cf_limit),
			
			IterativeLineGraph(
				None,
				datasource_name='max(|residual|)',
				axis_labels = (None, 'residual fabs value (red)'),
				static_frame=False,
				plt_kwargs = {'alpha':0.3, 'color':'red', 'linestyle':'--'},
				#ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(a7_2, deconvolver, lambda x: np.nanmax(np.fabs(x._residual))),
			IterativeLineGraph(
				None,
				datasource_name='rms(residual))',
				axis_labels = (None, 'residual rms (green)'),
				static_frame=False,
				plt_kwargs = {'alpha':0.3, 'color':'green', 'linestyle':'--'},
				ax_funcs=[lambda ax: ax.spines.right.set_position(('axes', 1.12)), lambda ax: ax.set_yscale('log')]
			).attach(a7_3, deconvolver, lambda x: np.sqrt(np.nanmean(x._residual**2))),
			IterativeLineGraph(
				None,
				datasource_name='max(|diff(residual)|)',
				axis_labels = (None, 'max |residual diff| (purple)'),
				axis_data_mappings = (
					AxisDataMapping(
						'iteration',
						'_x',
						limit_getter=plot_helper.LimFixed(0)
					),
					AxisDataMapping(
						'value',
						'_y',
						limit_getter=plot_helper.LimSymAroundCurrent(1, plot_helper.LimRememberNExtremes(10))
					)
				),
				static_frame=False,
				plt_kwargs = {'alpha':0.3, 'color':'purple', 'linestyle':':'},
				ax_funcs=[lambda ax: ax.spines.right.set_position(('axes', 1.24))],# lambda ax: ax.set_yscale('log')]
			).attach(a7_4, deconvolver, lambda x: np.nanmax(np.fabs(diff_instance_1(x._residual)))),
			IterativeLineGraph(
				None,
				datasource_name='rms(|diff(residual)|)',
				axis_labels = (None, 'rms (residual diff) (brown)'),
				axis_data_mappings = (
					AxisDataMapping(
						'iteration',
						'_x',
						limit_getter=plot_helper.LimFixed(0)
					),
					AxisDataMapping(
						'value',
						'_y',
						limit_getter=plot_helper.LimSymAroundCurrent(1, plot_helper.LimRememberNExtremes(10))
					)
				),
				static_frame=False,
				plt_kwargs = {'alpha':0.3, 'color':'brown', 'linestyle':':'},
				ax_funcs=[lambda ax: ax.spines.right.set_position(('axes', 1.36))],# lambda ax: ax.set_yscale('log')]
			).attach(a7_5, deconvolver, lambda x: np.sqrt(np.nanmean(diff_instance_2(x._residual)**2))),
			
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
	output_fname = os.path.join(output_dir, f"{__name__}_test_lucy_richardson_on_example_data_output.fits")
	print(f'Outputting test result to {output_fname}')
	hdul_output.writeto(output_fname, overwrite=True)
