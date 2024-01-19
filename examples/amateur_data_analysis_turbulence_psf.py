

from __future__ import annotations

import sys, os
import os.path
import dataclasses as dc
from typing import Any, TypeVar, TypeVarTuple, Generic
import datetime as dt
from functools import partial

import numpy as np
import scipy as sp
import scipy.ndimage
import skimage as ski
import skimage.measure
import skimage.morphology

import scipy as sp
import scipy.optimize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches

import PIL
import PIL.Image

from geometry.bounding_box import BoundingBox
from optimise_compat import PriorParam, PriorParamSet
from algorithm.deconv.clean_modified import CleanModified

from optics.turbulence_model import phase_psd_von_karman_turbulence
from turbulence_psf_model import TurbulencePSFModel, SimpleTelescope, CCDSensor


import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
S = Generic[IntVar]
N = TypeVar('N',bound=int)
LabelArray = np.ndarray[S,int]

def select_labels_from_props(props, predicate = lambda prop: True):
	selected_labels = []
	for prop in props:
		if predicate(prop):
			selected_labels.append(prop.label)
	return selected_labels
		

def props_keep(props, labels_to_keep):
	props_to_keep = []
	for prop in props:
		if prop.label in labels_to_keep:
			props_to_keep.append(prop)
	return props_to_keep

def props_relabel(props, old_labels, new_labels):
	for prop in props:
		if prop.label in old_labels:
			prop.label = new_labels[old_labels.index(prop.label)]
	return props

def labels_keep(labels, labels_to_keep, background=0):
	for i in range(1, np.max(labels)+1):
		if i not in labels_to_keep:
			labels[labels==i] = background
	return labels

def labels_relabel(labels, old_labels, new_labels):
	for i, lbl in enumerate(old_labels):
		labels[labels==lbl] = new_labels[i]
	return labels

def labels_and_props_relabel(labels, props, old_labels, new_labels):
	for prop in props:
		lbl = prop.label
		if lbl in old_labels:
			new_lbl = new_labels[old_labels.index(lbl)]

			labels[labels==lbl] = new_lbl
			prop.label = new_lbl
	return(labels, props)


def get_default_arg_dict(func, locals_dict):
	n = func.__code__.co_argcount
	ndefault = len(func.__defaults__)
	default_vars = func.__code__.co_varnames[n-ndefault:n]
	return dict((k,locals_dict[k]) for k in default_vars)



def get_source_regions(
		data : np.ndarray[S[N], T], 
		name : str = '',
		signal_noise_ratio : float = 5, 
		smallest_obj_area_px : int = 5,
		bbox_inflation_factor : float = 1.1,
		bbox_inflation_px : int = 30,
	) -> tuple[LabelArray[S[N]], tuple[BoundingBox[N,T],...], dict[str,Any]]:
	"""
	Return regions, bounding boxes, and parameters used to find emitting regions in `data`
	"""

	data_object_mask = data >= (np.median(data)*signal_noise_ratio)

	
	data_object_mask = ski.morphology.opening(
		data_object_mask,
		ski.morphology.disk(3, decomposition='sequence')
	)


	labels = ski.measure.label(data_object_mask)
	
	props = ski.measure.regionprops(labels)

	# only keep regions larger than `smallest_obj_area_px`
	large_region_labels = select_labels_from_props(props, lambda prop: prop.num_pixels >= smallest_obj_area_px)
	props = props_keep(props, large_region_labels)
	labels = labels_keep(labels, large_region_labels)

	props = sorted(props, key= lambda x: x.num_pixels, reverse=True)
	
	labels = labels_keep(labels, [p.label for p in props])

	labels, props = labels_and_props_relabel(labels, props, [p.label for p in props], tuple(range(1,len(props)+1)))
	
	bboxes = tuple(BoundingBox.from_min_max_tuple(prop.bbox).inflate(bbox_inflation_factor, bbox_inflation_px) for prop in props)

	return labels, bboxes, get_default_arg_dict(get_source_regions, locals())


def plot_source_regions(
		data : np.ndarray[S[N],T],
		labels : LabelArray[S[N]],
		bboxes : tuple[BoundingBox[N,T],...],
		params : dict[str,Any],
		output_file
	):
	noise_estimate = np.median(data[labels==0])
	nplots = len(source_bounding_boxes)+2
	nr, nc = int(nplots // np.sqrt(nplots)), int(np.ceil(nplots/(nplots//np.sqrt(nplots))))
	
	f,ax = plt.subplots(nr, nc, squeeze=False, figsize=(12,8))
	f.suptitle(f'Regions: S/N >= {params["signal_noise_ratio"]}; n_pixels >= {params["smallest_obj_area_px"]}')

	ax=ax.flatten()
	for a in ax:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	ax[0].set_title(f'{params["name"]}')
	ax[0].imshow(data)

	ax[1].set_title(f'Labels')
	ax[1].imshow(labels)

	for i, bbox in enumerate(bboxes):
		rect = mpl.patches.Rectangle(*bbox.to_mpl_rect(),edgecolor='r', facecolor='none', lw=1)
		text = f'region {i}'

		region_data = data[bbox.to_slices()]
		conservative_sig_noise = np.max(region_data)/np.median(region_data)
		region_sig_noise = np.max(region_data)/noise_estimate


		ax[0].add_patch(rect)
		ax[0].text(*bbox.mpl_min_corner, text, color='r', horizontalalignment='left', verticalalignment='top')
		ax[i+2].imshow(region_data)
		ax[i+2].set_title(f'{text}: S/N={region_sig_noise:0.2g} {"x".join(str(x) for x in bbox.extent)}')

	plt.savefig(output_file)




def model_badness_of_fit_callable_factory(model_flattened_callable, data, err):
	def model_badness_of_fit_callable(*args, **kwargs):
		residual = model_flattened_callable(*args, **kwargs) - data
		result = np.nansum((residual/err)**2)
		return np.log(result)
	
	return model_badness_of_fit_callable

def save_array_as_tif(data_in, path, nan='zero', ninf='zero', pinf='zero', scale='false'):
	data = np.array(data_in) # copy input data
	type = np.uint16
	
	nan_mask = np.isnan(data)
	ninf_mask = np.isneginf(data)
	pinf_mask = np.isposinf(data)
	
	ignore_mask = nan_mask | ninf_mask | pinf_mask
	
	if scale:
		data = (data - np.min(data[~ignore_mask]))/np.max(data[~ignore_mask]) * np.iinfo(type).max
	
	for mode, mask in ((nan,nan_mask),(ninf,ninf_mask),(pinf,pinf_mask)):
		match mode:
			case 'zero':
				data[mask] = 0
			case 'max':
				data[mask] = np.iinfo(type).max
			case 'min':
				data[mask] = np.iinfo(type).min
	PIL.Image.fromarray(((deconv_components-np.nanmin(deconv_components))/np.nanmax(deconv_components))*np.iinfo(type).max, mode='I;16')
	
	
	PIL.Image.fromarray(data.astype(type), mode='I;16').save(path, 'tiff')
	
		

if __name__=='__main__':

	import example_data_loader
	import psf_data_ops

	if len(sys.argv) <= 1:
		files = example_data_loader.get_amateur_data_set(0)
	else:
		files = sys.argv[1:]

	# note, labels are 1-indexed (0 is background)
	target_label = 1
	psf_label = 3


	mpl.rc('image', cmap='gray')


	for file in files:
		_lgr.debug(f'Operating on {file}')
		with PIL.Image.open(file) as image:
			data = np.array(image)

		output_dir = example_data_loader.get_amateur_data_set_output_directory(0)
		fname = os.path.splitext(os.path.split(file)[1])[0]

		source_labels, source_bounding_boxes, parameters = get_source_regions(data, name=file.split(os.sep)[-1])
		_lgr.debug(f'{len(source_bounding_boxes)=}')
		
		plot_source_regions(
			data, 
			source_labels, 
			source_bounding_boxes, 
			parameters, 
			output_file= output_dir / (fname+'_source_regions.png')
		)
		
		
		_lgr.debug(f'{parameters=}')
		_lgr.debug(f'{len(source_bounding_boxes)=}')
		
		
		# Get PSF data
		psf_data = data[source_bounding_boxes[psf_label-1].to_slices()].astype(float)
		
		psf_data = psf_data_ops.normalise(psf_data)
		
		# Define and fit PSF model
		psf_model = TurbulencePSFModel(
			SimpleTelescope(
				8, 
				200, 
				CCDSensor.from_shape_and_pixel_size(psf_data.shape, 2.5E-6)
			),
			phase_psd_von_karman_turbulence
		)
		
		
		params = PriorParamSet(
			PriorParam(
				'r0',
				(0, 1),
				True,
				0.1
			),
			PriorParam(
				'turb_ndim',
				(0, 3),
				False,
				1.5
			),
			PriorParam(
				'L0',
				(0, 50),
				False,
				8
			)
		)
		
		wavelength = 750E-9
		
		flattened_psf_model_callable = lambda r0, turb_ndim, L0: psf_model(wavelength, r0, turb_ndim, L0)
		model_scipyCompat_callable, var_param_name_order, const_var_param_name_order = params.wrap_callable_for_scipy_parameter_order(flattened_psf_model_callable)
		
		print(f'{tuple(params[p_name].const_value for p_name in var_param_name_order)=}')
		test_result = model_scipyCompat_callable(tuple(params[p_name].const_value for p_name in var_param_name_order))
		print(f'{test_result.shape=}')
		print(f'{test_result=}')
		if False:
			plt.close()
			f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			
			ax[0].imshow(test_result)
			
			ax[1].plot(psf_model.telescope_model.f_axis(wavelength)[0], test_result[:,test_result.shape[1]//2])
			ax[1].set_yscale('log')
			
			ax[2].plot(psf_model.telescope_model.f_axis(wavelength)[1], test_result[test_result.shape[0]//2,:])
			ax[2].set_yscale('log')
			
			plt.show()
		
		model_badness_of_fit_callable = model_badness_of_fit_callable_factory(model_scipyCompat_callable, psf_data, np.nanmax(psf_data)*1E-3)
		
		# callback for sp.optminize.minimise to track progress
		def callback(intermediate_result=None):
			print(f'{intermediate_result.x=}')
			print(f'{intermediate_result.fun=}')
		
		# Do PSF fitting
		result = sp.optimize.minimize(
			model_badness_of_fit_callable,
			tuple(params[p_name].const_value for p_name in var_param_name_order),
			bounds = tuple(params[p_name].domain for p_name in var_param_name_order),
			method=None,
			callback=callback
		)
		
		print(f'{result.success=}')
		print(f'{result.x=}')
		print(f'{result.message=}')
		
		fitted_psf = model_scipyCompat_callable(result.x)
		
		if True:
			psf_residual = psf_data - fitted_psf
		
			f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			
			f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")}\n'+fname + f' PSF region={psf_label}')
		
			im0 = ax[0].imshow(psf_data)
			vmin, vmax = im0.get_clim()
			ax[0].set_title(f'Original PSF [{vmin:0.2g}, {vmax:0.2g}]')
			
			im1 = ax[1].imshow(fitted_psf)
			vmin, vmax = im1.get_clim()
			ax[1].set_title(f'Fitted PSF [{vmin:0.2g}, {vmax:0.2g}]')
			
			im2 = ax[2].imshow(psf_residual)
			vmin, vmax = im2.get_clim()
			ax[2].set_title(f'Residual [{vmin:0.2g}, {vmax:0.2g}]')
			
			im3 = ax[3].imshow(np.log(np.abs(psf_residual)))
			vmin, vmax = im3.get_clim()
			ax[3].set_title(f'Log(|Residual|) [{vmin:0.2g}, {vmax:0.2g}]')
			
			for a in ax:
				a.xaxis.set_visible(False)
				a.yaxis.set_visible(False)
			
			plt.show()
			
			#plt.savefig(output_dir / (fname+'_psf_plot.png'))
		continue # DEBUGGING
		
		# PSF is fitted at this point
		
		for psf_type in ('fitted','original'):
			match psf_type:
				case 'fitted':
					psf_for_deconv = fitted_psf
				case 'original':
					psf_for_deconv = psf_data
				case _:
					raise NotImplementedError
			
			# Get and deconvolve target data
			for region_label in range(1,len(source_bounding_boxes)+1):
				
				output_file_fmt = "{fname}_region_{region_label}_psf_{psf_type}_test_deconv_{tag}.{ext}"
				
			
				target_data = data[source_bounding_boxes[region_label-1].to_slices()].astype(float)
				# subtract background emission
				bg_region_slice = (s//10 for s in target_data.shape)
				target_data -= np.median(target_data[*bg_region_slice])
				
				n_iter = 10000#5000
				threshold = 0.1 #0.3
				loop_gain = 0.2 #0.02
				max_stat_increase = 1E-3
				rms_frac_threshold = 1E-3#3E-3
				fabs_frac_threshold = 1E-3#3E-3
				
				if '890' in fname:
					threshold = 0.1#-1
					loop_gain=0.2
					#rms_frac_threshold = 5E-2
					#fabs_frac_threshold = 5E-2
				
				elif region_label in (2,3):
					rms_frac_threshold = 2E-2
					fabs_frac_threshold = 2E-2
					
				if psf_type == 'original':
					rms_frac_threshold /= 2
					fabs_frac_threshold /= 2
				
				deconvolver = CleanModified()
				deconvolver(
					target_data, 
					psf_for_deconv,
					n_iter=n_iter,
					threshold = threshold,
					loop_gain = loop_gain,
					max_stat_increase=max_stat_increase,
					rms_frac_threshold = rms_frac_threshold,
					fabs_frac_threshold = fabs_frac_threshold
				)
				
				n_iters = deconvolver.get_iters()
				iter_stat_names = deconvolver._iter_stat_names
				iter_stats = deconvolver._iter_stat_record
				
				deconv_components = deconvolver.get_components()
				deconv_residual = deconvolver.get_residual()
				
				# Save the data
				save_array_as_tif(deconv_components, output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_type=psf_type, tag='components', ext='tif'))
				save_array_as_tif(deconv_residual, output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_type=psf_type, tag='residual', ext='tif'))
				
				# Plot data
				f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
				ax = ax.flatten()
				
				f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")}\n'+fname+f'\n{n_iters=}')
			
				im0 = ax[0].imshow(target_data)
				vmin, vmax = im0.get_clim()
				ax[0].set_title(f'target ({vmin:0.2g}, {vmax:0.2g})')
			
				if ('890' in fname 
						or ('750' in fname and '1957' in fname and region_label == 2)
					):
					sorted_components = np.sort(deconv_components, axis=None)
					fmin, fmax = int(0.005*sorted_components.size), int(0.995*sorted_components.size)
					im1 = ax[1].imshow(deconv_components, vmin=sorted_components[fmin], vmax=sorted_components[fmax])
				else:
					im1 = ax[1].imshow(deconv_components)
				vmin, vmax = im1.get_clim()
				ax[1].set_title(f'deconvolved target ({vmin:0.2g}, {vmax:0.2g})')
				
				
				im2 = ax[2].imshow(deconv_residual)
				vmin, vmax = im2.get_clim()
				ax[2].set_title(f'residual ({vmin:0.2g}, {vmax:0.2g})')
				
				
				im3 = ax[3].imshow(np.log(np.abs(deconv_residual)))
				vmin, vmax = im3.get_clim()
				ax[3].set_title(f'log(|residual|) ({vmin:0.2g}, {vmax:0.2g})')
				
				for a in ax:
					a.xaxis.set_visible(False)
					a.yaxis.set_visible(False)
				
				plt.savefig(output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_type=psf_type, tag=f'plot', ext='png'))
				
				# Plot iteration statistics
				f, ax = plt.subplots(1,3,squeeze=False,figsize=(18,9))
				ax = ax.flatten()
				
				f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")}\n'+fname+f'\n{n_iters=} {fabs_frac_threshold=} {rms_frac_threshold=} {threshold=} {loop_gain=}')
				
				x = range(n_iters)
				j = 0
				for i, stat_name in enumerate(iter_stat_names):
					if stat_name == 'UNUSED': continue
					
					ax[j].plot(x, iter_stats[:n_iters,i])
					ax[j].set_title(stat_name)
					ax[j].set_yscale('log')
					j+=1
				
				plt.savefig(output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_type=psf_type, tag=f'plot_deconv_stats', ext='png'))
		
		
		
		
		
		


