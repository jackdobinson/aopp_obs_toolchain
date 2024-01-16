

from __future__ import annotations

import sys, os
import dataclasses as dc
from typing import Any, TypeVar, TypeVarTuple, Generic

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
from gaussian_psf_model import GaussianPSFModel
from optimise_compat import PriorParam, PriorParamSet
from algorithm.deconv.clean_modified import CleanModified

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




def model_flattened_callable_factory(model):
	def model_flattened_callable(x,y,sigma, const,factor):
		return model(np.array([x,y]), np.array([sigma,sigma]), const)*factor
	return model_flattened_callable

def model_badness_of_fit_callable_factory(model_flattened_callable, data, err):
	def model_badness_of_fit_callable(*args, **kwargs):
		residual = model_flattened_callable(*args, **kwargs) - data
		return np.nansum((residual/err)**2)
	
	return model_badness_of_fit_callable


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


	for file in files:
		print(f'Operating on {file}')
		with PIL.Image.open(file) as image:
			data = np.array(image)

		source_labels, source_bounding_boxes, parameters = get_source_regions(data, name=file.split(os.sep)[-1])
		plot_source_regions(
			data, 
			source_labels, 
			source_bounding_boxes, 
			parameters, 
			output_file= example_data_loader.get_amateur_data_set_directory(0) / ('.'.join((file.split(os.sep)[-1]).split('.')[:-1])+'_source_regions.png')
		)
		
		print(f'{parameters=}')
		
		
		target_data = data[source_bounding_boxes[target_label-1].to_slices()].astype(float)
		# subtract background emission
		bg_region_slice = (s//10 for s in target_data.shape)
		target_data -= np.median(target_data[*bg_region_slice])
		
		psf_data = data[source_bounding_boxes[psf_label-1].to_slices()].astype(float)
		
		psf_data = psf_data_ops.normalise(psf_data)
		psf_model = GaussianPSFModel(psf_data.shape, float)
		
		flattened_psf_model_callable = model_flattened_callable_factory(psf_model)
		
		params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				True,
				psf_data.shape[0]//2
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				True,
				psf_data.shape[1]//2
			),
			PriorParam(
				'sigma',
				(0, np.sum([x**2 for x in psf_data.shape])),
				False,
				5
			),
			PriorParam(
				'const',
				(0, 1),
				False,
				0
			),
			PriorParam(
				'factor',
				(0, 2),
				False,
				1
			)
		)
		
		model_scipyCompat_callable, var_param_name_order, const_var_param_name_order = params.wrap_callable_for_scipy_parameter_order(flattened_psf_model_callable)
		
		test_result = model_scipyCompat_callable(tuple(params[p_name].const_value for p_name in var_param_name_order))
		if False:
			plt.imshow(test_result)
			plt.show()
		
		model_badness_of_fit_callable = model_badness_of_fit_callable_factory(model_scipyCompat_callable, psf_data, np.nanmax(psf_data)*1E-3)
		
		
		def callback(intermediate_result=None):
			print(f'{intermediate_result.x}')
			print(f'{intermediate_result.fun}')
		
		
		
		result = sp.optimize.minimize(
			model_badness_of_fit_callable,
			tuple(params[p_name].const_value for p_name in var_param_name_order),
			method=None,
			callback=callback
		)
		
		print(f'{result.success=}')
		print(f'{result.x=}')
		print(f'{result.message=}')
		
		fitted_psf = model_scipyCompat_callable(result.x)
		
		if False:
			f, ax = plt.subplots(2,2,squeeze=False)
			ax = ax.flatten()
		
			ax[0].imshow(psf_data)
			ax[1].imshow(fitted_psf)
			ax[2].imshow((psf_data - fitted_psf)**2)
			ax[3].imshow(np.log((psf_data - fitted_psf)**2))
			
			
			plt.show()
		
		
		deconvolver = CleanModified()
		deconvolver(
			target_data, 
			fitted_psf,
			n_iter=2000,
			threshold = 0.3 if '890' not in file else -1,
			rms_frac_threshold = 5E-3 if '890' not in file else 1E-2,
			fabs_frac_threshold = 5E-3 if '890' not in file else 1E-2
		)
		
		deconv_components = deconvolver.get_components()
		deconv_residual = deconvolver.get_residual()
		
		f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
		ax = ax.flatten()
		
		vmin, vmax = np.nanmin(target_data), np.nanmax(target_data)
	
		f.suptitle(file)
	
		ax[0].set_title(f'target ({vmin:0.2g}, {vmax:0.2g})')
		ax[0].imshow(target_data, vmin=vmin, vmax=vmax)
		
		ax[1].set_title(f'deconvolved target ({vmin:0.2g}, {vmax:0.2g})')
		ax[1].imshow(deconv_components, vmin=vmin, vmax=vmax)
		
		
		im2 = ax[2].imshow((deconv_residual))
		vmin, vmax = im2.get_clim()
		ax[2].set_title(f'residual ({vmin:0.2g}, {vmax:0.2g})')
		
		
		im3 = ax[3].imshow(np.log(np.abs(deconv_residual)))
		vmin, vmax = im3.get_clim()
		ax[3].set_title(f'log(residual) ({vmin:0.2g}, {vmax:0.2g})')
		
		for a in ax:
			a.xaxis.set_visible(False)
			a.yaxis.set_visible(False)
		
		
		#plt.show()
		plt.savefig(example_data_loader.get_amateur_data_set_directory(0) / ('.'.join((file.split(os.sep)[-1]).split('.')[:-1])+'deconv_test.png'))
		
		
		
		
		
		


