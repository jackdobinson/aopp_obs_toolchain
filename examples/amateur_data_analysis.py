

from __future__ import annotations

import sys, os
import os.path
import dataclasses as dc
from typing import Any, TypeVar, TypeVarTuple, Generic, ParamSpec, Callable, Protocol
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







import psf_data_ops


import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
S = Generic[IntVar]
N = TypeVar('N',bound=int)
M = TypeVar('M',bound=int)
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
		text = f'region {i+1}'

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

def save_array_as_tif(data_in, path, nan='zero', ninf='zero', pinf='zero', neg='zero', scale=False, type=np.uint16):
	
	_lgr.debug(f'{path=}')
	data = np.array(data_in, dtype=data_in.dtype) # copy input data
	
	nan_mask = np.isnan(data)
	ninf_mask = np.isneginf(data)
	pinf_mask = np.isposinf(data)
	
	ignore_mask = nan_mask | ninf_mask | pinf_mask
	
	# see: https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
	tiff_info = {
		259:1 # No compression
	}
	
	if scale:
		data_min, data_max = np.min(data[~ignore_mask]), np.max(data[~ignore_mask])
		_lgr.debug(f'{data_min=} {data_max=}')
		data = (data - data_min)/(data_max - data_min) * np.iinfo(type).max
		_lgr.debug(f'{np.min(data)=} {np.max(data)=}')
	
	for mode, mask in ((nan,nan_mask),(ninf,ninf_mask),(pinf,pinf_mask),(neg,data <0)):
		match mode:
			case 'zero':
				data[mask] = 0
			case 'max':
				data[mask] = np.iinfo(type).max
			case 'min':
				data[mask] = np.iinfo(type).min	
	
	match type:
		case np.float32:
			image_mode = 'F'
			tiff_info[339] = 3
		case np.int32:
			image_mode = 'I'
			tiff_info[339] = 2
		case np.uint16:
			image_mode = 'I;16'
			tiff_info[339] = 1
		case _:
			raise NotImplementedError(f'Unknown PIL image mode for numpy type {type}')
	
	PIL.Image.fromarray(data.astype(type), mode=image_mode).save(path, 'tiff', tiffinfo=tiff_info)
	
def print_tiff_tags(path):
	# print out tags of image
	with PIL.Image.open(path) as im:
		print(f'{path=}')
		print(f'{im.mode=}')
		exif = im.getexif()
		for k,v in exif.items():
			tag = PIL.TiffTags.lookup(k)
			print(f'{tag} {v}')
			




# Need to have some way to describe how to communicate with the dependency injection mechanism,
# this isn't the best, but it's what I could come up with. 
# `ParamsAndPsfModelDependencyInjector` is a base class that defines the interface
# I could probably do this with protocols, but I think it would be harder to communicate my intent

Ts_PSF_Data_Array_Shape = [N,M] # We don't know the shape of the PSF data, but it must be a have two integer values
P_ArgumentsLikePriorParamSet = ParamSpec('ArgumentsLikePriorParamSet') # we require that this argument set is compatible with parameters specified in a 'PriorParamSet' instance
T_PSF_Data_NumpyArray = np.ndarray[Ts_PSF_Data_Array_Shape, T] # PSF data is a numpy array of some specified shape and type 'T'
T_PSF_Model_Flattened_Callable = Callable[P_ArgumentsLikePriorParamSet, T_PSF_Data_NumpyArray] # We want the callable we are given to accept parameters in a way that is compatible with 'PriorParamSet', and return a numpy array that is like our PSF Data
T_Fitted_Variable_Parameters = dict[str,Any] # Fitted varaibles from `psf_data_ops.fit_to_data(...)` are returned as a dictionary
T_Constant_Parameters = dict[str,Any] # Constant paramters to `psf_data_ops.fit_to_data(...)` are returned as a dictionary
P_PSF_Result_Postprocessor_Arguments = [PriorParamSet, T_PSF_Model_Flattened_Callable, T_Fitted_Variable_Parameters, T_Constant_Parameters] # If we want to postprocess the fitted PSF result, we will need to know the PriorParamSet used, the callable used, the fitted variables, and the constant paramters resulting from the fit.
T_PSF_Result_Postprocessor_Callable = Callable[P_PSF_Result_Postprocessor_Arguments, T_PSF_Data_NumpyArray] # If we preprocess the fitted PSF, we must return something that is compatible with the PSF data.

class ParamsAndPsfModelDependencyInjector:
	def __init__(self, psf_data : PSF_Data_NumpyArray):
		self.psf_data = psf_data
		self._psf_model = NotImplemented
		self._params = NotImplemented # PriorParamSet()
	
	def get_psf_model_name(self):
		return self._psf_model.__class__.__name__

	def get_parameters(self) -> PriorParamSet:
		return self._params
	
	def get_psf_model_flattened_callable(self) -> PSF_Model_Flattened_Callable : 
		NotImplemented
	
	def get_psf_result_postprocessor(self) -> None | T_PSF_Result_Postprocessor_Callable : 
		NotImplemented


class RadialPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from radial_psf_model import RadialPSFModel
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				False,
				psf_data.shape[0]//2
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				False,
				psf_data.shape[1]//2
			),
			PriorParam(
				'nbins',
				(0, np.inf),
				True,
				50
			)
		)
		
		self._psf_model = RadialPSFModelDependencyInjector.RadialPSFModel(
			psf_data
		)
		
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		return self._psf_model
	
	def get_psf_result_postprocessor(self): 
		def psf_result_postprocessor(params, psf_model_flattened_callable, fitted_vars, consts):
			params.apply_to_callable(
				psf_model_flattened_callable, 
				fitted_vars,
				consts
			)
			return psf_model_flattened_callable.centered_result
			
		return psf_result_postprocessor


class GaussianPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from gaussian_psf_model import GaussianPSFModel
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
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
		
		self._psf_model = GaussianPSFModelDependencyInjector.GaussianPSFModel(psf_data.shape, float)
	
	
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		def psf_model_flattened_callable(x, y, sigma, const, factor):
			return self._psf_model(np.array([x,y]), np.array([sigma,sigma]), const)*factor
		return psf_model_flattened_callable
	
	def get_psf_result_postprocessor(self): 
		return None


class TurbulencePSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from turbulence_psf_model import TurbulencePSFModel, SimpleTelescope, CCDSensor
	from optics.turbulence_model import phase_psd_von_karman_turbulence as turbulence_model
	
	def __init__(self, psf_data):
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'wavelength',
				(0, np.inf),
				True,
				750E-9
			),
			PriorParam(
				'r0',
				(0, 1),
				True,
				0.1
			),
			PriorParam(
				'turbulence_ndim',
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
		
		self._psf_model = TurbulencePSFModelDependencyInjector.TurbulencePSFModel(
			TurbulencePSFModelDependencyInjector.SimpleTelescope(
				8, 
				200, 
				TurbulencePSFModelDependencyInjector.CCDSensor.from_shape_and_pixel_size(psf_data.shape, 2.5E-6)
			),
			TurbulencePSFModelDependencyInjector.turbulence_model
		)
		
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		return self._psf_model
	
	def get_psf_result_postprocessor(self): 
		return None




if __name__=='__main__':

	import example_data_loader
	import psf_data_ops

	# Set some defaults for matplotlib
	mpl.rc('image', cmap='gray', origin='upper')

	# Choose dataset to operate upon
	data_set_index = 0

	if len(sys.argv) <= 1:
		files = example_data_loader.get_amateur_data_set(data_set_index)
	else:
		files = sys.argv[1:]

	# We use scikit-image to label regions of emission from largest to smallest (0 is background, 1 is largest emission source, etc.)
	# The actual labelling is done later, this is just the configuration
	# note, labels are 1-indexed (0 is background)
	target_label = 1
	psf_label = -1 # last label is PSF as we want the smallest object.

	# set PSF model type
	psf_model_type = ('turbulence', 'gaussian', 'radial')[2]


	# Use the correct dependency injector for the psf model type
	match psf_model_type:
		case 'turbulence':
			params_and_psf_model_dependency_injector : ParamsAndPsfModelDependencyInjector = TurbulencePSFModelDependencyInjector
		case 'gaussian':
			params_and_psf_model_dependency_injector : ParamsAndPsfModelDependencyInjector = GaussianPSFModelDependencyInjector
		case 'radial':
			params_and_psf_model_dependency_injector : ParamsAndPsfModelDependencyInjector = RadialPSFModelDependencyInjector
		case _:
			raise NotImplementedError(f'Unknown option {psf_model_type=}')
	
	# Loop over the files in our input dataset
	for file in files:
		_lgr.debug(f'Operating on {file}')
		with PIL.Image.open(file) as image:
			image = image.convert(mode='F')
			data = np.array(image)

		output_dir = example_data_loader.get_amateur_data_set_output_directory(data_set_index)
		output_dir.mkdir(parents=True, exist_ok=True)
		fname = os.path.splitext(os.path.split(file)[1])[0]

		source_labels, source_bounding_boxes, parameters = get_source_regions(data, name=file.split(os.sep)[-1])
		_lgr.debug(f'{len(source_bounding_boxes)=}')
		
		assert psf_label != 0, "Labelled PSF region cannot be the background"
		psf_label = psf_label if psf_label >=0 else np.max(source_labels) + 1 + psf_label
		psf_bbox_index = psf_label-1
		
		
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
		psf_data = psf_data_ops.normalise(data[source_bounding_boxes[psf_bbox_index].to_slices()].astype(float))
		
		# Get psf model, PriorParamSet, and any postprocessing function from dependency injection
		di = params_and_psf_model_dependency_injector(psf_data)
		psf_model_name = di.get_psf_model_name()
		params = di.get_parameters()
		psf_model = di.get_psf_model_flattened_callable()
		psf_result_postprocess = di.get_psf_result_postprocessor()
		
		# fit PSF model
		fitted_psf, fitted_vars, consts = psf_data_ops.fit_to_data(
			params, 
			psf_model, 
			psf_data, 
			np.ones_like(psf_data)*np.nanmax(psf_data)*1E-3, 
			psf_data_ops.scipy_fitting_function_factory(sp.optimize.minimize),
			partial(psf_data_ops.objective_function_factory, mode='minimise'),
			plot_mode=None
		)
		_lgr.debug(f'{fitted_vars=}')
		_lgr.debug(f'{consts=}')
		
		# Do any postprocessing if we need to
		if psf_result_postprocess is not None:
			fitted_psf = psf_result_postprocess(params, psf_model, fitted_vars, consts)
		
		# Plot the PSF data so we know what we are using for later
		if True:
			psf_residual = psf_data - fitted_psf
		
			f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			
			f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")} {psf_model_name}\n'+fname + f' PSF region={psf_label}')
		
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
			
			plt.savefig(output_dir / (fname+f'_psf_name_{psf_model_name}_plot.png'))

		
		# PSF is fitted at this point
		# Now we do the deconvolution
		
		# Want to compare fitted PSF with the observed psf, so loop over the two different cases
		for psf_type in ('fitted','original'):
			match psf_type:
				case 'fitted':
					psf_name = psf_model_name
					psf_for_deconv = fitted_psf
				case 'original':
					psf_name = 'ObservedPSF'
					psf_for_deconv = psf_data
				case _:
					raise NotImplementedError
			
			psf_for_deconv /= np.nansum(psf_for_deconv)
			
			# Loop over the regions that define the target data
			for region_label in [x for x in range(1,len(source_bounding_boxes)+1)]+[0]:
				
				output_file_fmt = "{fname}_region_{region_label}_psf_{psf_name}_test_deconv_{tag}.{ext}"
				
				save_array_as_tif(
					psf_for_deconv, 
					output_dir / "{fname}_region_{region_label}_psf_{psf_name}.{ext}".format(fname=fname, region_label=region_label, psf_name=psf_name, ext='tif'),
					scale=True
				)
				
				np.save(
					output_dir / "{fname}_region_{region_label}_psf_{psf_name}.{ext}".format(fname=fname, region_label=region_label, psf_name=psf_name, ext='npy'),
					psf_for_deconv
				)
				
				if region_label == 0:
					target_data = data.astype(float)
				else:
					target_data = data[source_bounding_boxes[region_label-1].to_slices()].astype(float)
				
				# subtract background emission
				bg_region_slice = (s//10 for s in target_data.shape)
				target_data -= np.median(target_data[*bg_region_slice])
				
				# Set deconvolution paramters
				n_iter = 1000#5000
				threshold = 0.1#0.483#0.1 #0.3
				loop_gain = 0.2 #0.02
				rms_frac_threshold = 1E-3#1E-3
				fabs_frac_threshold = 1E-3#1E-3
				min_frac_stat_delta = loop_gain*5E-2#1E-2
				
				# Peform deconvolution
				deconvolver = CleanModified(give_best_result=False)
				deconvolver(
					target_data, 
					psf_for_deconv,
					n_iter=n_iter,
					threshold = threshold,
					loop_gain = loop_gain,
					rms_frac_threshold = rms_frac_threshold,
					fabs_frac_threshold = fabs_frac_threshold,
					min_frac_stat_delta = min_frac_stat_delta
				)
				
				# Save the parameters used on this run
				param_file = output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_name=psf_name, tag='parameters', ext='txt')
				with open(param_file, 'w') as f:
					for k,v in deconvolver.get_parameters().items():
						f.write(f'{k}\n\t{v}\n\n')				
				
				n_iters = deconvolver.get_iters()
				iter_stat_names = deconvolver._iter_stat_names
				iter_stats = deconvolver._iter_stat_record
				
				deconv_components = deconvolver.get_components()
				deconv_residual = deconvolver.get_residual()
				
				# Save the results of the deconvolution
				save_array_as_tif(
					deconv_components, 
					output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_name=psf_name, tag='components', ext='tif')
				)
				save_array_as_tif(
					deconv_residual, 
					output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_name=psf_name, tag='residual', ext='tif'), 
					scale=True
				)
				
				# Plot results of the deconvolution
				f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
				ax = ax.flatten()
				
				f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")} {psf_name}\n'+fname+f'\n{n_iters=}')
			
				im0 = ax[0].imshow(target_data)
				vmin, vmax = im0.get_clim()
				ax[0].set_title(f'target \n({vmin:0.2g}, {vmax:0.2g}) sum {np.nansum(target_data):0.4g}')
			
				# For some input data, we want to use 99% of the range as speckles can break things.
				if ('890' in fname 
						or ('750' in fname and '1957' in fname and region_label == 2)
					):
					sorted_components = np.sort(deconv_components, axis=None)
					fmin, fmax = int(0.005*sorted_components.size), int(0.995*sorted_components.size)
					im1 = ax[1].imshow(deconv_components, vmin=sorted_components[fmin], vmax=sorted_components[fmax])
				else:
					im1 = ax[1].imshow(deconv_components)
				vmin, vmax = im1.get_clim()
				ax[1].set_title(f'deconvolved target\n ({vmin:0.2g}, {vmax:0.2g}) sum {np.nansum(deconv_components):0.4g}')
				
				
				im2 = ax[2].imshow(deconv_residual)
				vmin, vmax = im2.get_clim()
				ax[2].set_title(f'residual ({vmin:0.2g}, {vmax:0.2g})  sum {np.nansum(deconv_residual):0.4g}')
				
				
				im3 = ax[3].imshow(np.log(np.abs(deconv_residual)))
				vmin, vmax = im3.get_clim()
				ax[3].set_title(f'log(|residual|) ({vmin:0.2g}, {vmax:0.2g})')
				
				for a in ax:
					a.xaxis.set_visible(False)
					a.yaxis.set_visible(False)
				
				plt.savefig(output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_name=psf_name, tag=f'plot', ext='png'))
				
				# Plot iteration statistics
				f, ax = plt.subplots(1,1,squeeze=False,figsize=(18,9))
				ax = ax.flatten()
				
				ax = [ax[0], ax[0].twinx(), ax[0].twinx()]
				ax[2].spines.right.set_position(('axes',1.05))
				
				f.suptitle(f'{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f%z")} {psf_name}\n'+fname+f'\n{n_iters=} {fabs_frac_threshold=} {rms_frac_threshold=} {threshold=} {loop_gain=}')
				
				x = range(n_iters)
				p_handles = []
				for i, stat_name in enumerate(iter_stat_names):
					p, = ax[i].plot(x, iter_stats[:n_iters,i], alpha=0.5, color=f"C{i}", label=stat_name)
					p_handles.append(p)
					if i==0:
						ax[i].set(xlabel='Iteration')
					ax[i].set(ylabel=stat_name)
					ax[i].set_yscale('log')
					ax[i].yaxis.label.set_color(p.get_color())
					ax[i].tick_params(axis='y', colors=p.get_color())
				
				ax[0].legend(handles=p_handles)
				
				plt.savefig(output_dir / output_file_fmt.format(fname=fname, region_label=region_label, psf_name=psf_name, tag=f'plot_deconv_stats', ext='png'))
		
		
		
		


