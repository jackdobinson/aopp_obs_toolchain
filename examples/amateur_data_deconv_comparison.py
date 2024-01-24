
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import PIL
import scipy as sp
import scipy.ndimage
import scipy.signal

import example_data_loader
import psf_data_ops

from amateur_data_analysis_radial_psf import get_source_regions

import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def load_image_as_numpy_array(fpath):
	with PIL.Image.open(fpath) as image:
		array = np.array(image).astype(float)
	return array




"""
	ymin, ymax = np.inf, -np.inf
	for a in ax:
		ylo, yhi = a.get_ylim()
		if ylo < ymin: ymin = ylo
		if yhi > ymax: ymax = yhi
	for a in ax:
		a.set_ylim(ymin, ymax)
	"""
def set_same_limits(
		axes_set, 
		limit_getter = lambda ax: ax.get_ylim(), 
		limit_setter = lambda ax,a,b: ax.set_ylim(a,b)
	):
	min, max = np.inf, -np.inf
	for ax in axes_set:
		a, b = limit_getter(ax)
		if a < min: min = a
		if b > max: max = b
	for ax in axes_set:
		limit_setter(ax, min, max)


if __name__=='__main__':
	data_dir = example_data_loader.get_amateur_data_set_directory(0)
	
	obs_file = data_dir / "2024-01-11-1917_1-Jupiter_750nm.tif"
	deconv_file = data_dir / "output" / "2024-01-11-1917_1-Jupiter_750nm_region_0_psf_original_test_deconv_components.tif"
	comp_deconv_files = (
		data_dir / "comparison_deconv" / "enhanced_a 2024-01-11-1917_1-Jupiter_750nm.tif",
		data_dir / "comparison_deconv" / "enhanced_b 2024-01-11-1917_1-Jupiter_750nm.tif",
	)
	
	obs_data = load_image_as_numpy_array(obs_file)
	deconv_data = load_image_as_numpy_array(deconv_file)
	comp_deconv_dataset = [load_image_as_numpy_array(fpath) for fpath in comp_deconv_files]
	
	
	deconv_data = sp.ndimage.median_filter(deconv_data, 3)
	for i in range(len(comp_deconv_dataset)):
		comp_deconv_dataset[i] = sp.ndimage.median_filter(comp_deconv_dataset[i], 3)
	
	
	psf_label = 3
	source_labels, source_bounding_boxes, parameters = get_source_regions(obs_data, name=str(obs_file).split(os.sep)[-1])
	psf_data = obs_data[source_bounding_boxes[psf_label-1].to_slices()].astype(float)
	psf_data = psf_data_ops.normalise(psf_data)
	
	
	region = lambda : mpl.patches.Rectangle(
		(472,233), 
		120, 
		1,#30, 
		angle=22, 
		facecolor='none', 
		edgecolor='red', 
		lw=1
	)
	
	# Plot data and region
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('data with region')
	
	ax[0].imshow(obs_data)
	ax[0].set_title(f'observation sum {np.nansum(obs_data)}')
	
	ax[1].imshow(deconv_data)
	ax[1].set_title(f'MC deconv sum {np.nansum(deconv_data)}')
	
	
	for i in range(len(comp_deconv_dataset)):
		ax[2+i].imshow(comp_deconv_dataset[i])
		ax[2+i].set_title(f'comparison deconv {i} sum {np.nansum(comp_deconv_dataset[i])}')
	
	for _a in ax:
		_a.add_patch(region())
	
	plt.show()
	
	
	# Plot residuals before convolving with psf
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('residuals before convolving with psf')
	
	ax[0].imshow(obs_data-obs_data)
	ax[0].set_title(f'obs_data - obs_data')
	
	ax[1].imshow(deconv_data-obs_data)
	ax[1].set_title(f'MC deconv - obs_data')
	
	
	for i in range(len(comp_deconv_dataset)):
		ax[2+i].imshow(comp_deconv_dataset[i]-obs_data)
		ax[2+i].set_title(f'comparison deconv {i} - obs_data')
	
	set_same_limits(ax, lambda x: x.get_images()[0].get_clim(), lambda x, a, b: x.get_images()[0].set_clim(a,b))
	
	plt.show()
	
	
	
	# Plot residuals after convolving with psf
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('residuals after convolving with psf')
	
	ax[0].imshow(obs_data-obs_data)
	ax[0].set_title(f'obs_data - obs_data')
	
	ax[1].imshow(sp.signal.fftconvolve(deconv_data, psf_data, mode='same')-obs_data)
	ax[1].set_title(f'MC deconv - obs_data')
	
	
	for i in range(len(comp_deconv_dataset)):
		ax[2+i].imshow(sp.signal.fftconvolve(comp_deconv_dataset[i], psf_data, mode='same')-obs_data)
		ax[2+i].set_title(f'comparison deconv {i} - obs_data')
	
	set_same_limits(ax, lambda x: x.get_images()[0].get_clim(), lambda x, a, b: x.get_images()[0].set_clim(a,b))
	
	plt.show()
	
	
	
	
	a_region = region()
	
	transform = a_region.get_patch_transform()
	
	
	mpl_indices = np.flip(np.indices(obs_data.shape), axis=0)
	
	points = np.moveaxis(mpl_indices,0,-1)
	_lgr.debug(f'{points.shape=}')
	t_points = transform.inverted().transform(points.reshape((-1,points.shape[-1]), order='C'))
	_lgr.debug(f'{points=}')
	_lgr.debug(f'{t_points=}')
	t_points = np.moveaxis(t_points.reshape((*obs_data.shape,-1), order='C'), -1,0)
	_lgr.debug(f'{t_points.shape=}')
	
	mask = np.ones_like(obs_data,dtype=bool)
	for x in t_points:
		mask &= ((0 <= x) & (x <= 1))
	
	
	
	# Plot region data only
	zeros = np.zeros_like(obs_data)
	
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('region data (other data zeroed)')
	
	zeros *= 0
	zeros[mask] = obs_data[mask]
	ax[0].imshow(zeros)
	ax[0].set_title('observation')
	
	zeros *= 0
	zeros[mask] = deconv_data[mask]
	ax[1].imshow(zeros)
	ax[1].set_title('MC deconv')
	
	for i in range(len(comp_deconv_dataset)):
		zeros *= 0
		zeros[mask] = comp_deconv_dataset[i][mask]
		ax[2+i].imshow(zeros)
		ax[2+i].set_title(f'comparison deconv {i}')
	
	for _a in ax:
		_a.add_patch(region())
	
	plt.show()
	
	
	
	# Plot histogram of region data
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('histograms of region data')
	
	ax[0].hist(obs_data[mask].flatten(), bins=50)
	ax[0].set_title('observation')
	
	ax[1].hist(deconv_data[mask].flatten(), bins=50)
	ax[1].set_title('MC deconv')
	
	
	for i in range(len(comp_deconv_dataset)):
		ax[2+i].hist(comp_deconv_dataset[i][mask].flatten(), bins=50)
		ax[2+i].set_title(f'comparison deconv {i}')
	
	plt.show()
	
	
	# Plot line plot of region data
	f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
	ax = ax.flatten()
	f.suptitle('line plot of region data, region of data convovled with psf')
	
	ax[0].plot(obs_data[mask].flatten())
	ax[0].set_title('observation')
	
	ax[1].plot(obs_data[mask].flatten())
	ax[1].plot(deconv_data[mask].flatten(), label='deconv data')
	ax[1].plot(sp.signal.fftconvolve(deconv_data, psf_data, mode='same')[mask].flatten(), label='deconv data convolved with psf')
	ax[1].legend()
	ax[1].set_title('MC deconv')
	
	
	for i in range(len(comp_deconv_dataset)):
		ax[2+i].plot(obs_data[mask].flatten())
		ax[2+i].plot(comp_deconv_dataset[i][mask].flatten())
		ax[2+i].plot(sp.signal.fftconvolve(comp_deconv_dataset[i], psf_data, mode='same')[mask].flatten())
		ax[2+i].set_title(f'comparison deconv {i}')
	
	
	set_same_limits(ax, lambda x: x.get_ylim(), lambda x, a, b: x.set_ylim(a,b))

	plt.show()
	
	
	
	