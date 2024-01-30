
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
import plot_helper

from amateur_data_analysis_radial_psf import get_source_regions

import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def load_image_as_numpy_array(fpath):
	with PIL.Image.open(fpath) as image:
		array = np.array(image).astype(float)
	return array


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
	# Set some defaults for matplotlib
	mpl.rcParams['image.origin'] = 'upper'
	
	for psf_type in ('fitted', 'original'):
		
		data_dir = example_data_loader.get_amateur_data_set_directory(0)
		
		obs_file = data_dir / "2024-01-11-1917_1-Jupiter_750nm.tif"
		deconv_file = data_dir / "output" / f"2024-01-11-1917_1-Jupiter_750nm_region_0_psf_{psf_type}_test_deconv_components.tif"
		psf_file = data_dir / "output" / f"2024-01-11-1917_1-Jupiter_750nm_region_0_psf_{psf_type}.npy"
		
		comp_deconv_dir = data_dir / "comparison_deconv"
		comp_deconv_files = (
			comp_deconv_dir / "enhanced_a 2024-01-11-1917_1-Jupiter_750nm.tif",
			comp_deconv_dir / "enhanced_b 2024-01-11-1917_1-Jupiter_750nm.tif",
		)
		output_dir = comp_deconv_dir / "output"
		output_dir.mkdir(parents=True, exist_ok=True)
		plot_fname_fmt = '{fname}'+f'_psf_{psf_type}'+'_{plot_info}.png'
		
		obs_data = load_image_as_numpy_array(obs_file)
		deconv_data = load_image_as_numpy_array(deconv_file)
		psf_data = np.load(psf_file)
		_lgr.debug(f'{np.nansum(psf_data)=}')
		comp_deconv_dataset = [load_image_as_numpy_array(fpath) for fpath in comp_deconv_files]
		
		
		deconv_data = sp.ndimage.median_filter(deconv_data, 3)
		for i in range(len(comp_deconv_dataset)):
			comp_deconv_dataset[i] = sp.ndimage.median_filter(comp_deconv_dataset[i], 3)
		
		
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
		
		plot_helper.output(
			fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'component_residuals_unconvolved'),
			figure=f
		)
		
		
		
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
		
		plot_helper.output(
			fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'component_residuals_convolved'),
			figure=f
		)
		
		
		regions = [
			lambda : mpl.patches.Rectangle(
				(465,265), 
				40, 
				20, 
				angle=22, 
				facecolor='none', 
				edgecolor='red', 
				lw=1
			),
			lambda : mpl.patches.Rectangle(
				(455,290), 
				120, 
				60, 
				angle=22, 
				facecolor='none', 
				edgecolor='red', 
				lw=1
			),
			lambda : mpl.patches.Rectangle(
				(472,233), 
				120, 
				30, 
				angle=22, 
				facecolor='none', 
				edgecolor='red', 
				lw=1
			),
			lambda : mpl.patches.Rectangle(
				(485,233), 
				120, 
				5, 
				angle=22, 
				facecolor='none', 
				edgecolor='red', 
				lw=1
			)
		]
		
		for region_idx, region in enumerate(regions):
		
			# Plot data and region
			f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			f.suptitle('data with region')
			
			ax[0].imshow(obs_data)
			ax[0].set_title(f'observation sum {np.nansum(obs_data):0.4g}')
			
			ax[1].imshow(deconv_data)
			ax[1].set_title(f'MC deconv sum {np.nansum(deconv_data):0.4g}')
			
			
			for i in range(len(comp_deconv_dataset)):
				ax[2+i].imshow(comp_deconv_dataset[i])
				ax[2+i].set_title(f'comparison deconv {i} sum {np.nansum(comp_deconv_dataset[i]):0.4g}')
			
			for _a in ax:
				_a.add_patch(region())
			
			#plt.show()
			plot_helper.output(
				fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'region_{region_idx}_data_context'),
				figure=f
			)
			
			
			
			
			
			
			# Get a mask that selects the region
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
			
			region_x_values = t_points[0][mask]
			region_y_values = t_points[1][mask]
			
			n_bins = a_region.get_width()
			region_x_bin_edges = np.linspace(0, int(np.ceil(np.max(region_x_values))), n_bins+1)
			print(f'{region_x_bin_edges=}')
			region_x_bin_values = np.zeros((n_bins,))
			region_x_bin_masks = np.zeros((n_bins,*region_x_values.shape), dtype=bool)
			for i in range(n_bins):
				region_x_bin_masks[i] = (region_x_bin_edges[i] < region_x_values) & (region_x_values <= region_x_bin_edges[i+1])
			
			print(f'{region_x_bin_masks.shape=}')
			print(f'{[np.mean(obs_data[mask][bin_mask]) for bin_mask in region_x_bin_masks]=}')
			
			
			region_corners = np.array([x for x in a_region.get_corners()])
			print(f'{region_corners=}')
			
			
			
			
			# Plot region data only
			region_extent = (
				region_corners[:,0].min(),
				region_corners[:,0].max(),
				region_corners[:,1].max(),
				region_corners[:,1].min(),
			)
			region_slices = (
				slice(int(np.floor(region_corners[:,1].min())), int(np.ceil(region_corners[:,1].max()))), 
				slice(int(np.floor(region_corners[:,0].min())), int(np.ceil(region_corners[:,0].max())))
			)
			_lgr.debug(f'{region_extent=}')
			zeros = np.zeros_like(obs_data)
			
			f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			f.suptitle('region data (other data zeroed)')
			
			zeros *= 0
			zeros[mask] = obs_data[mask]
			ax[0].imshow(zeros[region_slices], extent=region_extent, vmin=np.min(zeros[mask]), vmax=np.max(zeros[mask]))
			ax[0].set_title('observation')
			
			zeros *= 0
			zeros[mask] = deconv_data[mask]
			ax[1].imshow(zeros[region_slices], extent=region_extent, vmin=np.min(zeros[mask]), vmax=np.max(zeros[mask]))
			ax[1].set_title('MC deconv')
			
			for i in range(len(comp_deconv_dataset)):
				zeros *= 0
				zeros[mask] = comp_deconv_dataset[i][mask]
				ax[2+i].imshow(zeros[region_slices], extent=region_extent, vmin=np.min(zeros[mask]), vmax=np.max(zeros[mask]))
				ax[2+i].set_title(f'comparison deconv {i}')
			
			for _a in ax:
				_a.add_patch(region())
				
			set_same_limits(ax, lambda x: x.get_images()[0].get_clim(), lambda x, a, b: x.get_images()[0].set_clim(a,b))
			
			plot_helper.output(
				fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'region_{region_idx}_zoom'),
				figure=f
			)
			
			
			
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
			
			set_same_limits(ax, lambda x: x.get_ylim(), lambda x, a, b: x.set_ylim(a,b))
			set_same_limits(ax, lambda x: x.get_xlim(), lambda x, a, b: x.set_xlim(a,b))
			
			plot_helper.output(
				fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'region_{region_idx}_histograms'),
				figure=f
			)
			
			
			# Plot line plot of region data
			std_alpha = 0.2
			f, ax = plt.subplots(2,int(1+np.ceil(len(comp_deconv_dataset)/2)),squeeze=False,figsize=(12,8))
			ax = ax.flatten()
			f.suptitle('line plot of region data, region of data convovled with psf')
			
			x = range(region_x_bin_masks.shape[0])
			
			obs_data_region_x_mean = np.array([np.mean(obs_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			obs_data_region_x_std = np.array([np.std(obs_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			
			p, = ax[0].plot(obs_data_region_x_mean)
			ax[0].fill_between(x, obs_data_region_x_mean - obs_data_region_x_std, obs_data_region_x_mean + obs_data_region_x_std, color = p.get_color(), alpha=std_alpha)
			ax[0].set_title('observation')
			
			
			
			
			p, = ax[1].plot(obs_data_region_x_mean)
			ax[1].fill_between(x, obs_data_region_x_mean - obs_data_region_x_std, obs_data_region_x_mean + obs_data_region_x_std, color = p.get_color(), alpha=std_alpha)
			#ax[1].plot(deconv_data[mask].flatten(), label='deconv data')
			#ax[1].plot(sp.signal.fftconvolve(deconv_data, psf_data, mode='same')[mask].flatten(), label='deconv data convolved with psf')
			
			deconv_data_region_x_mean = np.array([np.mean(deconv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			deconv_data_region_x_std = np.array([np.std(deconv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			
			p, = ax[1].plot(deconv_data_region_x_mean, label='mean(deconv)')
			ax[1].fill_between(
				x, 
				deconv_data_region_x_mean - deconv_data_region_x_std, 
				deconv_data_region_x_mean + deconv_data_region_x_std, 
				color = p.get_color(), 
				alpha=std_alpha,
				label='std(deconv)'
			)
			
			conv_data = sp.signal.fftconvolve(deconv_data, psf_data, mode='same')
			conv_data_region_x_mean = np.array([np.mean(conv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			conv_data_region_x_std = np.array([np.std(conv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
			
			p, = ax[1].plot(conv_data_region_x_mean, label='mean(deconv data convolved with psf)')
			ax[1].fill_between(
				x, 
				conv_data_region_x_mean - conv_data_region_x_std, 
				conv_data_region_x_mean + conv_data_region_x_std, 
				color = p.get_color(), 
				alpha=std_alpha,
				label='std(deconv data convolved with psf)'
			)
			ax[1].legend()
			ax[1].set_title('MC deconv')
			
			
			for i in range(len(comp_deconv_dataset)):
				p, = ax[2+i].plot(obs_data_region_x_mean)
				ax[2+i].fill_between(x, obs_data_region_x_mean - obs_data_region_x_std, obs_data_region_x_mean + obs_data_region_x_std, color = p.get_color(), alpha=std_alpha)
				
				comp_deconv_dataset_i_region_x_mean = np.array([np.mean(comp_deconv_dataset[i][mask][bin_mask]) for bin_mask in region_x_bin_masks])
				comp_deconv_dataset_i_x_std = np.array([np.std(comp_deconv_dataset[i][mask][bin_mask]) for bin_mask in region_x_bin_masks])
				
				p, = ax[2+i].plot(comp_deconv_dataset_i_region_x_mean)
				ax[2+i].fill_between(
					x, 
					comp_deconv_dataset_i_region_x_mean - comp_deconv_dataset_i_x_std, 
					comp_deconv_dataset_i_region_x_mean + comp_deconv_dataset_i_x_std, 
					color = p.get_color(), 
					alpha=std_alpha
				)
				
				conv_data = sp.signal.fftconvolve(comp_deconv_dataset[i], psf_data, mode='same')
				conv_data_region_x_mean = np.array([np.mean(conv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
				conv_data_region_x_std = np.array([np.std(conv_data[mask][bin_mask]) for bin_mask in region_x_bin_masks])
				
				p, = ax[2+i].plot(conv_data_region_x_mean)
				ax[2+i].fill_between(x, conv_data_region_x_mean - conv_data_region_x_std, conv_data_region_x_mean + conv_data_region_x_std, color = p.get_color(), alpha=std_alpha)
				ax[2+i].set_title(f'comparison deconv {i}')
			
			
			set_same_limits(ax, lambda x: x.get_ylim(), lambda x, a, b: x.set_ylim(a,b))

			plot_helper.output(
				fname=output_dir / plot_fname_fmt.format(fname=obs_file.stem, plot_info=f'region_{region_idx}_marginalised_line_along_region_x_direction'),
				figure=f
			)
	
	
	
	