#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, "DEBUG")

import sys, os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import utilities as ut
import utilities.plt
import utilities.args
import utilities.dict
import utilities.str
#import fitscube.header
import utilities.fits
import py_ssa
import fitscube.deconvolve.flag_bad_pixels
import utilities.sp

def plot_radiance_histogram(fig, ax, data, bins=100, **kwargs):
	#print(data.shape)
	d = data.flatten()
	#h1 = ax.hist(d, bins=n, range=(1E-10, 1E-7), histtype='step', density=True, **kwargs)
	h1 = ax.hist(d, bins=bins, histtype='step', **kwargs)
	ax.set_xlabel('Radiance')
	ax.set_ylabel('Counts')
	#ax.set_xscale('log')
	ax.set_yscale('log')
	return(h1) 

def plot_mask(fig, ax, data, mask, **kwargs):
	#print('HERE')
	im1 = ax.imshow(data, origin='lower', interpolation='none', cmap='viridis')
	ys, xs = np.nonzero(mask)
	#print(xs, ys)
	s1 = ax.plot(xs, ys, marker='.', color='tab:red', markersize=1, linestyle='none')
	return(im1, s1)


def fit_poly(x, y, order=1, aslice=np.s_[...]):
	A = np.vstack([x[aslice]**n for n in range(0,order+1)]).T
	#print(A.shape)
	c = np.linalg.lstsq(A, y[aslice], rcond=None)[0]
	#print(c.shape)
	return(x[aslice], np.matmul(A,c))


def plot_mask_diagnostics(data, disk_mask, mask, source_fpath, fmt, mode, make_dirs):
	"""
	Plot some mask diagnostics
	"""
	data_chosen = np.array(data)
	data_chosen[disk_mask] = np.nan

	# plot mask diagnostics
	#print('data_chosen.shape', data_chosen.shape)
	#print('mask.shape', mask.shape)
	f0 = plt.figure(figsize=[12,12])
	a0 = [f0.add_axes([0.05, 0.6, 0.3, 0.3])]
	a0 += [f0.add_axes([0.35, 0.6, 0.3, 0.3])]
	a0 += [f0.add_axes([0.65, 0.6, 0.3, 0.3])]
	a0 += [f0.add_axes([0.1,0.05,0.8,0.5])]
	
	plot_mask(f0, a0[0], data_chosen, np.zeros_like(mask))
	a0[0].set_title('Object Disk')
	plot_mask(f0, a0[1], data_chosen, mask) 
	a0[1].set_title('Object Disk With Mask Applied')
	plot_mask(f0, a0[2], data_chosen, ~mask)
	a0[2].set_title('Object Disk With Inverse Mask Applied')
	
	a0[3].set_title('Radiance Histogram of Disk and Masked Region')
	bins = np.linspace(np.nanmin(data_chosen), np.nanmax(data_chosen), 101)
	plot_radiance_histogram(f0, a0[3], data_chosen, label='DISK', bins=bins)
	plot_radiance_histogram(f0, a0[3], data_chosen[mask], label='MASKED REGION', bins=bins)
	plot_radiance_histogram(f0, a0[3], data_chosen[~mask], label='UNMASKED REGION', bins=bins)
	
	a0[3].legend()
	#ut.plt.save_show_plt(f0, 'auto_mask_regions_1.png', plot_dir, show_plot=args['plots.show'], save_plot=args['plots.save'])
	ut.plt.save_show_plt(
		f0, 
		ut.plt.get_plot_fpath(fmt, source_fpath=os.path.abspath(source_fpath), name='auto_mask_regions_1'),
		mode=mode, 
		make_dirs=make_dirs
	)
	
	
	# plot more mask diagnostics
	f1 = plt.figure(figsize=[6,8])
	#a11 = f1.subplots(1,1)
	a11 = f1.add_axes([0.1, 0.05, 0.8, 0.2])
	#print(hdul.info())
	bins = np.linspace(np.nanmin(data), np.nanmax(data), 101)
	plot_radiance_histogram(f1, a11, data[disk_mask], label='DISK', bins=bins)
	plot_radiance_histogram(f1, a11, data, label='ALL REGIONS', bins=bins)
	plot_radiance_histogram(f1, a11, data[~disk_mask], label='BACKGROUND', bins=bins)
	plot_radiance_histogram(f1, a11, data[disk_mask & (~mask)], label='DISK NO CLOUDS')
	plot_radiance_histogram(f1, a11, data[disk_mask & mask], label='DISK ONLY CLOUDS')
	a11.legend()
	
	a2 = 4*[None]
	a2[0] = f1.add_axes([0.05, 0.3, 0.4, 0.3])
	a2[1] = f1.add_axes([0.55, 0.3, 0.4, 0.3])
	a2[2] = f1.add_axes([0.05, 0.65, 0.4, 0.3])
	a2[3] = f1.add_axes([0.55, 0.65, 0.4, 0.3])
	
	plot_mask(f1, a2[0], data, disk_mask)
	a2[0].set_title('DISK MASK')
	
	plot_mask(f1, a2[1], data, ~disk_mask)
	a2[1].set_title('NOT DISK MASK')
	
	plot_mask(f1, a2[2], data, disk_mask & (~mask))
	a2[2].set_title('NOT CLOUD MASK')
	
	plot_mask(f1, a2[3], data, disk_mask & mask)
	a2[3].set_title('CLOUD MASK')

	#ut.plt.save_show_plt(f1, 'auto_mask_regions_2.png', plot_dir, show_plot=args['plots.show'], save_plot=args['plots.save'])
	ut.plt.save_show_plt(
		f1, 
		ut.plt.get_plot_fpath(fmt, source_fpath=os.path.abspath(source_fpath), name='auto_mask_regions_2'),
		mode=mode, 
		make_dirs=make_dirs
	)

	return	

def main(argv):
	args = parse_args(argv)
	_lgr.INFO(ut.str.wrap_in_tag(ut.dict.to_str(args), 'ARGUMENTS'))

	for tc in args['target_cubes']:
		_lgr.INFO(f'Finding an auto-calculated mask for target cube {tc}')
		with fits.open(tc) as hdul:
			_lgr.INFO(f'Selecting cloudy sections of cube in wavelength range {args["mask.wav_range"]}')

		
			spec_ax_idx = ut.fits.hdr_get_spectral_axes(hdul[args['cube.extension']].header)
			wavs = ut.fits.hdr_get_axis_world_coords(hdul[args['cube.extension']].header, spec_ax_idx)

			wavs *= ut.fits.hdr_get_unit_si(hdul[args['cube.extension']].header, spec_ax_idx)


			widxs = np.nonzero(np.logical_and(wavs>args['mask.wav_range'][0], wavs < args['mask.wav_range'][1]))[0]
			
			data_chosen = np.nanmean(hdul[args['cube.extension']].data[widxs,:,:], axis=(0))
			_lgr.DEBUG(f'{wavs = }')
			_lgr.DEBUG(f'{widxs = }')
			_lgr.DEBUG(f"{hdul[args['cube.extension']].shape = }")
			_lgr.DEBUG(f"{hdul[args['cube.extension']].data[widxs,:,:].shape = }")
			_lgr.DEBUG(f'{data_chosen.shape = }')
			_lgr.DEBUG(f"{hdul[args['cube.ext.disk_mask']].header.get('EXTNAME', '') = }")

			if hdul[args['cube.ext.disk_mask']].header.get('EXTNAME', '')=="ZENITH":
				disk_mask = ~np.isnan(hdul[args['cube.ext.disk_mask']].data)
			else:
				disk_mask = hdul[args['cube.ext.disk_mask']].data==0
			
			# remove infs and nans for SSA to work
			data_chosen[np.isnan(data_chosen)] = 0

			data_chosen = ut.sp.interpolate_at_mask(data_chosen, np.isnan(data_chosen), edges='convolution')


			ssa = py_ssa.SSA2D(data_chosen, args['ssa.w_shape'], svd_strategy='eigval') 
	
			ssa_masking_function = fitscube.deconvolve.flag_bad_pixels.ssa2d_sum_prob_map

			mask1 = ssa_masking_function(ssa, start=3, stop=12, value=0.98, 
				show_plots=1, transform_value_as=[], weight_by_evals=True
			)
			mask2 = ssa_masking_function(ssa, start=3, stop=12, value=0.992, 
				show_plots=1, transform_value_as=[], weight_by_evals=True
			)

			mask = mask1 & ~mask2
			plt.figure()
			plt.imshow(mask)

			plt.show()

			

			#plot_mask_diagnostics(data_chosen, disk_mask, mask, tc, args['plots.fmt'], args['plots.mode'], args['plots.make_dirs'])

			# CREATE 3D VERSION OF MASK
			mask_3d = np.array(np.broadcast_to(mask, hdul[args['cube.extension']].data.shape), dtype=np.uint32)
			mask_hdu = fits.PrimaryHDU(mask_3d)
			mask_hdul = fits.HDUList([mask_hdu])
			mask_hdul.writeto(
				os.path.join(
					os.path.dirname(tc), 
					args['mask.out']
				), 
				overwrite=(not args['mask.out.overwrite'])
			)
	return()	

def parse_args(argv):
	import argparse as ap
	
	parser = ap.ArgumentParser(description=__doc__)
	
	parser.add_argument('target_cubes', type=str, nargs='+', help='Target cubes to operate on')
	parser.add_argument('--cube.extension', type=int, help='Extension of the target_cubes to operate on', default=0)
	parser.add_argument('--cube.ext.disk_mask', type=int, help='Extension of the target_cubes that contains a disk mask', default=1)
	parser.add_argument('--mask.out', type=str, help='File to output mask to (relative to target cube)', default='./auto_mask_cloud.fits')
	parser.add_argument('--mask.out.overwrite', action=ut.args.ActiontF, prefix='mask.out.', help='If present, will not overwrite existing "--mask.out" file')
	
	ut.plt.add_plot_arguments(parser)
	
	parser.add_argument('--mask.wav_range', nargs=2, type=float, help='Range of wavelengths to use when autodetecting cloud', default=[0.72E-6, 0.735E-6]) # 1.65 um, 1.66 um is ok, so is 720, 735 nm

	parser.add_argument('--ssa.w_shape', type=int, nargs=2, help='Size of window for SSA2D algorithm', default=[5,5])
	
	parsed_args = vars(parser.parse_args(argv))
	return(parsed_args)
		
		
if __name__=='__main__':
	main(sys.argv[1:])	
		
		






