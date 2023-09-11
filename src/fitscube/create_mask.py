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
		
			data_chosen[~disk_mask] = np.nan
			
			#dc = np.log(data_chosen)
			dc = data_chosen
			dc_max, dc_min = np.nanmax(dc), np.nanmin(dc)
			
			#print(dc_min, dc_max)
			
			f3 = plt.figure(figsize=[6,12])
			a3 = [f3.add_axes([0.1,0.55,0.8,0.4])]
			a3 += [f3.add_axes([0.1,0.05,0.8,0.4])]
			
			# CLOUD SELECTION PARAMETERS
			fit_order = args['fit_order']
			n_sections = args['n_sections']
			n_thresholds = args['n_thresholds']
			cutoff_step = args['cutoff_step']
			
			thresholds = np.linspace(0,1,n_thresholds)
			n_in_mask = lambda x: np.nansum(dc >= (x*(dc_max-dc_min)+dc_min))
			
			# find "n_sections" line sections that best fit the pixel survival distribution (1-cumulative probability)
			# plotting and data generation is all mixed togehter.
			num_in_mask = np.array([n_in_mask(_x) for _x in thresholds])
			a3[0].plot(thresholds, num_in_mask, label='Number of pixels in mask')
			a3[0].set_title(f'Number of pixels in mask as masking threshold increases\nfit with {n_sections} order {fit_order} polynomials')
			a3[0].set_xlabel('Masking Thresholds (fraction of range from min to max)')
			a3[0].set_ylabel('Number of pixels in mask')
		
			if n_sections == 2:
				cutoff_vals = np.linspace(0,1,n_thresholds)
				chisqs = np.vstack(fit_order*[np.full_like(cutoff_vals, fill_value=np.nan)])
				for j in range(fit_order):
					for i in range(1, n_thresholds, cutoff_step):
						c = cutoff_vals[i]
						afit = fit_poly(thresholds, num_in_mask, order=j, aslice=thresholds>=c)
						bfit = fit_poly(thresholds, num_in_mask, order=j, aslice=thresholds<=c)
						chisqs[j,i] = np.sum((num_in_mask[thresholds>=c] - afit[1])**2)/len(afit[1]) \
										+ np.sum((num_in_mask[thresholds<=c] - bfit[1])**2)/len(bfit[1])
				
				#a3[1].plot(np.vstack(fit_order*[cutoff_vals]).T, chisqs.T, label=[f'order {i+1}' for i in range(fit_order)])
				for i in range(fit_order):
					a3[1].plot(cutoff_vals, chisqs[i], label=f'order {i+1}')
				a3[1].set_title(f'ChiSq for different polynomial fits (using fit order {fit_order}, 2 line fit)')
				a3[1].set_xlabel('2 line join position (fraction of range)')
				a3[1].set_ylabel('Chi Squared')
				a3[1].legend()
				best_cutoffs = [cutoff_vals[np.nanargmin(chisqs[fit_order-1])]]
				#print(chisqs)
				
			elif n_sections == 3:
				cutoff_vals = np.linspace(0,1,n_thresholds)
				chisqs = np.full((n_thresholds, n_thresholds), fill_value=np.nan)
				for i in range(0, n_thresholds, cutoff_step):
					for j in range(i+cutoff_step, n_thresholds, cutoff_step):
						cs = [0, cutoff_vals[i], cutoff_vals[j], 1]
						ss = [np.logical_and(cs[_i-1]<=thresholds, thresholds<=cs[_i]) for _i in range(1,len(cs))]
						fitl = [fit_poly(thresholds, num_in_mask, order=fit_order, aslice=s) for s in ss]
						chisqs[i,j] = np.sum([np.sum((num_in_mask[sx] - fitx[1])**2)/len(fitx[1]) for sx, fitx in zip(ss,fitl)])
						
				a3[1].set_title('ChiSq for different cutoffs (3 line fit)')
				a3[1].set_xlabel('Join position between line 2 and 3')
				a3[1].set_ylabel('Join position between line 1 and 2')
				a3[1].imshow(chisqs, origin='lower', extent=(0,1,0,1))
				
				#print(chisqs)
				c1_idx, c2_idx = np.unravel_index(np.nanargmin(chisqs), chisqs.shape)
				best_cutoffs = [cutoff_vals[c1_idx], cutoff_vals[c2_idx]]
			
			else:
				_lgr.ERROR('More than three sections not implemented, exiting...')
				sys.exit()
	


			min_cutoff_idx = int(args['use_section_threshold']//1)
			max_cutoff_idx = int(1+args['use_section_threshold']//1) if min_cutoff_idx < (len(best_cutoffs)-1) else -1
			cutoff_frac = args['use_section_threshold'] - min_cutoff_idx
			cloud_cutoff = cutoff_frac*best_cutoffs[max_cutoff_idx] + best_cutoffs[min_cutoff_idx]*(1 - cutoff_frac)
			#cloud_cutoff = best_cutoffs[args['use_section_threshold']] # we want the last one for cloud mask
		
			#print(f'best_cutoffs {best_cutoffs}')
			#print(f'cloud_cutoff {cloud_cutoff}')

			cutoff = cloud_cutoff
			mask = dc >= (cutoff*(dc_max- dc_min) + dc_min)

			# plot found thresholds
			#print(f'best_cutoffs {best_cutoffs}')
			line_sections = [0]+best_cutoffs+[1]
			for i in range(1,len(best_cutoffs)+2):
				aslice = np.logical_and(line_sections[i-1]<=thresholds, thresholds<=line_sections[i])
				a3[0].plot(*fit_poly(thresholds, num_in_mask, order=fit_order, aslice=aslice), label=f'Fit line section {i}')
			for ls in best_cutoffs:
				a3[0].axvline(ls, linestyle='--', color='tab:red', linewidth=0.5)
			a3[0].axvline(cutoff, linestyle='-', color='tab:green', linewidth=0.5)
			a3[0].legend()
			ut.plt.save_show_plt(
				f3, 
				ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='auto_mask_thresholds'),
				mode=args['plots.mode'], 
				make_dirs=args['plots.make_dirs']
			)
			
			
	
		
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
				ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='auto_mask_regions_1'),
				mode=args['plots.mode'], 
				make_dirs=args['plots.make_dirs']
			)
		
			
			
			
			
			# plot more mask diagnostics
			f1 = plt.figure(figsize=[6,8])
			#a11 = f1.subplots(1,1)
			a11 = f1.add_axes([0.1, 0.05, 0.8, 0.2])
			#print(hdul.info())
			data1 = np.nanmedian(hdul[args['cube.extension']].data, axis=0)
			bins = np.linspace(np.nanmin(data1), np.nanmax(data1), 101)
			plot_radiance_histogram(f1, a11, data1[disk_mask], label='DISK', bins=bins)
			plot_radiance_histogram(f1, a11, data1, label='ALL REGIONS', bins=bins)
			plot_radiance_histogram(f1, a11, data1[~disk_mask], label='BACKGROUND', bins=bins)
			plot_radiance_histogram(f1, a11, data1[disk_mask & (~mask)], label='DISK NO CLOUDS')
			plot_radiance_histogram(f1, a11, data1[disk_mask & mask], label='DISK ONLY CLOUDS')
			a11.legend()
			
			a2 = 4*[None]
			a2[0] = f1.add_axes([0.05, 0.3, 0.4, 0.3])
			a2[1] = f1.add_axes([0.55, 0.3, 0.4, 0.3])
			a2[2] = f1.add_axes([0.05, 0.65, 0.4, 0.3])
			a2[3] = f1.add_axes([0.55, 0.65, 0.4, 0.3])
			
			plot_mask(f1, a2[0], data1, disk_mask)
			a2[0].set_title('DISK MASK')
			
			plot_mask(f1, a2[1], data1, ~disk_mask)
			a2[1].set_title('NOT DISK MASK')
			
			plot_mask(f1, a2[2], data1, disk_mask & (~mask))
			a2[2].set_title('NOT CLOUD MASK')
			
			plot_mask(f1, a2[3], data1, disk_mask & mask)
			a2[3].set_title('CLOUD MASK')
		
			#ut.plt.save_show_plt(f1, 'auto_mask_regions_2.png', plot_dir, show_plot=args['plots.show'], save_plot=args['plots.save'])
			ut.plt.save_show_plt(
				f1, 
				ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='auto_mask_regions_2'),
				mode=args['plots.mode'], 
				make_dirs=args['plots.make_dirs']
			)
		
			#plt.show()
			
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
	
	parser.add_argument('--mask.wav_range', nargs=2, type=float, help='Range of wavelengths to use when autodetecting cloud', default=[860E-9, 890E-9]) # 1.65 um, 1.66 um is ok, so is 860 nm, 890 nm
	
	parser.add_argument('--fit_order', type=int, help='Order of polynomial fit to the number of pixels in the mask with increaseing threshold', default=3)
	parser.add_argument('--n_sections', type=int, help='Number of line sections to fit to the num_of_pixels-vs-threshold plot', default=3)
	parser.add_argument('--n_thresholds', type=int, help='Number of thresholds to calcualte in the num_of_pixels-vs-threshold plot', default=50)
	parser.add_argument('--cutoff_step', type=int, help='How often should we test a cutoff for the thresholds', default=1)
	parser.add_argument('--use_section_threshold', type=float, help='Which threshold should we use for the mask (uses python indexing so -1 is the last one etc.), fractional values are between two sections', default=-1)
	
	parsed_args = vars(parser.parse_args(argv))
	return(parsed_args)
		
		
if __name__=='__main__':
	main(sys.argv[1:])	
		
		






