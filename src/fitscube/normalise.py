#!/usr/bin/env python3
"""
Takes in a set of fits files, and a filename relative to those fits files that
contains a spectra of a boring region of the target. Will normalise the fits
files to the average brightness of the spectra in a certain band.

# EXAMPLE #
$ cd ~/scratch/reduced_images
$ ~/Documents/code/python3/standalone/fitscube/normalise.py ./*/*/analysis/obj*.fits --exclude ./*/*/analysis/obj*renormed.fits ./SINFO.2018-08-19T04:23:27/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP096851.fits ./SINFO.2018-08-31T06:20:44/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP014898.fits  --plots.show --plots.save
"""

import sys, os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import subsample
import fitscube.process.cfg
import plotutils


def main(args):
	spectra_files = [os.path.join(os.path.dirname(tc), args['norm.spec_file']) for tc in args['target_cubes']]
	spectra = np.stack([np.loadtxt(spectra_file) for spectra_file in spectra_files])
	
	### PLOTTING ###
	# plot spectra we have been given and also plot the average
	spectra_smoothed = np.array([subsample.conv(spectra[i,:,0], spectra[i,:,1], conv_size=0.005) for i in range(spectra.shape[0])]).transpose(0,2,1)
	f0 = plt.figure(0,figsize=[12,6])
	a0 = f0.subplots(1,1,squeeze=False)
	plot_spectra(f0, a0[0,0], spectra_smoothed, linewidth=0.2)
	plot_spectra(f0, a0[0,0], np.nanmean(spectra_smoothed, axis=0)[None,:,:], linewidth=1, linestyle='--', label='mean of spectra', color='black')
	a0[0,0].set_title('Spectra chosen for renormalisation (smoothed over 0.005 um)')
	a0[0,0].legend()
	a0[0,0].set_ylim([1E-9,5E-7])
	#plt.show()
	plotutils.save_show_plt(f0, 'unnormalised_spectra.png', args['plots.dir'], 
						 show_plot=args['plots.show'], save_plot=args['plots.save'], 
						 overwrite=args['plots.overwrite'])
	### -------- ###
	
	spectra_mean = np.nanmean(spectra, axis=0)
	# find the normalisation factor, taking into account the desired band or wavelength range
	
	# get transmittance in desired band
	band_name = None
	if args['norm.wavs'] is not None:
		wmin, wmax = min(args['norm.wavs']), max(args['norm.wavs'])
		band_name = f'{wmin} to {wmax} um'
		widxs = np.nonzero((wmin <= spectra_mean[:,0]) & (spectra_mean[:,0] <= wmax))
		band_transmittance_interp = np.full_like(spectra_mean[:,0], fill_value=0)
		band_transmittance_interp[widxs] = 1
		
	elif args['norm.band.file'] is not None:
		band_name=args['norm.band.file']
		band_transmittance = np.loadtxt(args['norm.band.file'])
		band_transmittance_interp = np.interp(spectra_mean[:,0], band_transmittance[:,0], band_transmittance[:,1], 
											   left=np.nan, right=np.nan)
		
	elif args['norm.band'] is not None: # this should always be last as it is the default value, check others first
		# get transmittance of band as wavelengths vs transmittance
		if args['norm.band'].lower() == 'h_band':
			band_transmittance = fitscube.process.cfg.H_filter_transmittance[:,1:] 
		else:
			print(f'ERROR: No data for transmittance of band "{args["norm.band"]}" found. Exiting...')
			raise NotImplementedError
		band_name = args['norm.band']
		band_transmittance_interp = np.interp(spectra_mean[:,0], band_transmittance[:,0], band_transmittance[:,1], 
											   left=np.nan, right=np.nan)
	else:
		print('ERROR: Have not specification for what range of wavelengths we should base the normalisation on. Exiting...')
		sys.exit()
	
	# find sum of mean radiance and calculate a normalising factor for each spectra we have
	spectra_mean_sum_in_band = np.nansum(spectra_mean[:,1]*band_transmittance_interp)
	norm_factors = np.nansum(spectra[:,:,1]*band_transmittance_interp[None,:], axis=1)/spectra_mean_sum_in_band
	
	# divide spectra by norm_factor and display
	spectra_normed = spectra[:,:,:]/np.stack([np.ones_like(norm_factors), norm_factors]).T[:,None,:]
	#spectra_normed[:,:,1] = spectra_normed[:,:,1]/norm_factors[:,None]
	
	### PLOTTING ###
	# plot smoothed normed spectra so we can see what is happening
	spectra_normed_smoothed = np.array([subsample.conv(spectra_normed[i,:,0], spectra_normed[i,:,1], conv_size=0.005) for i in range(spectra_normed.shape[0])]).transpose(0,2,1)
	f1 = plt.figure(figsize=[12,6])
	a1 = f1.subplots(1,1,squeeze=False)
	a1_x = a1[0,0].twinx()
	plot_spectra(f1, a1[0,0], spectra_normed_smoothed, linewidth=0.2)
	plot_spectra(f1, a1[0,0], np.nanmean(spectra_normed_smoothed, axis=0)[None,:,:], linewidth=1, linestyle='--', label='mean of normed spectra', color='black')
	a1[0,0].set_title('Spectra renormalised to mean (smoothed over 0.005 um)')
	a1[0,0].set_ylim([1E-9,5E-7])
	plot_spectra(f1, a1_x, np.stack([spectra_mean[:,0], band_transmittance_interp]).T[None,:,:], label=f'Transmittance from {band_name}')
	hdls, lbls = [], []
	for ax in f1.axes:
		h, l = ax.get_legend_handles_labels()
		hdls += h
		lbls += l
	a1_x.legend(hdls, lbls)
	a1_x.set_ylabel('Transmittance')
	a1_x.set_yscale('linear')
	plotutils.save_show_plt(f1, 'normalised_spectra.png', args['plots.dir'], 
						 show_plot=args['plots.show'], save_plot=args['plots.save'], 
						 overwrite=args['plots.overwrite'])
	#plt.show()
	### -------- ###
	
	# apply normalisation to target cubes
	if not args['dry_run']:
		for i, tc in enumerate(args['target_cubes']):
			print(f'INFO: Operating on file {tc}')
			with fits.open(tc) as hdul:
				print(f'INFO: Normalising data in fits file by a factor of {1/norm_factors[i]:.2g}')
				# work out normalised data
				hdul['PRIMARY'].data /=norm_factors[i]
				# find adjusted error
				hdul['ERROR'].data /= norm_factors[i]
				hdul['ERROR_VS_SPECTRAL'].data /= norm_factors[i]
				# adjust psf
				hdul['PSF'].data /= norm_factors[i]
				hdul['UNADJUSTED_PSF'].data /= norm_factors[i]
				hdul['UNADJUSTED_PSF_RESIDUAL'].data /= norm_factors[i]
				# ajust calibrator observation
				hdul['CALIBRATOR_OBS'].data /=  norm_factors[i]
				hdul['CALIBRATOR_SPEC'].data /= norm_factors[i]
				new_fits_file = f'{tc.rstrip(".fits")}{args["norm.tag"]}.fits'
				print(f'INFO: Writing normalised fits file to {new_fits_file}')
				hdul.writeto(new_fits_file, overwrite=args['fits.overwrite'])

def plot_spectra(fig, ax, stacked_spectra_data, **kwargs):
	p1 = ax.plot(stacked_spectra_data[:,:,0].T, stacked_spectra_data[:,:,1].T, **kwargs)
	ax.set_yscale('log')
	ax.set_ylabel('Radiance')
	ax.set_xlabel('Wavelength (um)')
	return([p1])

def parse_args(argv):
	import argparse as ap
	
	parser = ap.ArgumentParser(description='Normalises targetcubes to their average')
	
	parser.add_argument('target_cubes', type=str, nargs='+', help='target cubes to normalise')
	parser.add_argument('--exclude', type=str, nargs='*', help='target cubes to exclude (useful for globbing)', default=[])
	parser.add_argument('--norm.tag', type=str, help='Tag to add to the end of a file that has been normalised', default='_renormed')
	parser.add_argument('--norm.spec_file', type=str, help='File (relative to "target_cubes") that contains a spectra that is used to normalise target cubes.', default='boring_region_spec.dat')
	
	# wavelength region for normalising arguments
	band_grp = parser.add_mutually_exclusive_group()
	band_grp.add_argument('--norm.band', type=str, help='Spectral band to normalise over ("--norm.spec_file" must cover the spectral range to be useful)', default='h_band')
	band_grp.add_argument('--norm.wavs', type=float, nargs=2, help='Wavelength range to normalise over ("--norm.spec_file" must cover the range to be useful)', default=None)
	band_grp.add_argument('--norm.band.file', type=str, help='File of a spectral band [must have columns wavlength and transmittance] to normalise over ("--norm.spec_file" mjust cover the range to be useful)', default=None)
	
	# other args
	parser.add_argument('--dry_run', action='store_true', help='If present, will not actually save files or perform normalisation, just show graphs and other diagnostic information')
	parser.add_argument('--fits.overwrite', action='store_true', help='If present, fits files will be overwritten')
	
	# plotting arguments
	plotutils.add_plot_arguments(parser)
	
	parsed_args = vars(parser.parse_args(argv))
	
	# exclude target cubes we don't want
	chosen_tcs = []
	normed_exclude_tcs = [os.path.normpath(_x) for _x in parsed_args['exclude']]
	for tc in parsed_args['target_cubes']:
		if os.path.normpath(tc) not in normed_exclude_tcs:
			chosen_tcs.append(tc)
	parsed_args['target_cubes'] = chosen_tcs
	
	
	"""
	# DEBUGGING
	for k, v in parsed_args.items():
		if hasattr(v, "__getitem__") and (type(v) not in (str,)):
			print(k)
			for x in v:
				print(f'\t{x}')
		else:
			print(f'{k}\n\t{v}') 
	sys.exit() 
	# END DEBUGGING
	"""
	
	return(parsed_args)


#%% run code


# if we have called the function from the command line then use the passed values
if __name__=='__main__':
	print('INFO: Running file as main')
	# setup interactive mode arguments, if run as a cell in spyder, the 0-element
	# of sys.argv is an empty string. In other cases it's the filename of the
	# script
	# This doesn't plot graphs when run as a cell, why??
	if sys.argv[0] == '':
		sys.argv = ['',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-18T04:00:14/MOV_Neptune---H+K_0.1_2_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
			#'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-19T04:23:27/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP096851.fits', # bad image incomplete and probably cloudy
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-19T09:43:00/MOV_Neptune---H+K_0.1_3_tpl/analysis/obj_NEPTUNE_cal_HIP038133.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T05:08:24/MOV_Neptune---H+K_0.1_1_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
			#'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T06:20:44/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP014898.fits', # bad image, probably cloudy
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP105633.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-26T04:35:04/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD216009.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-26T04:35:04/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP104320.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-27T03:21:50/MOV_Neptune---H+K_0.025_tpl/analysis/obj_NEPTUNE_cal_HD216009.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-27T03:21:50/MOV_Neptune---H+K_0.025_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-27T04:39:52/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD216009.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-27T05:53:05/MOV_Neptune---H+K_0.025_2_tpl/analysis/obj_NEPTUNE_cal_HD216009.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-28T03:05:40/MOV_Neptune---H+K_0.025_3_tpl/analysis/obj_NEPTUNE_cal_HIP094378.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-28T03:05:40/MOV_Neptune---H+K_0.025_3_tpl/analysis/obj_NEPTUNE_cal_HIP105164.fits',
			'/home/dobinsonl/scratch/reduced_images/SINFO.2018-09-30T01:02:39/MOV_Neptune---H+K_0.025_3_tpl/analysis/obj_NEPTUNE_cal_HD216009.fits'
		]
		sys.argv += ['--plots.no_save', '--plots.show', '--dry_run']
	
	args = parse_args(sys.argv[1:])
	main(args)