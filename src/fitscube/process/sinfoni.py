#!/usr/bin/env python3
"""
Processes SINFONI datacubes for later analysis requires the IDL routine "xtellcor_general.pro" has been run
on calibration files to create telluric spectrum files contained in the "--rel_tellspec_folder".

The `if __name__=='__main__':` statement allows execution of code if the script is called directly.
eveything else not in that block will be executed when a script is imported. 
Import statements that the rest of the code relies upon should not be in the if statement, python
is quite clever and will only import a given package once, but will give it multiple names if it
has been imported under different names.

Standard library documentation can be found at https://docs.python.org/3/library/

Packages used in this program are:
	sys
	os
	*utils
	astropy
	numpy
	*fitscube
	astroquery
	matplotlib
	scipy
	textwrap
	copy

	* - indicated package is written by me.
"""
# import standard packages
import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
import copy
import textwrap
import logging
# import 3rd party packages
import astropy as ap
import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.optimize as spo
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.modeling import models, fitting
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from astropy import units as u

# import my packages
import utils as ut # used for convenience functions
import logging_setup
import fitscube.process.cfg
import fitscube.header
import fitscube.fit_region
import const
import psf
import plotutils

def main(argv):
	args = parse_args(argv)
	logging_setup.logging_setup()
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	for tc in args['target_cubes']:
		logging.info(73*'=')
		logging.info('Operating on cube "{}"'.format(tc))
	
		logging.info('Grabbing directory, plot directory, calibrator file, telluric spectrum directory')	
		targ_dir = os.path.dirname(tc)
		targ_plot_dir = os.path.join(targ_dir, './analysis/plots')
		targ_calibs_file = os.path.join(targ_dir, args['rel_calibrators_file'])
		targ_tellspec_dir = os.path.join(targ_dir, args['rel_tellspec_folder'])

		logging.info('Opening fits file as header data unit list')
		with fits.open(tc) as targ_hdul:
			logging.info('Getting target attributes from hdul and updating with attributes from JPLHorizons')
			targ_attrs = get_datacube_attributes(targ_hdul) # attributes of target
			targ_attrs.update(	query_jplhorizons_eph_to_ids(	
									targ_attrs['object'],
									targ_attrs['origin_observatory'],
									targ_attrs['observation_date'],
									[	'r',
										'PDObsLat',
										'delta',
										'delta_rate',
										'ang_width'
									],
									[	'sun_dist',
										'sub_obs_lat',
										'obs_dist',
										'obs_vel',
										'angular_width'
									]
								)
							)

			logging.info('Opeining calibrator fits files as header data unit lists')
			with open(targ_calibs_file, 'r') as f:
				if args['calibrators_relative_fix']: 
					targ_calib_hduls = [fits.open(os.path.relpath(calib_file.strip(), start='/network/group/aopp/planetary/PGJI002_IRWIN_TELESCOP/dobinson/sinfoni/esorex_test/neptune_4')) for calib_file in f]
				else:
					targ_calib_hduls = [fits.open(calib_file.strip()) for calib_file in f]
			targ_n_calibs = len(targ_calib_hduls)
			targ_calib_names = [targ_calib_hdul[0].header['ESO OBS TARG NAME'].lower() for targ_calib_hdul in targ_calib_hduls]

			# TODO: Make a backup function that takes a key-value file and outputs a dictonary for each
			# calibrator just like 'vector_query_simbad_to_ids' just incase we cannot acces SIMBAD
			logging.info('Creating calibrator attributes from SIMBAD, and getting telluric spectrum from each calibrator')
			targ_calib_attrs = vector_query_simbad_to_ids(	targ_calib_names, 
															[	'coordinates',
																'propermotions', 
																'rv_value',
																'fluxdata(H)',
																'fluxdata(V)',
																'fluxdata(B)',
																'typed_id'
															],
															[	'FLUX_H', 
																'FLUX_B', 
																'FLUX_V', 
																'RV_VALUE', 
																'RA',
																'DEC', 
																'PMRA', 
																'PMDEC',
																'TYPED_ID'
															],
															[	'h_flux', 
																'b_flux', 
																'v_flux', 
																'radial_velocity',
																'ra', 
																'dec', 
																'proper_motion_ra', 
																'proper_motion_dec',
																'name'
															]
														)
			for _tca, _tch in zip(targ_calib_attrs, targ_calib_hduls):
				# get attributes for each calibration observation
				_tca.update(get_datacube_attributes(_tch))
				# get data from a previous run of 'xtellcor_general'		
				_tca['xtellcor_A0V'] = os.path.normpath(
											os.path.join(
												targ_tellspec_dir, 
												'xtellcor_{}_adj_A0V.dat'.format(_tca['object'].upper())
											)
										)
				_tca['xtellcor'] = 	os.path.normpath(
										os.path.join(
											targ_tellspec_dir, 
											'xtellcor_{}_adj.dat'.format(_tca['object'].upper())
										)
									)
				_tca['xtellcor_tellspec'] = os.path.normpath(
												os.path.join(
													targ_tellspec_dir, 
													'xtellcor_{}_adj_tellspec.dat'.format(_tca['object'].upper())
												)	
											)

			# make a copy of the target data for each calibrator we have for it, we will calibrate using
			# all calibrators independently
			norm_targ_hduls = [copy.deepcopy(targ_hdul) for _x in range(targ_n_calibs)]
			norm_targ_attrs = [copy.deepcopy(targ_attrs) for _x in range(targ_n_calibs)]


		for cal_hdul, cal_attr, nt_hdul, nt_attr in zip(targ_calib_hduls, targ_calib_attrs, 
														norm_targ_hduls, norm_targ_attrs):
			logging.info('Finding PSF of calibrator')
			
			logging.info(f'DEBUG: cal_hdul[0].data[300, 41:43, 18:22]\n{cal_hdul[0].data[300, 41:43, 18:22]}')
			
			# we are fitting to the median here, but the star location drifts with wavelength so
			# maybe we should fit each wavelength separately?
			#psf, seeing, starpos, background, psf_func, psf_opts = find_star_psf_seeing_background_median(
			#																			cal_hdul[0].data,
			#																			cal_attr['px_scale_arcsec'],
			#																			show_plots=args['show_plots'])
			cached_psf_found=False
			if args['rel_cached_psf']:
				cached_psf_file = os.path.join(os.path.dirname(tc), args['rel_cached_psf']) # should be a location relative to the current target cube
			else:
				cached_psf_file = False
			if args['use_cached_psf']:
				try:
					cached_psf_dat = np.load(cached_psf_file)
					psf = cached_psf_dat['psf']
					seeing = cached_psf_dat['seeing']
					starpos = cached_psf_dat['starpos']
					background = cached_psf_dat['background']
					psf_func = cached_psf_dat['psf_func']
					psf_popt = cached_psf_dat['psf_popt']
					psf_func_name = cached_psf_dat['psf_func_name']
					popt_names = cached_psf_dat['popt_names']
					psf_cube = cached_psf_dat['psf_cube']
					popt_cube = cached_psf_dat['popt_cube']
					pconv_cube = cached_psf_dat['pconv_cube']
					popt_est_cube = cached_psf_dat['popt_est_cube']
					residual_cube = cached_psf_dat['residual_cube']
				except:
					print(f'ERROR: Could not find cached psf located at {cached_psf_file}')
					print(f'-----: Falling back on creating file from scratch')
					cached_psf_found = False

			if not cached_psf_found:
				(	psf, seeing, 
					starpos, background, 
					psf_func, psf_popt, 
					psf_func_name, popt_names, 
					psf_cube, popt_cube, 
					pconv_cube, popt_est_cube, 
					residual_cube
				) = find_star_psf_seeing_background_full(	cal_hdul[0].data, 
															cal_attr['px_scale_arcsec'], 
															show_plots=args['show_plots'], 
															plot_dir=targ_plot_dir, 
															save_plots=True, 
															calibrator_name=cal_attr['name']
														)
			if cached_psf_file:
				try:
					np.savez(cached_psf_file, psf=psf, seeing=seeing, starpos=starpos, background=background, 
							psf_func=psf_func, psf_popt=psf_popt, psf_func_name=psf_func_name, popt_names=popt_names, 
							psf_cube=psf_cube, pconv_cube=pconv_cube, popt_est_cube = popt_est_cube, residual_cube=residual_cube)
				except:
					print(f'ERROR: Something went wrong when saveing psf data to {cached_psf_file}')
				

			logging.info('Saving PSF attributes for target and calibrator')
			cal_attr['psf'] = psf # psf has been adjusted to be centered with no background offset
			cal_attr['seeing'] = seeing
			nt_attr['seeing'] = seeing # assume calibrator and target have the same seeing
			nt_attr['see_sig'] = 1/3*seeing # just use 1/3 of seeing for now # np.sqrt(np.sum([s**2 for s in psf_opts[3:5]])/2) # get standard deviation of seeing # ADJUST FOR NEW PSF CALCULATION
			nt_attr['see_fwhm'] = nt_attr['see_sig']*np.sqrt((2*np.log(2))/2) # get fwhm of seeing
			cal_attr['star_x'] = starpos[:,1]
			cal_attr['star_y'] = starpos[:,0]
			cal_attr['star_x_0based'] = cal_attr['star_x']-1 # make a version of star position for 0-based indexing
			cal_attr['star_y_0based'] = cal_attr['star_y']-1
			cal_attr['background'] = background
			cal_attr['norm_psf'] = cal_attr['psf']/np.nansum(cal_attr['psf'])
			cal_attr['psf_popt'] = psf_popt # parameters to centered and backgrounded psf
			cal_attr['psf_func_name'] = psf_func_name # name of psf used
			cal_attr['popt_names'] = popt_names # names of psf parameters
			cal_attr['psf_cube'] = psf_cube # unadjusted psf
			cal_attr['popt_cube'] = popt_cube # parameters to unadjusted psf
			cal_attr['pconv_cube'] = pconv_cube # covariance matrix of unadjusted psf
			cal_attr['popt_est_cube'] = popt_est_cube # initial parameter estimates of unadjusted psf
			cal_attr['residual_cube'] = residual_cube # residual of the unadjusted psf

			# subtract background from calibrator and target, possibly make this a prompt?
			#print(cal_attr['background'].shape)
			#if args['subtract_cal_background']:
			if args['calibrator.subtract_psf_background']:
				logging.info(' Subtracting PSF background from calibrator')
				cal_hdul[0].data -= cal_attr['background'][:,None,None]
			if args['target.subtract_psf_background']:
				logging.info(' Subtracting PSF background from target')
				nt_hdul[0].data -= cal_attr['background'][:,None,None]

			logging.info(f'DEBUG: cal_hdul[0].data[300, 41:43, 18:22]\n{cal_hdul[0].data[300, 41:43, 18:22]}')

			# subtract any detected drift in the PSF, can apply this to the 
			# target, but probably not a good idea unless there's a really good 
			# reason
			if args['calibrator.remove_psf_spectroscopic_drift']:
				logging.info(' Removing PSF drift from calibrator')
				nz,ny,nx = cal_hdul[0].data.shape
				for k in range(nz):
					cal_hdul[0].data[k,:,:] = sp.ndimage.shift(
													cal_hdul[0].data[k,:,:], 
													(	cal_attr['star_y_0based'][0]-cal_attr['star_y_0based'][k],
														cal_attr['star_x_0based'][0]-cal_attr['star_x_0based'][k]
													), 
													order=1, 
													mode='constant', 
													cval=np.nan, 
													prefilter=True
												)
			# only apply detected spectroscopic drift to the target if there's
			# a really good reason.
			if args['target.remove_psf_spectroscopic_drift']:
				logging.info(' Removing PSF drift from target')
				nz,ny,nx = nt_hdul[0].data.shape
				for k in range(nz):
					nt_hdul[0].data[k,:,:] = sp.ndimage.shift(
													nt_hdul[0].data[k,:,:], 
													(	cal_attr['star_y_0based'][0]-cal_attr['star_y_0based'][k], 
														cal_attr['star_x_0based'][0]-cal_attr['star_x_0based'][k]
													),
													order=1, 
													mode='constant', 
													cval=np.nan, 
													prefilter=True
												)

			logging.info('Converting target and calibrator from photon counts to spectral radiance')
			# get factor to convert to spectral radiance
			counts_per_sec_to_spectral_radiance = get_counts_per_sec_in_h_filter2spectral_radiance(
																					cal_hdul[0].data,
																					cal_attr['wavgrid'],
																					cal_attr['exposure_time'],
																					cal_attr['h_flux'],
																					cal_attr['px_ster']
																					)
			# convert calibrator and target to spectral radiance
			cal_hdul[0].data *= counts_per_sec_to_spectral_radiance/cal_attr['exposure_time']
			cal_attr['psf'] *= counts_per_sec_to_spectral_radiance/cal_attr['exposure_time']
			cal_attr['psf_cube'] *=  counts_per_sec_to_spectral_radiance/cal_attr['exposure_time']
			#cal_attr['residual_cube'] *=  counts_per_sec_to_spectral_radiance/cal_attr['exposure_time'] # don't need to do residual as it's normalised already
			nt_hdul[0].data *= counts_per_sec_to_spectral_radiance/nt_attr['exposure_time']

			logging.info('Adjusting target observation for telluric lines')
			# Assuming airmass is approximately equal between target and calibrator,
			# we can use optical depth per airmass to adjust for telluric lines using the
			# data from 'xtellcor_general.pro'
			optical_depth_per_airmass, polyfit = optical_depth_per_airmass_from_xtellcor(
																cal_attr['xtellcor_tellspec'],
																cal_attr['airmass'],
																xtellcor_A0V = cal_attr['xtellcor_A0V'],
																xtellcor = cal_attr['xtellcor'],
																show_plots=args['show_plots'])
			nt_hdul = apply_xtellcor_tellspec(nt_hdul, nt_attr['wavgrid'], nt_attr['airmass'],
												 optical_depth_per_airmass, polyfit, show_plots=args['show_plots'])
			
			logging.info(f'DEBUG: cal_hdul[0].data[300, 41:43, 18:22]\n{cal_hdul[0].data[300, 41:43, 18:22]}')
			
			logging.info('Fit an ellipsoid and find disk location of target')
			# Fit an ellipsoid to the target's disk, we use a simple 'maximum contained signal' method to
			# determine the ellipsoid's location
			#print(ut.str_wrap_in_tag(repr(nt_attr), 'NORM TARG ATTRIBUTES'))
			(ecx, ecy, Re, Rp, Rpp, disk, lats, lons, zens, 
				xx, yy) = project_lat_lon(	nt_hdul,
											nt_attr['angular_width'],
											fitscube.process.cfg.planet_data[nt_attr['object']]['oblateness'],
											nt_attr['sub_obs_lat'],
											nt_attr['px_scale_arcsec'],
											show_plots=args['show_plots'],
											find_disk_manually=args['find_disk_manually']
										)
			# define a couple of helper functions	
			def shift_3d(im_3d_to_shift):
				nz,ny,nx = im_3d_to_shift.shape
				for k in range(nz):
					im_3d_to_shift[k,:,:] = sp.ndimage.shift(im_3d_to_shift[k,:,:], (ny/2-ecy, nx/2-ecx), order=1, mode='constant', cval=np.nan, prefilter=True)
				return(im_3d_to_shift)

			def shift_2d(im_2d_to_shift, fill_value=np.nan):
				ny, nx = im_2d_to_shift.shape
				im_2d_to_shift = sp.ndimage.shift(im_2d_to_shift, (ny/2-ecy, nx/2-ecx), order=1, mode='constant', cval=fill_value, prefilter=True)
				return(im_2d_to_shift)

			# if we want to, recenter things
			if args['calibrator.recenter_using_psf']:
				logging.info(' Recentering the calibrator using the detected PSF center')
				nz,ny,nx = cal_hdul[0].data.shape
				for k in range(nz):
					cal_hdul[0].data[k,:,:] = sp.ndimage.shift(cal_hdul[0].data[k,:,:], 
																(nx/2 - cal_attr['star_x'], ny/2 - cal_attr['star_y']), 
																order=1, mode='constant', cval=np.nan, prefilter=True)
				#cal_hdul[0].data = shift_3d(cal_hdul[0].data)
			if args['target.recenter_using_psf']:
				logging.info(' Recentering the target using the detected PSF center')
				nt_hdul[0].data = shift_3d(nt_hdul[0].data)
				tnz,tny,tnx = nt_hdul[0].data.shape # normed target sizes
				disk = shift_2d(disk, fill_value=0)
				lats = shift_2d(lats)
				lons = shift_2d(lons)
				zens = shift_2d(zens)
				xx = shift_2d(xx)
				yy = shift_2d(yy)
				# we have adjusted the center of the data so the center of the ellipse will shift also
				ecy, ecx = tny/2, tnx/2
					

			logging.info(f'DEBUG: cal_hdul[0].data[300, 41:43, 18:22]\n{cal_hdul[0].data[300, 41:43, 18:22]}')
			
			# put some of the data into the attributes dictionary to make passing at the end easier
			fill_with_keys = ('ecx','ecy','Re','Rp','Rpp')
			fill_with_values = (ecx,ecy,Re,Rp,Rpp)
			for k, v in zip(fill_with_keys, fill_with_values):
				nt_attr[k] = v
					
			logging.info('Calculating target pixel disk fraction')				
			# What fraction of a pixel's flux is from the disk
			disk_frac = find_disk_fraction(nt_hdul, disk, psf_func, psf_func_name, psf_popt, popt_names, show_plots=args['show_plots'])				
			
			logging.info('Get statistical noise of target from low disk-fraction regions')
			# Use a region far from the disk (low disk_frac) to choose pixels which we can sample 
			# for statistical noise
			stat_error, stat_error_mask = find_stat_error(nt_hdul, disk_frac, show_plots=args['show_plots'])
		
			logging.info('Increasing error in wavelength bins that have poor atmospheric transmission')
			# we should now account for errors due to low-transmittance due to earth's atmosphere
			stat_error = apply_xtellcor_tellspec_to_errors(	stat_error, nt_attr['airmass'],
															optical_depth_per_airmass)

			logging.info('Adjusting for target doppler shift')
			# adjust the wavelength grid of target cube to account for doppler shift due to radial motion of target
			# must do this after any adjustements due to tellurics and other observer-frame effects.
			nt_hdul, wavgrid = adjust_for_doppler_shift(nt_hdul, nt_attr['wavgrid'], nt_attr['obs_vel'])
			nt_attr['wavgrid'] = wavgrid	

			# create an output name for our new fits file
			outname = os.path.join(	targ_dir,
									args['rel_output_folder'],
									'obj_{}_cal_{}.fits'.format(nt_attr['object'].upper(), cal_attr['object'].upper())
									)
			
			logging.info('Assembling output fits file header data units')
			# Assemble the new fits file
			
			logging.info(f'DEBUG: cal_hdul[0].data[300, 41:43, 18:22]\n{cal_hdul[0].data[300, 41:43, 18:22]}')
			
			assemble_norm_targ_fits(nt_hdul, nt_attr, cal_hdul, cal_attr,
					stat_error, stat_error_mask, lats, lons, zens,
					xx, yy, disk, disk_frac, psf)
			# write the new fits file
			logging.info('Writing out fits file to {}'.format(outname))
			nt_hdul.writeto(outname, overwrite=True)

		logging.info('Closing opened fits files')
		# close fits files
		[calib_hdul.close() for calib_hdul in targ_calib_hduls]
		[norm_targ_hdul.close() for norm_targ_hdul in norm_targ_hduls]

		logging.info(73*'=')
	return	

def assemble_norm_targ_fits(nt_hdul, nt_attr, cal_hdul, cal_attr, stat_error, 
								stat_error_mask, lats, lons, zens, xx, yy, disk, disk_frac, psf,
								outfname='norm_targ.fits', overwrite=True):
	"""
	Assembles a fits file that is congruent to those created by IDL routine 'process_sinf.pro'

	NOTE: IDL routines make an 80x80 image with the planetary disk center and the image center aligned, I haven't done that here, but if some routines fail that may be why

	hdul[0] = data calibrated for spectral radiance using the calibrator
	hdul[1] = error on data (each pixel has a 1:1 correspondence, could change this to be a single wavelength dependet column but IDL routines don't do that (and therefore have loads of redundant data)
	hdul[2] = latitude of pixel
	hdul[3] = longitude of pixel
	hdul[4] = angle LOS makes to zenith of surface
	hdul[5] = arcsec in x-direction from the center of disk
	hdul[6] = arcsec in y-direction from the center of disk

	My extras
	hdul[7] = on-off planetary disk mask
	hdul[8] = pixel fraction that is emission from disk
	hdul[9] = pixels used for statistical errors on data
	hdul[10]= Point spread function
	hdul[11]= 1D version of error on data (1 pixel per wavelength)
	hdul[12]= Calibration observation
	"""
	logging.info('In "assemble_norm_targ_fits()"')
	onetruepath = lambda x: os.path.normpath(os.path.abspath(x))

	#print(f'nt_hdul {nt_hdul}')
	#print(f'nt_attr {nt_attr}')
	#print(f'cal_hdul {cal_hdul}')
	#print(f'cal_attr {cal_attr}')

	# Keys to copy from dictionary
	ks = 	[	'sun_dist',
				'sub_obs_lat',
				'obs_dist',
				'obs_vel',
				'angular_width',
				'seeing',
				'see_sig',
				'see_fwhm',
				'Re',
				'Rp',
				'Rpp'
			]
	# Name to give keys in fits file, try to make <=8 characters
	fks =	[	'SUN_DIST',
				'SOB_LAT',
				'OB_DIST',
				'OB_VEL',
				'A_WIDTH',
				'SEEING',
				'SEE_SIG',
				'SEE_FWHM',
				'RAD_EQ',
				'RAD_POL',
				'RAD_PROJ'
			]
	comments = 	[ 	'Distance from sun to object (AU or km)',
					'Sub observer latitude (deg)',
					'Distance from observer to object (AU or km)',
					'Velocity of object w.r.t observeri (km/s)',
					'Angular width of object w.r.t observer (arcsec)',
					'Seeing (arcsec), usually 3 sigma of a gaussian',
					'standard deviation of seeing (pixels)',
					'full width half maximum of seeing (pixels)',
					'Equatorial radius (arcsec)',
					'Polar radius (arcsec)',
					'Projected polar radius (arcsec)'
				]
	for k,fk,c in zip(ks,fks,comments):
		nt_hdul[0].header[fk] = (nt_attr[k],c)
	# what about 1-based indexing?
	nt_hdul[0].header['ELL_CX'] = (nt_attr['ecx']+1,'Disk ellipse center x-coord (pixels)')
	nt_hdul[0].header['ELL_CY'] = (nt_attr['ecy']+1,'Disk ellipse center y-coord (pixels)')


	hdr_3d = fitscube.header.get_coord_fields(nt_hdul[0].header)
	hdr_cel = hdr_3d
	hdr_spec = hdr_3d

	#print(hdr_3d)

	stat_error_3d = np.zeros_like(nt_hdul[0].data)
	stat_error_3d[:,:,:] = stat_error[:,None,None]
	err_hdu = fits.ImageHDU(data=stat_error_3d,
							header=hdr_3d,
							name = 'ERROR'
							)

	# may have to change the headers to account for no spectral axis
	lat_hdu = fits.ImageHDU(data=lats,
							header=hdr_cel,
							name='LATITUDE'
							)

	lon_hdu = fits.ImageHDU(data=lons,
							header=hdr_cel,
							name='LONGITUDE'
							)

	zen_hdu = fits.ImageHDU(data=zens,
							header=hdr_cel,
							name='ZENITH'
							)

	xarcsec_hdu = fits.ImageHDU(data=xx,
								header=hdr_cel,
								name='X_ARCSEC'
								)

	yarcsec_hdu = fits.ImageHDU(data=yy,
								header=hdr_cel,
								name='Y_ARCSEC'
								)

	planmask_hdu = fits.ImageHDU(data=disk.astype('float'),
								header=hdr_cel,
								name='DISK_MASK'
								)

	diskfrac_hdu = fits.ImageHDU(data=disk_frac,
								header=hdr_cel,
								name='DISK_FRAC'
								)

	errmask_hdu = fits.ImageHDU(data=stat_error_mask.astype('float'),
								header=hdr_cel,
								name='ERROR_MASK'
								)

	psf_hdu = fits.ImageHDU(data=psf, 
							header=hdr_cel, 
							name='PSF'
							)
	
	#stat_error_unmask = np.full_like(stat_error, np.nan) # fill with nan's initially
	#stat_error_unmask[~stat_error.mask] = stat_error[~stat_error.mask]
	#print(stat_error_unmask)
	#print(hdr_spec)
	err_sp_hdu = fits.ImageHDU(data=stat_error.data[:,None,None],
								header=hdr_spec,
								name='ERROR_VS_SPECTRAL'
								)


	ks = 	[	'seeing',
				'h_flux',
				'b_flux',
				'v_flux',
				'radial_velocity',
			]
	fks = 	[	'SEEING',
				'H_FLUX',
				'B_FLUX',
				'V_FLUX',
				'RV',
			]
	comments = 	[	'seeing (arcsec) ususally 3 sigma of a gaussian',
					'H-band flux (mag)',
					'B-band flux (mag)',
					'V-band flux (mag)',
					'Radial veocity of star',
				]
	for k, fk, c in zip(ks, fks, comments):
		#print(f'{fk} {k} {c}')
		cal_hdul[0].header[fk] = (cal_attr[k], c)
	#print('DEBUG: After cal_hdul[0].header loop')
	#print(cal_hdul[0].header)

	cal_hdu = fits.ImageHDU(data=cal_hdul[0].data,
							header = cal_hdul[0].header,
							name='CALIBRATOR_OBS'
							)

	cal_sp_hdu = fits.ImageHDU(data=np.nansum(cal_hdul[0].data, axis=(1,2), keepdims=True),
								header = fitscube.header.get_coord_fields(cal_hdul[0].header),
								name='CALIBRATOR_SPEC'
								)

	ahdr = fitscube.header.get_coord_fields(cal_hdul[0].header)
	ahdr['PSF_FUNC'] = (cal_attr['psf_func_name'], 'Function used to calculate psf')
	psf_unadj_hdu = fits.ImageHDU(	data=cal_attr['psf_cube'], 
									header=ahdr,
									name='UNADJUSTED_PSF')

	ahdr = fitscube.header.get_coord_fields(cal_hdul[0].header)
	psf_unadj_resid_hdu = fits.ImageHDU(	data=cal_attr['residual_cube'],
											header=ahdr,
											name='UNADJUSTED_PSF_RESIDUAL')

	tbl_names = ['STAR_X', 'STAR_Y', 'BACKGROUND_PER_PIX']
	tbl_formats = ['D', 'D', 'D']
	tbl_values = [cal_attr['star_x'], cal_attr['star_y'], cal_attr['background']]
	tbl_cols = [fits.Column(name=n, format=f, array=v) for n, f, v in zip(tbl_names, tbl_formats, tbl_values)]
	psf_prop_tbl_hdu = fits.BinTableHDU.from_columns(tbl_cols, name='PSF_DERIVED_PROPERTIES')

	tbl_names = cal_attr['popt_names']
	tbl_formats=['D']*len(tbl_names)
	tbl_values = [cal_attr['popt_cube'][:,i] for i in range(len(tbl_names))]
	tbl_cols = [fits.Column(name=n, format=f, array=v) for n, f, v in zip(tbl_names, tbl_formats, tbl_values)]
	psf_param_tbl_hdu = fits.BinTableHDU.from_columns(tbl_cols, name='UNADJUSTED_PSF_PARAMS')

	
	tbl_names = cal_attr['popt_names']
	tbl_formats=['D']*len(tbl_names)
	tbl_values = [cal_attr['popt_est_cube'][:,i] for i in range(len(tbl_names))]
	tbl_cols = [fits.Column(name=n, format=f, array=v) for n, f, v in zip(tbl_names, tbl_formats, tbl_values)]
	psf_est_param_tbl_hdu = fits.BinTableHDU.from_columns(tbl_cols, name='UNADJUSTED_PSF_ESTIMATED_PARAMS')


	list_of_hdus =  [	err_hdu,
						lat_hdu, 
						lon_hdu, 
						zen_hdu, 
						xarcsec_hdu, 
						yarcsec_hdu, 
						planmask_hdu, 
						diskfrac_hdu, 
						errmask_hdu, 
						psf_hdu,
						err_sp_hdu,
						cal_hdu,
						cal_sp_hdu,
						psf_unadj_hdu,
						psf_unadj_resid_hdu,
						psf_prop_tbl_hdu,
						psf_param_tbl_hdu,
						psf_est_param_tbl_hdu
					]
	for hdu in list_of_hdus:
		nt_hdul.append(hdu)

	return(nt_hdul)

def adjust_for_doppler_shift(hdul, wavgrid, radvel):
	"""
	Adjusts a wavelength grid to take doppler shift into account
	Assuming that radial velocity is given in km/s, +ve is away from observer
	Must apply after any operations that deal with telluric corrections as telluric correction involves light as
	detected from Earth, and so the adjustments should be made in the observer frame. But we want to know what 
	wavelengths are emitted from our target, so have to adjust to target frame at the end.
	"""
	dwav = hdul[0].header['CRVAL3']*radvel*1E3/1E8 # get change in wavelength, assume spectral axis is axis 3
	print(f'dwav {dwav}')
	hdul[0].header['CRVAL3'] -= dwav # just have to shift the reference value
	return(hdul, datacube_wavelength_grid(hdul[0]))

def apply_xtellcor_tellspec_to_errors(stat_error, airmass, odepth_per_airmass):
	"""
	Adjusts the statistical error of a fitscube due to atmospheric absorption
	"""
	odepth = odepth_per_airmass*airmass
	trans = np.exp(-odepth)
	stat_error /= trans # divide by transmittance, low transmittance = lots of error
	return(stat_error)

def find_stat_error(hdul, disk_frac, cutoff=0.02, show_plots=True):
	#show_plots = True # FOR DEBUGGING
	stat_error_mask = np.zeros_like(disk_frac, dtype=bool)
	stat_error_mask[np.where(disk_frac > cutoff)] = True
	
	mask = np.zeros_like(hdul[0].data, dtype=bool)
	mask[:, :, :] = stat_error_mask[None, :, :]
	masked_data = ma.array(hdul[0].data, mask=mask)

	stat_error = np.nanstd(masked_data, axis=(1,2))

	if show_plots:
		nx, ny = (2,1)
		cm_size = 12
		f1 = plt.figure(figsize=[_x/2.54 for _x in (cm_size*nx, cm_size*ny)])

		a11 = f1.add_subplot(ny,nx,1)
		a12 = f1.add_subplot(ny,nx,2)
		#a13 = f1.add_subplot(ny,nx,3)
		img11 = a11.imshow(stat_error_mask, origin='lower')
		a11.set_title('stat error mask')

		mask[:, :, :] = stat_error_mask[None, :, :]
		masked_data = ma.array(hdul[0].data, mask=mask)

		img12 = a12.imshow(masked_data[300,:,:], origin='lower')
		a12.set_title('masked data')

		#a13.plot(err)
		#a13.set_title('err')

		plt.show()
	return(stat_error, stat_error_mask)

def find_disk_fraction(hdul, disk, psf_func, psf_func_name, psf_popts, popt_names, show_plots=True):
	nz,ny,nx = hdul[0].data.shape
	#y,x = np.mgrid[:ny,:nx]
	df = np.zeros((ny,nx))
	pd = dict(zip(popt_names, np.nanmedian(psf_popts, axis=0))) # use the median of the psf_popt's as opposed to treating the whole wavelength-space independently, it will save time
	logging.info(f' pd {pd}')
	for i in range(ny):
		for j in range(nx):
			if psf_func_name in ('mofgaus_8',):
				pd['cx'] = i
				pd['cy'] = j
				pd['atot'] = 1
				psf_arguments = [pd[name] for name in popt_names]
			else:
				print(f'ERROR: No prescription for using point spread function {psf_func_name} to calculate disk fraction')
				sys.exit()
			psf_at_ij = psf.func_as(psf_func, disk.shape, *psf_arguments)
			df[i,j] = np.sum(psf_at_ij*disk)
	if show_plots:
		f1 = plt.figure()
		a11 = f1.add_subplot(1,1,1)
		img111 = a11.imshow(df, origin='lower')
		a11.set_title('disk fraction')

		#a12 = f1.add_subplot(1,2,2)
		#img121 = a12.imshow(psf, origin='lower')
		#a12.set_title('last psf example')

		plt.show()
	return(df)

def project_lat_lon(hdul, angular_width, oblateness, sub_obs_lat, px_scale, show_plots=True, find_disk_manually=True):
	"""
	Fits an ellipsoid to the passed data

	ARGUMENTS:
		hdul [nz,ny,nx]
			<header data unit list> A list of header data units, the 0th hdu will have an ellipse fitted to it
		angular_width
			<float> (arcsec) The major axis of the ellipse to fit
		oblateness
			<float> The oblateness of the ellipse to fit
		sub_obs_lat
			<float> (deg) The latitide of the sub-observer point (this is the point in the direct center of an image)
		px_scale
			<float> (arcsec) How large a pixel of our image is in arcseconds
		show_plots
			<bool> If true, will show plots useful for debugging

	RETURNS:
		ecx
			<int> The center of the fitted ellipse in the x-direction, is an int due to fitting algorithm
		ecy
			<int> The center of the fitted ellipse in the y-direction, is an int due to fitting algorithm
		Re
			<float> (arcsec) The equatorial radius of the fitted ellipsoid
		Rp
			<float> (arcsec) The polar radius of the fitted ellipsoid
		Rpp
			<float> The projected polar radius of the fitted ellipsoid (as observed in the image)
		ellipse [ny,nx]
			<array, <array, float>> The ellipse that is fitted to the planetary disk
		lats [ny,nx]
			<array,<array,float>> A map of latitudes across the disk of the planet, locations off-disk are filled with NANs
		lons [ny,nx]
			<array, <array, float>> A map of longitudes across the disk of the planet, locations off-disk are filled with NANs
		zens [ny,nx]
			<array, <array, float>> A map of 'observer-to-zenith' angles across the disk of the planet, locations off-disk are filled with NANs
		xx [ny,nx]
			<array, <array, float>> A map of angular distance (arcsec) in the x-direction from the center of the fitted ellipse
		yy [ny,nx]
			<array,<array,float>> A map of angular distance (arcsec) in the y-direction from the center of the fitted ellipse
	"""
	Re = angular_width/2 # equatorial radius
	Rp = Re*(1.0 - oblateness)

	# define an ellipsoid with same oblateness, but major axis = 1
	a = 0.5 # major axis of 1, sma of 0.5
	b = a*(1.0 - oblateness)
	m = np.tan(sub_obs_lat)
	if m != 0.0:
		xdiff_sq = 4.0/((1.0/a**2) + (b**2/(m**2*a**4)))
		ydiff_sq = (4.0*b**4)/((m*a)**2 + b**2)
		proj_dist = np.sqrt(xdiff_sq + ydiff_sq)
		Rpp = proj_dist*Re # major axis = 1, so just scale by equatorial diameter
	else:
		Rpp = Rp
	print(f'Equatorial diameter {Re} arcsec')
	print(f'Polar diameter {Rp} arcsec')
	print(f'Projected polar diameter {Rpp} arcsec')

	# auto-find pointing by finding where the greatest signal enclosed by
	# an ellipse with a=Re/2, b=Rpp/2 ?	

	a = Re/px_scale
	b = Rpp/px_scale
	e_idx, ellipse = find_max_ellipse_top_hat(hdul, a, b, t=0, find_disk_manually=find_disk_manually)
	#e_idx, ellipse = find_ellipse_gradient(np.nanmedian(hdul[0].data, axis=(0)), a, b, t=0)

	print(f'ellipse a {a} b {b} t {0}')
	print(f'ellipse center indices {e_idx}')

	#### Plot found ellipse	
	if show_plots:
		f1 = plt.figure(figsize=[_x/2.54 for _x in (36,12)])

		a11 = f1.add_subplot(1,3,1)
		# create an ellipse patch, will need to copy it to reuse it because matplotlib is weird
		disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, 0, facecolor='none', edgecolor='tab:green')
		data = np.nanmean(hdul[0].data, axis=0)
		#data = np.abs(np.diff(data, axis=0)[:,1:] + np.diff(data, axis=1)[1:,:])
		#data = np.log(np.sum(np.abs(np.gradient(data)), axis=0))
		#data = np.log(np.nanmean(hdul[0].data, axis=0))
		cmin = np.nanmin(data)
		cmax = np.nanmax(data)
		img1 = a11.imshow(data, origin='lower', vmin=cmin, vmax=cmax)
		a11.set_title('Median of data')
		a11.add_patch(copy.copy(disk_ellipse))

		print(data.shape)
		print(ellipse.shape)
		#ec = np.array([e_idx[1],e_idx[0]])
		ec = e_idx
		aa = np.array([a,0])
		bb = np.array([0,b])
		print(ec, aa, bb)
		print(ec+aa, ec+bb)
		al = np.stack([ec,ec+aa], axis=0)
		bl = np.stack([ec,ec+bb], axis=0)
		print(al)
		print(bl)

		#DEBUGGING
		#plt.show()

		a12 = f1.add_subplot(1,3,2)
		img2 = a12.imshow(ma.array(data,mask=np.logical_not(np.array(ellipse, dtype='bool'))), origin='lower', vmin=cmin, vmax=cmax)
		#img2 = a12.imshow(ellipse, origin='lower', zorder=0)
		a12.scatter(e_idx[0], e_idx[1], edgecolor='red', facecolor='none', zorder=1, label='ellipse center')
		a12.plot(al[:,0], al[:,1], color='red', zorder=2, label='semi-major axis')
		a12.plot(bl[:,0], bl[:,1], color='green', zorder=3, label='semi-minor axis')
		a12.legend()
		a12.set_title('Ellipse fitted to planet disc')
		#disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, 0, facecolor='none', edgecolor='tab:green')
		a12.add_patch(copy.copy(disk_ellipse))

		a13 = f1.add_subplot(1,3,3)
		img3 = a13.imshow(data - ellipse*data, origin='lower', vmin=cmin, vmax=cmax)
		a13.set_title('data median minus fitted ellipse')
		#disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, 0, facecolor='none', edgecolor='tab:green')
		a13.add_patch(copy.copy(disk_ellipse))

		plt.show()
	##### end plotting

	ny, nx = hdul[0].data.shape[1:]
	lats = np.ones((ny,nx))*np.nan # fill with junk value
	lons = np.ones((ny,nx))*np.nan # fill with junk value
	zens = np.ones((ny,nx))*np.nan # fill with junk value
	ecx, ecy = e_idx
	print(f'ellipse center x {ecx} y {ecy}')
	for i in range(nx):
		for j in range(ny):
			eoffA = (i-ecx)*px_scale
			poffA = (j-ecy)*px_scale
			#print(f'i {i} j {j} eoffA {eoffA} poffA {poffA} px_scale {self.px_scale} py_scale {self.px_scale} nx {nx} ny {ny}')
			iflag, xlat, xlon, zen = projpos_ellipse(Re, Rp, sub_obs_lat, eoffA, poffA)
			#print(f'Re {Re} Rp {Rp} ecx {ecx} ecy {ecy} iflag {iflag} xlat {xlat} xlon {xlon} zen {zen}')
			if iflag:
				lats[j,i] = xlat
				lons[j,i] = xlon
				zens[j,i] = zen

	# calculate row-column pixel scale offsets from center of ellipse
	yy, xx = np.mgrid[:ny, :nx]
	yy = (yy-ecy)*px_scale
	xx = (xx-ecx)*px_scale

	### plot for debugging
	if show_plots:
		print(f'lats.shape {lats.shape}')
		print(f'lons.shape {lons.shape}')
		print(f'zens.shape {zens.shape}')

		nx, ny = (4,1)
		f2 = plt.figure(figsize=[_x/2.54 for _x in (12*nx, 12*ny)])

		a21 = f2.add_subplot(ny,nx,1)
		im21 = a21.imshow(data, origin='lower')
		a21.set_title('Median of data')	
		a21.add_patch(copy.copy(disk_ellipse))

		a22 = f2.add_subplot(ny,nx,2)
		#im22 = a22.imshow(lats, origin='lower')
		im22 = a22.imshow(ma.array(lats, mask=np.isnan(lats)), origin='lower')
		a22.set_title('Latitude')
		a22.add_patch(copy.copy(disk_ellipse))

		a23 = f2.add_subplot(ny,nx,3)
		#im23 = a23.imshow(lons, origin='lower')
		im23 = a23.imshow(ma.array(lons, mask=np.isnan(lons)), origin='lower')
		a23.set_title('longitude')
		a23.add_patch(copy.copy(disk_ellipse))

		a24 = f2.add_subplot(ny,nx,4)
		#im24 = a24.imshow(zens, origin='lower')
		im24 = a24.imshow(ma.array(zens, mask=np.isnan(zens)), origin='lower')
		a24.set_title('angle of LOS w.r.t zenith')
		a24.add_patch(copy.copy(disk_ellipse))

		plt.show()
	### end plotting

	# store relavent data in class
	#self.ellipse_center_x = ecx
	#self.ellipse_center_y = ecy
	#self.Re = Re
	#self.Rp = Rp
	#self.Rpp = Rpp
	#self.disk = ellipse
	#self.lats = lats
	#self.lons = lons
	#self.zens = zens
	#self.xx = xx
	#self.yy = yy
	return(ecx, ecy, Re, Rp, Rpp, ellipse, lats, lons, zens, xx, yy)

def find_ellipse_gradient(data,a,b,t):
	graddata = np.log(np.sum(np.abs(np.gradient(data)),axis=0))
	a_vec = fitEllipse(graddata)
	ecx,ecy = ellipse_center(a_vec)
	a,b = ellipse_axis_length(a_vec)
	t = ellipse_angle_of_rotation2(a_vec)

	print(ecx, ecy, a, b, t)

	nx, ny = (1,1)
	f1 = plt.figure(figsize=[_x/2.54 for _x in (12*nx, 12*ny)])

	a11 = f1.add_subplot(ny,nx,1)
	im11 = a11.imshow(graddata, origin='lower')
	a11.set_title('Median of data')	
	disk_ellipse = ptc.Ellipse((ecx,ecy), 2*a, 2*b, t, facecolor='none', edgecolor='tab:green')
	a11.add_patch(disk_ellipse)

	plt.show()

def find_max_ellipse_top_hat(hdul, a=1, b=1, t=0, find_disk_manually=True):
	cmin, cmax, chmin, chmax = (0.07, 0.2, 200, 400)
	data = np.nanmedian(hdul[0].data[chmin:chmax,:,:], axis=0)
	data_range = np.nanmax(data) - np.nanmin(data)
	data_min = np.nanmin(data)
	data = np.clip(data, cmin*data_range+data_min, cmax*data_range+data_min)
	top_hat_sum = np.zeros_like(data)
	print('data.shape {}'.format(data.shape))
	y, x = np.mgrid[:data.shape[0], :data.shape[1]] # fortran y-x ordering from data
	# simple grid search, make better if I need to
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			eth = ellipse_top_hat_as(data.shape, x[i,j], y[i,j], a, b, t)
			top_hat_sum[i,j] = np.nansum(eth*data)
	max_top_hat_sum_idx = np.unravel_index(np.nanargmax(top_hat_sum), data.shape)
	ecx, ecy = (x[max_top_hat_sum_idx], y[max_top_hat_sum_idx])
	"""
	ny, nx = data.shape
	popt, pconv = fit_ellipse(data, params=[nx/2, ny/2, a, b, t], fixed=[False, False, True, True, True])
	ecx, ecy, a, b, t = popt
	"""
	if find_disk_manually:
		diskFitter = fitscube.fit_region.FitscubeDiskFinder(hdul, ecx=ecx, ecy=ecy, a=a, b=b, icmapmin=cmin, 
															icmapmax=cmax, ifreqmin=chmin, ifreqmax=chmax)
		diskFitter.run()
		ecx, ecy = diskFitter.getEllipseCenter()
	return((ecx,ecy), ellipse_top_hat_as(data.shape, ecx, ecy,a,b,t,error=1E-3))
	
def projpos_ellipse(Re, Rp, eps, eoff, poff):
	"""
	Finds the longitude, latitude, and zenith angles of the intercept point of an ellipse and a
	line fired offset from an observers line of sight.

	ARGUMENTS:
		Re
			<float> Equatorial radius of ellipse
		Rp
			<float> Polar radius of ellipse
		eps
			<float> Sub-observer (planetocentric) latitude
		eoff
			<float> equatorial offset of beam (arcsec)
		poff
			<float> polar offset of beam (arcsec)

	RETURNS:
		iflag
			<bool> True if line intercepts ellipse, False otherwise
		xlat
			<float> Latitude of intercept point
		xlon
			<float> Longitude of intercept point
		zen
			<float> Angle of line w.r.t Zenith of intercept point
	"""
	const.deg2rad = np.pi/180
	# using line in symmetric form (x-x0)/alpha = (y-y0)/beta = (z-z0)/gamma
	# line goes through the point (x0,y0,z0)
	# line has the direction vector d = (alpha, beta, gamma), d should be a unit vector

	x0 = 0.0 # x-point on the beam-line (zero x-coord means the point the line goes through is in the yz-plane
	y0 = eoff # y-point on the beam-line, equatorial offset of beam (arcsec)
	z0 = poff/np.cos(eps*const.deg2rad) # z-point on the beam-line

	alpha = np.sin(np.pi/2.0 - eps*const.deg2rad) # x-dimension of unit direction vector
	beta = 0.0 # y-dimension of unit direction vector, no change in y-direction so y = eoff always, so the line is contained within the xz-plane at y=eoff
	gamma = np.cos(np.pi/2 - eps*const.deg2rad)# z-dimension of unit direction vector

	iflag, x, y, z = intercept_ellipse(Re, Rp, alpha, beta, gamma, x0, y0,z0)
	#; iflag - does our line intercept (True) or not (False) the ellipsoid? (tangents count as not intercepting)
	#; x - the x-values of the two intercepting points
	#; y - the y-values of the two intercepting points
	#; z - the z-values of the two intercepting points

	# initialise variables with junk
	xlat = -999
	xlon = -999
	zen = -999
	if iflag:
		# find closest point to observer. LOS is not neccesarily along the x-axis but if we assume we 
		# are on the "right hand side" of the graph, then we just want the point with the larger x-value
		dx = (x-x0)/alpha
		inear = np.argmax(dx) # find closest point
		x1 = x[inear]
		y1 = y[inear]
		z1 = z[inear]
		
		# (x1,y1,z1) now contains the "intercept closest to observer" or ICO_point
		r = np.sqrt(x1**2 + y1**2 + z1**2) # find distance of ICOpoint from origin

		# theta is the angle between the x-axis and the origin->ICOpoint line
		theta = np.arccos(z1/r)
		xlat = 90 - theta/const.deg2rad
		# planetocentric latiude is the angle between the equatorial plane and a line joining
		# the ICOpoint to the origin

		# the component of orign->ICOpoint in the xy-plane is r*sin(theta), so the angle between the
		# x-axis and the ICOpoint is arccos(x1/r*sin(theta)) (c.f. spherical polars)
		cphi = x1/(r*np.sin(theta))
		cphi = np.clip(cphi, -1, 1)
		phi = np.arccos(cphi)
		if y1 < 0:
			phi*=-1
		xlon = phi/const.deg2rad

		v1 = np.array([x1/r, y1/r, z1/r]) # unit vector of ICOpoint
		#v1 = np.array([x1/Re, y1/Re, z1/Rp]) # unit vector of ICOpoint
		v2 = np.array([alpha, beta, gamma]) # unit vector defining direction of LOS

		zen = np.arccos(np.dot(v1,v2))/const.deg2rad
		# however, the zenith is the normal to the surface, v1 dot v2 != angle normal to the surface
		# possibly this is good enough when oblateness is small, but when it is large this will be wrong
		# shoud use v1 = 2*(x1/a^2, y1/a^2, z1/b^2) for correct answer, I think...
		# see https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid

	return(iflag, xlat, xlon, zen) 

def intercept_ellipse( a, b, alpha, beta, gamma, x0, y0, z0):
	"""
	procedure to find the intercepts (if any) between the line
	
	 (x-x0)       (y-y0)      (z-z0)
	 ------  =    ------   =  ------     [1]
	 alpha         beta       gamma
	
	and the ellipsoid
	
	
	 x^2    y^2    z^2   
	 --- +  --- +  ---  = 1              [2]
	 a^2    a^2    b^2
	
	Input variables
		a
			<real> ellipsoid semi-major axis
		b    	
			<real> ellipsoid semi-minor axis
		alpha	
			<real> line x-gradient
		beta 	
			<real> line y-gradient
		gamma	
			<real> line z-gradient
		x0	
			<real> line x-intercept
		y0	
			<real> line y-intercept
		z0	
			<real> line z-intercept
	
	Output variables
	  iflag	integer	Set to 1 if line intercepts, set to -1  otherwise
	  x(2)	real	x-intercepts
	  y(2)	real	y-intercepts
	  z(2)	real	z-intercepts
	
	Pat Irwin	11/2/07
	
	**********************************************************
	
	
	Breaking [1] into two equations gives
	
	(x-x0)/alpha = (y-y0)/beta          [3]
	
	(x-x0)/alpha = (z-z0)/gamma         [4]
	
	
	rearranging to make y and z in terms of x
	
	y = (beta/alpha)(x-x0) + y0         [5]
	
	z = (gamma/alpha)(x-x0) + z0        [6]
	
	
	substituting [5] and [6] into [2] gives a big
	long equation that can eventually be simplified
	down to a quadratic in x of the form
	
	a1 x^2 + b1 x + c1 = 0              [7]
	
	and can be solved by the quadratic formula, giving
	                _______________  
	               /               |
	    -b1 +/-  \/ b1^2 - 4 a1 c1
	x = ----------------------------    [8]
	               2 a1
	"""
	# The equations below give the values of the coefficients a1, b1, c1

	a1 = 1.0/a**2 + (beta/(a*alpha))**2 + (gamma/(b*alpha))**2

	b1 = (-2*x0*beta**2/alpha**2 + 2*beta*y0/alpha)/a**2
	b1 = b1 + (-2*x0*gamma**2/alpha**2 + 2*gamma*z0/alpha)/b**2

	c1 = ((beta*x0/alpha)**2 - 2*beta*y0*x0/alpha + y0**2)/a**2
	c1 = c1+ ((gamma*x0/alpha)**2 - 2*gamma*x0*z0/alpha + z0**2)/b**2 -1


	xtest = b1**2 - 4*a1*c1 # determinant of [7]

	# in the case of the determinant > 0 we can have two
	# real roots, therefore the line [1] intercepts the
	# ellipsoid [2] at two places
	x = np.zeros([2])
	y = np.zeros([2])
	z = np.zeros([2])

	# if the determinant of [7] is > 0, we have two real roots
	# therefore calculate the value of x using [8], and then
	# find y and z using [5] and [6]
	if(xtest >= 0.0):
		iflag=True
		x[0]=(-b1 + np.sqrt(xtest))/(2*a1) # positive root
		x[1]=(-b1 - np.sqrt(xtest))/(2*a1) # negative root

		y = y0 + (beta/alpha)*(x-x0) # [5]
		z = z0 + (gamma/alpha)*(x-x0)# [6]

		# here, x, y, z have two components each.
		# each of them has a value for the positive root (in index 0)
		# and the negative root (in index 1)

		test = (x/a)**2 + (y/a)**2 + (z/b)**2 # calcuate [2] with our found numbers
		# [2] should always equal 1, so subtract 1 to find the error
		xtest1 = np.abs(test[0]-1.0)
		xtest2 = np.abs(test[1]-1.0)

		err = 1e-5 # deterimine our error threshold
		if(xtest1 > err or xtest2 > err):
			# if the error is larger than our threshold, inform the user
			print('Problem in interceptellip - solution not on ellipsoid')
			print(f'test = {test}')
	else:
		iflag=False # if we don't have two real roots, we don't intercept the ellipsoid

	return(iflag, x,y,z)

def ellipse_top_hat_as(shape, x0, y0, a, b, t, error=1E-3):
	y, x = np.mgrid[:shape[0], :shape[1]]
	return(ellipse_top_hat_2D((y,x), y0, x0, b, a, t, error).reshape(shape))
		
def ellipse_top_hat_2D(pos, x0=0, y0=0, a=1, b=1, t=0, error=1E-3):
	"""
	pos - x, y positions on a 2d grid to evaluate ellipse at
	x0 - x position of center of ellipse
	y0 - y position of center of ellipse
	a - semi-major axis
	b - semi-minor axis
	t - angle of rotation
	see https://en.wikipedia.org/wiki/Ellipse#General_ellipse
	"""
	x, y = pos
	A = (a*np.sin(t))**2 + (b*np.cos(t))**2
	B = 2*(b**2-a**2)*np.sin(t)*np.cos(t)
	C = (a*np.cos(t))**2 + (b*np.sin(t))**2
	D = -2*A*x0 - B*y0
	E = -B*x0 - 2*C*y0
	F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
	
	bg = np.zeros_like(x)
	center_value = A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F 
	#print(f'center_value {center_value}')
	#print(f'F {F}')
	ellipse_idxs = np.where(A*x**2 + B*x*y + C*y**2 + D*x + E*y <= -(1-error)*F)
	bg[ellipse_idxs] = 1.0
	return(bg.flatten())

def fit_ellipse(data, params=[0,0,1,1,0], fixed=[False]*5):
	"""
	DOES NOT WORK
	"""
	nx, ny = data.shape
	x, y = np.mgrid[:nx, :ny]
	variables = [p  for p,f in zip(params, fixed) if not f]
	constants = [p for p,f in zip(params, fixed) if f]
	#print(f'variables {variables}')
	#print(f'constants {constants}')
	popt, pconv = spo.curve_fit(ellipse_top_hat_2D, (x,y), np.nan_to_num(data, copy=True).ravel(), p0=variables)
	popt_full = []
	i=0
	j=0
	for f in fixed: 
		if f:
			popt_full.append(constants[i])
			i+=1
		else:
			popt_full.append(popt[j])
			j+=1
	#print(f'popt_full {popt_full}')
	return(popt_full, pconv)

def apply_xtellcor_tellspec(hdul, wavgrid, airmass, cal_od_per_am, pf, show_plots=True):
	logging.info('Applying xtellcor tellspec')
	print(cal_od_per_am)
	print(pf)
	print(wavgrid)
	print(wavgrid.shape)
	hft = interpolate_h_filter_to_wavgrid(wavgrid)
	print(hft)
	odepth = cal_od_per_am*airmass
	trans = np.exp(-odepth)
	polyline = polynomial(pf, wavgrid)
	spec = trans*polyline
	tellmean = np.nansum(spec*hft)/np.nansum(hft)
	print(f'tellmean {tellmean}')

	normspec = np.nan_to_num(spec/tellmean, nan=1.0)
	recip_normspec = 1.0/normspec

	if show_plots:
		nz,ny,nx = hdul[0].data.shape
		before_tellspec_norm = hdul[0].data[:,int(ny/2),int(nx/2)].copy()

	hdul[0].data *= recip_normspec[:,None,None] #1.0/normspec[:,None,None]

	if show_plots:		
		f1 = plt.figure(figsize=[_x/2.5 for _x in (48,24)])
		a11 = f1.add_subplot(2,4,1)
		a11.plot(wavgrid, before_tellspec_norm)
		a11.set_title(f"pixel ({int(nx/2)},{int(ny/2)}) before tellspec normalisation")

		a12 = f1.add_subplot(2,4,2)
		a12.plot(wavgrid, odepth, label='optical depth')
		a12.plot(wavgrid, trans, label='transmittance')
		a12.legend()

		a13 = f1.add_subplot(2,4,3)
		a13.plot(wavgrid, normspec)
		a13.set_title("normalised tellspec")

		a14 = f1.add_subplot(2,4,4)
		a14.plot(wavgrid, hdul[0].data[:,int(ny/2),int(nx/2)])
		a14.set_title(f"pixel ({int(nx/2)},{int(ny/2)}) after normalised by tellspec")

		a15 = f1.add_subplot(2,4,5)
		a15.plot(wavgrid, spec, label='transmittance*polyline')
		a15.plot(wavgrid, polyline, label='polyline')
		a15.legend()

		a16 = f1.add_subplot(2,4,6)
		a16.plot(wavgrid, recip_normspec, label='reciprocal of normspec')
		a16.legend()

		plt.show()

	return(hdul)

def polynomial(coeffs, xs):
	n_plus_1 = len(coeffs)
	poly = np.sum([coeffs[_i]*xs**(n_plus_1-_i-1) for _i in range(n_plus_1)], axis=0)
	logging.info('IN "polynomial()"')
	print(poly)
	print(poly.shape)
	return(poly)

def optical_depth_per_airmass_from_xtellcor(xtellcor_tellspec, airmass, transmittance_cutoff=0.99, xtellcor_A0V=None,
											 xtellcor=None, show_plots=True):
	"""
	Finds the optical depth per airmass from the telluric correction spectrum found using 'xtellcor_general.pro'
	
	ARGUMENTS:
		xtellcor_tellspec
			<str> Path to the telluric correction spectrum from 'xtellcor_general.pro'
		airmass
			<float> Airmass of the observation
		transmittance_cutoff
			<float> A value used to define '100% transmission' when calculating a simple polynomial fit to a '100% transmitted spectrum'
		show_plots
			<bool> If true will show plots for visualisation and debugging.

	RETURNS:
		optical_depth_per_airmass [nspec]
			<array,float> An array that details the optical depth per airmass for each wavelength in 'xtellcor_tellspec'
		polyfit [3]
			<array, float> An array of coefficients to a polynomial that define a simple '100% transmitted' spectrum
	"""
	xtellcor_tellspec_data = np.loadtxt(xtellcor_tellspec, comments='#')
	if type(xtellcor_A0V) != type(None): xtellcor_A0V_data = np.loadtxt(xtellcor_A0V, comments='#')
	if type(xtellcor) != type(None): xtellcor_data = np.loadtxt(xtellcor, comments='#')

	# xtellcor_general outputs the reciprocal of the corrected solar spectrum, so invert here
	recip_tellspec_data = xtellcor_tellspec_data.copy()
	recip_tellspec_data[:,1] = 1.0/recip_tellspec_data[:,1]

	# get a solar transmission spectrum for our wavelength range
	trans_spec_raw = np.loadtxt(os.path.expanduser('~/Documents/reference_data/telluric_features/mauna_kea/sky_transmission/mktrans_zm_50_20.dat'))
	trans_spec = np.zeros(xtellcor_tellspec_data.shape[:2])
	trans_spec[:,0] = xtellcor_tellspec_data[:,0]
	trans_spec[:,1] = np.interp(trans_spec[:,0], trans_spec_raw[:,0], trans_spec_raw[:,1])
	trans_idxs = np.where(trans_spec[:,1] > transmittance_cutoff) # get indices where transmittance is nearly 100%
	#print('trans idxs')
	#print(trans_idxs)
	#print('trans_spec[trans_idxs]')
	#print(trans_spec[trans_idxs,1][0])
	# assume for transmission close to 1 that the real spectrum is equal to tellspec/transmittance
	real_spec = trans_spec.copy()
	real_spec[:,1] = 1.0/(real_spec[:,1])*recip_tellspec_data[:,1]
	real_spec = real_spec[trans_idxs]
	# fit a 2nd order polynomial to the 'real' spectrum
	polyfit = np.polyfit(real_spec[:,0], real_spec[:,1], 2)
	print(polyfit)
	# shift the fitted polynomial so it is just touching the tips of the fitted data
	polyline_small = polyfit[0]*real_spec[:,0]**2 + polyfit[1]*real_spec[:,0] + polyfit[2]
	polymax = np.nanmax(real_spec[:,1]/polyline_small)
	polyline = ((trans_spec[:,0]**2)*polyfit[0] + trans_spec[:,0]*polyfit[1] + polyfit[2])*polymax
	print(polyline)
	flattened_spec = recip_tellspec_data.copy()
	flattened_spec[:,1] = flattened_spec[:,1]/polyline # de-trend the tellspec data using the polynomial

	# by now we are at the equivalent point of line 177 in 'sub_scale_airmass_sinf.pro'
	# "flattened_spec" is the equivalent of "trans", "polyline" is the equivalent of "gain",
	# "self.recip_tellspec_data" is the equivalent of "spec"

	odepth = -np.log(flattened_spec[:,1])

	hfilter = flattened_spec.copy()
	hfilter[:,1] = np.interp(	hfilter[:,0], 
								fitscube.process.cfg.H_filter_transmittance[:,1], 
								fitscube.process.cfg.H_filter_transmittance[:,2]
							)

	odepth_per_airmass = odepth/airmass

	print(xtellcor_tellspec_data)
	######### plot for visualisation #############
	if show_plots:
		plot_x = 4
		plot_y = 2
		plot_cm_size = 12
		f1 = plt.figure(figsize=[_x/2.54 for _x in (plot_x*plot_cm_size, plot_y*plot_cm_size)])

		a11 = f1.add_subplot(plot_y,plot_x,1)
		a11.set_title(os.path.basename(xtellcor_tellspec))
		a11.plot(xtellcor_tellspec_data[:,0], xtellcor_tellspec_data[:,1])
		if type(xtellcor_A0V) != type(None):
			a12 = f1.add_subplot(plot_y,plot_x,2)
			a12.plot(xtellcor_A0V_data[:,0], xtellcor_A0V_data[:,1])
			a12.set_title(os.path.basename(xtellcor_A0V))
		if type(xtellcor) != type(None):
			a13 = f1.add_subplot(plot_y,plot_x,3)
			a13.plot(xtellcor_data[:,0], xtellcor_data[:,1])
			a13.set_title(os.path.basename(xtellcor))

		a14 = f1.add_subplot(plot_y,plot_x,4)
		a14.set_title('reciprocal of tellspec')
		a14_1 = a14.twinx()
		a14_1.plot(trans_spec[trans_idxs,0][0], trans_spec[trans_idxs,1][0], color='tab:orange', lw=1,zorder=1)
		a14_1.set_ylim([0,1])
		a14.plot(recip_tellspec_data[:,0], recip_tellspec_data[:,1], color='tab:blue', lw=1, zorder=0)
		a14.plot(real_spec[:,0], real_spec[:,1], color='tab:red', lw=1, zorder=2)
		a14.plot(recip_tellspec_data[:,0], polyline, lw=1, color='tab:green', zorder=3)

		a15 = f1.add_subplot(plot_y,plot_x,5)
		a15.plot(flattened_spec[:,0], flattened_spec[:,1], lw=1, color='tab:purple')
		a15.set_title('flattened reciprocal of tellspec')
		
		a16 = f1.add_subplot(plot_y,plot_x,6)
		a16.plot(hfilter[:,0], hfilter[:,1], lw=1, color='tab:blue')
		a16.set_title('H-filter')
		plt.show()
	########### end plotting ##################
	return(odepth_per_airmass, polyfit)

def get_counts_per_sec_in_h_filter2spectral_radiance(data, wavgrid, exposure_time, h_mag, px_ster):
	"""
	Gets a factor that converts from counts per second in the H-filter to spectral radiance
	"""
	hft = interpolate_h_filter_to_wavgrid(wavgrid) # transmittance of H-filter at data wavelengths
	h_band_data_counts = np.nansum(data*hft[:,None,None])/np.nansum(hft) # avg counts in H-band
	h_band_data_counts_per_sec = h_band_data_counts/exposure_time # avg counts s-1 in H-band
	h_band_lit_spectral_radiance = fitscube.process.cfg.hmag2spflux(h_mag) # convert magnitudes (from literature) to flux 
	h_band_lit_spectral_radiance_per_ster = h_band_lit_spectral_radiance/px_ster # W cm-2 um -1 ster-1 from literature
	counts2spectralradiance = h_band_lit_spectral_radiance_per_ster/h_band_data_counts_per_sec # W cm-2 um -1 ster-1 counts-1 seconds-1
	return(counts2spectralradiance)

def interpolate_h_filter_to_wavgrid(wavgrid):
	hft = fitscube.process.cfg.H_filter_transmittance
	return(np.interp(wavgrid, hft[:,1], hft[:,2]))

def find_star_psf_seeing_background_full(data3d, px_scale_arcsec, show_plots=True, interpolate_psf_n=100, use_simplified_psf_func=False, save_plots=False, plot_dir='./plots', calibrator_name='calibrator'):
	# TODO; This is failing for some psf observations, it could possibly be a x,y axis swap problem
	#       Probably need to put in some plots so I can see what's going on, also see .../psf.py
	logging.info("Fitting a function to the calibrator star's PSF")
	nz,ny,nx = data3d.shape
	psf_adjusted = np.full_like(data3d, np.nan)
	seeing = np.full((nz), np.nan)
	starpos = np.full((nz,2), np.nan)
	background = np.full((nz), np.nan)

	# calculate some useful starting points
	data_median = np.nanmedian(data3d, axis=0)
	s_median = np.nansum(data_median)
	argmax_median = np.unravel_index(np.nanargmax(data_median), data_median.shape)
	
	# define function to fit to point spread function
	psf_func = psf.mofgaus_8 # function definition
	psf_func_name = 'mofgaus_8' # name of function
	popt_names = 	('atot',      'fg',       'cx',            'cy',              's',    'a',    'beta',    'const')   # function argument names (try to be fairly descriptive)
	p0 = 			(s_median,    0.7,        argmax_median[0], argmax_median[1], 1,      6,      2,      0)         # initial guess at argument values, will get better guesses later
	ubounds = 		(10*s_median, 0.99,       nx,               ny,               nx,     ny,     100,       s_median)  # upper bounds on argument values
	lbounds = 		(0,           0.01,       0,                0,                0,      0,      2,         -s_median) # lower bounds on argument values
	clamp_after_est=(False,       True,       False,            False,            True,   True,   True,      True)      # which arguments should we 'clamp' (not let vary) after we have estimated their values?
	
	# keyword arguments to scipy.curve_fit
	sp_curvefit_kwargs = {
		'loss':'linear', 
		'f_scale':1E-4, # no effect when 'loss' is 'linear' 
		'bounds':(lbounds, ubounds), 
		'x_scale':'jac',
		'jac':'2-point'
	}

	# get initial estimate by fitting to the median
	popt_median, pconv_median, func_eval_median = psf.fit_func_to(psf_func, data_median, *p0, **sp_curvefit_kwargs)
	print(f'popt_median [{", ".join([f"{x:#0.3E}" for x in popt_median])}]')

	### PLOTTING ###
	if show_plots:
		logging.info(f' Plotting initial median point spread function estimate metrics')
		# plot the initial guess that uses the data median
		f1 = doplot_psf_metrics(data_median, 'Median psf data', func_eval_median, 'psf_model', popt_median, p0, (lbounds,ubounds), popt_names, psf_func_name)
		plotutils.save_show_plt(f1, '_'.join(['median_cal', f'{calibrator_name}_{psf_func_name}_psf'])+'.png', outfolder=plot_dir, show_plot=True, save_plot=save_plots)
	### -------- ###

	# clamp psf parameters as we now have a good estimate of them
	bounds_median = psf.clamp_param(popt_median, sp_curvefit_kwargs['bounds'], clamp_after_est, verbose=False)
	sp_curvefit_kwargs['bounds'] = bounds_median

	# compute some useful results
	x, y = np.mgrid[:nx, :ny]                               # positions to evaluate psf at
	s = np.nansum(data3d, axis=(1,2))
	argmax = [np.unravel_index(np.nanargmax(data3d[i,:,:]), data3d.shape[1:]) if not np.isnan(data3d[i,:,:]).all() else (np.nan,np.nan) for i in range(nz)]

	# create holders for psf data
	psf_cube = np.full_like(data3d, np.nan)                 # holds psf evaluation
	popt_cube = np.full((nz, *popt_median.shape), np.nan)   # holds parameters to psf for each wavelength
	pconv_cube = np.full((nz, *pconv_median.shape), np.nan) # holds covariance matrix of psf for each wavelength
	popt_adjusted = np.full((nz, *popt_median.shape), np.nan) # holds centered and background-zeroed version of popt for each wavelength
	popt_est_cube = np.zeros((nz, *popt_median.shape))      # holds initial 'estimated' parameters to psf for each wavelength


	if interpolate_psf_n == 0:
		# define simplified function if we want to use it
		if use_simplified_psf_func:
			param_idxs = (0, 2, 3)
			psf_func_simp = lambda pos, atot, cx, cy: psf_func(pos, atot, popt_median[1], cx, cy, *popt_median[4:])

		print('Beginning psf fit')
		for i in range(nz):
			if np.isnan(data3d[i,:,:]).all(): # if everything in the wavelength slice is NAN, then just use median as estimate
				popt_est_cube[i,:] = popt_median
				sp_curvefit_kwargs['bounds'] = bounds_median
			else: # if we have some data, calculate some good guesses
				cy, cx = argmax[i][0], argmax[i][1]
				if psf_func_name in ('mofgaus_8',): # if we have a bespoke method for estimating the parameters for this psf, then do so
					popt_est_cube[i,:] = (s[i], popt_median[1], cx, cy, *popt_median[4:])
					clamps = list(clamp_after_est)
					#clamps[0] = True # fix 'atot' parameter to always be the total counts in the wavelength slice
					bounds_new = bounds_median[:] #  copy previous bounds
					bounds_new[0][0] = 0.5*s[i] # change 'atot' lower bound
					bounds_new[1][0] = 2*s[i] # change 'atot' upper bound
					bounds_new = psf.clamp_param(popt_est_cube[i], bounds_new, clamps, verbose=False)
					sp_curvefit_kwargs['bounds'] = bounds_new
				else: # if we don't have a bespoke method for estimating parameters of psf, fall back to using median
					popt_est_cube[i,:] = popt_median
					sp_curvefit_kwargs['bounds'] = bounds_median

			if use_simplified_psf_func:
				popt_est = (popt_est_cube[i][0], popt_est_cube[i][2], popt_est_cube[i][3])
				sp_curvefit_kwargs['bounds'] = [[a[j] for j in param_idxs] for a in sp_curvefit_kwargs['bounds']]
				#print(popt_est)
				#print(sp_curvefit_kwargs['bounds'])

			# print progress message
			plotutils.progress(i, nz, message=f'popt_est [{", ".join([f"{x:#0.3E}" for x in popt_est_cube[i,:]])}]')

			# perform curve fitting
			try:
				if use_simplified_psf_func:
					popt_small, pconv_small, psf_cube[i,:,:] = psf.fit_func_to(psf_func_simp, data3d[i,:,:], *popt_est,
																				**sp_curvefit_kwargs)
					popt_cube[i,:] = popt_est_cube[i]
					for k, j in enumerate(param_idxs): popt_cube[i,j] = popt_small[k]
				else:
					popt_cube[i,:], pconv_cube[i,:], psf_cube[i,:,:] = psf.fit_func_to(psf_func, data3d[i,:,:],
																						*popt_est_cube[i],
																						**sp_curvefit_kwargs)
					
			except RuntimeError:
				print('WARNING: Could not fit a psf to this wavelength bin')
				pass # do nothing if scipy.optimize.curve_fit could not find a solution, we already have default values set to NAN

	else: # we are interpolating the psf, i.e. we can assume it is a slowly varying funciton of wavelength
		# find where we have >80% of pixels not be NAN, and we have a decent amount of signal
		npix = ny*nx
		nnan = np.count_nonzero(np.isnan(data3d), axis=(1,2))
		nan_frac = nnan/npix
		sig_sum = np.nansum(data3d, axis=(1,2))
		max_sig_frac = sig_sum/np.nanmax(sig_sum) # low signal makes the fit difficult as there is too much noise, therefore choose a minimum signal level
		#print('nan_frac', nan_frac)

		# choosing wavelength slices that have <20% NAN data, and that have at least 20% of maximum signal
		ok_idxs = np.nonzero((nan_frac < 0.2)*(max_sig_frac > 0.2))[0] # always 1D so can do this no problem
		#print(ok_idxs)
		nelem = interpolate_psf_n if len(ok_idxs)>interpolate_psf_n else len(ok_idxs)
		interp_idxs = ok_idxs[np.round(np.linspace(0, len(ok_idxs)-1, nelem)).astype(int)]
		logging.info(f' Interpolating psf at wavelength slices {interp_idxs}')

		# find psf parameters at interpolation points
		for i in interp_idxs:
			s_x = np.nansum(data3d[i]*x)/s[i]
			s_y = np.nansum(data3d[i]*y)/s[i]
			cy, cx = s_y, s_x #argmax[i][0], argmax[i][1]
			# create estimates of psf parameters specifically for each psf function used
			if psf_func_name in ('mofgaus_8',):
				popt_est_cube[i,:] = (s[i], popt_median[1], cx, cy, *popt_median[4:])
				clamps = list(clamp_after_est)
				#clamps[0] = True # fix 'atot' parameter to always be the total counts in the wavelength slice
				bounds_new = bounds_median[:] #  copy previous bounds
				bounds_new[0][0] = 0.5*s[i] # change 'atot' lower bound
				bounds_new[1][0] = 2*s[i] # change 'atot' upper bound
				bounds_new = psf.clamp_param(popt_est_cube[i], bounds_new, clamps, verbose=False)
				sp_curvefit_kwargs['bounds'] = bounds_new
			else: # if we don't have a bespoke method for estimating parameters of psf, fall back to using median
				popt_est_cube[i,:] = popt_median
				sp_curvefit_kwargs['bounds'] = bounds_median
			# print progress message
			plotutils.progress(i, nz, message=f'popt_est [{", ".join([f"{x:#0.3E}" for x in popt_est_cube[i,:]])}]')

			# perform curve fitting
			try:
				popt_cube[i,:], pconv_cube[i,:], psf_cube[i,:,:] = psf.fit_func_to(psf_func, data3d[i,:,:], *popt_est_cube[i], 
																				   **sp_curvefit_kwargs)	
			except RuntimeError:
				print(f'WARNING: Could not fit a psf to wavelength slice {i}') # do nothing if scipy.optimize.curve_fit could not find a solution, we already have default values set to NAN

		print('') # print a new line because the progress counter won't (as we technically don't reach the end because we are interpolating and extrapolating
		# interpolate (and extrapolate) results over nans
		for j in range(popt_cube.shape[1]):
			popt_cube[:,j] = np.interp(np.arange(0, nz), interp_idxs, popt_cube[interp_idxs,j])
			popt_est_cube[:,j] = np.interp(np.arange(0, nz), interp_idxs, popt_est_cube[interp_idxs,j])

		# do some extra processing if we can to make the results a bit more exact
		if psf_func_name in ('mofgaus_8',):
			for i in range(nz):
				popt_cube[i, 0] = s[i] # set the 'atot' parameter to the sum of the wavelength slice, as 'atot' is the total signal in the PSF

		#print(popt_cube)

		# calculate full psf based on interpolated parameters
		for i in range(nz):
			psf_cube[i,:,:] = psf.func_as(psf_func, data3d.shape[1:], *popt_cube[i])

		#sys.exit() # DEBUGGING
		

	# calculate and store residuals
	residual_cube = ((data3d - psf_cube)**2)/(data3d**2)


	logging.info(' Finding centered and background zeroed version of psf')
	for i in range(nz):
		# try to estimate seeing, background, star position, etc. based on fitted parameters
		pd = dict(zip(popt_names, popt_cube[i]))
		if psf_func_name in ('mofgaus_8',):
			# 3 'standard deviations', where sd is estimated using a weighiting of 's' for the gaussian part, and 'a' for the moffat part
			# the weighting is based off of 'fg' the fraction of the total psf that is made up of the gaussian
			seeing[i] = 3*np.sqrt(pd['fg']*pd['s']**2 + (1-pd['fg'])*pd['a']**2)*px_scale_arcsec
			starpos[i] = (pd['cx']+1, pd['cy']+1) # fortran uses 1-based indexing
			background[i] = pd['const']/(nx*ny) # the constant portion of the fit for each pixel
			# adjust 'pd' so that we can have a centered psf with zero background
			pd['cx'] = nx/2 # want to be in the 'centre' of the image, for python the 'n.0' point of a pixel is the bottom left, and 'n.5' is the middle so this should work ok
			pd['cy'] = ny/2
			pd['const'] = 0
			popt_adjusted[i,:] = [pd[name] for name in popt_names] 
			psf_adjusted[i,:,:] = psf_func((x,y), *popt_adjusted[i]).reshape((ny,nx))
		else:
			print(f'ERROR: No prescription present for getting seeing, star_pos, background, from psf function {psf_func_name}')
			# default values are already NANs

	### Plotting ###
	if show_plots:
		logging.info(f' Plotting point spread function metrics')
		
		# now plot the median of all the wavelength dependent psfs
		psf_cube_median = np.nanmedian(psf_cube, axis=0)
		popt_cube_median = np.nanmedian(popt_cube, axis=0)
		popt_est_cube_median = np.nanmedian(popt_est_cube, axis=0)
		f2 = doplot_psf_metrics(data_median, 'median psf data', psf_cube_median, 'median psf model', popt_cube_median, popt_est_cube_median, sp_curvefit_kwargs['bounds'], popt_names, psf_func_name)
		plotutils.save_show_plt(f2, '_'.join(['median_cal', f'{calibrator_name}_{psf_func_name}_psf_median'])+'_stacked.png', outfolder=plot_dir, show_plot=True, save_plot=save_plots)

		# now plot the average of all wavelength dependent plots		
		psf_cube_mean = np.nanmean(psf_cube, axis=0)
		popt_cube_mean = np.nanmean(popt_cube, axis=0)
		popt_est_cube_mean = np.nanmean(popt_est_cube, axis=0)
		f3 = doplot_psf_metrics(np.nanmean(data3d, axis=0), 'mean psf data', psf_cube_mean, 'mean psf model', popt_cube_mean, popt_est_cube_mean, sp_curvefit_kwargs['bounds'], popt_names, psf_func_name)
		plotutils.save_show_plt(f3, '_'.join(['median_cal', f'{calibrator_name}_{psf_func_name}_psf_mean'])+'_stacked.png', outfolder=plot_dir, show_plot=True, save_plot=save_plots)
		
	### -------- ###

	return(psf_adjusted, np.nanmean(seeing), starpos, background, psf_func, popt_adjusted, psf_func_name, popt_names, psf_cube, popt_cube, pconv_cube, popt_est_cube, residual_cube)


def doplot_psf_metrics(psf_data, psf_data_label, psf_model, psf_model_label, popt_values, popt_est_values, popt_bounds, popt_names, psf_func_name):
	import io
	import pretty_table as pt
	f1, a1 = plot_psf_model(psf_data, psf_data_label, psf_model, psf_model_label, scale_func = np.log, scale_func_label='log')
	psf.plot_psf_components(a1[1,2], psf_model, popt_values, popt_names, psf_func_name, ssfac=1)
	a1[1,2].set_title(f'Marginalisation of {psf_func_name}_psf over x')
	a1[1,2].set_ylim(a1[1,1].get_ylim())
	astr = io.StringIO()
	tbl_struct = {
		'PARAMETERS':{
			'header':(('param', 'estimate', 'lbound', 'ubound', 'value'),),
			'data':[f'{n} {e:#0.3E} {l:#0.3E} {u:#0.3E} {p:#0.3E}'.split(' ') for n, e, l, u, p in zip(popt_names, popt_est_values, *popt_bounds, popt_values)]
		}
	}
	pt.write(tbl_struct, astr)
	f1.text(0.02,0.05, astr.getvalue())
	astr.close()
	return(f1)

def plot_psf_model(d, ld, p, lp, r=None, lr=None, scale_func=lambda x: x, scale_func_label='', f1=None, a1=None, cax=None, xline=None, yline=None):
		"""
			d
				data
			ld
				label for data
			p
				psf fit
			lp
				label for psf fit
			r
				residual
			lr
				label for residual
		"""
		marginalise_function = lambda x, axis=None: np.nansum(x, axis=axis)
		if r is None:
			r = (d - p)**2/d**2 
		if lr is None:
			lr = f'({ld} - {lp})^2/{ld}^2'

		d_x, p_x, r_x = [marginalise_function(x, axis=0) for x in (d, p, r)]
		d_y, p_y, r_y = [marginalise_function(y, axis=1) for y in (d, p, r)]

		d_sum = np.nansum(d)
		p_sum = np.nansum(p)
		r_sum = np.nansum(r)
		#print(f'sum({ld})', d_sum)
		#print(f'sum({lp})', p_sum)
		#print(f'sum({lr})', r_sum)

		#x = lambda x: x
		f , lf= scale_func, scale_func_label
		#f, lf = x, ''
		d, p, r = map(f, [d,p,r])

		nr, nc = 2,5
		if f1 is None:
			f1 = plt.figure(figsize=[_x/2.54 for _x in (nc*12, nr*12)])
		if a1 is None:
			a1 = f1.subplots(nrows=nr, ncols=nc, squeeze=False, gridspec_kw={'hspace':0.25, 'wspace':0.25})
		if cax is None:
			cax = f1.add_axes([0.29, 0.515, 0.30, 0.02]) # [l, b, w, h]
			cax2 = f1.add_axes([0.6, 0.515, 0.13, 0.02])

		vmin, vmax = np.nanmin(d[~np.isnan(d)]), np.nanmax(d[~np.isnan(d)])

		im11 = a1[0,1].imshow(d, origin='lower', vmin=vmin, vmax=vmax)
		#f1.colorbar(im11, ax=a1[0,1])
		f1.colorbar(im11, cax=cax, orientation='horizontal')
		a1[0,1].set_title(f'{lf}({ld})\nsum {d_sum}')

		im12 = a1[0,2].imshow(p, origin='lower', vmin=vmin, vmax=vmax)
		#f1.colorbar(im12, ax=a1[0,2])
		a1[0,2].set_title(f'{lf}({lp})\nsum {p_sum}')

		im13 = a1[0,3].imshow(r, origin='lower')#, vmin=vmin, vmax=vmax)
		f1.colorbar(im13, cax=cax2, orientation='horizontal')
		a1[0,3].set_title(f'{lf}({lr})\nsum {r_sum}')
		
		l11 = a1[1,1].plot(d_x, label=ld)
		a1[1,1].set_title(f'Marginalisation over x of\n{ld} and {lp}')
		l11_ylim = a1[1,1].get_ylim()
		l12 = a1[1,1].plot(p_x, label=lp)
		a1[1,1].set_ylim(l11_ylim)
		if not (yline is None): # x and y swapped because of fortran ordering
			a1[1,1].axvline(yline, color='green', label='initial_cy')
		a1[1,1].legend(loc='upper right')

		l000 = a1[0,0].plot(d_y, range(len(d_y)), label=ld)
		l001 = a1[0,0].plot(p_y, range(len(p_y)), label=lp)
		a1[0,0].set_xlim(l11_ylim[::-1])
		a1[0,0].set_title(f'Marginalisation over y of\n{ld} and {lp}')
		if not (xline is None): # x and y swapped because of fortran ordering
			a1[0,0].axhline(xline, color='green', label='initial_cx')
		a1[0,0].legend(loc='upper left')

		l10 = a1[0,4].plot(r_y, range(len(r_y)))
		a1[0,4].set_title(f'Marginalisation over y of\n{lr}')
		#a1[0,4].set_xlim(l11_ylim)

		#l12 = a1[1,2].plot(p_x)
		#a1[1,2].set_title(f'Marginalisation of {lp} over x')
		#a1[1,2].set_ylim(l11_ylim)

		l13 = a1[1,3].plot(r_x)
		a1[1,3].set_title(f'Marginalisation over x of\n{lr}')
		#a1[1,3].set_ylim(l11_ylim)

		a1[1,4].remove()
		a1[1,0].remove()

		return(f1, a1)


def find_star_psf_seeing_background_median(data3d, px_scale_arcsec, show_plots=True):
	logging.info('Fitting 2D gaussian')
	data = np.nanmedian(data3d,axis=0)
	#data = data3d[300,:,:]
	xy = np.unravel_index(np.nanargmax(data), data.shape)
	amp = data[xy]
	popt, pcov = fit_gaussian(data, amp, xy[0], xy[1], 1, 1, 0, 0)
	amp, x, y, sx, sy, t, o = popt # options for fitted gaussian
	#fitted_gaussian = gaussian_as(data.shape, *popt)
	seeing = 3*np.sqrt(sx*sx + sy*sy)*px_scale_arcsec # use 3 standard deviations as resolution element, in arcseconds
	starpos = (y+1,x+1) # fits uses fortran style ordering, and 1-based indexing
	background = o
	psf = gaussian_as(data.shape,amp,data.shape[0]/2,data.shape[1]/2,sx,sy,t,0)
	psf_sum = np.sum(psf)
	psf /= psf_sum
	def point_spread_function(i, j, shape, opts):
		return(gaussian_as(shape, opts[0], i, j, *opts[1:]))
	psf_func = point_spread_function
	psf_opts = (amp/psf_sum, sx,sy,t,0)
	
	if show_plots:
		nx, ny = (2,2)
		f1 = plt.figure(figsize=[x/2.54 for x in (10*ny,10*nx)])
		a11 = f1.add_subplot(ny,nx,1)
		im11 = a11.imshow(psf, origin='lower')
		f1.colorbar(im11,ax=a11)
		a11.set_title('psf, seeing {:0.2}",\nsx {:0.2}" sy {:0.2}"'.format(seeing, sx*px_scale_arcsec, sy*px_scale_arcsec))

		a12 = f1.add_subplot(ny,nx,2)
		im12 = a12.imshow(data-gaussian_as(data.shape, *popt), origin='lower')
		#im12 = a12.imshow(data - fitted_gaussian, origin='lower')
		f1.colorbar(im12,ax=a12)
		a12.set_title('residual')

		a13 = f1.add_subplot(ny,nx,3)
		a13.plot(range(data3d.shape[0]), data3d[:,30,30], label='cal_data[:,30,30]')
		a13.axhline(background, color='tab:orange', label='background from fit of median')
		a13.legend()

		a14 = f1.add_subplot(ny,nx,4)
		im14 = a14.imshow(data, origin='lower')
		# Have to minus 1 from star postions as python works on 0-based indexing
		a14.scatter(starpos[0]-1,starpos[1]-1, label='star_center', marker='o', edgecolor='tab:orange', lw=1, facecolor='none')
		f1.colorbar(im14,ax=a14)
		a14.legend()

		plt.show()
		
	return(psf, seeing, starpos, background, psf_func, psf_opts)

def gaussian_as(shape, amp=1, mx=0, my=0, sx=1, sy=1, t=0, o=0):
	nx,ny = shape
	x,y = np.mgrid[:nx,:ny]
	return(gaussian_2d((x,y), amp, mx, my, sx, sy, t, o).reshape(*shape))

def fit_gaussian(data, amp=1, mx=0, my=0, sx=1, sy=1, t=0, o=0):
	#x = np.linspace(0, data.shape[0]-1, data.shape[0])
	#y = np.linspace(0, data.shape[0]-1, data.shape[0])
	#x,y = np.meshgrid(x,y)
	#print(x)
	nx,ny = data.shape
	x,y=np.mgrid[:nx,:ny]
	popt, pcov = spo.curve_fit(gaussian_2d, (x,y), np.nan_to_num(data, copy=True).ravel(), p0=(amp, mx, my, sx, sy, t, o))
	return(popt, pcov)

def gaussian_2d(pos,amp, mx, my, s11, s22, t, o):
	"""
	pos - position to evaluate gaussian at (x,y)
	amp - amplitude of gaussian
	mx  - mean in x direction
	my  - mean in y direction
	s11 - standard deviation in x direction
	s22 - standard deviation in y direction
	t   - angle of rotation of the gaussian (theta)
	o   - offset to base of gaussian (offset)
	See https://en.wikipedia.org/wiki/Gaussian_function
	"""
	x, y = pos
	a = (np.cos(t)**2)/(2*(s11**2)) + (np.sin(t)**2)/(2*(s22**2))
	b = -np.sin(2*t)/(4*(s11**2)) + np.sin(2*t)/(4*(s22**2))
	c = (np.sin(t)**2)/(2*(s11**2)) + (np.cos(t)**2)/(2*(s22**2))
	g = amp*np.exp(-(a*(x-mx)**2 + 2*b*(x-mx)*(y-my) + c*(y-my)**2)) + o
	return(g.ravel())
			
def get_datacube_attributes(hdul):
	attrs = {}
	attrs['origin_observatory'] = hdul[0].header['ORIGIN'].lower()
	attrs['object'] = hdul[0].header['ESO OBS TARG NAME'].lower()
	attrs['observation_date'] = hdul[0].header['DATE-OBS']
	attrs['airmass'] = 0.5*(hdul[0].header['ESO TEL AIRM START'] 
	   						+ hdul[0].header['ESO TEL AIRM END'])
	attrs['exposure_time'] = hdul[0].header['EXPTIME']
	attrs['px_scale_deg'] = np.abs(hdul[0].header['CDELT1'])
	attrs['px_scale_arcsec'] = np.abs(hdul[0].header['CDELT1'])*const.deg2arcsec
	attrs['px_ster'] = (attrs['px_scale_deg']**2)*const.degSq2ster
	attrs['wavgrid'] = datacube_wavelength_grid(hdul[0])
	return(attrs)

def datacube_wavelength_grid(hdu):
	i = 3
	naxis_i = hdu.header['NAXIS{}'.format(i)]
	# assume spectral axis is 3rd axis
	p = np.linspace(1, naxis_i, num=naxis_i, dtype=int) # do FITS files use 0 or 1 indexing? I think 1.
	#print(p)
	#assume 1d
	ltmi_i = float(hdu.header.get('LTM{}_{}'.format(i,i),1))
	ltvi = float(hdu.header.get('LTV{}'.format(i), 0))
	l = ltmi_i*p + ltvi
	#print(l)

	crpix_i = int(hdu.header['CRPIX{}'.format(i)])
	crval_i = float(hdu.header['CRVAL{}'.format(i)])
	cdelt_i = float(hdu.header['CDELT{}'.format(i)])
	cdi_i = float(hdu.header.get('CD{}_{}'.format(i,i),cdelt_i))
	#print(crpix_i, crval_i, cdelt_i, cdi_i)
	#w = np.zeros([naxis1])
	w = crval_i + cdi_i*(l-crpix_i)
	return(w)

def query_jplhorizons_eph_to_ids(obj, origin_observatory, observation_date, fields, ids):
	"""
	Returns a dictionary of fields selected from the jplhorizons database and renamed using list 'ids'
	""" 
	jpl_obj = jplh_query_object_at_instant(origin_observatory, obj, observation_date)
	jpl_eph = jpl_obj.ephemerides()	
	adict = {}
	for f, i in zip(fields, ids):
		adict[i] = jpl_eph[f][0]
	return(adict)

def jplh_query_object_at_instant(observer, target, date):
	tgt = fitscube.process.cfg.object_ids[target]
	obs_code = fitscube.process.cfg.observatory_codes[observer]
	t1 = Time(date, format='fits')
	#print(t1.mjd)
	return(Horizons(id=tgt[0], location=obs_code, epochs=[t1.jd], id_type=tgt[1]))

def vector_query_simbad_to_ids(objs, fields, sids, ids):
	"""
	Queries the SIMBAD archive about each object in list 'objs', and requests each field in list 'fields'.
	Then returns a dictionary populated with data from the archive, where each simbad id 'sid' is assigned
	a key in the returned dictonary. The keys are associated by position in the list 'ids'
	
	ARGUMENTS:
		objs	[n]
			<str> A list of object names understood by SIMBAD
		fields [m]
			<str> A list of field names understood by SIMBAD
		sids [l]
			<str> A list of data keywords understood by SIMBAD
		ids [l]
			<str> A list of keys to rename 'sids' to when repacked into a dictionary

	RETURNS:
		A dictionary containing the keys passed in 'ids'

	EXAMPLE:
		attribute_dict = vector_query_simbad_to_ids(['VEGA'], 
													['coordinates','fluxdata(V)'], 
													['RA','DEC','FLUX_V'], 
													['RightAscention','Declination','V_band_flux']
													)
	"""
	Simbad.add_votable_fields('typed_id')
	Simbad.remove_votable_fields('coordinates', 'main_id')
	Simbad.add_votable_fields(*fields)
	sbd_obj = Simbad.query_objects(objs)
	#print(type(sbd_obj))
	#sbd_obj = sbd_obj.filled('NAN')
	sbd_dict_list=[]
	for sbd in sbd_obj:
		sd = {}
		for s,i in zip(sids, ids):
			if np.ma.is_masked(sbd[s]):
				sd[i] = 'UNKNOWN'
			else:
				sd[i] = sbd[s]
		sbd_dict_list.append(sd)
	print(sbd_dict_list)
	return(sbd_dict_list)

def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	# A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	# see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	# of them do.
	# Quick reference:
	# ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	# ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	# ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	# ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	class RawDefaultTypeFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class RawDefaultFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap(__doc__), wrapsize=79),
									formatter_class = RawDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('target_cubes', type=str, nargs='+', help='data cubes to operate on', default=[])
	parser.add_argument('--rel_calibrators_file', type=str, help='Relative location (w.r.t target_cube) that points to a file that contains a list of fits files of calibration sources (standard stars)', default='./calibrators.txt')
	#parser.add_argument('--optional', help='A description of the optional argument')

	parser.add_argument('--rel_tellspec_folder', type=str, help='Relative location (w.r.t target_cube) that points to a folder that contains telluric spectrum files (created by xtellcor_general) that are named in the format "xtellcor_<STD_STAR_NAME_CAPS>_{A0V,adj,tellspec}.dat', default='./analysis/telluric_correction')

	parser.add_argument('--rel_output_folder', type=str, help='Relative location (w.r.t. target_cube) to output calibrated target cube to', default='./analysis')
	#parser.add_argument('--show_plots', action='store_true', help='If present, will show progress plots')
	parser.add_argument('--no_plots', action='store_false', dest='show_plots',help='If present, will not show progress plots')
	parser.add_argument('--autofind_disk', action='store_false', dest='find_disk_manually', help='If present will attempt to automatically find the disk (to nearest pixel)')

	parser.add_argument('--require_relative', type=str, nargs='*', help='Only include target cubes that have a file with this name relative to it', default=[])

	parser.add_argument('--calibrators_relative_fix', action='store_true', help='When transferring files, it is sometimes useful to treat part of a path as a different path because of filesystem structure differences.')
	
	parser.add_argument('--use_cached_psf', action='store_true', help='If present will look for a cached version of the psf (as specifiec by "--rel_cached_psf") to avoid calculation')
	parser.add_argument('--rel_cached_psf', type=str, default='./analysis/cached_psf.npz', help='Location (relative to target cube) to save psf properties to for caching, set to "" to disable.')

	parser.add_argument('--calibrator.recenter_using_psf', action='store_true', help='If present, will use the point-spread-function to recenter the calibrator')
	parser.add_argument('--target.recenter_using_psf', action='store_true', help='If present will use the PSF offset from the center of field to center the target (use with caution)')
	parser.add_argument('--calibrator.no_subtract_psf_background', action='store_false', dest='calibrator.subtract_psf_background', help='If present, will not subtract the PSF background from the calibrator')
	parser.add_argument('--target.subtract_psf_background', action='store_true', help='If present, will subtract the PSF background from the target')
	parser.add_argument('--calibrator.no_remove_psf_spectroscopic_drift', action='store_false', dest='calibrator.remove_psf_spectroscopic_drift', help='If present, will not subtract any spectroscopic drift detected in PSF modelling')
	parser.add_argument('--target.remove_psf_spectroscopic_drift', action='store_true', help='If present, will subtract any spectroscopic drift detected in PSF modelling (use with caution)')

	# parse the arguments
	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface

	# process arguments
	filtered_tcs = []
	for tc in parsed_args['target_cubes']:
		#reqs = [os.path.join(tc, rr) for rr in parsed_args['require_relative']]
		#print(reqs)
		reqs_exist = [os.path.exists(os.path.join(os.path.dirname(tc), rr)) for rr in parsed_args['require_relative']]
		if all(reqs_exist):
			filtered_tcs.append(os.path.abspath(os.path.normpath(tc)))
	parsed_args['target_cubes'] = filtered_tcs

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
