#!/usr/bin/env python3

import sys, os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import fitscube.header
import fitscube.process.sinfoni
import const
import scipy as sp
import scipy.signal

# assume that most normalising has been done

#file_associations = [	(	os.path.expanduser('~/scratch/transfer/Neptune_SINFONI_20131009_1_H_renorm.fits'), # observation
#							os.path.expanduser('~/scratch/transfer/09.STD.H.T0144.T0145.fits') # standard star
#						)
#					]

file_associations = [(os.path.expanduser("~/scratch/transfer/MUSE_OBS/V_BAND/DATACUBE_STD_0001.fits"), os.path.expanduser("~/scratch/transfer/MUSE_OBS/V_BAND/DATACUBE_FINAL.fits"))]


show_plots=True
find_psf_flag = True
std_hdu_idx = 1
obs_hdu_idx = 1

for obsf, stdf in file_associations:
	# alter Pat's cubes to my standards
	with fits.open(stdf) as std_hdul:
		print(std_hdul.info())
		# turn non-image pixels into NANs in the primary HDU
		# in Pat's cubes, there is a frame of junk data around the observation, use top left pixel as mask
		non_image_pixels = std_hdul[std_hdu_idx].data == std_hdul[std_hdu_idx].data[:,0,0][:,None,None]
		std_hdul[std_hdu_idx].data[non_image_pixels] = np.nan
		if find_psf_flag:
			(	psf, seeing, 
				starpos, background, 
				psf_func, psf_popt, 
				psf_func_name, popt_names, 
				psf_cube, popt_cube, 
				pconv_cube, popt_est_cube, 
				residual_cube
			) = fitscube.process.sinfoni.find_star_psf_seeing_background_full(	
														std_hdul[std_hdu_idx].data, 
														np.abs(std_hdul[std_hdu_idx].header['CDELT1'])*const.deg2arcsec, 
														show_plots=True, 
														plot_dir='./plots', 
														save_plots=True, 
														calibrator_name=std_hdul[std_hdu_idx].header['HIERARCH ESO OBS TARG NAME']
													)
			
		diff = std_hdul[std_hdu_idx].data - psf_cube
		std_data = np.array(std_hdul[std_hdu_idx].data)
		std_header = std_hdul[std_hdu_idx].header
		
		if show_plots:
			f1, a1 = plt.subplots(1, 3, squeeze=False, figsize=(3*6, 1*6))
			vmin, vmax = (-np.nanmax(psf_cube, axis=(1,2)), np.nanmax(psf_cube, axis=(1,2)))
			norms = [mpl.colors.SymLogNorm(linthresh=1, vmin=amin, vmax=amax) for amin, amax in zip(vmin, vmax)]
			im10 = a1[0,0].imshow(data[0], origin='lower')
			im11 = a1[0,1].imshow(data[0], origin='lower')
			im12 = a1[0,2].imshow(data[0], origin='lower')
			def update(n):
				#a1[0,0].imshow(data[n,:,:], origin='lower', norm=norms[n])
				#a1[0,1].imshow(psf_cube[n,:,:], origin='lower', norm=norms[n])
				#a1[0,2].imshow(diff[n,:,:], origin='lower', norm=norms[n])
				im10.set_data(std_data[n])
				im10.set_norm(norms[n])
				im11.set_data(psf_cube[n])
				im11.set_norm(norms[n])
				im12.set_data(diff[n])
				im12.set_norm(norms[n])
				f1.suptitle(f'{n}')
				return
			
			ani = mpl.animation.FuncAnimation(f1, update, range(std_data.shape[0]), interval=10)
			plt.show()
#%%
	
	with fits.open(obsf) as obs_hdul:
		print(obs_hdul.info())
		# rename HDUs to my names
		# make sure all non-primary HDUs are the correct type
		obs_hdul[1] = fits.ImageHDU(data = obs_hdul[1].data, header=obs_hdul[1].header, name='PIX_OFFSET')
		obs_hdul[2] = fits.ImageHDU(data = obs_hdul[2].data, header=obs_hdul[2].header, name='LATITUDE')
		obs_hdul[3] = fits.ImageHDU(data = obs_hdul[3].data, header=obs_hdul[3].header, name='LONGITUDE')
		obs_hdul[4] = fits.ImageHDU(data = obs_hdul[4].data, header=obs_hdul[4].header, name='ZENITH')
		obs_hdul[5] = fits.ImageHDU(data = obs_hdul[5].data, header=obs_hdul[5].header, name='X_ARCSEC')
		obs_hdul[6] = fits.ImageHDU(data = obs_hdul[6].data, header=obs_hdul[6].header, name='Y_ARCSEC')
		
		
		# turn non-image pixels into NANs in the primary HDU
		# in Pat's cubes, there is a frame of junk data around the observation, use top left pixel as mask
		non_image_pixels = obs_hdul['PRIMARY'].data == obs_hdul['PRIMARY'].data[:,0,0][:,None,None]
		obs_hdul['PRIMARY'].data[non_image_pixels] = np.nan
		
		# turn junk coded pixels into NANs in other HDUs
		# Pat uses -999.9 as a 'no data' code for hdu's 2,3,4
		no_data_value = -999.9
		obs_hdul['LATITUDE'].data[obs_hdul['LATITUDE'].data == no_data_value] = np.nan
		obs_hdul['LONGITUDE'].data[obs_hdul['LONGITUDE'].data == no_data_value] = np.nan
		obs_hdul['ZENITH'].data[obs_hdul['ZENITH'].data == no_data_value] = np.nan
		
		# create disk mask
		disk_mask = ~np.isnan(obs_hdul['LATITUDE'].data)
		
		# create disk fraction
		norm_psf = psf/np.nansum(psf, axis=(1,2))[:,None,None]
		disk_frac = sp.signal.convolve2d(disk_mask.astype('float'), np.nanmedian(norm_psf, axis=0), mode='same')
		
		# create statistical error
		stat_err_mask = np.ones_like(obs_hdul['PRIMARY'].data, dtype='bool')*(disk_frac < 1E-2)[None,:,:]
		print('HERE')
		temp_arr = np.array(obs_hdul['PRIMARY'].data)
		temp_arr[stat_err_mask] = np.nan
		error = np.ones_like(obs_hdul['PRIMARY'].data)*np.nanstd(temp_arr, axis=(1,2))[:,None,None]
		
		# assemble HDUs
		disk_mask_hdu = fits.ImageHDU(	data = disk_mask.astype('float'), 
										header=fitscube.header.get_coord_fields(obs_hdul['PRIMARY'].header), 
										name='DISK_MASK'
									)
		disk_frac_hdu = fits.ImageHDU(	data = disk_frac, 
										header=fitscube.header.get_coord_fields(obs_hdul['PRIMARY'].header), 
										name='DISK_FRAC'
									)
		error_hdu = fits.ImageHDU(	data = error,
									header=fitscube.header.get_coord_fields(obs_hdul['PRIMARY'].header), 
									name='ERROR'
								)
		
		psf_hdu = fits.ImageHDU(	data=psf,
									header=fitscube.header.get_coord_fields(std_header),
									name='PSF'
								)
		
		# put HDUs into file
		for _hdu in [disk_mask_hdu, disk_frac_hdu, error_hdu, psf_hdu]:
			obs_hdul.append(_hdu)
		
		out_obsf = '_jformat.'.join(obsf.rsplit('.',1))
		obs_hdul.writeto(out_obsf, overwrite=True)
		
		
		#plt.imshow(obs_hdul['PRIMARY'].data[1000,:,:], origin='lower')
		#plt.imshow(obs_hdul['LATITUDE'].data, origin='lower')
		#plt.imshow(obs_hdul['DISK_FRAC'].data, origin='lower')
		plt.plot(error[:,0,0])
		plt.show()
		
		print(obs_hdul.info())
	
	