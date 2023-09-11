#!/usr/bin/env python3

import sys, os
import numpy as np
from astropy.io import fits
import astropy.convolution
import fitscube.header
import scipy as sp
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% functions

def regrid_to(arr, i_step, o_step, i_origin=None):
	# how many new pixels can i fit in the array
	arr_size = np.array(arr.shape)*i_step
	o_shape = np.array(arr_size//o_step, dtype=int)

	if i_origin is None:
		i_origin=np.array(arr.shape)/2
		o_origin = o_shape/2

	i_data_pos = (np.array([np.arange(s) for s in arr.shape]) - i_origin[:,None])*i_step[:,None]
	o_data_pos = (np.array([np.arange(s) for s in o_shape]) - o_origin[:,None])*o_step[:,None]
	
	i_grid = np.meshgrid(*i_data_pos)
	o_grid = np.meshgrid(*o_data_pos)
	
	regrid_arr = sp.interpolate.griddata(tuple([_x.ravel() for _x in i_grid]), 
											arr.ravel(), 
											tuple([_x.ravel() for _x in o_grid]), 
											method='linear')
	
	return(regrid_arr.reshape(*o_shape))
	

#%% set up inputs

synth_fits = os.path.expanduser('~/scratch/reduced_images/SINFO.synthetic/test.fits')
psf_fits = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811_renormed.fits')

synth_wavgrid = np.array([1.55, 1.75, 1.95])

#%% compute convolution

with fits.open(synth_fits) as synth_hdul:
	with fits.open(psf_fits) as psf_hdul:
		#print(synth_hdul['DATA'].header)
		print(synth_hdul['DATA'].data.shape)
		synth_data = np.array(synth_hdul['DATA'].data)
		synth_dx = synth_hdul['DATA'].header['CD1_1']
		synth_dy = synth_hdul['DATA'].header['CD2_2']
		psf_dx = psf_hdul['PSF'].header['CDELT1']
		psf_dy = psf_hdul['PSF'].header['CDELT2']
		psf_wavgrid = fitscube.header.get_wavelength_grid(psf_hdul['PRIMARY'])
		psf_synth_wavgrid_idx = np.nanargmin(np.fabs(psf_wavgrid[:,None] - synth_wavgrid[None,:]), axis=0)
		convolved_synth_data = np.array(synth_hdul['DATA'].data)
		original_psf_data = []
		regrid_psf_data = []
		for i, widx in enumerate(psf_synth_wavgrid_idx):
			original_psf = psf_hdul['PSF'].data[widx,:,:]
			original_psf_data.append(original_psf)
			w_psf = regrid_to(original_psf, np.array([psf_dx,psf_dy]), np.array([synth_dx, synth_dy]))
			regrid_psf_data = w_psf
			w_psf /= np.nansum(w_psf) # ensure integral is zero
			convolved_synth_data[i,:,:] = astropy.convolution.convolve(synth_hdul['DATA'].data[i,:,:], w_psf[:,:], boundary=None)
		original_psf_hdu = fits.ImageHDU(data=np.array(original_psf_data), header=psf_hdul['PSF'].header, name='ORIGINAL PSF')
		regrid_psf_hdu = fits.ImageHDU(data=np.array(regrid_psf_data), header=synth_hdul['DATA'].header, name='REGRID PSF')
		convolved_synth_hdu = fits.ImageHDU(data=convolved_synth_data, header=synth_hdul['DATA'].header, name='CONVOLVED_SYNTH')
		synth_hdul.append(convolved_synth_hdu)
		synth_hdul.append(regrid_psf_hdu)
		synth_hdul.append(original_psf_hdu)
		synth_hdul.writeto(os.path.join(os.path.dirname(synth_fits), 'test_convolved.fits'))
		
#%% plots
nr,nc = (2,4)
f1 = plt.figure(figsize=(nc*5, nr*5))
a1 = f1.subplots(nr,nc,squeeze=False)

norm1 = norm = mpl.colors.Normalize(vmin=0, vmax=5E-7)
norm2 = norm = mpl.colors.Normalize(vmin=0, vmax=5E-8)
norm3 = norm = mpl.colors.Normalize(vmin=0, vmax=5E-8)

im100 = a1[0,0].imshow(convolved_synth_data[0,:,:], origin='lower', norm=norm1)
f1.colorbar(im100, ax=a1[0,0])
a1[0,0].set_title(f'Convolved image\nwav {synth_wavgrid[0]:04.2f} (um) sum {np.nansum(convolved_synth_data[0,:,:]):07.2E}')

im101 = a1[0,1].imshow(convolved_synth_data[1,:,:], origin='lower', norm=norm2)
f1.colorbar(im101, ax=a1[0,1])
a1[0,1].set_title(f'Convolved image\nwav {synth_wavgrid[1]:04.2f} (um) sum {np.nansum(convolved_synth_data[1,:,:]):07.2E}')

im102 = a1[0,2].imshow(convolved_synth_data[2,:,:], origin='lower', norm=norm3)
f1.colorbar(im102, ax=a1[0,2])
a1[0,2].set_title(f'Convolved image\nwav {synth_wavgrid[2]:04.2f} (um) sum {np.nansum(convolved_synth_data[2,:,:]):07.2E}')

im110 = a1[1,0].imshow(synth_data[0,:,:], origin='lower', norm=norm1)
f1.colorbar(im110, ax=a1[1,0])
a1[1,0].set_title(f'raw image\nwav {synth_wavgrid[0]:04.2f} (um) sum {np.nansum(synth_data[0,:,:]):07.2E}')

im111 = a1[1,1].imshow(synth_data[1,:,:], origin='lower', norm=norm2)
f1.colorbar(im111, ax=a1[1,1])
a1[1,1].set_title(f'raw image\nwav {synth_wavgrid[1]:04.2f} (um) sum {np.nansum(synth_data[1,:,:]):07.2E}')

im112 = a1[1,2].imshow(synth_data[2,:,:], origin='lower', norm=norm3)
f1.colorbar(im112, ax=a1[1,2])
a1[1,2].set_title(f'raw image\nwav {synth_wavgrid[2]:04.2f} (um) sum {np.nansum(synth_data[2,:,:]):07.2E}')

im103 = a1[0,3].imshow(w_psf, origin='lower')
f1.colorbar(im103, ax=a1[0,3])
a1[0,3].set_title(f'Example regridded PSF wav {synth_wavgrid[2]:04.2f} (um)\nsum {np.nansum(w_psf[:,:]):07.2E} dx {synth_dx*3600:07.2E} (arcsec)')

original_psf/=np.nansum(original_psf)
im113 = a1[1,3].imshow(original_psf, origin='lower')
f1.colorbar(im113, ax=a1[1,3])
a1[1,3].set_title(f'Example original PSF wav {synth_wavgrid[2]:04.2f} (um)\nsum {np.nansum(original_psf[:,:]):07.2E} dx {psf_dx*3600:07.2E} (arcsec)')

plt.show()

#%% psf plots
#psf_test = np.array(original_psf)
psf_test = np.array(w_psf)
#psf_test[30,:] = np.nan

nr, nc = (2,2)

f2 = plt.figure(figsize=(nc*5, nr*5))
a2 = f2.subplots(nr,nc, squeeze=False)

im200 = a2[0,0].imshow(psf_test, origin='lower')
a2[0,0].set_title('PSF')

x_margin = np.nansum(psf_test, axis=0)
p210 = a2[1,0].plot(x_margin)
a2[1,0].set_title('x marginalisation')

y_margin = np.nansum(psf_test, axis=1)
p201 = a2[0,1].plot(y_margin, np.arange(y_margin.size))
a2[0,1].set_title('y marginalisation')

a2[1,1].remove()

plt.show()