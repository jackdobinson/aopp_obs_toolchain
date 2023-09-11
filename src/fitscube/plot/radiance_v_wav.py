#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import fitscube.header

import glob


obs_dir = os.path.expanduser('~/scratch/reduced_images')
target_cubes = glob.glob(f'{obs_dir}/SINFO*/*/analysis/*renormed.fits')

f1 = plt.figure(figsize=(12,12))
a1 = f1.subplots(1,1,squeeze=False)

for i, tc in enumerate(target_cubes):
	print(f'INFO: Operating on cube [{i}/{len(target_cubes)}] "{tc}"')
	tc_dir = os.path.dirname(tc)
	mask_fits = os.path.join(tc_dir, 'auto_mask_cloud.fits')
	with fits.open(mask_fits) as mask_hdul:
		mask_data = mask_hdul['PRIMARY'].data
	with fits.open(tc) as hdul:
		radiance = hdul['PRIMARY'].data
		disk_mask = np.broadcast_to(hdul['DISK_MASK'].data[None,:,:], radiance.shape)
		wavgrid = fitscube.header.get_wavelength_grid(hdul['PRIMARY'])
	radiance[mask_data==1] = np.nan
	radiance[disk_mask==0] = np.nan
	
	# SANITY CHECK PLOT
	#plt.imshow(np.nanmedian(radiance, axis=0), origin='lower')
	#plt.show()
	wrange_idx = np.nonzero((wavgrid > 1.445) & (wavgrid < 2.455))
	a1[0,0].plot(wavgrid[wrange_idx], np.nanmedian(radiance, axis=(1,2))[wrange_idx], linewidth=0.2)
	
	
	# DEBUGGING
	#break

plt.show()
