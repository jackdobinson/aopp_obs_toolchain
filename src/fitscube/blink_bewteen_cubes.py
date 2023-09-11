#!/usr/bin/env python3

import sys, os
import glob
import numpy as np
import fitscube.limb_darkening.minnaert
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import utils as ut
import plotutils
import fitscube.stack
from astropy.io import fits


#%%
if __name__=='__main__':
	target_cubes = glob.glob('/home/dobinsonl/scratch/reduced_images/*/*0.1*/analysis/*clean.fits')[:2]
	
	obs_bands = np.array(((1.5,1.6),(1.7,1.8),(1.9,2.0)))
	
	plt.ioff()
	
	blink_data = []
	
	for i, tc in enumerate(target_cubes):
		blink_data.append([])
		with fits.open(tc) as hdul:
			for j, obs_band in enumerate(obs_bands):
				hdu_stacked = fitscube.stack.stack_hdu_along_spectral_axis(hdul['COMPONENTS'], obs_band, stack_func=np.nanmedian)
				#print(hdu_stacked.data.shape, blink_data)
				blink_data[i].append(hdu_stacked.data[0])
	
	# create animated plot
	nc, nr, s = (3, 1, 5)
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr, nc, squeeze=False)
	
	tc_dates = ", ".join([tc.split(os.sep)[-4] for tc in target_cubes])
	f1.suptitle(f'Blinking between observations; {tc_dates}')
	
	dummy_data = np.zeros_like(blink_data[0][0])
	mins = np.nanmin(np.array([[np.nanmin(d) for d in x] for x in blink_data]), axis=0)
	mins *= 0
	maxs = np.nanmax(np.array([[np.nanmax(d) for d in x] for x in blink_data]), axis=0)
	
	im10 = a1[0,0].imshow(dummy_data, origin='lower', vmin=mins[0], vmax=maxs[0])
	a1[0,0].set_title(f'Median of {obs_bands[0][0]} um to {obs_bands[0][1]} um')
	a1[0,0].get_xaxis().set_visible(False)
	a1[0,0].get_yaxis().set_visible(False)
	
	im11 = a1[0,1].imshow(dummy_data, origin='lower', vmin=mins[1], vmax=maxs[1])
	a1[0,1].set_title(f'Median of {obs_bands[1][0]} um to {obs_bands[1][1]} um')
	a1[0,1].get_xaxis().set_visible(False)
	a1[0,1].get_yaxis().set_visible(False)
	
	im12 = a1[0,2].imshow(dummy_data, origin='lower', vmin=mins[2], vmax=maxs[2])
	a1[0,2].set_title(f'Median of {obs_bands[2][0]} um to {obs_bands[2][1]} um')
	a1[0,2].get_xaxis().set_visible(False)
	a1[0,2].get_yaxis().set_visible(False)
	
	def update(i):
		im10.set_data(blink_data[i][0])
		im11.set_data(blink_data[i][1])
		im12.set_data(blink_data[i][2])
		return()
	
	ani = mpl.animation.FuncAnimation(f1, update, range(0, len(blink_data)), interval=500, repeat=True, repeat_delay=None)
	plt.show()
		
	