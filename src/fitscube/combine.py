#!/usr/bin/env python3
"""
Trying to combine a set of observations together into one fitscube by creating a latitude-longitude map
of a new cube, then finding corresponding radiances from the same latitude-longitude regions in observations.

TODO:
* The definition of llw uses np.float16 datatype, does this work?
* I am using a gaussian function to determine contribution of a lat-lon point of an observation
  to a specific lat-lon point on the combined cube. At the moment the lat-lon point on a combined cube
  is the weighted average of the nearby lat-lon points according the the gaussian function. Is there a better way to do this?
* At small values of 'lat_std' and 'lon_std', get a wierd aliasing effect in 'combined_cube' this is almost certainly due to
  something to do with the gaussian kernel but I'm not sure why it happens.
"""

import sys, os, glob
import numpy as np
from astropy.io import fits
import fitscube.process.cfg
import fitscube.process.sinfoni
import matplotlib.pyplot as plt
import time

combined_dir = os.path.expanduser('~/scratch/reduced_images/SINFO.COMBINED')
fits_dir = os.path.expanduser('~/scratch/reduced_images')
target_cubes = glob.glob(f'{fits_dir}/*/*/analysis/obj_*_renormed.fits')

# want to combine all of the target_cubes so that they have the disk in the same place
px_scale = 0.025 # arcsec
angular_width = 200*px_scale # choose an angular width for the object
sub_obs_lat = -20 # sub observer latitude point
oblateness = fitscube.process.cfg.planet_data['neptune']['oblateness']
nz,ny,nx = (2168, 256, 256)

Re = angular_width/2 # equatorial radius
Rp = Re*(1.0 - oblateness)

combined_cube = np.zeros((nz,ny,nx))
cc_lats = np.full((ny,nx), fill_value=np.nan)
cc_lons = np.full((ny,nx), fill_value=np.nan)
cc_zens = np.full((ny,nx), fill_value=np.nan)
ecx, ecy = (ny/2, nx/2)

print(f'ellipse center x {ecx} y {ecy}')
for i in range(nx):
	for j in range(ny):
		eoffA = (i-ecx)*px_scale
		poffA = (j-ecy)*px_scale
		#print(f'i {i} j {j} eoffA {eoffA} poffA {poffA} px_scale {self.px_scale} py_scale {self.px_scale} nx {nx} ny {ny}')
		iflag, xlat, xlon, zen = fitscube.process.sinfoni.projpos_ellipse(Re, Rp, sub_obs_lat, eoffA, poffA)
		#print(f'Re {Re} Rp {Rp} ecx {ecx} ecy {ecy} iflag {iflag} xlat {xlat} xlon {xlon} zen {zen}')
		if iflag:
			cc_lats[j,i] = xlat
			cc_lons[j,i] = xlon
			cc_zens[j,i] = zen

#%% plot setup

f1 = plt.figure(figsize=(18,6))
a1 = f1.subplots(1,3, squeeze=False)
im11 = a1[0,0].imshow(cc_lats, origin='lower', vmin=-90, vmax=90)
a1[0,0].set_title('latitude')
im12 = a1[0,1].imshow(cc_lons, origin='lower', vmin=-180, vmax=180)
a1[0,1].set_title('longitude')
im13 = a1[0,2].imshow(cc_zens, origin='lower', vmin=0, vmax=90)
a1[0,2].set_title('zenith')

plt.show()

#%% try to get radiances from all the target cubes

# for each lat-lon point
# get a weighted contribution from each target cube
# put it in an array
# do some stats on the array
# put it into the combined cube

radiance_array = np.zeros(len(target_cubes))
coverage = np.zeros_like(combined_cube)
#weights_sum = np.zeros((ny,nx))
lat_std = 1
lon_std = 1

t_overall_start = time.monotonic()
for l, tc in enumerate(target_cubes[:2]):
	print(f'INFO: Adding radiance from target cube {l}')	
	with fits.open(tc) as hdul:
		#lat_weights = np.full((ny,nx,*hdul['LATITUDE'].shape), fill_value=np.nan, dtype=np.float32)
		#lon_weights = np.full((ny,nx,*hdul['LONGITUDE'].shape), fill_value=np.nan, dtype=np.float32)
		llw = np.full((ny,nx,*hdul['LATITUDE'].shape), fill_value=np.nan, dtype=np.float16) # lat weights*lon weights
		
		# calculate latitude-longitude weights for each combined cube pixel
		print('INFO: Finding latitude-longitude weights')
		for j in range(ny):
			for i in range(nx):
				lat = cc_lats[j,i]
				lon = cc_lons[j,i]
				if np.isnan(lat) or np.isnan(lon):
					continue
				#lat_weights[j,i] = (1/(lat_std*np.sqrt(2*np.pi)))*np.exp(-0.5*((hdul['LATITUDE'].data-lat)/lat_std)**2)
				#lon_weights[j,i] = (1/(lat_std*np.sqrt(2*np.pi)))*np.exp(-0.5*((hdul['LONGITUDE'].data-lon)/lon_std)**2)
				lat_weight = (1/(lat_std*np.sqrt(2*np.pi)))*np.exp(-0.5*((hdul['LATITUDE'].data-lat)/lat_std)**2)
				lon_weight = (1/(lat_std*np.sqrt(2*np.pi)))*np.exp(-0.5*((hdul['LONGITUDE'].data-lon)/lon_std)**2)
				llw[j,i] = lat_weight*lon_weight
		
		# put data from target cube into combined cube
		t_start = time.monotonic()
		for k in range(nz):
			dt = time.monotonic() - t_start
			tf = (nz-k)*dt/(k+1)
			print(f'INFO: wavelength slice {k}/{nz} time since start {dt//3600:02.0f}:{((dt-dt%60)/60)%60:02.0f}:{dt%60:05.2f} approx end time {tf//3600:02.0f}:{((tf-tf%60)/60)%60:02.0f}:{tf%60:05.2f} (h:m:s)')
			for j in range(ny):
				#print(f'\tINFO: {j}/{ny}')
				for i in range(nx):
					#print(f'\t\tINFO: {i}/{nx}')
					#llw = np.array(lat_weights[j,i]*lon_weights[j,i], dtype=np.float32)
					if np.isnan(llw[j,i]).all(): # don't do anything if the lat-lon weighting is zero (we are outside disk)
						continue
					combined_cube[k,j,i] += np.nansum(llw[j,i]*hdul['PRIMARY'].data[k])
					coverage[k,j,i] += np.nansum(llw[j,i])
		
		dt = time.monotonic() - t_start
		print(f'INFO: Actual time for this cube {dt//3600:02.0f}:{((dt-dt%60)/60)%60:02.0f}:{dt%60:05.2f} (h:m:s)')

dt = time.monotonic() - t_overall_start
print(f'INFO: Actual time for all cubes {dt//3600:02.0f}:{((dt-dt%60)/60)%60:02.0f}:{dt%60:05.2f} (h:m:s)')

#coverage[coverage==0] = 1 # avoid divide by zero
normed_cube = np.full_like(combined_cube, fill_value=np.nan)
normed_cube[coverage!=0] = combined_cube[coverage!=0]/coverage[coverage!=0] # np.ones_like(combined_cube)*len(target_cubes)

#%% plot combined cube raw
print('INFO: Plotting combined cube raw')
f3 = plt.figure(figsize=(6,6))
a3 = f3.subplots(1,1, squeeze=False)
im31 = a3[0,0].imshow(combined_cube[300,:,:], origin='lower', vmin=0, vmax=1E-7)
a3[0,0].set_title('Combined Cube Raw')

plt.show()

#%% get median of combined cube
cc_median = np.nanmedian(normed_cube, axis=0)

#%% plot combined cube median
print('INFO: Plotting normed cube median')

f2 = plt.figure(figsize=(6,6))
a2 = f2.subplots(1,1, squeeze=False)
im21 = a2[0,0].imshow(cc_median, origin='lower')#, vmin=0, vmax=1E-5)
a2[0,0].set_title('Combined cube median')

plt.show()

#%% plot coverage
print('INFO: Plotting coverage cube')

f3 = plt.figure(figsize=(6,6))
a3 = f3.subplots(1,1, squeeze=False)
im31 = a3[0,0].imshow(coverage[0,:,:], origin='lower', vmin=0, vmax=1)
a3[0,0].set_title('Coverage cube')

plt.show()

#%% save combined cube
print('Saving combined cube')
os.makedirs(combined_dir, exist_ok=True)
combined_cube_file = os.path.join(combined_dir, 'combined_cube_006.npz')
np.savez(combined_cube_file, 
			combined_cube=combined_cube, 
			target_cubes=target_cubes, 
			px_scale=px_scale,
			angular_width=angular_width,
			sub_obs_lat= sub_obs_lat,
			oblateness=oblateness,
			Re=Re,
			Rp=Rp,
			cc_lats=cc_lats,
			cc_lons=cc_lons,
			cc_zens=cc_zens,
			ecx=ecx,
			ecy=ecy,
			lat_std=lat_std,
			lon_std=lon_std
		)

