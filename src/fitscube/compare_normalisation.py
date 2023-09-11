#!/usr/bin/env python3
"""
Docstring for module
"""
#%% import modules
import sys, os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import fitscube
import fitscube.header
import fitscube.create_mask
import matplotlib.animation
import matplotlib as mpl
import time

#%% create list of target cubes
target_cubes = [
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-18T04:00:14/MOV_Neptune---H+K_0.1_2_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-18T04:00:14/MOV_Neptune---H+K_0.1_2_tpl/analysis/obj_NEPTUNE_cal_HIP001115_no_correction.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-19T04:23:27/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP096851.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-19T09:43:00/MOV_Neptune---H+K_0.1_3_tpl/analysis/obj_NEPTUNE_cal_HIP038133.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T05:08:24/MOV_Neptune---H+K_0.1_1_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115.fits',
	'/home/dobinsonl/scratch/reduced_images/SINFO.2018-08-31T06:20:44/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP014898.fits',
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

#%% create list of calibrators
import const
calib_cubes = []
top_dir = '/home/dobinsonl/scratch/reduced_images/'
tcs = {}
for tc in target_cubes:
	adir = tc.rsplit(os.sep, 2)[0]
	calibrators_file = os.path.join(adir, 'calibrators.txt')
	calibs = []
	with open(calibrators_file, 'r') as f:
		for aline in f:
			calibs.append(os.path.join(top_dir, os.sep.join(aline.strip().rsplit(os.sep, 3)[1:])))
	calib_cubes.append(calibs)
	
def vector_query_simbad_to_ids(objs, fields, sids, ids):
	from astroquery.simbad import Simbad
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
	print(f'INFO: Finding SIMBAD data for objects {objs}')
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


simbad_calib_attrs = []
calibrator_names = []
calibrator_paths = []
for i, cals in enumerate(calib_cubes):
	cal_names = []
	calibrator_paths += cals
	for cal in cals:
		#if 'Telluric' in cal: continue
		with fits.open(cal) as cal_hdul:
			#print(cal_hdul[0].header['HIERARCH ESO OBS TARG NAME'])
			#sys.exit()
			cal_names.append(cal_hdul[0].header['HIERARCH ESO OBS TARG NAME'])
	calibrator_names += cal_names

def unique_list(alist, idxs=True):
	u = []
	idxs = []
	for i, item in enumerate(alist):
		if item in u: continue
		u.append(item)
		idxs.append(i)
	return(u, idxs)

calibrator_names, uidxs = unique_list(calibrator_names)
calibrator_paths = [calibrator_paths[i] for i in uidxs]

simbad_calib_attrs = (vector_query_simbad_to_ids(calibrator_names, 
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
								))
	

for i, cals in enumerate(calib_cubes):	
	tcs[target_cubes[i]] = {'calibrators':{}}
	for j, cal in enumerate(cals):
		with fits.open(cal) as cal_hdul:
			cal_name = cal_hdul[0].header['HIERARCH ESO OBS TARG NAME']
		#if 'Telluric' in cal: continue
		idx = calibrator_names.index(cal_name)
		tcs[target_cubes[i]]['calibrators'][cal_name] = simbad_calib_attrs[idx]
		tcs[target_cubes[i]]['calibrators'][cal_name]['file'] = cal
		
#%% update with extra info
import fitscube.process.sinfoni

for tc, tc_attrs in tcs.items():
	for cal, cal_attrs in tc_attrs['calibrators'].items():
		with fits.open(cal_attrs['file']) as cal_hdul:
			cal_attrs['obs_date_mjd'] = cal_hdul[0].header['MJD-OBS']
			cal_attrs['obs_date'] = cal_hdul[0].header['DATE-OBS']
			cal_attrs['wavgrid'] = fitscube.header.get_wavelength_grid(cal_hdul[0])
			cal_attrs['exposure_time'] = cal_hdul[0].header['EXPTIME']
			cal_attrs['detector_integration_time'] = cal_hdul[0].header['HIERARCH ESO DET DIT']
			cal_attrs['px_deg1'] = np.abs(cal_hdul[0].header['CDELT1'])
			cal_attrs['px_deg2'] = np.abs(cal_hdul[0].header['CDELT2'])
			cal_attrs['px_arcsec1'] = np.abs(cal_hdul[0].header['CDELT1'])*const.deg2arcsec
			cal_attrs['px_arcsec2'] = np.abs(cal_hdul[0].header['CDELT2'])*const.deg2arcsec
			cal_attrs['px_ster'] = np.abs(cal_hdul[0].header['CDELT1']*cal_hdul[0].header['CDELT2']*const.degSq2ster)
			cal_attrs['resolution'] = cal_hdul[0].header['HIERARCH ESO INS OPTI1 NAME']
			cal_attrs['calibration_factor'] = fitscube.process.sinfoni.get_counts_per_sec_in_h_filter2spectral_radiance(cal_hdul[0].data, cal_attrs['wavgrid'], cal_attrs['exposure_time'], cal_attrs['h_flux'], cal_attrs['px_ster'])
			hft = fitscube.process.sinfoni.interpolate_h_filter_to_wavgrid(cal_attrs['wavgrid']) # transmittance of H-filter at data wavelengths
			cal_attrs['h_band_counts'] = np.nansum(cal_hdul[0].data*hft[:,None,None])/np.nansum(hft) # avg counts in H-band
			cal_attrs['h_band_counts_per_sec'] = cal_attrs['h_band_counts']/cal_attrs['exposure_time']

#%% work out normalisation factors
h_mags = []
h_counts_per_secs = []
labels = []
shape_list = iter(['.', 's', '^', 'v', '<', '>', '+', 'x', 'd', '*', 'h', 'P', 'X', '1', '2', '3', '4'])
shapes = []
used_std_stars = []
c_resolutions = []
for tc, tc_attrs in tcs.items():
	for cal, c in tc_attrs['calibrators'].items():
		#print(c)
		print(f"\nAttributes for {c['name']}:")
		print(f"\tobs_date {c['obs_date']}")
		print(f"\tpx_deg1 {c['px_deg1']}")
		print(f"\tpx_deg2 {c['px_deg1']}")
		print(f"\tpx_arcsec1 {c['px_arcsec1']}")
		print(f"\tpx_arcsec2 {c['px_arcsec2']}")
		print(f"\tpx_ster {c['px_ster']}")
		print(f"\tresolution {c['resolution']}")
		print(f"\th_flux {c['h_flux']}")
		print(f"\tcalibration_factor {c['calibration_factor']}")
		print(f"\texposure_time {c['exposure_time']}")
		print(f"\tdetector_integration_time {c['detector_integration_time']}")
		print(f"\th_band_counts {c['h_band_counts']}")
		print(f"\th_band_counts_per_sec {c['h_band_counts_per_sec']}")
		h_mags.append(float(c['h_flux']))
		h_counts_per_secs.append(float(c['h_band_counts_per_sec']))
		lbl = f"{c['name']}, {c['obs_date']}"
		labels.append(lbl)
		c_resolutions.append(c['resolution'])
		if all([lbl != uss for uss in used_std_stars]):
			shape = next(shape_list)
		used_std_stars.append(lbl)
		shapes.append(shape)
		
used_std_stars = []
for i in range(len(h_mags)):
	ss = labels[i]
	if ss not in used_std_stars:
		lbl = labels[i]
	else:
		lbl=None
	used_std_stars.append(ss)
	if float(c_resolutions[i]) == 0.1:
		colour = 'tab:green'
	elif float(c_resolutions[i]) == 0.025:
		colour = 'tab:blue'
	plt.plot(h_counts_per_secs[i], h_mags[i], linestyle='none', marker=shapes[i], label=lbl, color=colour)

import scipy.optimize
h_mags = np.array(h_mags)
h_counts_per_secs = np.array(h_counts_per_secs)
A = np.array([np.ones_like(h_mags), h_mags]).T
y = np.log(np.array(h_counts_per_secs))
m = -np.log(10)/2.5
x = scipy.optimize.lsq_linear(A, y, bounds=((-np.inf, m),(np.inf, np.sign(m)*m*(1+1E-9))), method='trf').x
h_mag_grid = np.linspace(6.0, 8.5, 50)
yfit = x[0] + h_mag_grid*x[1]

plt.plot(np.exp(yfit), h_mag_grid, color='tab:orange', label=f'counts/sec = a x 10^(-h_mag/2.5), a = {x[0]}')
#for a, b, c in zip(h_counts_per_secs, h_mags, labels):
#	plt.annotate(c, (a,b))
print(x)
print(np.exp(yfit))

#plt.yscale('log')
plt.xlabel('counts/sec')
plt.ylabel('h band magnitude')
plt.legend()
plt.show()


#%% open all fits files and extract data
medians = np.full(len(target_cubes), fill_value=np.nan)
sums = np.full(len(target_cubes), fill_value=np.nan)
means = np.full(len(target_cubes), fill_value=np.nan)
dates = np.full(len(target_cubes), fill_value=np.nan)
resolutions = np.full(len(target_cubes), fill_value=np.nan)
num_of_nans = np.full(len(target_cubes), fill_value=np.nan)
num_of_pixels = np.full(len(target_cubes), fill_value=np.nan)
maxs = np.full(len(target_cubes), fill_value=np.nan)

histogram_full = [None]*len(target_cubes)
histogram_nocloud = [None]*len(target_cubes)
histogram_cloud = [None]*len(target_cubes)
date_readables = [None]*len(target_cubes)
calibrators = [None]*len(target_cubes)



f2 = plt.figure()
a2 = f2.subplots(1,1)
bins = np.linspace(-2E-7, 28E-7, 200)

for i, tc in enumerate(target_cubes):
	print(f'INFO: Operating on cube {tc}')
	with fits.open(tc) as hdul:
		num_of_nans[i] = np.count_nonzero(np.isnan(hdul[0].data))
		num_of_pixels[i] = hdul[0].data.size #np.prod(hdul[0].data.shape)
		if num_of_nans[i] < 1*num_of_pixels[i]: # < 0.25 is all full disk obs, < 0.5 is disk obs with holes, < 1.0 is all obs
			wavs = fitscube.header.get_wavelength_grid(hdul[0])
			widxs = np.nonzero(np.logical_and(wavs>1.55, wavs < 1.56))
			
			disk_mask = np.broadcast_to(np.array(hdul['DISK_MASK'].data, dtype=bool), hdul[0].data.shape)
			cloud_mask_file = os.path.join(os.path.dirname(tc), './auto_mask_cloud.fits')
			with fits.open(cloud_mask_file) as cloud_mask_hdul:
				cloud_mask = np.array(cloud_mask_hdul[0].data, dtype=bool)
			disk_and_no_cloud_mask = disk_mask & (~cloud_mask)
			disk_and_cloud_mask = disk_mask & cloud_mask
			
			histogram_full[i] = np.zeros((hdul[0].data.shape[0], bins.shape[0]-1))
			histogram_nocloud[i] = np.zeros((hdul[0].data.shape[0], bins.shape[0]-1))
			histogram_cloud[i] = np.zeros((hdul[0].data.shape[0], bins.shape[0]-1))
			nocloud_data = hdul[0].data[disk_and_no_cloud_mask].reshape(disk_and_no_cloud_mask.shape[0], -1)
			cloud_data = hdul[0].data[disk_and_cloud_mask].reshape(disk_and_cloud_mask.shape[0], -1)
			for j in range(hdul[0].data.shape[0]):
				if not(j % 100): sys.stdout.write('.')
				histogram_full[i][j,:] = np.histogram(hdul[0].data[j], bins)[0]
				histogram_nocloud[i][j,:] = np.histogram(nocloud_data[j], bins)[0]
				histogram_cloud[i][j,:] = np.histogram(cloud_data[j], bins)[0]
			sys.stdout.write('\n')
			"""
			f1 = plt.figure()
			a1 = f1.subplots(1,1)
			fitscube.create_mask.plot_mask(f1,a1,hdul[0].data[300], apply_mask[300])
			plt.show()
			"""
			
			#apply_mask = disk_and_no_cloud_mask
			apply_mask = disk_mask
			
			data = hdul[0].data[apply_mask].reshape(apply_mask.shape[0], -1)[widxs]
			#data= hdul[0].data[300:330]
			dates[i] = float(hdul[0].header['MJD-OBS'])
			resolutions[i] = float(hdul[0].header['HIERARCH ESO INS OPTI1 NAME'])
			medians[i] = np.nanmedian(data)
			sums[i] = np.nansum(data)
			means[i] = np.nanmean(data)
			maxs[i] = np.nanmax(data)
			calibrators[i] = hdul['CALIBRATOR_OBS'].header['HIERARCH ESO OBS TARG NAME']
			
			date_readables[i] = tc.split('/')[-4]
			fitscube.create_mask.plot_radiance_histogram(f2, a2, data, bins=bins, density=False, label=f'{date_readables[i]}, {resolutions[i]}')
a2.legend()
plt.show()

#%% plot histogram of radiances vs wavelength

f3 = plt.figure(figsize=(16,8))
a3 = f3.subplots(1,1)
bin_mids = 0.5*(bins[:-1]+bins[1:])

def update(j):
	#print(j)
	a3.clear()
	colours = iter(plt.cm.rainbow(np.linspace(0,1,len(target_cubes))))
	for i, hd in enumerate(histogram_full):
		if hd is None:
			continue
		a3.plot(bin_mids, hd[j,:], markersize=4, linewidth=1, marker='.', label=f'{date_readables[i]}, {resolutions[i]}, {calibrators[i]}, {num_of_nans[i]/num_of_pixels[i]:04.2f}', c=next(colours))
	a3.set_title(f'All Radiance histogram at wavelength {wavs[j]}')
	a3.set_xlim([np.min(bins), np.max(bins)])
	a3.set_yscale('log')
	a3.set_xlabel('Radiance')
	a3.set_ylabel('Counts')
	a3.legend(loc='upper right')
	
	return

ani = mpl.animation.FuncAnimation(f3, update, frames=range(100,2168), interval=100)
plt.show()

#%% radiance vs wavelength histogram for clouds
f3 = plt.figure(figsize=(16,8))
a3 = f3.subplots(1,1)
bin_mids = 0.5*(bins[:-1]+bins[1:])
def update(j):
	#print(j)
	a3.clear()
	colours = iter(plt.cm.rainbow(np.linspace(0,1,len(target_cubes))))
	for i, hd in enumerate(histogram_cloud):
		if hd is None:
			continue
		a3.plot(bin_mids, hd[j,:], markersize=4, linewidth=1, marker='.', label=f'{date_readables[i]}, {resolutions[i]}, {calibrators[i]}, {num_of_nans[i]/num_of_pixels[i]:04.2f}', c=next(colours))
	a3.set_title(f'Cloud Radiance histogram at wavelength {wavs[j]}')
	a3.set_xlim([np.min(bins), np.max(bins)])
	a3.set_yscale('log')
	a3.set_xlabel('Radiance')
	a3.set_ylabel('Counts')
	a3.legend(loc='upper right')
	
	return

ani = mpl.animation.FuncAnimation(f3, update, frames=range(100,2168), interval=100)
plt.show()

#%% radiance vs wavelength histogram for not clouds
f3 = plt.figure(figsize=(16,8))
a3 = f3.subplots(1,1)
bin_mids = 0.5*(bins[:-1]+bins[1:])
def update(j):
	#print(j)
	a3.clear()
	colours = iter(plt.cm.rainbow(np.linspace(0,1,len(target_cubes))))
	for i, hd in enumerate(histogram_nocloud):
		if hd is None:
			continue
		a3.plot(bin_mids, hd[j,:], markersize=4, linewidth=1, marker='.', label=f'{date_readables[i]}, {resolutions[i]}, {calibrators[i]}, {num_of_nans[i]/num_of_pixels[i]:04.2f}', c=next(colours))
	a3.set_title(f'Non-cloud Radiance histogram at wavelength {wavs[j]}')
	a3.set_xlim([np.min(bins), np.max(bins)])
	a3.set_yscale('log')
	a3.set_xlabel('Radiance')
	a3.set_ylabel('Counts')
	a3.legend(loc='upper right')
	
	return

ani = mpl.animation.FuncAnimation(f3, update, frames=range(100,2168), interval=100)
plt.show()

#%% Get total radiance for each observation and compare them

avg_01_mean = np.nanmean([means[i] for i in range(len(target_cubes)) if resolutions[i]==0.1])
avg_01_median = np.nanmean([medians[i] for i in range(len(target_cubes)) if resolutions[i]==0.1])
avg_01_sum = np.nanmean([sums[i] for i in range(len(target_cubes)) if resolutions[i]==0.1])

def table(columns, head=None, fmts=None, title=None, frame=True, rstart= '| ', rend=' |', csep=' | ', fhdl=None, text=''):
	import io
	import textwrap
	if fhdl is None:
		fhdl = io.StringIO('')
		
	if not frame:
		rstart, rend= '',''
	
	cw_maxs = np.zeros((len(columns)), dtype=int)
	nr = max([len(c) for c in columns])
	
	# get column widths
	for i, c in enumerate(columns):
		cw_maxs[i] = max([len(str(_x)) if (fmts is None) or (fmts[i] is None) else len(fmts[i].format(_x)) for _x in c])
		if (head is not None) and (head[i] is not None) and (len(head[i]) > cw_maxs[i]):
			cw_maxs[i] = len(head[i])
	
	# get table width
	t_width = len(rstart)+len(rend)+len(csep)*(len(columns)-1)+sum(cw_maxs)
	tb_width = len(csep)*(len(columns)-1)+sum(cw_maxs) # table body width
	
	# write the table title string
	title_string = f'# {title} #' if title is not None else ''
	fhdl.write(title_string)
	if frame:
		fhdl.write('-'*(t_width-len(title_string)))
	fhdl.write('\n')
	
	# write header if needed
	if head is not None:
		hfmt = csep.join(['{:^'+f'{cw_max}'+'}' for cw_max in cw_maxs])
		hs = hfmt.format(*head)
		fhdl.write(rstart+hs+rend+'\n')
		fhdl.write(rstart+'-'*tb_width+rend+'\n')
		
	# get row format
	rfmt = csep.join(['{:>'+f'{cw_max}'+'}' for cw_max in cw_maxs])	
	# write rows
	for i in range(nr):
		rs = rfmt.format(*[str(c[i]) if (fmts is None) or (fmts[j] is None) else fmts[j].format(c[i]) for j, c in enumerate(columns)])
		fhdl.write(rstart+rs+rend+'\n')
	
	
	# write text at end if needed
	if len(text)>0:
		fhdl.write(rstart+'-'*tb_width+rend+'\n')
		text_fmt = '{:<'+f'{tb_width}'+'}'
		for aline in textwrap.wrap(text, width=tb_width, tabsize=4):
			aline = aline.rstrip('\n')
			fhdl.write(rstart + text_fmt.format(aline) + rend+'\n')
		
	
	if frame:
		fhdl.write('-'*t_width+'\n')
			
	
	# if we are using a stringIO the reutrn the resulting string
	if type(fhdl) is io.StringIO:
		return(fhdl.getvalue())

# table of statistics
title = 'Radiance Statistics'
head = ['date', 'res', 'calibrator', 'frac_of_nans', 'sum', 'mean', 'median']
columns = [date_readables, resolutions, calibrators, num_of_nans/num_of_pixels, sums, means, medians]
fmts = [None]*3+['{:>6.3g}']*4
print(table(columns, head, fmts, title))

# table of normalised statistics
title='Radiance Statistics Normed'
head = ['date', 'res', 'calibrator', 'frac_of_nans', 'sum_norm', 'mean_norm', 'median_norm']
columns = [date_readables, resolutions, calibrators, num_of_nans/num_of_pixels, sums/avg_01_sum, means/avg_01_mean, medians/avg_01_median]
fmts = [None]*3+['{:>6.3f}']*4
print(table(columns, head, fmts, title, text='NOTE: Sums, means, and medians are divided by the average values of the res=0.1" observations for comparison purposes'))

#%% plot boring regions
import subsample
boring_specs = [os.path.join(os.path.dirname(tc), 'boring_region_spec.dat') for tc in target_cubes]
f0 = plt.figure(figsize=[16,8])
a0 = f0.subplots(1,1)
colours = iter(plt.cm.rainbow(np.linspace(0,1,len(target_cubes))))
for i, boring_spec in enumerate(boring_specs):
	spec = np.loadtxt(boring_spec)
	if resolutions[i]==0.1:
		marker = '^'
		ms = 2
		ls = '--'
	else:
		marker=''
		ms = 1
		ls='-'
	a0.plot(*subsample.conv(spec[:,0], spec[:,1], 0.005, conv_type='norm_triangle'), c=next(colours), markersize=ms, linewidth=1, marker=marker, linestyle=ls, label=f'{date_readables[i]}, {resolutions[i]}, {calibrators[i]}, {num_of_nans[i]/num_of_pixels[i]:04.2f}')
a0.legend(loc='upper right')
a0.set_ylim([1E-9,5E-7])
a0.set_yscale('log')
plt.show()


#%% plot medians as a line
f0 = plt.figure(figsize=[4*4, 4])
a0 = f0.subplots(1,4,squeeze=False, gridspec_kw={'hspace':0.2, 'wspace':0.3})
c = ['b' if r==0.1 else 'r' for r in resolutions]
print(f'INFO: medians range {np.nanmin(medians)} {np.nanmax(medians)}')
s1 = a0[0,0].scatter(dates, medians, c=c)
a0[0,0].set_ylim([0.9*np.nanmin(medians), 1.1*np.nanmax(medians)])
a0[0,0].set_xlabel('dates (MJD)')
a0[0,0].set_ylabel('Median Radiance')
a0[0,0].set_title('Colour denotes 0.1" (blue)\n or 0.025" (red) resolution')

s1 = a0[0,1].scatter(dates, sums, c=c)
a0[0,1].set_ylim([0.9*np.nanmin(sums), 1.1*np.nanmax(sums)])
a0[0,1].set_xlabel('dates (MJD)')
a0[0,1].set_ylabel('Sum of  Radiance')
a0[0,1].set_title('Colour denotes 0.1" (blue)\n or 0.025" (red) resolution')

s1 = a0[0,2].scatter(dates, means, c=c)
a0[0,2].set_ylim([0.9*np.nanmin(means), 1.1*np.nanmax(means)])
a0[0,2].set_xlabel('dates (MJD)')
a0[0,2].set_ylabel('Mean Radiance')
a0[0,2].set_title('Colour denotes 0.1" (blue)\n or 0.025" (red) resolution')

s1 = a0[0,3].scatter(dates, maxs, c=c)
a0[0,3].set_ylim([0.9*np.nanmin(maxs), 1.1*np.nanmax(maxs)])
a0[0,3].set_xlabel('dates (MJD)')
a0[0,3].set_ylabel('Max Radiance')
a0[0,3].set_title('Colour denotes 0.1" (blue)\n or 0.025" (red) resolution')

plt.show()

