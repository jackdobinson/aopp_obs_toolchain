#!/usr/bin/env python3
"""
Creates text files that hold minnaert inputs and minnaert parameters for specified fits files.

TODO:
* Write a routine that combines minnaert input files for different observations and creates a
  combined minnaert parameter file
"""

import sys, os
import glob
import numpy as np
from astropy.io import fits
import fitscube.limb_darkening.minnaert
import fitscube.header
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import plotutils
import utils as ut


# create files to contain minnaert input files by latitude range
# TODO:
# * Add in cloud masking
def save_minnaert_inputs(target_cubes, mask_files, minnaert_dirs, lat_limits, cube_extension, overwrite=True):
	minnaert_input_files = []
	for l, tc in enumerate(target_cubes):
		print(f'INFO: Operating on target_cube {tc}')
		tc_name = os.path.basename(tc).rsplit('.')[0]
		mask_file = mask_files[l]
		with fits.open(mask_file) as mask_hdul:
			mask_data = mask_hdul['PRIMARY'].data[0] # find a better way to do this bit
		minnaert_input_files.append([])
		
		
		for lat_min, lat_max in lat_limits:
			print(f'INFO: slicing latitude range {lat_min} to {lat_max}')
			m_input_lat_file = os.path.join(minnaert_dirs[l], f'{tc_name}_minnaert_inputs_lat_{lat_min:+07.2f}_to_{lat_max:+07.2f}.npz')
			
			if os.path.exists(m_input_lat_file) and (not overwrite):
				print(f'INFO: File {m_input_lat_file} exists and will not be overwritten, skipping...')
				minnaert_input_files[l].append(m_input_lat_file)
				continue
			
			with fits.open(tc) as hdul:
				# for each latitude range get minnaert inputs (u, u0, IperF) vs wavelength and save in <path>_minnaert_inputs_<lat_lims>.txt
				lat_range_mask = (lat_min <= hdul['LATITUDE'].data) & (hdul['LATITUDE'].data <= lat_max)
				not_masked = mask_data==0
				lat_idxs = np.nonzero(lat_range_mask & not_masked)
				#print(lat_idxs)
				
				# DEBUGGING
				"""
				if lat_min < 0: continue
				nz_mask = np.nonzero(mask_data)
				#imgdat = np.nanmedian(hdul['PRIMARY'].data, axis=0)
				imgdat = hdul['PRIMARY'].data[300]
				latmask = (lat_min <= hdul['LATITUDE'].data) & (hdul['LATITUDE'].data <= lat_max)
				nz_lats = np.nonzero((lat_min <= hdul['LATITUDE'].data) & (hdul['LATITUDE'].data <= lat_max))
				plt.imshow(imgdat, origin='lower', interpolation=None)
				plt.scatter(*nz_lats[::-1], marker='o', c='white', s=4)
				plt.scatter(*nz_mask[::-1], marker='x', c='red', s=1)
				plt.scatter(*lat_idxs[::-1], marker='x', c='green', alpha=1, s=1)
				
				#plt.imshow(mask_data, origin='lower')
				#plt.imshow(np.nanmedian(hdul['PRIMARY'].data/(1-mask_data[None,:,:]), axis=0), origin='lower')
				#imgdat[not_masked] = np.nan
				#imgdat[lat_range_mask] = np.nan
				#imgdat[lat_range_mask & not_masked] = np.nan
				#imgdat[lat_idxs[0],lat_idxs[1]] = np.nan
				#imgdat[nz_lats[0],nz_lats[1]] = np.nan
				#imgdat[nz_mask[0],nz_mask[1]] = np.nan
				#plt.imshow(imgdat, origin='lower')
				
				plt.show()
				sys.exit()
				"""
				
				
				# if we don't have any latitudes in the range we ask for, just skip and go to the next set
				if lat_idxs[0].size == 0:
					print('INFO: No latitudes in range found in file, skipping...')
					continue 
				
				# filter out zeros as they mean that we don't have any emission (data) for that pixel
				# filter out -ves as they mean that we effectively have zeros and therefore no data for that pixel
				hdul[cube_extension].data[hdul[cube_extension].data<=0] = np.nan

				IperF = hdul[cube_extension].data[:,lat_idxs[0],lat_idxs[1]]
				u = np.broadcast_to(np.cos(hdul['ZENITH'].data[lat_idxs[0],lat_idxs[1]]*np.pi/180), IperF.shape) # observer zenith
				u0 = np.broadcast_to(np.cos(hdul['ZENITH'].data[lat_idxs[0],lat_idxs[1]]*np.pi/180), IperF.shape) # solar zenith (equals observer zenith for outer solar system)
				wavs = fitscube.header.get_wavelength_grid(hdul[cube_extension])
				os.makedirs(os.path.dirname(m_input_lat_file), exist_ok=True)
				np.savez(m_input_lat_file, wavs=wavs, u=u, u0=u0, IperF=IperF)
				minnaert_input_files[l].append(m_input_lat_file) # add file to written file list

	return(minnaert_input_files)

# save minnaert coefficients to file
def save_minnaert_coefficients(target_cubes, minnaert_input_files, overwrite=True):
	minnaert_coefficient_files = []
	for l, tc in enumerate(target_cubes):
		print(f'INFO: working on cube {l}/{len(target_cubes)}')
		tc_name = os.path.basename(tc).rsplit('.')[0]
		minnaert_coefficient_files.append([])
		
		m_input_lat_files = minnaert_input_files[l]
		for m, m_input_lat_file in enumerate(m_input_lat_files):
			m_coefficients_lat_file = m_input_lat_file.replace('_inputs_', '_coefficients_')
			if os.path.exists(m_coefficients_lat_file) and (not overwrite):
				print(f'INFO: File {m_coefficients_lat_file} already exists and will not be overwritten, skipping...')
				minnaert_coefficient_files[l].append(m_coefficients_lat_file)
				continue
			
			print(f'INFO: Operating on file {m_input_lat_file}')
			print(f'INFO: Progress {l}/{len(target_cubes)} {m}/{len(m_input_lat_files)}')
			# unpack latitude limits from filename
			lat_max, lat_min = map(float,os.path.basename(m_input_lat_file).rsplit('.',1)[0].split('_')[-1:-4:-2])
			print(f'INFO: latitude range {lat_min} to {lat_max}')
			ldata = np.load(m_input_lat_file)
			IperF, u, u0, wavs = (ldata['IperF'], ldata['u'], ldata['u0'], ldata['wavs'])
			print(f'IperF.shape {IperF.shape} u.shape {u.shape} u0.shape {u0.shape} wavs.shape {wavs.shape}')
			
			#if IperF.size < 3:
			#	print('INFO: Less than three points, cannot make a meaningful fit, skipping...')
			#	continue
				
			IperF0s, ks, log_IperF0s_var, ks_var, n_valid_idxs = fitscube.limb_darkening.minnaert.get_coefficients_v_wavelength(u0, u, IperF)	
			IperF0s_var = (np.exp(np.log(IperF0s)+np.sqrt(log_IperF0s_var)) - np.exp(np.log(IperF0s)-np.sqrt(log_IperF0s_var)))**2	

			print(f'IperF0s.shape {IperF0s.shape} ks.shape {ks.shape} log_IperF0s_var.shape {log_IperF0s_var.shape} ks_var.shape {ks_var.shape} n_valid_idxs.shape {n_valid_idxs.shape}')
						
			print(f'INFO: Saving minnaert coefficients to file {m_coefficients_lat_file}')
			np.savez(m_coefficients_lat_file, 
				IperF0s=IperF0s, 
				ks=ks, 
				IperF0s_var=IperF0s_var, 
				ks_var=ks_var, 
				wavs=wavs, 
				log_IperF0s_var=log_IperF0s_var, 
				n_points=IperF.shape[1]*np.ones_like(wavs), 
				n_used_points=n_valid_idxs
			)
			minnaert_coefficient_files[l].append(m_coefficients_lat_file) # add file to written files list
			
			# DEBUGGING
			#sys.exit('DEBUG: Exiting for test cases')
	return(minnaert_coefficient_files)

# create lat v wav v k and lat v wav v IperF0 colourbar plot
def plot_k_IperF0_wav(target_cubes, minnaert_coefficient_files, overwrite=True):
	lat_v_wav_v_IperF0 = None
	lat_v_wav_v_k = None
	lat_grid = np.linspace(-90,90,180)
	lat_v_wav_coverage = None
	lat_v_wav_v_IperF0_std = None
	lat_v_wav_v_k_std = None
	
	nr, nc, s = (2, 3, 6)
	f0 = plt.figure(figsize=(nc*s,nr*s))
	for l, tc in enumerate(target_cubes):
		print(f'INFO: working on cube {l}/{len(target_cubes)}')
		print(f'INFO: cube path {tc}')
		tc_name = os.path.basename(os.path.splitext(tc)[0])
		plot_file_name = f'{os.path.dirname(tc)}/minnaert/plots/{tc_name}_lat_wav_IperF0_k.png'
		if os.path.exists(plot_file_name) and (not overwrite):
			print(f'INFO: File {plot_file_name} already exists and will not be overwritten, skipping...')
			continue
		
		#glob_str = f'{os.path.dirname(tc)}/minnaert/{tc_name}_minnaert_coefficients_lat_*_to_*.npz'
		m_coefficients_lat_files = minnaert_coefficient_files[l] #glob.glob(glob_str)
		for m, m_coefficients_lat_file in enumerate(m_coefficients_lat_files):
			print(f'INFO: Operating on file {m_coefficients_lat_file}')
			lat_max, lat_min = map(float,os.path.basename(m_coefficients_lat_file).rsplit('.',1)[0].split('_')[-1:-4:-2])
			print(f'INFO: latitude range {lat_min} to {lat_max}')
			dl = np.load(m_coefficients_lat_file)
			wavs, IperF0s, ks, log_IperF0s_var, ks_var, IperF0s_var, n_points, n_used_points = (dl['wavs'], dl['IperF0s'], dl['ks'], 
																		dl['log_IperF0s_var'], dl['ks_var'], dl['IperF0s_var'],
																		dl['n_points'], dl['n_used_points'])
			print(f'INFO: IperF0s.shape {IperF0s.shape}')
			print(f'INFO: ks.shape {ks.shape}')
			print(f'INFO: IperF0s_var.shape {IperF0s_var.shape}')
			print(f'INFO: ks_var.shape {ks_var.shape}')
			print(f'INFO: wavs.shape {wavs.shape}')
			# create numpy arrays to hold data
			if lat_v_wav_v_IperF0 is None:
				lat_v_wav_v_IperF0 = np.zeros((lat_grid.size,wavs.size))
			if lat_v_wav_v_k is None:
				lat_v_wav_v_k = np.zeros((lat_grid.size,wavs.size))
			if lat_v_wav_coverage is None:
				lat_v_wav_coverage = np.zeros((lat_grid.size, wavs.size))
			if lat_v_wav_v_IperF0_std is None:
				lat_v_wav_v_IperF0_std = np.zeros((lat_grid.size, wavs.size))
			if lat_v_wav_v_k_std is None:
				lat_v_wav_v_k_std = np.zeros((lat_grid.size, wavs.size))
			
			# fill numpy arrays with data
			lat_idxs = np.nonzero(np.logical_and(lat_min <= lat_grid, lat_grid <= lat_max))
			lat_v_wav_v_IperF0[lat_idxs,:] += IperF0s[None,:]
			lat_v_wav_v_k[lat_idxs,:] += ks[None,:]
			lat_v_wav_coverage[lat_idxs,:] += 1
			lat_v_wav_v_IperF0_std[lat_idxs,:] += np.sqrt(IperF0s_var)[None,:]
			lat_v_wav_v_k_std[lat_idxs,:] += np.sqrt(ks_var)[None,:]
			
	
		covered_idxs = np.nonzero(lat_v_wav_coverage)
		not_covered_idxs = np.nonzero(lat_v_wav_coverage == 0)
		lat_v_wav_v_IperF0[covered_idxs] /= lat_v_wav_coverage[covered_idxs]
		lat_v_wav_v_k[covered_idxs] /= lat_v_wav_coverage[covered_idxs]
		lat_v_wav_v_IperF0_std[covered_idxs] /= lat_v_wav_coverage[covered_idxs]
		lat_v_wav_v_k_std[covered_idxs] /= lat_v_wav_coverage[covered_idxs]
		lat_v_wav_v_IperF0[not_covered_idxs] = np.nan
		lat_v_wav_v_k[not_covered_idxs] = np.nan
		lat_v_wav_v_IperF0_std[not_covered_idxs] = np.nan
		lat_v_wav_v_k_std[not_covered_idxs] = np.nan
	
		f0.clear()
		a0 = f0.subplots(nr,nc, squeeze=False).T
		
		norm = mpl.colors.SymLogNorm(1E-3*np.nanmax(lat_v_wav_v_IperF0), linscale=1, vmin=0, vmax=np.nanmax(lat_v_wav_v_IperF0))
		extent = [wavs[0], wavs[-1], lat_grid[0], lat_grid[-1]]
		
		im000 = a0[0,0].imshow(lat_v_wav_v_IperF0, origin='lower', cmap='viridis', 
						   norm=norm, extent=extent, aspect='auto')
		im001 = a0[0,1].imshow(lat_v_wav_v_k, origin='lower', vmin=0, vmax=1, cmap='viridis', 
						   extent=extent, aspect='auto')
		im010 = a0[1,0].imshow(lat_v_wav_v_IperF0_std, origin='lower', cmap='viridis',
							norm=norm, extent = extent, aspect='auto')
		im011 = a0[1,1].imshow(lat_v_wav_v_k_std, origin='lower', vmin=0, vmax=1, cmap='viridis',
							extent=extent, aspect='auto')
		
		norm2 = mpl.colors.SymLogNorm(1E-3*np.nanmax(lat_v_wav_v_IperF0/lat_v_wav_v_IperF0_std), 
								linscale=1, vmin=0, vmax=np.nanmax(lat_v_wav_v_IperF0/lat_v_wav_v_IperF0_std))
		im020 = a0[2,0].imshow(lat_v_wav_v_IperF0/lat_v_wav_v_IperF0_std, origin='lower', cmap='viridis',
							norm=norm2, extent = extent, aspect='auto')
		norm3 = mpl.colors.SymLogNorm(1E-3*np.nanmax(lat_v_wav_v_k/lat_v_wav_v_k_std), 
								linscale=1, vmin=0, vmax=np.nanmax(lat_v_wav_v_k/lat_v_wav_v_k_std))
		im021 = a0[2,1].imshow(lat_v_wav_v_k/lat_v_wav_v_k_std, origin='lower', cmap='viridis',
							extent=extent, aspect='auto', norm=norm3)
		
		a0[0,0].set_title('IperF0')
		a0[0,0].set_xlabel('wavlength (um)')
		a0[0,0].set_ylabel('latitude (deg)')
		a0[0,1].set_title('k')
		a0[0,1].set_xlabel('wavelength (um)')
		a0[0,1].set_ylabel('latitude (deg)')
		a0[1,0].set_title('IperF0 std')
		a0[1,0].set_xlabel('wavlength (um)')
		a0[1,0].set_ylabel('latitude (deg)')
		a0[1,1].set_title('k std')
		a0[1,1].set_xlabel('wavelength (um)')
		a0[1,1].set_ylabel('latitude (deg)')
		a0[2,0].set_title('IperF0/error')
		a0[2,0].set_xlabel('wavlength (um)')
		a0[2,0].set_ylabel('latitude (deg)')
		a0[2,1].set_title('k/error')
		a0[2,1].set_xlabel('wavelength (um)')
		a0[2,1].set_ylabel('latitude (deg)')
		f0.colorbar(im000, ax=a0[0,0])
		f0.colorbar(im001, ax=a0[0,1])
		f0.colorbar(im010, ax=a0[1,0])
		f0.colorbar(im011, ax=a0[1,1])
		f0.colorbar(im020, ax=a0[2,0])
		f0.colorbar(im021, ax=a0[2,1])
		os.makedirs(os.path.dirname(plot_file_name), exist_ok=True)
		f0.savefig(plot_file_name, bbox_inches='tight')
		#plt.show()
	plt.close(f0)
	return()

def plot_k_IperF0_wav_per_lat(minnaert_coefficient_files, lat_grid=np.linspace(-90,90,180), wav_plot_lims=(1.455, 2.455), overwrite=True):
	n_mcf_list = len(minnaert_coefficient_files)
	
	for j, mcf_list in enumerate(minnaert_coefficient_files):
		n_mcf = len(mcf_list)
		wav_v_lat_v_IperF0 = None
		wav_v_lat_v_k = None
		wav_v_lat_coverage = None
		wavelength_plot_lim_range = wav_plot_lims  # wavelength range to set the plot limits with
		
		for i, mcf in enumerate(mcf_list):
			print(f'INFO: creating plots of {mcf} {i}/{n_mcf} of {j}/{n_mcf_list}')
			plot_file_name = f'{os.path.basename(mcf)[:-4]}.png'
			plot_dir = f'{os.path.dirname(mcf)}/plots'
			plot_file = os.path.join(plot_dir, plot_file_name)
			if os.path.exists(plot_file) and (not overwrite):
				print(f'INFO: File {plot_file} already exists and will not be overwritten, skipping...')
				continue
			
			lat_min = float(mcf.split('_')[-3])
			lat_max = float(mcf.split('_')[-1].rsplit('.',1)[0])
			print(lat_min, lat_max)
			
			
			dl = np.load(mcf)
			wavs, IperF0s, ks, log_IperF0s_var, ks_var, IperF0s_var, n_points, n_used_points = (dl['wavs'], dl['IperF0s'], dl['ks'], 
																		dl['log_IperF0s_var'], dl['ks_var'], dl['IperF0s_var'],
																		dl['n_points'], dl['n_used_points'])
			"""
			print(f'INFO: wavs {wavs.shape}')
			print(wavs)
			
			print(f'INFO: IperF0s {IperF0s.shape}')
			print(IperF0s)
			
			print(f'INFO: ks {ks.shape}')
			print(ks)
			
			print(f'INFO: log_IperF0s_var {log_IperF0s_var.shape}')
			print(log_IperF0s_var)
			
			print(f'INFO: ks_var {ks_var.shape}')
			print(ks_var)
			
			print(f'INFO: IperF0s_var {IperF0s_var.shape}')
			print(IperF0s_var)
			
			print(f'INFO: n_points {n_points.shape}')
			print(n_points)
			
			print(f'INFO: n_used_points {n_used_points.shape}')
			print(n_used_points)
			"""
			
			shape = (lat_grid.size, wavs.size)
			if wav_v_lat_coverage is None:
				wav_v_lat_coverage = np.zeros(shape)
			if wav_v_lat_v_IperF0 is None:
				wav_v_lat_v_IperF0 = np.zeros(shape)
			if wav_v_lat_v_k is None:
				wav_v_lat_v_k = np.zeros(shape)
			lat_idx = np.nonzero((lat_min < lat_grid) & (lat_grid < lat_max))
			wav_v_lat_v_IperF0[lat_idx,:] += IperF0s[None,:]
			wav_v_lat_v_k[lat_idx,:] += ks[None,:]
			wav_v_lat_coverage[lat_idx,:] += 1
			
			wr = (wavelength_plot_lim_range[0]<wavs) & (wavs<wavelength_plot_lim_range[1])
			nr,nc = (2,2)
			f1 = plt.figure(figsize=(nc*5,nr*5))
			a1 = f1.subplots(nr,nc,squeeze=False)
			
			a1[0,0].plot(wavs, IperF0s)
			a1[0,0].fill_between(wavs, IperF0s+np.sqrt(IperF0s_var), IperF0s-np.sqrt(IperF0s_var), color='tab:orange', alpha=0.5)
			a1[0,0].set_xlabel('Wavelength (um)')
			a1[0,0].set_ylabel('IperF0')
			if all(~np.isnan(IperF0s)):
				a1[0,0].set_ylim((np.nanmin(IperF0s[wr]),np.nanmax(IperF0s[wr])))
			a1[0,0].set_title('Value (line) Error (region)')
			
			a1[0,1].plot(wavs, ks)
			a1[0,1].fill_between(wavs, ks+np.sqrt(ks_var), ks-np.sqrt(ks_var), color='tab:orange', alpha=0.5)
			a1[0,1].set_xlabel('Wavelength (um)')
			a1[0,1].set_ylabel('k')
			a1[0,1].set_ylim((0,1))
			a1[0,1].set_title('Value (line) Error (region)')
			
			a1[1,0].remove()
			"""
			IperF0s_std = np.sqrt(IperF0s_var)
			ks_std = np.sqrt(ks_var)
			#print(IperF0s_var)
			#print(IperF0s_std)
			a1[1,0].plot(wavs, IperF0s_std, color='tab:orange')
			a1100 = a1[1,0].twinx()
			a1100.plot(wavs, ks_std, color='tab:green')
			a1100.set_ylabel('k_std (green)')
			a1[1,0].set_xlabel('Wavelength (um)')
			a1[1,0].set_ylabel('IperF0_std (orange)')
			if all(~np.isnan(IperF0s_std)):
				a1[1,0].set_ylim((np.nanmin(IperF0s_std[wr]),np.nanmax(IperF0s_std[wr])))
			"""
			
			a1[1,1].plot(wavs, n_points, label='n_points')
			a1[1,1].plot(wavs, n_used_points, label='n_used_points')
			a1[1,1].set_xlabel('Wavelength (um)')
			a1[1,1].set_ylabel('Number of points input to minnaert curve fit')
			a1[1,1].set_title('Number of pixels in minnaert fit')
			#a1[1,1].set_ylim((np.nanmin(ks_std[wr]),np.nanmax(ks_std[wr])))
			a1[1,1].legend()
			
			f1.suptitle(f'minnaert coefficients from lat {lat_min:+07.2f} to {lat_max:+07.2f}')
			
			print(f'INFO: {plot_file}')
			f1.savefig(plot_file, bbox_inches='tight')
			plt.close(f1)
		
		if wav_v_lat_coverage is not None:
			covered = wav_v_lat_coverage != 0
			wav_v_lat_v_IperF0[covered] /= wav_v_lat_coverage[covered]
			wav_v_lat_v_k[covered] /= wav_v_lat_coverage[covered]
			wav_v_lat_v_IperF0[~covered] = np.nan
			wav_v_lat_v_k[~covered] = np.nan
	return(wav_v_lat_v_IperF0, wav_v_lat_v_k)

# create animated plots of minnaert data and minnaert fit for each latitude slice
def plot_animated_minnaert_fit_each_lat(target_cubes, minnaert_input_files, minnaert_coefficient_files, overwrite=True):
	f2 = plt.figure()
	for l, tc in enumerate(target_cubes):
		print(f'INFO: working on cube {l}/{len(target_cubes)}')
		print(f'INFO: cube path {tc}')
		tc_name = os.path.basename(os.path.splitext(tc)[0])
		#glob_str = f'{os.path.dirname(tc)}/minnaert/{tc_name}_minnaert_coefficients_lat_*_to_*.npz'
		m_coefficients_lat_files = minnaert_coefficient_files[l] #glob.glob(glob_str)
		#glob_str = f'{os.path.dirname(tc)}/minnaert/{tc_name}_minnaert_inputs_lat_*_to_*.npz'
		m_input_lat_files = minnaert_input_files[l] #glob.glob(glob_str)
		
		for m, m_coefficients_lat_file in enumerate(m_coefficients_lat_files):
			m_input_lat_file = m_input_lat_files[m] #m_coefficients_lat_file.replace('_coefficients_', '_inputs_')
			print(f'INFO: Working on animation {m}/{len(m_coefficients_lat_files)} of {l}/{len(target_cubes)}')
			print(f'INFO: input file {os.path.basename(m_input_lat_file)}')
			print(f'INFO: coefficient file {os.path.basename(m_coefficients_lat_file)}')
			lat_max, lat_min = map(float,os.path.basename(m_input_lat_file).rsplit('.',1)[0].split('_')[-1:-4:-2])
			outfolder=f'{os.path.dirname(m_coefficients_lat_file)}/plots/'
			outfile_name = f'{tc_name}_lat_{lat_min:+04.1f}_to_{lat_max:+04.1f}_IperF0_k_fit.mov'
			outfile = os.path.join(outfolder,outfile_name)
			if os.path.exists(outfile) and (not overwrite):
				print(f'INFO: File {outfile} already exists and will not be overwritten, skipping...')
				continue
			
			lcdata = np.load(m_coefficients_lat_file)
			IperF0s, ks, IperF0s_var, ks_var, wavs, log_IperF0s_var = (lcdata['IperF0s'], 
																 lcdata['ks'], 
																 lcdata['IperF0s_var'], 
																 lcdata['ks_var'], 
																 lcdata['wavs'], 
																 lcdata['log_IperF0s_var'])
			
			lidata = np.load(m_input_lat_file)
			IperF, u, u0, wavs = (lidata['IperF'], lidata['u'], lidata['u0'], lidata['wavs'])
			
			zen_grid = np.linspace(0,85,50)
			u_grid = np.linspace(np.min(u),np.max(u)) # np.cos(zen_grid*np.pi/180)
			u0_grid = u_grid[:]
			#log_u0u_grid = np.broadcast_to(np.log(u_grid*u0_grid), (ks.shape[0], u_grid.shape[0]))
			log_u0u_grid = np.log(u_grid*u0_grid)
			
			log_uIperF_fit = np.log(IperF0s[:,None]) + ks[:,None]*log_u0u_grid
			
			log_uIperF = np.log(u*IperF)
			log_u0u = np.log(u0*u)
			
			worst_log_uIperF_fits = np.zeros((ks.shape[0], u_grid.shape[0], 4))
			ks_lims = (ks+np.sqrt(ks_var), ks-np.sqrt(ks_var))
			log_IperF0s_lims = (np.log(IperF0s)+np.sqrt(log_IperF0s_var), np.log(IperF0s)-np.sqrt(log_IperF0s_var))
			_i = 0
			for _k in ks_lims:
				for _l in log_IperF0s_lims:
					worst_log_uIperF_fits[:,:,_i] = _l[:,None] + _k[:,None]*log_u0u_grid
					_i += 1
					
			wf_min = np.nanmin(worst_log_uIperF_fits, axis=-1)
			wf_max = np.nanmax(worst_log_uIperF_fits, axis=-1)
			
			f2.clear()
			a2 = f2.subplots(1,1,squeeze=False)
			def update(i):
				a2[0,0].clear()
				a2[0,0].plot(log_u0u[i,:], log_uIperF[i,:], '.')
				a2[0,0].plot(log_u0u_grid, log_uIperF_fit[i,:], '-', color='tab:pink', label='Minnaert fitted line')
				a2[0,0].fill_between(log_u0u_grid, wf_min[i,:], wf_max[i,:], color='tab:pink', alpha=0.5, label='Minnaert fitted line error region')
				
				f2.suptitle(f'{tc[tc.find("SINFO"):]}', fontsize=6)
				a2[0,0].set_title(f'Minnaert limb darkening at wav {wavs[i]:05.3f} um\nlat {lat_min:+#04.1f} to {lat_max:+#04.1f} IperF0 {IperF0s[i]:05.2E} k {ks[i]:03.2f}')
				a2[0,0].set_xlabel('log[u0 u]')
				a2[0,0].set_ylabel('log[u (I/F)]')
				a2[0,0].legend()
				return
			ani = mpl.animation.FuncAnimation(f2, update, range(0,wavs.size), interval=50)
			
			os.makedirs(outfolder, exist_ok=True)
			ani.save(outfile, progress_callback=plotutils.progress)
	return()
	
def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	"""
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	of them do.
	Quick reference:
	ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	"""
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

	parser.add_argument('target_cubes', type=str, nargs='+', help='A list of fitscubes to operate on')
	
	parser.add_argument('--cube.extension', type=str, help='Extension of the target cube to operate on', default='PRIMARY')
	
	parser.add_argument('--save.dir', type=str, help='Directory to save results to. If preceded with a "./" will be relative to each target cube', 
					 default='./minnaert')
	parser.add_argument('--mask.file', type=str, help='Mask to apply to target cube. If preceded with a "./" will be relative to each target cube',
					 default='./auto_mask_cloud.fits')
	parser.add_argument('--lat.spacing.min', type=float, help='Minimum value of latitude spacing', default=-90)
	parser.add_argument('--lat.spacing.max', type=float, help='Maximum value of latitude spacing', default=+90)
	parser.add_argument('--lat.spacing.n', type=int, help='Number of latitude bins', default=37)
	parser.add_argument('--lat.spacing.microstep', action=plotutils.ActionTf, prefix='lat.spacing.', help='Should we microstep the latitude bins?')
	
	parsed_args = vars(parser.parse_args(argv))
	
	return(parsed_args)

# setup inputs

def main(argv):
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	lat_limits = np.linspace(args['lat.spacing.min'], args['lat.spacing.max'], args['lat.spacing.n'])
	if args['lat.spacing.microstep']:
		lat_limits = np.vstack([lat_limits[:-2:1], lat_limits[2::1]])
	else:
		lat_limits = np.vstack([lat_limits[:-1:2], lat_limits[1::2]])
	
	print(lat_limits)
	
	minnaert_dirs = [os.path.join(os.path.dirname(_x), args['save.dir']) for _x in args['target_cubes']]
	mask_files = [os.path.join(os.path.dirname(_x), args['mask.file']) for _x in args['target_cubes']]
	
	minnaert_input_files = save_minnaert_inputs(target_cubes, mask_files, minnaert_dirs, lat_limits.T, args['cube.extension'], overwrite=True)
	
	minnaert_coefficient_files = save_minnaert_coefficients(target_cubes, minnaert_input_files, overwrite=True)
	
	plot_k_IperF0_wav(target_cubes, minnaert_coefficient_files, overwrite=True)
	
	plot_k_IperF0_wav_per_lat(minnaert_coefficient_files, overwrite=True)
	
	# this takes a long time so don't bother unless you need it
	#plot_animated_minnaert_fit_each_lat(target_cubes, minnaert_input_files, minnaert_coefficient_files, overwrite=False)

if __name__=='__main__':
	# DEBUGGING create arguments
	if sys.argv[0] == '':
		plt.ioff()
		fits_dir = os.path.expanduser('~/scratch/reduced_images')
		target_cubes = glob.glob(f'{fits_dir}/*/*/analysis/obj_*_renormed_cleanMR2_200.fits')
		sys.argv = ['', *target_cubes, 
						'--save.dir', './minnaert', 
						'--mask.file', './auto_mask_clean_cloud.fits', 
						'--cube.extension', 'COMPONENTS', 
						'--lat.spacing.min', '-90', 
						'--lat.spacing.max', '90', 
						'--lat.spacing.microstep', # --lat.spacing.no_microstep,
						'--lat.spacing.n', '37' #'2'
						]
		print(sys.argv)
	main(sys.argv[1:])
	
	
	
	#lat_limits = np.vstack([np.linspace(-90, 80, 35), np.linspace(-80,90,35)]).T
	