#!/usr/bin/env python3
"""
Contains routines for calculating Minnaert limb darkeing coefficients.

Minnaert limb darkening equation:

			 k  k-1
 I     /I\  u  u
--- = |---|  0                                            (1)
 F     \F/
		 0

Where:
	u = cos(zenith)

	u_0 = cos(solar zenith)
	
	k, (I/F)_0 are fixed parameters
	
	I/F is the flux at the zenith angle we are interested in

In out case, the solar zenith angle and the zenith angle are approximately the 
same as the earth and sun are very close when seen from Neptune

Taking the logarithm of (1) gives:

	ln[u(I/F)] = ln[(I/F)_0] + k ln[u_0 u]                (2)

-------------------------------------------------------------------------------

The `if __name__=='__main__':` statement allows execution of code if the script
is called directly. Eveything else not in that block will be executed when a 
script is imported. Import statements that the rest of the code relies upon 
should not be in the if statement, python is quite clever and will only import 
a given package once, but will give it multiple names if it has been imported 
under different names.

Standard library documentation can be found at 
https://docs.python.org/3/library/

Packages used in this program are:
	sys
	os 

-------------------------------------------------------------------------------
"""

import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
#import utils as ut # used for convenience functions
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import scipy as sp
import scipy.interpolate
import scipy.optimize

import utilities as ut
#import utilities.plt as plotutils
import utilities.plt
#import plotutils
import utilities.fits
import utilities.args
from utilities.classes import Slice
#import fitscube.header
"""
TODO:
* Re-write main function to find the minnaert inputs from a cube and write them to a file (IperF, u, u0)
* Make the minnaert inputs data be groupable by latitude range
* Is there a way to simplify my *.spx creation routines?
"""

def main(argv):
	"""This code will be executed if the script is called directly"""

	args = parse_args(argv)
	print('# ARGUMENTS #')
	for k, v in sorted(args.items()):
		print(f'\t{k}')
		print(f'\t\t{v}')
	print('#############')

	if args['mode'] == 'individual':
		for tc in args['target_cubes']:
			tc_mask = os.path.join(os.path.dirname(tc), args['mask']) if args['mask'] is not None else None
			with fits.open(tc) as hdul:

				# Choose only those pixels that have an associated ZENITH value
				bd_mask = (hdul[args['fits.sci.exts']['SCI']].data < args['data.nan_threshold']) | (np.abs(hdul[args['fits.sci.exts']['SCI']].data) < args['data.nan_cutoff'])
				bz_mask = (np.abs(hdul[args['fits.sci.exts']['ZENITH']].data -45) > 45) | np.isnan(hdul[args['fits.sci.exts']['ZENITH']].data)
				hdul[args['fits.sci.exts']['SCI']].data[bd_mask | bz_mask[None,...]] = np.nan
				hdul[args['fits.sci.exts']['ZENITH']].data[bz_mask] = np.nan

				IperF, u, u_0 = get_minnaert_inputs_by_wav_mask_file(hdul, tc_mask, hdul_exts=args['fits.sci.exts'])

				# DEBUGGING
				"""
				print(f'{IperF.shape=}')
				print(f'{u.shape=}')
				fd1, dax = plt.subplots(1,3, figsize=(12,6),squeeze=False)
				idx = hdul[args['fits.sci.exts']['SCI']].data.shape[0]//2
				dax[0,0].imshow(hdul[args['fits.sci.exts']['SCI']].data[idx])
				dax[0,1].plot(IperF[idx])
				dax[0,2].plot(u[idx])

				plt.show()
				"""

				print(f'IperF.shape {IperF.shape}')
				print(f'u.shape {u.shape}')
				print(f'u_0.shape {u_0.shape}')

				(IperF0, k, residual, rank, s, 
				log_u0u, log_uIperF, valid_idxs, cov_mat) = get_coefficients(	u_0[u_0.shape[0]//2,:], u[u.shape[0]//2,:], IperF[IperF.shape[0]//2,:])
		
				print('Plotting minnaert fit')
				f1 = plt.figure()
				a11 = f1.add_subplot(1,1,1)
				ph1 = plot_coefficient_fit(f1, a11, log_u0u, log_uIperF, k, np.log(IperF0))
				f1.suptitle(f'{k=} {np.log(IperF0)=}')
				a11.set_ylim(args['plot.fit.ylim'])
				a11.set_xlim(args['plot.fit.xlim'])
				ut.plt.save_show_plt(f1, ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='minnaert_fit'), args['plots.mode'], args['plots.make_dirs'])
				#plotutils.save_show_plt(f1, 'minnaert_fit.png', 
				#						outfolder=plot_dir, 
				#						show_plot=args['plots.show'], 
				#						save_plot=args['plots.save'])
				#plt.show()
				

				"""
				f2 = plt.figure(figsize=[24,8])
				ax2 = f2.subplots(1, 3)
				ph2 = plot_coefficient_fit_grid(f2, ax2, u_0, u, IperF, k, IperF0)
				ut.plt.save_show_plt(f2, ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='coefficient_fit'), args['plots.mode'], args['plots.make_dirs'])
				plt.show()
				"""	

				print('Plotting minnaert coefficient variation with wavelength')
				IperF0s, ks, log_IperF0s_var, ks_var, n_valid_idxs  = get_coefficients_v_wavelength(u_0, u, IperF)
				print(f'IperF0s.shape {IperF0s.shape}')
				print(f'ks.shape {ks.shape}')
				f3 = plt.figure(figsize=[12,6])
				ax3 = f3.subplots(1,2)
				#wavs = fitscube.header.get_wavelength_grid(hdul[0])
				wavs = ut.fits.hdr_get_axis_world_coords(hdul[args['fits.sci.exts']['SCI']].header, 
					ut.fits.hdr_get_spectral_axes(hdul[args['fits.sci.exts']['SCI']].header)
				).squeeze()
				if wavs.ndim == 0: wavs=np.array([wavs])
				print(f'{wavs=}')
				ph3 = plot_coefficients_v_wavelength(f3, ax3, wavs, IperF0s, ks, wslice=np.s_[:])
				f3.suptitle('Minnaert Parameter Variation vs Wavelength')
				#plt.show()
				ut.plt.save_show_plt(f3, ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='minnaert_coefficient_vs_wavelength'), args['plots.mode'], args['plots.make_dirs'])
				#plotutils.save_show_plt(f1, 
				#			'minnaert_coefficient_vs_wavelength.png', 
				#			outfolder=plot_dir, show_plot=args['plots.show'], 
				#			save_plot=args['plots.save'])
				
				print('Animating minnaert fit with wavelength')
				f4 = plt.figure()
				a41 = f4.add_subplot(1,1,1)
				ani4 = animate_coefficient_fit(f4, a41, log_u0u, np.log(u*IperF), ks, np.log(IperF0s), wavs)
				ut.plt.save_animate_plt(ani4, ut.plt.get_plot_fpath(args['plots.fmt'], source_fpath=os.path.abspath(tc), name='minnaert_fit', ext='.mov'), args['plots.mode'], args['plots.make_dirs'])
				#plotutils.save_animate_plt(ani4, 'minnaert_fit.mov', plot_dir, save_plot=args['plots.save'], animate_plot=args['plots.show'])
	
				hdul = apply_correction(hdul, IperF0s, ks, args['fits.sci.exts']['SCI'], args['fits.sci.exts']['ZENITH'], args['fits.sci.exts'].get('SOLAR_ZENITH', args['fits.sci.exts']['ZENITH']), angles_in_radians=False)

				tc_root, tc_ext = os.path.splitext(tc)
				mc_filename = f'{tc_root}_minnaert_corrected{tc_ext}'
				print(f'Writing minnaert corrected cube to {mc_filename}')
				hdul.writeto(mc_filename, overwrite=True)


	elif args['mode'] == 'combined':
		IperFl = []
		ul = []
		u_0l = []
		for i, tc in enumerate(args['target_cubes']):
			tc_mask = os.path.join(os.path.dirname(tc), args['mask']) if args['mask'] is not None else None
			with fits.open(tc) as hdul:
				"""
				# how to slice by latitude or some other property
				IperF, u, u_0 = get_minnaert_inputs_by_wav(hdul, 
														tc_mask, 
														aslice = np.broadcast_to(np.logical_and(hdul['LATITUDE'].data >-10,
																								hdul['LATITUDE'].data < 0), 
																					hdul['PRIMARY'].data.shape)
														)
				"""
				IperF, u, u_0 = get_minnaert_inputs_by_wav_mask_file(hdul, tc_mask, hdul_exts=args['fits.sci.exts'])
				IperF, u, u_0 = make_arrays_smaller([IperF, u, u_0], reorder_by=u, nout_max=1000)
				IperFl.append(IperF)
				ul.append(u)
				u_0l.append(u_0)
				
			print(f'IperFl[{i}].shape {IperFl[i].shape} Memory {IperFl[i].itemsize*IperFl[i].size/(2**20):.3f} Mb')
			print(f'ul[{i}].shape {ul[i].shape} Memory {ul[i].itemsize*ul[i].size/(2**20):.3f} Mb')
			print(f'u_0l[{i}].shape {u_0l[i].shape} Memory {u_0l[i].itemsize*u_0l[i].size/(2**20):.3f} Mb')
			print(f'Sum of memory in IperFl, ul u_0l {np.sum([np.sum([_x.itemsize*_x.size for _x in _l]) for _l in [IperFl, ul, u_0l]])/(2**20):.3f} Mb\n')
		# DEBUG
		# calculate memory needed to concatenate
		print(f'Memory needed to concatenate IperFl {np.sum([_x.itemsize*_x.size for _x in IperFl])/(2**20):.3f} Mb')

		# Here we are combininng all arrays along their last axis,
		# this is for the situations where we don't care which target cube the data comes from,
		# just which wavelength it pertains to. We fit the parameters to these data, but we plot
		# the data in list form because then we can see if the different target cubes separate into
		# different populations or not.
		IperFs = np.concatenate(IperFl, axis=-1)
		us = np.concatenate(ul, axis=-1)
		u_0s = np.concatenate(u_0l, axis=-1)
		print(f'IperFs.shape {IperFs.shape}')
		print(f'us.shape {us.shape}')
		print(f'u_0s.shape {u_0s.shape}')
		
		(IperF0, k, residual, rank, s, 
		log_u0u, log_uIperF, valid_idxs, cov_mat) = get_coefficients(u_0s[300,:], us[300,:], IperFs[300,:])


		print('INFO: Minnaert parameters at wavelength bin 300')
		print(cov_mat)
		#sigma_IperF0 = (np.exp(cov_mat[0,0]) - 1)*np.exp(cov_mat[0,0])
		#sigma_IperF0 = cov_mat[0,0]/(np.log(IperF0)**2)
		sigma_IperF0 = (np.exp(np.log(IperF0)+np.sqrt(cov_mat[0,0])) - np.exp((np.log(IperF0)-np.sqrt(cov_mat[0,0]))))**2
		print(f'IperF0 {IperF0} {sigma_IperF0}')
		print(f'log_IperF0 {np.log(IperF0)} {cov_mat[0,0]}')
		print(f'k {k} {cov_mat[1,1]}')
			
		f1 = plt.figure()
		a11 = f1.add_subplot(1,1,1)
		#ph1 = plot_coefficient_fit(f1, a11, log_u0u, log_uIperF, k, np.log(IperF0))
		log_u0ul =  [np.log(u_0*u) for u_0, u in zip(u_0l,ul)]
		log_uIperFl = [np.log(u*IperF) for IperF, u in zip(IperFl, ul)]
		ph1 = plot_coefficient_fit_list(f1, a11, [_x[300] for _x in log_u0ul], [_x[300] for _x in log_uIperFl], k, np.log(IperF0), np.sqrt(cov_mat[1,1]), np.sqrt(cov_mat[0,0]))
		ut.plt.save_show_plt(f1, ut.plt.get_plot_fpath(args['plots.fmt'], name='combined_minnaert_fit'), args['plots.mode'], args['plots.make_dirs'])
		#plotutils.save_show_plt(f1, 'minnaert_fit.png', outfolder=args['plots.dir'], show_plot=args['plots.show'], save_plot=args['plots.save'])

		"""
		f2 = plt.figure(figsize=[24,8])
		ax2 = f2.subplots(1, 3)
		ph2 = plot_coefficient_fit_grid(f2, ax2, u_0, u, IperF, k, IperF0)
		plt.show()
		"""

		IperF0s, ks, log_IperF0s_var, ks_var = get_coefficients_v_wavelength(u_0s, us, IperFs)
		IperF0s_var = (np.exp(np.log(IperF0s)+np.sqrt(log_IperF0s_var)) - np.exp(np.log(IperF0s)-np.sqrt(log_IperF0s_var)))**2
		print(f'IperF0s.shape {IperF0s.shape}')
		print(f'ks.shape {ks.shape}')
		f3 = plt.figure(figsize=[12,6])
		ax3 = f3.subplots(1,2)
		wavs = fitscube.header.get_wavelength_grid(hdul[0])
		ph3 = plot_coefficients_v_wavelength(f3, ax3, wavs, IperF0s, ks, wslice=np.s_[100:2000], IperF0s_err=np.sqrt(IperF0s_var), ks_err=np.sqrt(ks_var))
		f3.suptitle('Minnaert Parameter Variation vs Wavelength')
		ut.plt.save_show_plt(f3, ut.plt.get_plot_fpath(args['plots.fmt'], name='combined_minnaert_coefficients_vs_wavelength'), args['plots.mode'], args['plots.make_dirs'])
		#plotutils.save_show_plt(f3, 'minnaert_coefficients_vs_wavelength.png', outfolder=args['plots.dir'], show_plot=args['plots.show'], save_plot=args['plots.save'])


		f4 = plt.figure()
		a41 = f4.add_subplot(1,1,1)
		ani4 = animate_coefficient_fit_list(f4, a41, log_u0ul, log_uIperFl, ks, np.log(IperF0s), wavs, log_IperF0s_err=np.sqrt(log_IperF0s_var), ks_err=np.sqrt(ks_var))
		ut.plt.save_show_ani(ani4, ut.plt.get_plot_fpath(args['plots.fmt'], name='combined_minnaert_fit', ext='.mov'), args['plots.mode'], args['plots.make_dirs'])
		#plotutils.save_animate_plt(ani4, 'minnaert_fit.mov', args['plots.dir'], save_plot=args['plots.save'], animate_plot=args['plots.show'])
	else:
		print('ERROR: Unknown "--mode" {}.'.format(args['mode']))
	
	return()

def make_arrays_smaller(array_list, reorder_by=None, nout_max=100):
	if reorder_by is not None:
		sort_idxs = np.argsort(reorder_by)

	return_list = []
	for arr in array_list:
		if arr.shape[1] < 1000:
			return_list.append(arr)
			continue
		if reorder_by is None:
			arr.sort()
		else:
			arr = np.take_along_axis(arr, sort_idxs,1)
			
		# try to reduce data by interpolation and averaging
		arr_interp = sp.interpolate.interp1d(np.linspace(0,1,arr.shape[1]), arr, kind='linear', assume_sorted=True)
		return_list.append(arr_interp(np.linspace(0,1,nout_max)))
		
	return(return_list)


def get_minnaert_inputs_by_wav_mask_file(hdul, mask=None, aslice=np.s_[...], show_plots=False, hdul_exts=0):
	#print(aslice.shape)
	#print(hdul['PRIMARY'].data.shape)
	#print(mask)
	if mask is None:
		mask_3d = np.zeros_like(hdul[hdul_exts['MASK']].data, dtype=bool)
	else:
		with fits.open(mask) as mask_hdul:
			mask_3d = np.array(mask_hdul[hdul_exts['MASK']].data, dtype=bool)
	if aslice==np.s_[...]:
		aslice = np.ones_like(hdul[hdul_exts['SCI']].data, dtype=bool)
	#print(mask_3d.shape)
	mask_3d = np.logical_or(mask_3d, ~aslice) # should this be and?
	return(get_minnaert_inputs_by_wav_mask_array(hdul, mask_3d=mask_3d, show_plots=show_plots, hdul_exts=hdul_exts))

def get_minnaert_inputs_by_wav_mask_array(hdul, mask_3d=None, show_plots=False, hdul_exts=0, angle2rad_factor=np.pi/180):
	nz,ny,nx = hdul[hdul_exts['SCI']].data.shape
	if mask_3d is None:
		mask_3d = np.zeros_like((nz,ny,nx), dtype=bool)
	if show_plots:
		plot_mask(hdul[hdul_exts['SCI']].data, mask_3d)

	IperF = hdul[hdul_exts['SCI']].data[~mask_3d].reshape(nz, -1)
	u = np.broadcast_to(np.cos(hdul[hdul_exts['ZENITH']].data*angle2rad_factor), (nz,ny,nx))[~mask_3d].reshape(nz, -1)
	u_0 = np.broadcast_to(np.cos(hdul[hdul_exts.get('SOLAR_ZENITH',hdul_exts['ZENITH'])].data*angle2rad_factor), (nz,ny,nx))[~mask_3d].reshape(nz, -1) # same as 'u' for outer planets for others change to 'SOLAR_ZENITH'

	return(IperF.astype(float), u.astype(float), u_0.astype(float))	

def plot_mask(data, mask):
	"""
	Plots mask so we can see if we're doing the right thing
	"""
	plt.imshow(np.nanmedian(data*(~mask), axis=0), origin='lower')
	plt.show()
	return

def animate_coefficient_fit_list(fig, ax, log_u0ul, log_uIperFl, ks, log_IperF0s, wavs, ks_err=None, log_IperF0s_err=None, err_scaling=None, **kwargs):
	"""
	An animated version of "plot_coefficient_fit_list()"
	"""
	import matplotlib as mpl
	import matplotlib.animation
	
	s = []
	colours = iter(plt.cm.rainbow(np.linspace(0,1,len(log_u0ul))))
	for i, (log_u0u, log_uIperF) in enumerate(zip(log_u0ul, log_uIperFl)):
		s_i = ax.plot([], [], label=f'minnaert data {i}', marker='.', markersize=0.5, linestyle='none', color=next(colours))
		s.append(s_i)
	p1 = ax.plot([], [], label='minnaert fitted line', color='tab:pink', **kwargs)
	
	log_u0u_span = np.linspace(min([np.nanmin(log_u0u) for log_u0u in log_u0ul]), max([np.nanmax(log_u0u) for log_u0u in log_u0ul]), 100)
	
	ax.set_xlabel('log[u_0 u]')
	ax.set_ylabel('log[u(I/F)]')
	ax.set_xlim([np.nanmin(log_u0u),np.nanmax(log_u0u)])
	ax.set_ylim([np.nanmin(log_uIperF), np.nanmax(log_uIperF)])

	pc1 = None
	
	def update(j):
		nonlocal pc1
		for i, (log_u0u, log_uIperF) in enumerate(zip(log_u0ul, log_uIperFl)):
			s[i][0].set_data(log_u0u[j], log_uIperF[j])
		p1[0].set_data(log_u0u_span, calculate_log_uIperF(log_IperF0s[j], ks[j], log_u0u_span))
		
		# do slow but working version of errors for now
		if (ks_err is not None) and (log_IperF0s_err is not None):
			if pc1 is not None:
				pc1.remove()
			k_lim = (ks[j]+ks_err[j], ks[j] - ks_err[j])
			l_lim = (log_IperF0s[j]+log_IperF0s_err[j], log_IperF0s[j]-log_IperF0s_err[j])
			worst_fits = np.zeros((log_u0u_span.shape[0],4))
			_i = 0
			for _k in k_lim:
				for _l in l_lim:
					worst_fits[:,_i] = calculate_log_uIperF(_l, _k, log_u0u_span)
					_i += 1
			wf_max = np.nanmax(worst_fits, axis=1)
			wf_min = np.nanmin(worst_fits, axis=1)
			if err_scaling is not None:
				fit = calculate_log_uIperF(log_IperF0s[j], ks[j], log_u0u_span)
				wf_max = fit + err_scaling*(wf_max- fit)
				wf_min = fit - err_scaling*(fit - wf_min)
			pc1 = ax.fill_between(log_u0u_span, wf_min, wf_max, color='tab:pink', alpha=0.5, label='Minnaert fitted line error')

		ax.set_title(f'Minnaert Combined Parameter Fit, Wavelength {wavs[j]:05.3f}')
		return
	
	ani = mpl.animation.FuncAnimation(fig, update, range(len(wavs)), interval=100)
	return(ani)
	
def plot_coefficient_fit_list(fig, ax, log_u0ul, log_uIperFl, k, log_IperF0, k_err=None, log_IperF0_err=None, err_scaling=None, **kwargs):
	"""
	Plot a scatter graph of the values passed as arguments along with a best fit derived from
	k and log_IperF0. Should plot each list element in a different colour
	"""
	s = []
	colours = iter(plt.cm.rainbow(np.linspace(0,1,len(log_u0ul))))
	for i, (log_u0u, log_uIperF) in enumerate(zip(log_u0ul, log_uIperFl)):
		s_i = ax.plot(log_u0u, log_uIperF, marker='.', markersize=0.5, c=next(colours), linestyle='none', label=f'data from input {i}')
		s.append(s_i)
	log_u0u_span = np.linspace(min([np.nanmin(log_u0u) for log_u0u in log_u0ul]), max([np.nanmax(log_u0u) for log_u0u in log_u0ul]), 100)
	p1 = ax.plot(log_u0u_span, calculate_log_uIperF(log_IperF0, k, log_u0u_span), label='minnaert fitted line', color='tab:pink', **kwargs)
	if (k_err is not None) and (log_IperF0_err is not None):
		k_err, log_IperF0_err = [_x if _x is not None else 0 for _x in (k_err, log_IperF0_err)] # if any of the errors are None then set them to zero
		k_lim = (k+k_err, k-k_err)
		log_IperF0_lim = (log_IperF0+log_IperF0_err, log_IperF0-log_IperF0_err)
		worst_fits = np.zeros((log_u0u_span.shape[0],4))
		_i = 0
		for _k in k_lim:
			for _l in log_IperF0_lim:
				worst_fits[:,_i] = calculate_log_uIperF(_l, _k, log_u0u_span)
				#print(_i, _k, _l)
				_i += 1
		wf_max = np.nanmax(worst_fits, axis=1)
		wf_min = np.nanmin(worst_fits, axis=1)
		if err_scaling is not None:
			fit =  calculate_log_uIperF(log_IperF0, k, log_u0u_span)
			wf_max = fit + err_scaling*(wf_max- fit)
			wf_min = fit - err_scaling*(fit - wf_min)
		#print('INFO:')
		#print(k_lim)
		#print(log_IperF0_lim)
		#print(wf_max)
		#print(wf_min)
		#print(worst_fits)
		p2 = ax.fill_between(log_u0u_span, wf_min, wf_max, color='tab:pink', alpha=0.5, label='Minnaert fitted line error region')

	ax.set_xlabel('log[u_0 u]')
	ax.set_ylabel('log[u(I/F)]')
	ax.set_title('Minnaert Limb Darkening Combined Parameter Fit')
	return(s, p1)
		
def get_coefficients_v_wavelength(u_0, u, IperF):
	"""
	Runs "get_coefficients()" for each wavelength and combines the output into a nice format

	# RETURNS #
		IperF0s
			<array, float> The values of IperF for each wavelength
		ks
			<array, float> The values of k for each wavelength
		log_IperF0s_var
			<array, float> The variance on the IperF0s values
		ks_va
			<array, float> The variance on the ks values
	"""
	#print('INFO: In "get_coefficients_v_wavelength()"')
	wavshape = IperF.shape[0]
	#print(f'INFO: wavshape {wavshape}')
	IperF0s = np.full(wavshape, fill_value = np.nan)
	ks = np.full(wavshape, fill_value = np.nan)
	log_IperF0s_var = np.full(wavshape, fill_value=np.nan)
	ks_var = np.full(wavshape, fill_value=np.nan)
	n_valid_idxs = np.full(wavshape, fill_value=0)
	for i in range(wavshape):
		#print(u_0[i,:], u[i,:], IperF[i,:])
		result = get_coefficients(u_0[i,:], u[i,:], IperF[i,:])
		IperF0s[i] = result[0]
		ks[i] = result[1]
		#IperF0s_err[i] = (np.exp(result[8][0,0]) - 1)*np.exp(result[8][0,0])
		#IperF0s_var[i] = (np.exp(np.log(result[0])+np.sqrt(result[8][0,0])) - np.exp(np.log(result[0])-np.sqrt(result[8][0,0])))**2
		log_IperF0s_var[i] = result[8][0,0]
		ks_var[i] = result[8][1,1]
		n_valid_idxs[i] = np.nansum(result[7])
	return(IperF0s, ks, log_IperF0s_var, ks_var, n_valid_idxs)
	
def plot_coefficients_v_wavelength(fig, ax, wavs, IperF0s, ks, wslice=np.s_[:], IperF0s_err=None, ks_err=None):
	"""
	Plots minnaert coefficients vs wavelength
	"""	
	IperF0s_err = np.zeros_like(IperF0s) if IperF0s_err is None else IperF0s_err
	ks_err = np.zeros_like(ks) if ks_err is None else ks_err
	print(f'{wavs=}')
	print(f'{IperF0s=}')
	print(f'{ks=}')
	print(f'{wslice=}')
	print(f'{IperF0s_err=}')
	print(f'{ks_err=}')


	s1 = ax[0].errorbar(wavs[wslice], IperF0s[wslice], yerr=IperF0s_err[wslice], marker='.', markersize=2, linestyle='-', linewidth=0.3)
	s2 = ax[1].errorbar(wavs[wslice], ks[wslice], yerr=ks_err[wslice], marker='.', markersize=2, linestyle='-', linewidth=0.3)

	lims = (np.nanmin(IperF0s[wslice]), np.nanmax(IperF0s[wslice]))
	print(f'{lims=}')

	ax[0].set_ylim(lims)
	ax[0].set_xlabel('Wavelength (um)')
	ax[0].set_ylabel('IperF0')
	ax[0].set_yscale('log')

	ax[1].set_ylim(0,1)
	ax[1].set_xlabel('Wavelength (um)')
	ax[1].set_ylabel('k')
	return(s1,s2)

def animate_coefficient_fit(fig, ax, log_u0u, log_uIperF, ks, log_IperF0s, wavs, **kwargs):
	"""
	An animated version of "plot_coefficient_fit()"
	"""
	import matplotlib as mpl
	import matplotlib.animation
	
	print(f'{log_u0u=}')
	print(f'{log_uIperF=}')
	print(f'{ks=}')
	print(f'{log_IperF0s=}')
	print(f'{wavs=}')

	s1 = ax.scatter([],[],label='minnaert data to fit', marker='.', color='tab:blue', s=0.5)
	p1 = ax.plot([], [], label='minnaert fitted line', color='tab:pink')

	ax.set_xlabel('log[u_0 u]')
	ax.set_ylabel('log[u(I/F)]')
	bad_value = lambda x: np.isnan(x) | np.isinf(x)
	xlim = (np.nanmin(log_u0u[~bad_value(log_u0u)]),np.nanmax(log_u0u[~bad_value(log_u0u)]))
	ylim = (np.nanmin(log_uIperF[~bad_value(log_uIperF)]), np.nanmax(log_uIperF[~bad_value(log_uIperF)]))

	vlines = []
	for _angle, _ls in ((30,'-'), (60,'--'), (80,'-.'), (89,':')):
		_x = np.log(np.cos(_angle*(np.pi/180))**2)
		l = ax.axvline(_x, color='red', ls=_ls, label=f'{_angle:.1f} degrees zenith')
		vlines.append(l)

	print(f'{xlim=}')
	print(f'{ylim=}')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	def update(i):
		s1.set_offsets(np.array([log_u0u, log_uIperF[i]]).T)
		p1[0].set_data(log_u0u, calculate_log_uIperF(log_IperF0s[i], ks[i], log_u0u))
		ax.set_title(f'Minnaert Parameter Fit\nWavelength {wavs[i]:09.2E} k {ks[i]:09.2E} IperF0 {np.exp(log_IperF0s[i]):09.2E}')
		ax.legend()
		return

	ani = mpl.animation.FuncAnimation(fig, update, range(len(wavs)), interval=100)
	return(ani)

def plot_coefficient_fit(fig, ax, log_u0u, log_uIperF, k, log_IperF0, **kwargs):
	"""
	Plots a scatter graph of the values passed as argument along with a best fit derived
	from k and log_IperF0
	"""
	
	s1 = ax.plot(log_u0u, log_uIperF, label='minnaert data to fit', marker='.', color='tab:blue', markersize=0.5, linestyle='none', **kwargs)
	p1 = ax.plot(log_u0u, calculate_log_uIperF(log_IperF0, k, log_u0u), label='minnaert fitted line', color='tab:orange', markersize=4, **kwargs)

	vlines = []
	for _angle, _ls in ((30,'-'), (60,'--'), (80,'-.'), (89,':')):
		_x = np.log(np.cos(_angle*(np.pi/180))**2)
		l = ax.axvline(_x, color='red', ls=_ls, label=f'{_angle:.1f} degrees zenith')
		vlines.append(l)

	ax.set_xlabel('log[u_0 u]')
	ax.set_ylabel('log[u(I/F)]')
	ax.set_title('Minnaert Limb Darkening Parameter Fit')
	ax.legend()
	
	return(s1, p1)


def plot_coefficient_fit_grid(fig, ax, u_0, u, IperF, k, IperF0, **kwargs):
	"""
	WORK IN PROGRESS
	Plots the fit to found Minnaert coefficients on a u. u_0 grid
	"""
	gu_0, gu = np.meshgrid(u_0, u)

	gIperF = np.full_like(gu, fill_value=np.nan)
	for i in range(len(IperF)): gIperF[i,i] = IperF[i]

	gIperF_fit = calculate(IperF0, k, gu_0, gu)
	gIperF_resid = gIperF - gIperF_fit

	im1 = ax[0].imshow(gIperF, origin='lower')
	im2 = ax[1].imshow(gIperF_fit, origin='lower')
	im3 = ax[2].imshow(gIperF_resid, origin='lower')

	return(im1, im2, im3)

def calculate(IperF0, k, u_0, u):
	"""
	Uses (1) to calculate I/F, should work just as well with arrays if they are all
	the same shape.

	# ARGUMENTS #
		IperF0
			<float> Brightness at zenith = 0
		k
			<float> Minnaert parameter that determines limb brightening/darkeing,
			usually between 0 and 1
		u_0
			<float> cos(solar zenith)
		u
			<float> cos(zenith)
	# RETURNS #
		IperF
			<float> Brightness according to (1)
	"""
	return(IperF0*(u_0**k)*(u**(k-1)))

def calculate_log(log_IperF0, k, log_u0u, u):
	"""
	Uses (2) to calculate log[(I/F)], should work fine with arrays
	"""
	return((log_IperF0 + k*log_u0u)**(1/u))

def calculate_log_uIperF(log_IperF0, k, log_u0u):
	"""
	Calculates (2) directly
	"""
	return(log_IperF0 + k*log_u0u)

def get_coefficients(u_0, u, IperF, two_points_ok_flag=False, weight=None, mode='least_squares'):
	"""
	Uses (2) to find k and (I/F)_0 for a set of u_0, u, (I/F) values

	re-write as a matrix equation of the form y=Ax, for each i^th triple of
	u_0, u, IperF we have:
		y = {... log[u(I/F)]_i ...}
		x = {log[(I/F)_0], k}
		A = {... 1_i ..., ... log[u_0 u]_i ...}
	We want to find the x that best predicts y.

	# ARGUMENTS #
		u_0 [n]
			<float, array> cos(solar zenith)
		u [n]
			<float, array> cos(zenith)
		IperF [n]
			<float, array> Radiance (or flux, some brightness measure)
		two_points_ok_flag
			<bool> If True, will let the algorithm operate on just two points.
			This means that the covariance matrix will not be informative
		weight
			<str> How should we weight the data. 
				None
					No weighting,
				"adjust_for_logscale"
					Apply weighting to adjust for logarithmic scale
		mode
			<str> How should we compute the coeffieicnts
				'least_squares'
					use sp.optimize.lst_squares
				'curve_fit'
					use sp.optimize.curve_fit

	# RETURNS #
		IperF0
			<float> The minnaert parameter that tells you the brightness at zenith=0
		k
			<float> The minnaert parameter that determines limb brightening vs limb 
			darkening. Should be between 0 and 1 (0.5 is no brightening or darkening).
		residual [m]
			<float, array> residual = y - Ax for the found values of x
		rank
			<int> Rank of matrix A
		s 
			<int, array> Singular values of A
		log_u0u [n]
			<float, array> The values of log[u_0 u] in (2)
		log_uIperF [n]
			<float, array> The values of log[u(I/F)] in (2)
		valid_idxs [m]
			<int, array> The indices of log_uIperF that are not NAN (as IperF could be
			less than zero due to background subtraction)
		cov_mat [2,2]
			<float, array> The covariance matrix for IperF0 and k
	"""
	# assume we have numpy arrays for all inputs
	# masked arrays don't play nice with scipy routines sometimes
	if type(IperF) == np.ma.MaskedArray:
		IperF = IperF.filled(fill_value=np.nan)

	# get inputs into the form of (2)
	log_u0u = np.log(u_0*u)
	log_uIperF = np.log(u*IperF)
	
	# sometimes will have -ve I/F values (noise and backgrounding) so find the valid indexes
	valid_idxs = (~np.isnan(log_uIperF)) & (~np.isinf(log_uIperF))
	
	# only operate on the valid indexes to avoid wierdness
	log_u0u_valid = log_u0u[valid_idxs]
	log_uIperF_valid = log_uIperF[valid_idxs]
	
	n_unique_log_u0u_valids = np.unique(log_u0u_valid).size
	if (n_unique_log_u0u_valids < 2) or (n_unique_log_u0u_valids<3 and (not two_points_ok_flag)):
		# then we can't get meaningful results so just return junk
		#print(n_unique_log_u0u_valids, n_unique_log_u0u_valids<2, (n_unique_log_u0u_valids<3 and (not two_points_ok_flag)))
		#print(f'DEBUG: Less than 2 or 3 valid indices, cannot get meaningful data, returning junk')
		return(np.nan, np.nan, np.array([]), np.nan, np.nan, log_u0u, log_uIperF, np.array([]), np.array([[np.nan,np.nan],[np.nan,np.nan]]))
	
	
	# find matrix A, A = {... 1_i ..., ... log[u_0 u]_i ...}
	A = np.vstack([np.ones_like(log_u0u_valid), log_u0u_valid]).T
	
	if A.shape == (0,2):
		# then we didn't have any valid indexes, so the whole thing is NANs or INFs so just return junk
		#print(f'DEBUG: Matrix A not of required shape, we have no valid indices, returning junk')
		return(np.nan, np.nan, np.nan, np.nan, np.nan, log_u0u, log_uIperF, [], np.full((2,2), fill_value=np.nan))
	
	# perform least squares fit to determine vector x
	# numpy version cannot constrain to bounds, so use scipy version.
	
	#x, residual, rank, s = np.linalg.lstsq(A, log_uIperF_valid, rcond=None)
	
	# Apply weights
	if weight is None:
		w = np.ones_like(log_uIperF_valid)
	elif weight == 'adjust_for_logscale':
		# adjusts for the logarithmic scale so that fitting in logspace is the
		# same as fitting in linear space
		w = (log_uIperF_valid - np.min(log_uIperF_valid))**2
	else:
		print(f'ERROR: Unknown value {weight} passed to argument "weight"')
		raise NotImplementedError

	# bounds in form (IperF0 min, k min), (IperF0 max, k max)
	bounds = ((-np.inf,0),(np.inf,1))

	if mode == 'least_squares':
		result = sp.optimize.lsq_linear(A*w[:,None], w*log_uIperF_valid, bounds=bounds)
		x = result.x
		residual = result.fun
		rank = np.nan
		s = np.nan
		
		# get the residuals of the fit
		fit_residuals = np.matmul(A, x) - log_uIperF_valid

		# == THOUGHTS ON COVARIANCE STUFF ==
		# note that cov_mat is the covariance of log(IperF0) and k, so need to take this into account
		# Assuming that the errors on log(IperF0) are distributed normally, then the errors on IperF0 are
		# distributed according to a log-normal distribution. So the variance of those errors is
		# sigma_IperF0 = np.exp(cov_mat[0,0] - 1)*np.exp(cov_mat[0,0])
		# see https://en.wikipedia.org/wiki/Log-normal_distribution
		# and assume that the mean of the errors on log(IperF0) is zero
		# BUT I DONT THINK THAT log(IperF0) is normally distributed, but IperF0 is,
		# Therefore may have to work out the errors by brute force, something like (exp(x+sqrt(var(x))) - exp(x-sqrt(var(x))))^2
		
		# == GETTING VARIANCE OF (I/F)0 FROM log[(I/F)0] ==
		# The error propagation that will give me the variance of IperF0 starting 
		# from log(IperF0) is confusing. Instead I can brute-force the problem
		# by calculating
		# IperF0s_var = (np.exp(np.log(IperF0s)+np.sqrt(log_IperF0s_var)) 
		#               - np.exp(np.log(IperF0s)-np.sqrt(log_IperF0s_var)))**2	
		
		AtA = np.matmul(A.transpose(), A)
		mat_inverted = False
		while not mat_inverted:
			try:
				AtA_inv = np.linalg.inv(AtA)
			except np.linalg.LinAlgError:
				# if we can't invert the matrix, make a small alteration until it's non-singular
				AtA += np.random.rand(2,2)*1E-10
				continue
			mat_inverted=True
			
		cov_mat = np.var(fit_residuals)*AtA_inv # see https://www.stat.purdue.edu/~boli/stat512/lectures/topic3.pdf
		IperF0_ret, k_ret = (np.exp(x[0]), x[1])
	elif mode == 'curve_fit':
		p0 = [np.nanmax(IperF), 0.5] # initial guesses
		curve_to_fit = lambda log_u0u, IperF0, k: np.exp(np.log(IperF0) + k*log_u0u)
		popt, pcov = sp.optimize.curve_fit(curve_to_fit, p0=p0, bounds=bounds)
		# change pcov to hold variance of log_IperF0
		pcov[0,0] = np.abs(0.5*(np.log(popt[0]+pcov[0,0])-np.log(popt[0]) + np.log(popt[0]-pcov[0,0])-np.log(popt[0])))
		cov_mat = pcov
		IperF0_ret, k_ret = (popt[0], popt[1])
		residual = np.exp(log_uIperF_valid) - curve_to_fit(log_u0u_valid, *popt)
		rank, s = (np.nan, np.nan)
	else:
		print(f'ERROR: Unknown value {mode} to argument "mode"')
		raise NotImplementedError

	return(IperF0_ret, k_ret, residual, rank, s, log_u0u, log_uIperF, valid_idxs, cov_mat)
	

def apply_correction(hdul, IperF0s, ks, radiance_ext, zenith_ext, solar_zenith_ext, angles_in_radians=False):
	"""
	Applies minnaert correction to a hdul, either appends, overwrites, or creates new FITS file to store result.
	"""
	radiances = hdul[radiance_ext].data
	zens = np.array(hdul[zenith_ext].data)
	sol_zens = np.array(hdul[solar_zenith_ext].data)

	if not angles_in_radians:
		zens *= np.pi/180
		sol_zens *= np.pi/180

	on_disk_map = np.nonzero(((zens >= 0) & (zens <=np.pi/2)))
	u = np.cos(zens)
	u_0 = np.cos(sol_zens)
	u_0_power_k = u_0[None,...]**(ks[...,None,None])
	u_power_k_minus_one = u[None,...]**(ks[...,None,None]-1)

	minnaert_corrected = np.zeros_like(radiances)
	slices_3d = (slice(None), *on_disk_map)
	minnaert_corrected[slices_3d] = u_0_power_k[slices_3d]*u_power_k_minus_one[slices_3d]
	minnaert_corrected = radiances/minnaert_corrected

	hdul.append(fits.ImageHDU(data =minnaert_corrected, header=hdul[radiance_ext].header, name='minnaert_corrected'))
	return(hdul)
	


def parse_args(argv):
	parser = ut.args.DocStrArgParser(description=__doc__)
	parser.add_argument('target_cubes', type=str, nargs='+', help='A list of fitscubes to operate on')
	parser.add_argument('--fits.sci.exts', 
		type=str, 
		action='extend',
		nargs='*',
		help=' '.join((
			'A list of <string> <integer> pairs that denote the extension name of numbered extensions in the fits file.',
			'Currently recognises "SCI" and "ZENITH" extension names to be used in internal functions.',
		)),
		default=['SCI', '1', 'ZENITH', '9', 'MASK', 0]
	)
	parser = Slice.add_args(parser, 'fits.sci.')

	parser.add_argument('--mask', type=str, help='A file relative to the fits cube that holds a mask to be applied to the cube')
	parser.add_argument('--mode', type=str, help='How should we use the target cube data [individual|combined]', default='individual')
	parser.add_argument('--plot.fit.ylim', type=float, nargs=2, help='Limits on the y axis for the minnaert fit plot', default=None)
	parser.add_argument('--plot.fit.xlim', type=float, nargs=2, help='Limits on the x axis for the minnaert fit plot', default=None)

	parser.add_argument('--data.nan_cutoff', type=float, help='If the absolute value of a pixel is smaller than this value, bad data (NAN) is assumed', default=0)
	parser.add_argument('--data.nan_threshold', type=float, help='If the value of a pixel is smaller than this value, bad data (NAN) is assumed', default=-np.inf)


	#plotutils.add_plot_arguments(parser)
	ut.plt.add_plot_arguments(parser)

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface


	parsed_args['fits.sci.exts'] = dict(tuple((x,int(y)) for x, y in zip(parsed_args['fits.sci.exts'][::2], parsed_args['fits.sci.exts'][1::2])))
	if parsed_args['plots.mode'] is None:
		parsed_args['plots.mode'] = ['overwrite', 'show']

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
