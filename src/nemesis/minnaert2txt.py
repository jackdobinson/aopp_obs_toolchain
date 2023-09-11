#!/usr/bin/env python3
"""
Creates a set of text files that hold minnaert inputs and coefficients sourced
from nemesis retrievals
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nemesis.read
import fitscube.limb_darkening.minnaert
import matplotlib.colors as mplc
import plotutils

#%% set up files to read from

runname = 'neptune'
nemesis_retrieval = os.path.expanduser('~/scratch/nemesis_run_results/slurm/combined_cbh_v_hsv_001/05')

#%% create minnaert inputs for apriori and retrieved spectra

mre_files = glob.glob(f'{nemesis_retrieval}/lat_*_to_*/{runname}.mre')
n_mre_files = len(mre_files)

for l, mre_file in enumerate(mre_files):
	print(f'INFO: Operating on mre_file "{mre_file}" file number {l}/{n_mre_files}')
	run_dir = os.path.dirname(mre_file)
	lat_max = float(run_dir.rsplit('_',1)[-1])
	lat_min = float(run_dir.rsplit('_',3)[1])
	print(f'INFO: lat_min {lat_min} lat_max {lat_max}')
	
	mre_data = nemesis.read.mre(os.path.join(run_dir, runname))
	spx_data = nemesis.read.spx(os.path.join(run_dir, runname))
	
	ngeom = mre_data['ngeom']
	wavs = mre_data['waveln'][0] # each wavelength grid should be the same
	IperF_apr = mre_data['radiance_meas']
	IperF_apr_err = mre_data['radiance_err']
	IperF_ret = mre_data['radiance_retr']
	
	if np.isnan(IperF_apr).all():
		print('INFO: All of the data in this file is NAN, skipping...')
		continue
	
	u = np.cos(np.broadcast_to(np.array([fov[:,3] for fov in spx_data['fov_averaging_record']]), IperF_apr.shape)*180/np.pi)
	u0 = np.cos(np.broadcast_to(np.array([fov[:,2] for fov in spx_data['fov_averaging_record']]), IperF_apr.shape)*180/np.pi)
	
	# save the transposes, we want the 1st axis to be wavelength
	np.savez(os.path.join(run_dir, 'minnaert_inputs.npz'), 
			  wavs = wavs.T, 
			  IperF_apr = IperF_apr.T, 
			  IperF_apr_err = IperF_apr_err.T, 
			  IperF_ret = IperF_ret.T, 
			  u = u.T, 
			  u0 = u0.T
			  )
	

#%% create minnaert coefficients

minnaert_inputs_files = glob.glob(f'{nemesis_retrieval}/lat_*_to_*/minnaert_inputs.npz')
n_minnaert_inputs_files = len(minnaert_inputs_files)

for l, minnaert_inputs_file in enumerate(minnaert_inputs_files):
	print(f'INFO: Operating on mre_file "{minnaert_inputs_file}" file number {l}/{n_minnaert_inputs_files}')
	run_dir = os.path.dirname(minnaert_inputs_file)
	lat_max = float(run_dir.rsplit('_',1)[-1])
	lat_min = float(run_dir.rsplit('_',3)[1])
	minnaert_inputs_file = os.path.join(run_dir, 'minnaert_inputs.npz')
	minnaert_coefficients_file = os.path.join(run_dir, 'minnaert_coefficients.npz')
	
	ldata = np.load(minnaert_inputs_file)
	wavs, IperF_apr, IperF_apr_err, IperF_ret, u, u0 = (ldata['wavs'],
														 ldata['IperF_apr'],
														 ldata['IperF_apr_err'],
														 ldata['IperF_ret'],
														 ldata['u'],
														 ldata['u0']
														 )
													
	(IperF0s_apr, 
	  ks_apr, 
	  log_IperF0s_apr_var, 
	  ks_apr_var
	  ) = fitscube.limb_darkening.minnaert.get_coefficients_v_wavelength(u0, u, IperF_apr)	
	IperF0s_apr_var = (np.exp(np.log(IperF0s_apr)+np.sqrt(log_IperF0s_apr_var)) 
						- np.exp(np.log(IperF0s_apr)-np.sqrt(log_IperF0s_apr_var)))**2
			
	(IperF0s_ret, 
	  ks_ret, 
	  log_IperF0s_ret_var,
	  ks_ret_var
	  ) = fitscube.limb_darkening.minnaert.get_coefficients_v_wavelength(u0, u, IperF_ret)	
	IperF0s_ret_var = (np.exp(np.log(IperF0s_ret)+np.sqrt(log_IperF0s_ret_var)) 
						- np.exp(np.log(IperF0s_ret)-np.sqrt(log_IperF0s_ret_var)))**2

	print(f'INFO: Saving minnaert coefficients to file {minnaert_coefficients_file}')
	np.savez(minnaert_coefficients_file,
			  wavs = wavs,
			  IperF0s_apr=IperF0s_apr,
			  ks_apr=ks_apr,
			  log_IperF0s_apr_var = log_IperF0s_apr_var,
			  ks_apr_var = ks_apr_var,
			  IperF0s_ret = IperF0s_ret,
			  ks_ret = ks_ret,
			  log_IperF0s_ret_var = log_IperF0s_ret_var,
			  ks_ret_var = ks_ret_var
			  )

	# DEBUGGING PLOT
	nr,nc=(2,2)
	f0 = plt.figure(figsize=(nc*6,nr*6))
	f0.suptitle(f'Latitude {lat_min:+#07.2f} to {lat_max:+#07.2f} debugging plots')
	a0 = f0.subplots(2,2,squeeze=False)
	w_idx = 30
	ln_IperF0s_apr = np.log(IperF0s_apr)
	ln_u0u = np.log(u0*u)
	ln_uIperF_apr_fit = ln_IperF0s_apr[:,None] + ks_apr[:,None]*ln_u0u
	ln_uIperF_apr = np.log(u*IperF_apr)
	a0[0,0].errorbar(ln_u0u[w_idx], ln_uIperF_apr[w_idx], yerr=IperF_apr_err[w_idx], linestyle='none', marker='.', capsize=2,
					label='apriori minnaert inputs')
	a0[0,0].plot(ln_u0u[w_idx], ln_uIperF_apr_fit[w_idx], label='apriori minnaert fit')
	
	ln_IperF0s_ret = np.log(IperF0s_ret)
	ln_u0u = np.log(u0*u)
	ln_uIperF_ret_fit = ln_IperF0s_ret[:,None] + ks_ret[:,None]*ln_u0u
	ln_uIperF_ret = np.log(u*IperF_ret)
	a0[0,0].plot(ln_u0u[w_idx], ln_uIperF_ret[w_idx], linestyle='none', marker='.', 
				label='retrieved minnaert inputs')
	a0[0,0].plot(ln_u0u[w_idx], ln_uIperF_ret_fit[w_idx], label='retrieved minnaert fit')
	a0[0,0].legend()
	a0[0,0].set_title('\n'.join((f'Minnaert fit at wavelength {wavs[w_idx]:05.3f} um',
									f'k_apr {ks_apr[w_idx]:+#04.2f} ln_IperF0_apr {ln_IperF0s_apr[w_idx]:+#04.2f} IperF0_apr {np.exp(ln_IperF0s_apr[w_idx]):+#04.2}',
									f'k_ret {ks_ret[w_idx]:+#04.2f} ln_IperF0_ret {ln_IperF0s_ret[w_idx]:+#04.2f} IperF0_ret {np.exp(ln_IperF0s_ret[w_idx]):+#04.2}'
						)))
	
	a0[0,1].plot(wavs, IperF0s_apr, label='apriori Nadir Radiance')
	a0[0,1].plot(wavs, IperF0s_ret, label='retrieved Nadir Radiance')
	a0[0,1].legend()
	a0[0,1].set_yscale('log')
	a0[0,1].set_title('I/F_0 vs wavelength')
	
	a0[1,0].plot(wavs, ks_apr, label='apriori minnaert factor')
	a0[1,0].plot(wavs, ks_ret, label='retrieved minnaert factor')
	a0[1,0].legend()
	a0[1,0].set_title('k vs wavelength')
	
	colours = list(mplc.TABLEAU_COLORS.keys())
	for i in range(IperF_ret.shape[1]):
		a0[1,1].plot(wavs, IperF_ret[:,i], label=f'geometry {i} retrieved radiance', lw=1, color=colours[i])
		a0[1,1].plot(wavs, IperF_apr[:,i], label=f'geometry {i} apriori radiance', lw=1, alpha=0.5,color=colours[i])
	
	a0[1,1].legend()
	a0[1,1].set_yscale('log')
	a0[1,1].set_title('Radiance vs wavelength')
	
	plotutils.save_show_plt(f0, f'lat_{lat_min:+#07.2f}_to_{lat_max:+#07.2f}_debug.png', 
							 os.path.join(run_dir, 'plots'), show_plot=False, save_plot=True)
	#f0.savefig(f'{run_dir}/plots/lat_{lat_min:+#07.2f}_to_{lat_max:+#07.2f}_debug.png', bbox_inches='tight')
	#plt.show()
			

#%% combine minnaert coefficients from all different latitudes into a single set of ensemble plots

lat_v_wav_v_IperF0_apr = None
lat_v_wav_v_k_apr = None
lat_v_wav_coverage_apr = None

lat_v_wav_v_IperF0_ret = None
lat_v_wav_v_k_ret = None
lat_v_wav_coverage_ret = None

lat_grid = np.linspace(-90,90,180)

minnaert_coefficients_files = glob.glob(f'{nemesis_retrieval}/lat_*_to_*/minnaert_coefficients.npz')
n_minnaert_coefficients_files = len(minnaert_coefficients_files)

for l, minnaert_coefficients_file in enumerate(minnaert_coefficients_files):
	print(f'INFO: Operating on file {mre_file} {l}/{n_minnaert_coefficients_files}')
	run_dir = os.path.dirname(minnaert_coefficients_file)
	lat_max = float(run_dir.rsplit('_',1)[-1])
	lat_min = float(run_dir.rsplit('_',3)[1])
	
	ldata = np.load(minnaert_coefficients_file)
	(wavs,
		IperF0s_apr,
		ks_apr,
		log_IperF0s_apr_var,
		ks_apr_var,
		IperF0s_ret,
		ks_ret,
		log_IperF0s_ret_var,
		ks_ret_var
		) =map(ldata.__getitem__,  ('wavs',
									  'IperF0s_apr',
									  'ks_apr',
									  'log_IperF0s_apr_var',
									  'ks_apr_var',
									  'IperF0s_ret',
									  'ks_ret',
									  'log_IperF0s_ret_var',
									  'ks_ret_var'
									  )
							  )

	# create numpy arrays to hold data
	if lat_v_wav_v_IperF0_apr is None:
		lat_v_wav_v_IperF0_apr = np.zeros((lat_grid.size, wavs.size))
	if lat_v_wav_v_k_apr is None:
		lat_v_wav_v_k_apr = np.zeros((lat_grid.size, wavs.size))
	if lat_v_wav_coverage_apr is None:
		lat_v_wav_coverage_apr = np.zeros((lat_grid.size, wavs.size))
		
	if lat_v_wav_v_IperF0_ret is None:
		lat_v_wav_v_IperF0_ret = np.zeros((lat_grid.size, wavs.size))
	if lat_v_wav_v_k_ret is None:
		lat_v_wav_v_k_ret = np.zeros((lat_grid.size, wavs.size))
	if lat_v_wav_coverage_ret is None:
		lat_v_wav_coverage_ret = np.zeros((lat_grid.size, wavs.size))
		
	# fill numpy arrays with data
	lat_idxs = np.nonzero((lat_min < lat_grid) & (lat_grid < lat_max))
	
	lat_v_wav_v_IperF0_apr[lat_idxs,:] += IperF0s_apr
	lat_v_wav_v_k_apr[lat_idxs,:] += ks_apr
	lat_v_wav_coverage_apr[lat_idxs,:] += 1
	
	lat_v_wav_v_IperF0_ret[lat_idxs,:] += IperF0s_ret
	lat_v_wav_v_k_ret[lat_idxs,:] += ks_ret
	lat_v_wav_coverage_ret[lat_idxs,:] += 1
	
	
#%%
# ensure areas that have zero coverage are NANs
lat_v_wav_coverage_apr[lat_v_wav_coverage_apr==0] = np.nan
lat_v_wav_coverage_ret[lat_v_wav_coverage_ret==0] = np.nan

# divide through by coverage
lat_v_wav_v_IperF0_apr /= lat_v_wav_coverage_apr
lat_v_wav_v_k_apr /= lat_v_wav_coverage_apr
lat_v_wav_v_IperF0_ret /= lat_v_wav_coverage_ret
lat_v_wav_v_k_ret /= lat_v_wav_coverage_ret

#%%
# CREATE PLOT
nr,nc = 2,2
f1 = plt.figure(figsize=(nc*6,nr*6))
a1 = f1.subplots(nr,nc,squeeze=False)

extent = (wavs[0],wavs[-1],lat_grid[0],lat_grid[-1])
def set_axis_labels(ax, xl='X axis', yl='Y axis'):
	ax.set_xlabel(xl)
	ax.set_ylabel(yl)
	return

for ax in a1.ravel():
	set_axis_labels(ax, 'wavelength (um)', 'Latitude (deg)')

im100 = a1[0,0].imshow(lat_v_wav_v_IperF0_apr, origin='lower', extent=extent, aspect='auto')
a1[0,0].set_title('apriori IperF0 vs latitude and wavelength')
im101 = a1[0,1].imshow(lat_v_wav_v_k_apr, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1)
a1[0,1].set_title('apriori k vs latitude and wavelength')

im110 = a1[1,0].imshow(lat_v_wav_v_IperF0_ret, origin='lower', extent=extent, aspect='auto')
a1[1,0].set_title('retrieved IperF0 vs latitude and wavlength')
im111 = a1[1,1].imshow(lat_v_wav_v_k_ret, origin='lower', extent=extent, aspect='auto', vmin=0, vmax=1)
a1[1,1].set_title('retrieved k vs latitude and wavelength')

cb100 = f1.colorbar(im100, ax=a1[0,0])
cb101 = f1.colorbar(im101, ax=a1[0,1])
cb110 = f1.colorbar(im110, ax=a1[1,0])
cb111 = f1.colorbar(im111, ax=a1[1,1])
cb100.set_label('[I/F]_0 (uW)')
cb101.set_label('k')
cb110.set_label('[I/F]_0 (uW)')
cb111.set_label('k')

#plotutils.save_show_plt(f1, f'{runname}_lat_wav_IperF0_k.png', 
#							 os.path.join(nemesis_retrieval, 'plots'), show_plot=False, save_plot=True)
f1.savefig(f'{nemesis_retrieval}/plots/{runname}_lat_wav_IperF0_k.png', bbox_inches='tight')
plt.show()