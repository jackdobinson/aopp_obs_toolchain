#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nemesis.read

"""
NOTE:
* When using forward model retrievals, we need to calculate chi-squared and phi-squared ourselves because nothing gets written to the *.itr file
"""

if __name__=='__main__':
	# create inputs
	iri_gridsearch_dir = os.path.expanduser('~/scratch/nemesis_run_results/slurm/combined_minnaert_clean_txt_002')
	iri_dir_tag = 'lat_-005.00_to_+005.00'
	#iri_dir_tag = 'lat_-040.00_to_-030.00'
	iri_gridsearch_run_dirs = glob.glob(f'{iri_gridsearch_dir}/*/*/{iri_dir_tag}')
	runname = 'neptune'
	cloud_1_irifname = 'cloud1.dat'
	cloud_2_irifname = 'haze1.dat'
	use_mre_for_chi_flag = True
	
	n_runs = len(iri_gridsearch_run_dirs)
	
	#make nan array constructor for convenience
	np.nans = lambda x, *a, **k: np.full(x, fill_value=np.nan)
	
	chi_finals = np.nans((n_runs,))
	phi_finals = np.nans((n_runs,))
	cloud_1_iri = np.nans((n_runs,))
	cloud_2_iri = np.nans((n_runs,))
	
	# get gridsearch data from files
	for i, run_dir in enumerate(iri_gridsearch_run_dirs):
		print(f'INFO: Operating on {run_dir}')
		
		if not use_mre_for_chi_flag:
			itr_d = nemesis.read.itr(os.path.join(run_dir, runname))
			print(itr_d)
			chi_finals[i] = itr_d['chisq_arr'][-1]/itr_d['nx']
			phi_finals[i] = itr_d['phi_arr'][-1]/itr_d['nx']
		else:
			mred = nemesis.read.mre(os.path.join(run_dir, runname))
			chi2 = np.nansum((mred['radiance_meas'] - mred['radiance_retr'])**2/mred['radiance_err'])
			chi_finals[i] = chi2
			phi_finals[i] = chi2
			
		c1_iri_d = nemesis.read.iri(os.path.join(run_dir, cloud_1_irifname))
		cloud_1_iri[i] = c1_iri_d['wie_array'][0,1] # take the first value as the imaginary refractive index is constant in this case
		
		c2_iri_d = nemesis.read.iri(os.path.join(run_dir, cloud_2_irifname))
		cloud_2_iri[i] = c2_iri_d['wie_array'][0,1] # take the first value as the imaginary refractive index is constant in this case
	
	#%% get gridsearch data into useful format
	
	c1_iri_u = np.unique(cloud_1_iri)[:11] #  restrict range if needed
	c2_iri_u = np.unique(cloud_2_iri)[:11] # restrict range if needed
	
	iri_grid_chi = np.nans((c1_iri_u.size, c2_iri_u.size))
	iri_grid_phi = np.nans((c1_iri_u.size, c2_iri_u.size))
	
	for i, c1_iri in enumerate(c1_iri_u):
		for j, c2_iri in enumerate(c2_iri_u):
			where = (cloud_1_iri==c1_iri) & (cloud_2_iri==c2_iri)
			iri_grid_chi[i,j] = chi_finals[where]
			iri_grid_phi[i,j] = phi_finals[where]
			# find out which way around x and y are
			#if i==0 and j==c2_iri_u.size-1:
			#	print(i,j,c1_iri, c2_iri)
			#	iri_grid[i,j,0] = np.nan
	
	chi_min_idx = np.unravel_index(np.nanargmin(iri_grid_chi), iri_grid_chi.shape)
	phi_min_idx = np.unravel_index(np.nanargmin(iri_grid_phi), iri_grid_phi.shape)
	
	# plot gridsearch data
	
	(nr,nc,s) = (2,3,6)
	f1 = plt.figure(figsize=(nc*s,nr*s))
	a1 = f1.subplots(nr,nc,squeeze=False, gridspec_kw={'hspace':0.3})
	
	f1.suptitle(f'Imaginary Refractive Index gridsearch for {iri_dir_tag}')
	
	im1 = a1[0,0].imshow(iri_grid_chi, origin='lower')
	
	a1[0,0].set_xticks(np.linspace(0,c2_iri_u.size-1,c2_iri_u.size))
	a1[0,0].set_yticks(np.linspace(0,c1_iri_u.size-1,c1_iri_u.size))
	a1[0,0].set_xticklabels(c2_iri_u, rotation=45)
	a1[0,0].set_yticklabels(c1_iri_u)
	a1[0,0].set_title(f'CHISQ/DOF min at {chi_min_idx[::-1]}\n{cloud_2_irifname} = {c2_iri_u[chi_min_idx[1]]} {cloud_1_irifname} = {c1_iri_u[chi_min_idx[0]]} ')
	a1[0,0].set_xlabel('')#f'imaginary refractive index from {cloud_2_irifname}')
	a1[0,0].set_ylabel(f'imaginary refractive index from {cloud_1_irifname}')
	
	im1 = a1[0,1].imshow(iri_grid_phi, origin='lower')
	
	a1[0,1].set_xticks(np.linspace(0,c2_iri_u.size-1,c2_iri_u.size))
	a1[0,1].set_yticks(np.linspace(0,c1_iri_u.size-1,c1_iri_u.size))
	a1[0,1].set_xticklabels(c2_iri_u, rotation=45)
	a1[0,1].set_yticklabels(c1_iri_u)
	a1[0,1].set_title(f'PHISQ/DOF min at {phi_min_idx[::-1]}\n{cloud_2_irifname} = {c2_iri_u[phi_min_idx[1]]} {cloud_1_irifname} = {c1_iri_u[phi_min_idx[0]]} ')
	a1[0,1].set_xlabel('')#f'imaginary refractive index from {cloud_2_irifname}')
	a1[0,1].set_ylabel('')#f'imaginary refractive index from {cloud_1_irifname}')
	
	im1 = a1[1,0].imshow(np.log(iri_grid_chi), origin='lower')
	
	a1[1,0].set_xticks(np.linspace(0,c2_iri_u.size-1,c2_iri_u.size))
	a1[1,0].set_yticks(np.linspace(0,c1_iri_u.size-1,c1_iri_u.size))
	a1[1,0].set_xticklabels(c2_iri_u, rotation=45)
	a1[1,0].set_yticklabels(c1_iri_u)
	a1[1,0].set_title(f'log(CHISQ/DOF), min at {chi_min_idx[::-1]}\n{cloud_2_irifname}= {c2_iri_u[chi_min_idx[1]]} {cloud_1_irifname} = {c1_iri_u[chi_min_idx[0]]} ')
	a1[1,0].set_xlabel(f'imaginary refractive index from {cloud_2_irifname}')
	a1[1,0].set_ylabel(f'imaginary refractive index from {cloud_1_irifname}')
	
	im1 = a1[1,1].imshow(np.log(iri_grid_phi), origin='lower')
	
	a1[1,1].set_xticks(np.linspace(0,c2_iri_u.size-1,c2_iri_u.size))
	a1[1,1].set_yticks(np.linspace(0,c1_iri_u.size-1,c1_iri_u.size))
	a1[1,1].set_xticklabels(c2_iri_u, rotation=45)
	a1[1,1].set_yticklabels(c1_iri_u)
	a1[1,1].set_title(f'log(PHISQ/DOF), min at {phi_min_idx[::-1]}\n{cloud_2_irifname} = {c2_iri_u[phi_min_idx[1]]} {cloud_1_irifname} = {c1_iri_u[phi_min_idx[0]]} ')
	a1[1,1].set_xlabel(f'imaginary refractive index from {cloud_2_irifname}')
	a1[1,1].set_ylabel('')#f'imaginary refractive index from {cloud_1_irifname}')
	
	a1[0,2].remove()
	
	a1[1,2].set_axis_off()
	a1[1,2].set_frame_on(False)
	keys = ["R0","Rerr", "V0", "Verr", "nwave", "clen", "vref", "nreal_vref", "v_od_norm"]
	c1_kvs = [f'{_x} {str(c1_iri_d[_x])}' for _x in keys]
	c2_kvs = [f'{_x} {str(c2_iri_d[_x])}' for _x in keys]
	sep = '\n    '
	txt = f'Cloud 1 parameters ({cloud_1_irifname}):{sep}{sep.join(c1_kvs)}\n\nCloud 2 parameters ({cloud_2_irifname}):{sep}{sep.join(c2_kvs)}'
	a1[1,2].text(0,0,txt)
	
	plt.show()