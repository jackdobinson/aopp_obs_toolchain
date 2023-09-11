#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import glob
import nemesis.read

np.nans = lambda x, *a, **k: np.full(x, fill_value=np.nan)

if __name__=='__main__':
	# set up input variables
	retrieval_set_dir = os.path.expanduser("~/scratch/nemesis_run_results/slurm/combined_minnaert_clean_txt_000")
	runname = 'neptune'
	
	# we want to group the chi and phi values in the same way as the directory
	# structure make the numpy arrays large enough to hold them
	chi = np.nans((3,3))
	phi = np.nans((3,3))
	
	# want to record the directory names, each level should  have the same
	# set of directories (i.e. .../00 has children (00, 03, 05) and so does .../02)
	d1_names = []
	d2_names = []
	for i, d1 in enumerate(glob.glob(f'{retrieval_set_dir}/??/')):
		# if we are looking at a new directory, add it to the name list
		if i>=len(d1_names):
			d1_names.append(os.path.basename(os.path.dirname(d1)))
		for j, d2 in enumerate(glob.glob(f'{d1}/*/')):
			# the first time we go through the children directories, add them to
			# the name list
			if j >= len(d2_names):
				d2_names.append(os.path.basename(os.path.dirname(d2)))
			
			# set accumulator variables to zero
			phi_per_dof_av = 0
			chi_per_dof_av = 0
			n = 0
			
			# this loop is different because we want to average over all of the
			# latitudes
			for k, d3 in enumerate(glob.glob(f'{d2}/lat_*/')):
				print(f'INFO: Averaging cost functions in {d3}')
				itr_file = os.path.join(d3, f'{runname}')
				#print(f'INFO: Attempting to open file {itr_file}')
				try:
					
					itrd = nemesis.read.itr(itr_file)
					#print(f'INFO: Opened {itr_file}')
					#print(f'INFO: n {n}')
					#print(itrd['chisq_arr'])
					chi_val = itrd['chisq_arr'][-1]/itrd['nx']
					phi_val = itrd['phi_arr'][-1]/itrd['nx']
					if np.isnan(chi_val) or np.isnan(phi_val):
						continue
					chi_per_dof_av += chi_val
					phi_per_dof_av += phi_val
					#print(chi_per_dof_av)
					n+=1
				except Exception:
					continue
			#print(f'INFO: n {n}')
			if n == 0:
				phi_per_dof_av = np.nan
				chi_per_dof_av = np.nan
			else:
				phi_per_dof_av /= n
				chi_per_dof_av /= n
			chi[i,j] = chi_per_dof_av
			phi[i,j] = phi_per_dof_av
		
	chi_min_idx = np.unravel_index(np.nanargmin(chi), chi.shape)
	phi_min_idx = np.unravel_index(np.nanargmin(phi), phi.shape)
	
	nr, nc, s = (1, 2, 6)
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr, nc, squeeze=False)
	
	f1.suptitle('\n'.join(	(	f'Sum of cost function per degrees of freedom for directories:',
								f'{retrieval_set_dir}/YY/XX/*',
								f'chi_min:...\{d1_names[chi_min_idx[0]]}\{d2_names[chi_min_idx[1]]}',
								f'phi_min:...\{d1_names[phi_min_idx[0]]}\{d2_names[phi_min_idx[1]]}'
							)))
	
	a1[0,0].imshow(chi, origin='lower')
	a1[0,0].set_title('chi_per_dof_av')
	a1[0,0].set_yticks(range(0,len(d1_names)))
	a1[0,0].set_yticklabels(d1_names)
	a1[0,0].set_xticks(range(0,len(d2_names)))
	a1[0,0].set_xticklabels(d2_names)
	a1[0,0].set_xlabel('XX directory')
	a1[0,0].set_ylabel('YY directory')
	
	
	a1[0,1].imshow(phi, origin='lower')
	a1[0,1].set_title('phi_per_dof_av')
	a1[0,1].set_yticks(range(0,len(d1_names)))
	a1[0,1].set_yticklabels(d1_names)
	a1[0,1].set_xticks(range(0,len(d2_names)))
	a1[0,1].set_xticklabels(d2_names)
	a1[0,1].set_xlabel('XX directory')
	a1[0,1].set_ylabel('YY directory')
	
	plt.show()