#!/usr/bin/env python3

import sys, os
import numpy as np
import nemesis.read
import nemesis.write
import glob
import matplotlib.pyplot as plt

#%% get setup info for grid search

nemesis_template_dir = os.path.expanduser('~/scratch/nemesis_run_results/nemesis_run_templates/neptune/2cloud_ch4_imajrefidx_std')
grid_search_dir = os.path.join(nemesis_template_dir, 'iri_grid_search_forward_model')

# holds the files that will have their imaginary refractive index changed, and the values it should take
iri_grid = [('cloud1.dat', 10**np.linspace(2, -5, 15)),
			('haze1.dat', 10**np.linspace(2, -5, 15))
			]

#print(iri_grid)

#%% write files for grid search
# create grid search folder
os.makedirs(grid_search_dir, exist_ok=True)

grid_search_this_lvl_dirs = [grid_search_dir]
grid_search_next_lvl_dirs = []

for k, (gs_filename, iri_values) in enumerate(iri_grid):
	print(f'INFO: Writing {gs_filename} files...')
	iri_data = nemesis.read.iri(os.path.join(nemesis_template_dir, gs_filename))
	n_iri_values = len(iri_values)
	n_digits = int(np.ceil(np.log10(n_iri_values)))
	file_fmt_str = '{:0'+str(n_digits+1)+'d}'
	for i, gs_dir in enumerate(grid_search_this_lvl_dirs):
		#print(f'INFO: In grid search directory {gs_dir}')
		for j, iri_value in enumerate(iri_values):
			this_dir = os.path.join(gs_dir, file_fmt_str.format(j))
			grid_search_next_lvl_dirs.append(this_dir)
			os.makedirs(this_dir, exist_ok=True)
			iri_data['wie_array'][:,1] = iri_value
			iri_data['wie_array'][:,2] = 1E-7*iri_value
			this_file = os.path.join(this_dir, gs_filename)
			nemesis.write.iri(iri_data, this_file)
			#print(f'INFO: Written file {this_file}')
	print(f'INFO: All level-{k} files written, moving to next grid search level...')
	grid_search_this_lvl_dirs = grid_search_next_lvl_dirs[:]
	grid_search_next_lvl_dirs = []

#%% get inputs for plotting a 2d map of the grid search

nemesis_retrieval_dir = os.path.expanduser('~/scratch/nemesis_run_results/slurm/combined_cbh_v_hsv_003')
mre_files = glob.glob(f'{nemesis_retrieval_dir}/*/*/*/neptune.mre')

gridsearch_data = []

for i, mre_file in enumerate(mre_files):
	cloud_iri_file = os.path.join(os.path.dirname(mre_file),'cloud1.dat')
	haze_iri_file = os.path.join(os.path.dirname(mre_file), 'haze1.dat')
	cloud_iri_dat = nemesis.read.iri(cloud_iri_file)
	haze_iri_dat = nemesis.read.iri(haze_iri_file)
	cloud_iri = cloud_iri_dat['wie_array'][0,1] # should be constant in this case so just take the first element
	haze_iri = haze_iri_dat['wie_array'][0,1]
	
	mre_data = nemesis.read.mre(mre_file)
	#retr_geom_diff = mre_data['radiance_retr'][0,:] - mre_data['radiance_retr'][1,:]
	#retr_geom_diff_rms = np.sqrt(np.nansum(retr_geom_diff**2)/retr_geom_diff.size)
	retr_geom_diff_rms = 0
	for rr in mre_data['radiance_retr'][1:]:
		retr_geom_diff_rms += np.sqrt(np.nansum((mre_data['radiance_retr'][0,:]-rr)**2)/rr.size)
	
	
	gridsearch_data.append([cloud_iri, haze_iri, retr_geom_diff_rms])
	

gs_data = np.array(gridsearch_data)
ux_vals = np.unique(gs_data[:,0])
uy_vals = np.unique(gs_data[:,1])
z_data = np.zeros((uy_vals.size, ux_vals.size))
z_coverage = np.zeros_like(z_data)

for x, y, z in gs_data:
	i = np.nanargmin(np.fabs(ux_vals-x))
	j = np.nanargmin(np.fabs(uy_vals-y))
	print(x, y, z, i, j)
	z_data[j,i] += z
	z_coverage[j,i] += 1
	
z_data[z_coverage==0] = np.nan
z_data /= z_coverage

i_min = np.nanargmin(gs_data[:,2])



#%% plot the data
print(ux_vals)
print(uy_vals)
nr,nc=(2,2)
f1 = plt.figure(figsize=(nc*5,nr*5))
a1 = f1.subplots(nr,nc, squeeze=False)

im100 = a1[0,0].imshow(z_data, origin='lower')
a1[0,0].set_xticks(np.arange(ux_vals.size))
a1[0,0].set_xticklabels([f'{_x:07.2E}' for _x in ux_vals], rotation=30)
a1[0,0].set_xlabel('cloud imaginary refractive index')

a1[0,0].set_yticks(np.arange(uy_vals.size))
a1[0,0].set_yticklabels([f'{_y:07.2E}' for _y in uy_vals])
a1[0,0].set_ylabel('haze imaginary refractive index')

a1[0,0].set_title('rms of geometry differences')
f1.colorbar(im100, ax=a1[0,0])

mre_dat_min_diff = nemesis.read.mre(mre_files[i_min])
for i, rr in enumerate(mre_data['radiance_retr'][1:]):
	rr_diff = (mre_data['radiance_retr'][0,:] - rr)
	a1[0,1].plot(mre_data['waveln'][i,:], rr_diff, label='geometry differences')
	a1[0,1].plot(mre_data['waveln'][i,:], rr_diff**2, label='(geometry differences)^2')
a1[0,1].set_title(f'radiance difference of geometries with\ncloud_iri {gs_data[i_min,0]} haze_iri {gs_data[i_min,1]} rms {gs_data[i_min,2]:07.2E}')
a1[0,1].legend()

a1[1,0].plot(mre_data['radiance_retr'][0,:])
a1[1,0].plot(mre_data['radiance_retr'][1,:])


a1[1,1].plot(mre_data['radiance_meas'][0,:])
a1[1,1].plot(mre_data['radiance_meas'][1,:])

plt.show()
os.makedirs(f'{nemesis_retrieval_dir}/plots', exist_ok=True)
f1.savefig(f'{nemesis_retrieval_dir}/plots/iri_gridsearch.png', bbox_inches='tight')


