#!/usr/bin/env python3

import sys, os
import nemesis.read
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import numpy as np


def create(spx_data, f0=None, a0=None):
	nr, nc = (2,2)
	if f0 is None:
		f0 = plt.figure(figsize=(6*nc,6*nr))
	if a0 is None:
		a0 = f0.subplots(nr,nc,squeeze=False)
	colours = list(mplc.TABLEAU_COLORS.keys())
	for i, spec_record_i in enumerate(spx_data['spec_record']):
		a0[0,0].plot(spec_record_i[:,0], spec_record_i[:,1], color=colours[i], label=f'ngeom {i}')
		a0[0,0].fill_between(spec_record_i[:,0], 
								spec_record_i[:,1]+spec_record_i[:,2], 
								spec_record_i[:,1]-spec_record_i[:,2],
								color=colours[i],
								alpha=0.5)
		a0[0,1].plot(spec_record_i[:,0], spec_record_i[:,1], color=colours[i], label=f'ngeom {i}')
		a0[0,1].fill_between(spec_record_i[:,0], 
								spec_record_i[:,1]+spec_record_i[:,2], 
								spec_record_i[:,1]-spec_record_i[:,2],
								color=colours[i],
								alpha=0.5)
		
		#a0[0,0].plot(spec_record_i[:,0], 1.1*spec_record_i[:,1], color=colours[i], linestyle='--', linewidth=1)
		#a0[0,0].plot(spec_record_i[:,0], 0.9*spec_record_i[:,1], color=colours[i], linestyle='--', linewidth=1)

		a0[1,0].plot(spec_record_i[:,0], np.abs(spec_record_i[:,2]/spec_record_i[:,1]), color=colours[i], label=f'ngeom {i}')

	a0[0,0].legend()
	a0[0,0].set_xlabel('Wavelength (um)')
	a0[0,0].set_ylabel('Radiance (W cm-2 sr-1 um-1)')
	#a0[0,0].set_title(f'')

	a0[0,1].legend()
	a0[0,1].set_xlabel('Wavelength (um)')
	a0[0,1].set_ylabel('Radiance (W cm-2 sr-1 um-1)')
	#a0[0,1].set_title(f'')
	a0[0,1].set_yscale('log')
	a0[0,1].set_ylim([	np.nanmin([5E-1*_x[:,1] for _x in spx_data['spec_record']]),
						np.nanmax([_x[:,1]+_x[:,2] for _x in spx_data['spec_record']])]) 

	a0[1,0].legend()
	a0[1,0].set_xlabel('Wavelength (um)')
	a0[1,0].set_ylabel('Fractional Error')

	# make text for parameters
	txt = ['GLOBAL PARAMETERS', f'\tfwhm {spx_data["fwhm"]} latitude {spx_data["latitude"]} longitude {spx_data["longitude"]} ngeom {spx_data["ngeom"]}']
	for i in range(spx_data['ngeom']):
		txt += [f'NGEOM {i}',
				f'\tnconvs {spx_data["nconvs"][i]} navs {spx_data["navs"][i]}',
				f'\tFIELD OF VIEW AVERAGING RECORDS']
		for j in range(spx_data['navs'][i]):
			fov_ar = spx_data['fov_averaging_record'][i][j]
			txt += [f'\t\tflat {fov_ar[0]} flon {fov_ar[1]} sol_zen {fov_ar[2]} obs_zen {fov_ar[3]} azi_ang {fov_ar[4]} wgeom {fov_ar[5]}']				

	# text will fill upwards not downwards so this works	
	#print('\n'.join(txt).replace('\t','    '))
	f0.text(a0[1,1].get_position().x0, a0[1,1].get_position().y0, '\n'.join(txt).replace('\t','    '), fontsize=8)
	a0[1,1].remove()

	return(f0, a0)

if __name__=='__main__':
	args = sys.argv[1:]
	for spx_file in args:
		spx_data = nemesis.read.spx(spx_file)
		spx_f, spx_ax = create(spx_data)
		spx_f.suptitle(f'{os.path.normpath(os.path.abspath(spx_file))}', fontsize=8)
		plt.show()
