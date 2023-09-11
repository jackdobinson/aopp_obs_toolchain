#!/usr/bin/env python3

import sys, os
import astropy.io.fits as fits
import pretty_table
import numpy as np

def main(argv):
	"""
	Assume all arguments are paths to "out_objnod.fits" files. I.e. are reduced sinfoni observations.
	"""

	obs_dict = {}
	tbl_struct = {}
	tbl_struct['SINFONI Observations'] = {'header':[['id', 'obs type', 'target', 'coverage', 'date', 'observation template start', 'folder', 'observation block name', 'data cube shape', 'pixel size', 'wavelength range', 'calibrator id']],'data':[]}
	absolute_path_argv = [os.path.abspath(_x) for _x in argv]
	#[print(_x) for _x in absolute_path_argv]
	for i, tc in enumerate(argv):
		with fits.open(tc) as hdul:
			folder = os.path.dirname(tc)
			print(folder, hdul[0].data.shape)
			calibrator_file = os.path.join(folder, 'calibrators.txt')
			calibrator_paths = []
			if os.path.isfile(calibrator_file):
				with open(calibrator_file, 'r') as f:
					for aline in f:
						bline = aline.strip()
						if bline != '':
							calibrator_paths.append(bline.replace(' ', '_')) # replace spaces with underscores in path
				#[print(_x) for _x in calibrator_paths]
				calibrator_ids = []
				for _x in calibrator_paths:
					try:
						calibrator_ids.append(absolute_path_argv.index(os.path.abspath(_x)))
					except ValueError:
						pass
			else:
				calibrator_ids = []
			# neptune varies from 2.2 to 2.4 arsec diameter, so to be safe use 2.6
			#field_size = np.abs(np.prod(hdul[0].data.shape[1:])*float(hdul[0].header['CDELT1'])*float(hdul[0].header['CDELT2']*3600**2))
			field_size = (np.abs(hdul[0].data.shape[1]*float(hdul[0].header['CDELT1'])*3600), np.abs(hdul[0].data.shape[2]*float(hdul[0].header['CDELT2'])*3600))
			print(field_size, 2.6, np.abs(float(hdul[0].header['CDELT1'])*3600), np.prod(hdul[0].data.shape[1:]))
			if hdul[0].header['OBJECT'] == 'Neptune':
				if all([_x>2.6 for _x in field_size]):
					coverage = 'Full'
				else:
					coverage = 'Partial'
			elif hdul[0].header['OBJECT'] == 'STD':
				coverage = 'Full' # point sources are always covered
			else:
				coverage = 'Unknown'

			tbl_struct['SINFONI Observations']['data'].append(
				[	f'{i}',
					'SCIENCE' if hdul[0].header['OBJECT']!='STD' else 'STD',
					hdul[0].header['HIERARCH ESO OBS TARG NAME'],
					coverage,
					hdul[0].header['DATE-OBS'],
					hdul[0].header['HIERARCH ESO TPL START'],
					os.path.basename(os.path.dirname(os.path.abspath(folder))),
					hdul[0].header['HIERARCH ESO OBS NAME'],
					'x'.join([f'{_x}' for _x in hdul[0].data.shape]),
					hdul[0].header['HIERARCH ESO INS OPTI1 NAME'],
					hdul[0].header['HIERARCH ESO INS GRAT1 NAME'],
					', '.join([f'{_x}' for _x in calibrator_ids])
				]
			)

	with open('summary.txt', 'w') as f:
		pretty_table.write(tbl_struct, f)
	return()


if __name__=='__main__':
	main(sys.argv[1:])
