"""
Given a badness map, will apply a value cut (or possibly a range of interpolated value cuts) to the badness map to give a boolean mask that defines "bad" pixels

# EXAMPLE
>>> python -m aopp_deconv_tool.create_bad_pixel_mask './example_data/ifu_observation_datasets/MUSE.2019-10-18T00:01:19.521_rebin_artifactmap.fits' --const_regions ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_const.reg --dynamic_regions 59 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_59.reg --dynamic_regions 147 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_147.reg --dynamic_regions 262 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_262.reg --dynamic_regions 431 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_431.reg

"""

import sys, os
from pathlib import Path
import dataclasses as dc
from typing import Literal, Any, Type, Callable
from collections import namedtuple

import numpy as np
import scipy as sp
import scipy.ndimage
from astropy.io import fits
from astropy.wcs import WCS
import regions

import matplotlib.pyplot as plt

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import get_bp_mask_from_badness_map

from aopp_deconv_tool.py_ssa import SSA

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')





def load_const_regions(const_regions : list[str]) -> list[regions.Regions]:
	region_list = []
	for fpath in const_regions:
		region_list.extend(regions.Regions.read(fpath))
	return region_list

def load_dynamic_regions(dynamic_regions : list[tuple[int,str]]) -> dict[int, list[regions.Regions]]:
	idx_region_dict = dict()
	for index, fpath in sorted(dynamic_regions, key=lambda x: x[0]):
		_lgr.debug(f'{index} {fpath=}')
		idx_region_dict[index] = idx_region_dict.get(index, []) + load_const_regions([fpath])
	return idx_region_dict

class DynamicRegionInterpolator:
	def __init__(self, dynamic_region_dict):
		
		self.indices = np.array(list(sorted(dynamic_region_dict.keys())), dtype=int)
		if len(dynamic_region_dict) !=0:
			self.dynamic_region_defaults = [x for x in [z for z in dynamic_region_dict.values()][0]]
			self.n_regions = len(dynamic_region_dict[self.indices[0]])
		else:
			self.dynamic_region_defaults = []
			self.n_regions = 0
		
		self.region_classes = []
		
		#check we always have the same number of regions
		for idx in self.indices:
			assert len(dynamic_region_dict[idx]) == self.n_regions, "Must have a constant number of dynamic regions"
			for j, r in enumerate(dynamic_region_dict[idx]):
				if len(self.region_classes) <= j:
					self.region_classes.append(r.__class__)
				else:
					assert r.__class__ is self.region_classes[j], "Dynamic regions must not change type"
		_lgr.debug(f'{self.indices=} {self.n_regions=} {self.region_classes=}')
		
		# get the names of attributes that vary with index for each region
		do_region_attrs_vary = {}
		for idx in self.indices:
			for j, r in enumerate(dynamic_region_dict[idx]):
				if j not in do_region_attrs_vary:
					do_region_attrs_vary[j] = {}
				for k,v in r.__dict__.items():
					if k in ('meta', 'visual') or k.startswith('_'): # Don't bother with these attributes
						continue
					if k not in do_region_attrs_vary[j]:
						do_region_attrs_vary[j][k] = {'last_value' : v, 'varies' : False}
					else:
						_lgr.debug(f"{j=} {k=}")
						_lgr.debug(f'{do_region_attrs_vary[j][k]['varies']=}')
						if not do_region_attrs_vary[j][k]['varies']:
							do_region_attrs_vary[j][k]['varies'] = do_region_attrs_vary[j][k]['last_value'] != v
							_lgr.debug(f"{do_region_attrs_vary[j][k]['last_value']=} {v=} {do_region_attrs_vary[j][k]['last_value'] != v}")
							do_region_attrs_vary[j][k]['last_value'] = v
					
		_lgr.debug(f'\n#####\n{do_region_attrs_vary=}\n#####')
		
		self.interp_attrs = []
		for j, do_attrs_vary in do_region_attrs_vary.items():
			if len(self.interp_attrs) <= j:
				self.interp_attrs.append(list())
			
			for k,v in do_attrs_vary.items():
				if v['varies']:
					self.interp_attrs[j].append(k)
		_lgr.debug(f'{self.interp_attrs=}')
		
		self.attr_value_dict = {}
		for j, attrs in enumerate(self.interp_attrs):
			if j not in self.attr_value_dict:
				self.attr_value_dict[j] = {}
			for attr in attrs:
				self.attr_value_dict[j][attr] = {'type' : None, 'values' : []}
				for idx in self.indices:
					r_attr_value = getattr(dynamic_region_dict[idx][j], attr)
					_lgr.debug(f'{r_attr_value.__class__}')
					if self.attr_value_dict[j][attr]['type'] is None:
						self.attr_value_dict[j][attr]['type'] = r_attr_value.__class__
					
					assert r_attr_value.__class__ is self.attr_value_dict[j][attr]['type'], "dynamic region attributes cannot change type"
					match r_attr_value.__class__:
						case regions.PixCoord:
							self.attr_value_dict[j][attr]['values'].append(np.array((r_attr_value.x, r_attr_value.y)))
						case builtins.int:
							self.attr_value_dict[j][attr]['values'].append(np.array((r_attr_value,)))
						case builtins.float:
							self.attr_value_dict[j][attr]['values'].append(np.array((r_attr_value,)))
						case _:
							raise RuntimeError(f'Cannot convert from unknown region attribute type {attr.__class__} to numpy array')
				self.attr_value_dict[j][attr]['values'] = np.array(self.attr_value_dict[j][attr]['values'])
		return
	
	
	def interp(self, idx):
	
		new_regions = []
	
		for j in range(self.n_regions):
			new_regions.append(self.dynamic_region_defaults[j])
			
			for attr, value in self.attr_value_dict[j].items():
				arg_lt_idx = np.argwhere(self.indices > idx)
				if np.any(arg_lt_idx):
					i_min =np.min(arg_lt_idx)
				else:
					i_min = len(self.indices)-1
				if i_min >=0 and i_min < (len(self.indices)-1):
					frac = (idx - self.indices[i_min]) / (self.indices[i_min+1] - self.indices[i_min])
					diff = value['values'][i_min+1] - value['values'][i_min]
				else:
					frac = 0
					diff = 0
				
				
				interp_value = value['values'][i_min] + frac*diff
				
				match value['type']:
					case regions.PixCoord:
						interp_attr = regions.PixCoord(*interp_value)
					case builtins.int:
						interp_attr = interp_value
					case builtins.float:
						interp_attr = interp_value
					case _:
						raise RuntimeError(f'Cannot convert from numpy to unknown region attribute type {value["type"]}')
				setattr(new_regions[j], attr, interp_attr)
		return new_regions
		
def run(
		fits_spec,
		output_path,
		index_cut_values : list[list[float,float],...] | None = None,
		const_regions : list[str] = [],
		dynamic_regions : list[tuple[int,str]] = []
	):
	
	if index_cut_values is None:
		index_cut_values = [[0,4.5]]
	
	const_region_list = load_const_regions(const_regions)
	dynamic_region_dict = load_dynamic_regions(dynamic_regions)
	
	print(const_region_list)
	

	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		
		bad_pixel_mask = np.zeros_like(data, dtype=bool)
		
		cv_pos = 0
		next_cv_pos = 1
		
		for const_region in const_region_list:
			bad_pixel_mask |= const_region.to_mask(mode='center').to_image(bad_pixel_mask.shape[1:], dtype=bool)[None,:,:]
		
		dynamic_region_interpolator = DynamicRegionInterpolator(dynamic_region_dict)

		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=}')
			current_data_idx = idx[0][tuple(0 for i in fits_spec.axes['CELESTIAL'])]
			
			# Don't bother working on all NAN slices
			if np.all(np.isnan(data[idx])):
				continue
			
			_lgr.debug(f'{current_data_idx=}')
			
			for r in dynamic_region_interpolator.interp(current_data_idx):
				bad_pixel_mask[idx] |= r.to_mask(mode='center').to_image(bad_pixel_mask[idx].shape, dtype=bool)
			
			while next_cv_pos < len(index_cut_values) and index_cut_values[next_cv_pos][0] < current_data_idx:
				next_cv_pos += 1
			cv_pos = next_cv_pos -1
			
			if next_cv_pos < len(index_cut_values):
				lo_cv_idx = index_cut_values[cv_pos][0]
				hi_cv_idx = index_cut_values[next_cv_pos][0]
				
				lo_cv_value = index_cut_values[cv_pos][1]
				hi_cv_value = index_cut_values[next_cv_pos][1]
				_lgr.debug(f'{lo_cv_idx=} {hi_cv_idx=} {lo_cv_value=} {hi_cv_value=}')
				cv_value = (current_data_idx-lo_cv_idx)*(hi_cv_value - lo_cv_value)/(hi_cv_idx - lo_cv_idx) + lo_cv_value
			else:
				cv_value = index_cut_values[cv_pos][1]
			
			_lgr.debug(f'{cv_value=}')
			
			
			
			# Large "badness values" should have a larger AOE than smaller "badness values"
			# Therefore, dilate according to pixel value, for every 1 larger than the
			# cut value, dilate the pixel once more.
			
			#bp_mask = np.zeros(data[idx].shape, dtype=bool)
			_lgr.debug(f'{(int(np.floor(np.nanmax(data[idx]))), int(np.ceil(cv_value+1)))=}')
			for t in range(int(np.floor(np.nanmax(data[idx]))), int(np.ceil(cv_value)), -1):
				_lgr.debug(f'{t=}')
				diff = t - np.ceil(cv_value)
				_lgr.debug(f'{diff=}')
				#plt.imshow(data[idx] >= t)
				#plt.show()
				bad_pixel_mask[idx] |= sp.ndimage.binary_dilation(data[idx] >= t, iterations = int(diff))
			
			bad_pixel_mask[idx] |= data[idx] >= cv_value
			#bp_mask = data[idx] >= cv_value
			
	
	
		hdr = data_hdu.header
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			#**dict((f'cut_value_of_index_{k}', v) for k,v in index_cut_values)
		}
		#for i, x in enumerate(bad_pixel_map_binary_operations):
		#	param_dict[f'bad_pixel_map_binary_operations_{i}'] = x
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='artefact_detection',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
				

	
	
	# Save the products to a FITS file
	hdu_bad_pixel_mask = fits.PrimaryHDU(
		header = hdr,
		data = bad_pixel_mask.astype(int)
	)
	hdu_cut_value_table = fits.BinTableHDU.from_columns(
		columns = [
			fits.Column(name='cut_index', format='I', array=[x[0] for x in index_cut_values]), 
			fits.Column(name=f'cut_value', format='D', array=[x[1] for x in index_cut_values])
		],
		name = 'CUT_VALUE_BY_INDEX',
		header = None,
	)
	
	hdul_output = fits.HDUList([
		hdu_bad_pixel_mask,
		hdu_cut_value_table
	])
	hdul_output.writeto(output_path, overwrite=True)
	
	


def parse_args(argv):
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_bpmask'
	DESIRED_FITS_AXES = ['CELESTIAL']
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		os.get_terminal_size().columns - 30
	)
	
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.RawTextHelpFormatter,
		epilog=FITS_SPECIFIER_HELP
	)
	
	parser.add_argument(
		'fits_spec',
		help = '\n'.join((
			f'FITS Specifier of the badness map to operate upon . See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	#parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as the `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	parser.add_argument(
		'-o', 
		'--output_path', 
		type=FPath,
		metavar='str',
		default='{parent}/{stem}{tag}{suffix}',
		help = '\n    '.join((
			f'Output fits file path, supports keyword substitution using parts of `fits_spec` path where:',
			'{parent}: containing folder',
			'{stem}  : filename (not including final extension)',
			f'{{tag}}   : script specific tag, "{DEFAULT_OUTPUT_TAG}" in this case',
			'{suffix}: final extension (everything after the last ".")',
			'\b'
		))
	)
	
	parser.add_argument('-x', '--value_cut_at_index', metavar='int float', type=float, nargs=2, action='append', default=[], help='[index, value] pair, for a 3D badness map `index` will be cut at `value`. Specify multiple times for multiple indices. The `value` for non-specified indices is interpolated with "extend" boundary conditions.')
	
	parser.add_argument('--const_regions', type=str, nargs='+', default=[], 
		help='DS9 region files that defines regions to be masked. Assumed to not move, and will be applied to all wavelengths. Must use IMAGE coordinates.'
	)
	parser.add_argument('--dynamic_regions', type=str, metavar='int path', nargs=2, action='append', default=[],
		help='[index, path] pair. Defines a set of region files that denote **dynamic** regions that should be masked. `index` denotes the wavelength index the regions in a file apply to, region parameters are interpolated between index values, and associated by order within a file. Therefore, set a region to have zero size to remove it, but keep the entry present. Must have IMAGE coordinates.'
	)
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	#if args.output_path is None:
	#	args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	other_file_path = Path(args.fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	if len(args.value_cut_at_index) == 0:
		args.value_cut_at_index = [[0,3]]
	for i in range(len(args.value_cut_at_index)):
		args.value_cut_at_index[i][0] = int(args.value_cut_at_index[i][0])
	args.dynamic_regions = [(int(index), path) for index, path in args.dynamic_regions]
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	_lgr.debug(f'{vars(args)=}')
	
	run(
		args.fits_spec, 
		output_path=args.output_path, 
		index_cut_values = args.value_cut_at_index,
		const_regions = args.const_regions,
		dynamic_regions = args.dynamic_regions
	)
	