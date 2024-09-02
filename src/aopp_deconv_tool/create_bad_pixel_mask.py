r"""
Given a badness map, will apply a value cut (or possibly a range of interpolated value cuts) to the badness map to give a boolean mask that defines "bad" pixels

# EXAMPLE
>>> python -m aopp_deconv_tool.create_bad_pixel_mask './example_data/ifu_observation_datasets/MUSE.2019-10-18T00:01:19.521_rebin_artifactmap.fits' --const_regions ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_const.reg --dynamic_regions 59 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_59.reg --dynamic_regions 147 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_147.reg --dynamic_regions 262 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_262.reg --dynamic_regions 431 ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_dynamic_431.reg

"""

import sys, os
import builtins
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
import aopp_deconv_tool.arguments

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import get_bp_mask_from_badness_map

from aopp_deconv_tool.py_ssa import SSA

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')





def load_const_regions(const_regions : list[str]) -> list[regions.Regions]:
	"""
	Read a list of regions from multiple region files, concatenate them all into one list and return that list.
	"""
	region_list = []
	for fpath in const_regions:
		region_list.extend(regions.Regions.read(fpath))
	return region_list

def load_dynamic_regions(dynamic_regions : list[tuple[int,str]]) -> dict[int|str, list[tuple[int,regions.Regions]]]:
	"""
	Read a list of index,region-file pairs. Assume all regions exist in each file. Create a list of index,regions-parameters for each regions, return a dictionary
	mapping the region name/number to the index,region-parameter list for each region.
	"""
	region_dict = {}
	for index, fpath in sorted(dynamic_regions, key=lambda x:x[0]):
		
		region_list = load_const_regions([fpath])
		
		j=0
		for region in region_list:
			if 'text' in region.meta:
				region_key = region.meta['text']
			else:
				region_key = j
				j+=1
			
			region_dict[region_key] = region_dict.get(region_key, []) + [(index, region)]
	return region_dict


@dc.dataclass
class Attr:
	"""
	Class that simulates an attribute of some `type` that has a value that changes based on an index.
	"""
	type : Type
	values : np.ndarray


class DynamicRegionInterpolator:
	"""
	Class that takes a list of index,region-parameter pairs and interpolates a region between the defined indices.
	"""

	def __init__(self, index_region_list : list[tuple[int, regions.Region]]):
		self.indices = np.array([x[0] for x in index_region_list])
		self.region_example = index_region_list[0][1] if len(index_region_list) > 0 else None
		
		self.region_class = self.region_example.__class__
		
		assert all([self.region_class is x[1].__class__ for x in index_region_list]), "A dynamic region must not change type"

		_lgr.debug(f'{self.indices=} {self.region_example=} {self.region_class=}')
		
		self.attr_vary_names = []
		attr_prev_values = {}
		
		self.attrs = {}
		
		for i, (idx, region) in enumerate(index_region_list):
			for attr_name, attr_value in region.__dict__.items():
				_lgr.debug(f'{attr_name=} {attr_value=}')
				if attr_name in ('meta', 'visual') or attr_name.startswith('_'):
					continue
				
				if attr_name not in self.attrs:
					match attr_value.__class__:
						case builtins.int:
							self.attrs[attr_name] = Attr(attr_value.__class__, np.zeros((*self.indices.shape,), dtype=int))
						case builtins.float:
							self.attrs[attr_name] = Attr(attr_value.__class__, np.zeros((*self.indices.shape,), dtype=float))
						case regions.PixCoord:
							self.attrs[attr_name] = Attr(attr_value.__class__, np.zeros((*self.indices.shape,2), dtype=float))
						case _:
							raise RuntimeError(f'Cannot find index-varying attribute information for attribute {attr_name} of type {attr_value.__class__}')

				match self.attrs[attr_name].type:
					case builtins.int:
						self.attrs[attr_name].values[i] = attr_value
					case builtins.float:
						self.attrs[attr_name].values[i] = attr_value
					case regions.PixCoord:
						self.attrs[attr_name].values[i][0] = attr_value.x
						self.attrs[attr_name].values[i][1] = attr_value.y
					case _:
						raise RuntimeError(f'Cannot assign values to index-varying attribute information for attribute {attr_name} of type {sefl.attrs[attr_name].type}')

		_lgr.debug(f'{self.attrs=}')
		
	def interp(self, index):
		if self.indices.size == 0 or index < self.indices[0] or self.indices[-1] < index:
			return None
		
		i_gt = np.sum(index >= self.indices)
		
		
		for attr_name, attr in self.attrs.items():
			if i_gt < self.indices.size:
				frac = (index - self.indices[i_gt-1])/(self.indices[i_gt] - self.indices[i_gt-1])
				diff = attr.values[i_gt] - attr.values[i_gt-1]
			else:
				frac = 0
				diff = attr.values[i_gt-1] - attr.values[i_gt-1]
			
			_lgr.debug(f'{i_gt=} {attr.values[i_gt-1]=} {frac=} {diff=}')
			interp_value = attr.values[i_gt-1] + frac*diff
			
			match attr.type:
				case builtins.int:
					interp_attr = interp_value
				case builtins.float:
					interp_attr = interp_value
				case regions.PixCoord:
					interp_attr = regions.PixCoord(*interp_value)
				case _:
					raise RuntimeError(f'Cannot interpolate value for attribute {attr_name} of unrecognised type {attr.type}')
			
			setattr(self.region_example, attr_name, interp_attr)
		return self.region_example
		

	
def run(
		fits_spec,
		output_path,
		index_cut_values : list[list[float,float],...] | None = None,
		const_regions : list[str] = [],
		dynamic_regions : list[tuple[int,str]] = []
	):
	"""
	Perform the operation associated with this module.
	"""
	
	if index_cut_values is None:
		index_cut_values = [[0,4.5]]
	
	const_region_list = load_const_regions(const_regions)
	dynamic_region_dict = load_dynamic_regions(dynamic_regions)
	
	for r in const_region_list:
		_lgr.debug(f'{r=} {r.meta=} {r.visual=}')
		
	for k in dynamic_region_dict:
		_lgr.debug(f'dynamic_region_dict[{k}] = {dynamic_region_dict[k]}')
		
	dynamic_region_interpolators = [DynamicRegionInterpolator(v) for v in dynamic_region_dict.values()]

	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		
		bad_pixel_mask = np.zeros_like(data, dtype=bool)
		
		cv_pos = 0
		next_cv_pos = 1
		
		for const_region in const_region_list:
			bad_pixel_mask |= const_region.to_mask(mode='center').to_image(bad_pixel_mask.shape[1:], dtype=bool)[None,:,:]
		

		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=}')
			current_data_idx = idx[0][tuple(0 for i in fits_spec.axes['CELESTIAL'])]
			
			# Don't bother working on all NAN slices
			if np.all(np.isnan(data[idx])):
				continue
			
			_lgr.debug(f'{current_data_idx=}')
			
			for dri in dynamic_region_interpolators:
				r = dri.interp(current_data_idx)
				if r is not None:
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
	OUTPUT_COLUMNS=80
	try:
		OUTPUT_COLUMNS = os.get_terminal_size().columns - 30
	except Exception:
		pass
	
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		OUTPUT_COLUMNS
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


def go(
		fits_spec,
		output_path=None,
		value_cut_at_index=None,
		const_regions=None,
		dynamic_regions=None
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""
	# Add stuff to kwargs here if needed
	
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=1)
	exec_with_args(arglist)

def exec_with_args(argv):
	args = parse_args(argv)
	
	_lgr.debug(f'{vars(args)=}')
	
	run(
		args.fits_spec, 
		output_path=args.output_path, 
		index_cut_values = args.value_cut_at_index,
		const_regions = args.const_regions,
		dynamic_regions = args.dynamic_regions
	)

if __name__ == '__main__':
	exec_with_args(sys.argv[1:])
	