import sys, os
from pathlib import Path
from typing import Any
import argparse

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import skimage as ski
import skimage.measure
import skimage.morphology


from aopp_deconv_tool import plot_helper
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
from aopp_deconv_tool.geometry.bounding_box import BoundingBox

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')



def select_labels_from_props(props, predicate = lambda prop: True):
	selected_labels = []
	for prop in props:
		if predicate(prop):
			selected_labels.append(prop.label)
	return selected_labels
		

def props_keep(props, labels_to_keep):
	props_to_keep = []
	for prop in props:
		if prop.label in labels_to_keep:
			props_to_keep.append(prop)
	return props_to_keep

def props_relabel(props, old_labels, new_labels):
	for prop in props:
		if prop.label in old_labels:
			prop.label = new_labels[old_labels.index(prop.label)]
	return props

def labels_keep(labels, labels_to_keep, background=0):
	for i in range(1, np.max(labels)+1):
		if i not in labels_to_keep:
			labels[labels==i] = background
	return labels

def labels_relabel(labels, old_labels, new_labels):
	for i, lbl in enumerate(old_labels):
		labels[labels==lbl] = new_labels[i]
	return labels

def labels_and_props_relabel(labels, props, old_labels, new_labels):
	for prop in props:
		lbl = prop.label
		if lbl in old_labels:
			new_lbl = new_labels[old_labels.index(lbl)]

			labels[labels==lbl] = new_lbl
			prop.label = new_lbl
	return(labels, props)



def get_source_regions(
		data : np.ndarray, 
		name : str = '',
		signal_noise_ratio : float = 5, 
		smallest_obj_area_px : int = 5,
		bbox_inflation_factor : float = 1.1,
		bbox_inflation_px : int = 30,
		morphology_opening : int = 3
	):
	"""
	Return regions, bounding boxes, and parameters used to find emitting regions in `data`
	"""
	_lgr.debug(f'{name=}')
	_lgr.debug(f'{signal_noise_ratio=}')
	_lgr.debug(f'{smallest_obj_area_px=}')
	_lgr.debug(f'{bbox_inflation_factor=}')
	_lgr.debug(f'{bbox_inflation_px=}')
	

	data_object_mask = data >= (np.median(data)*signal_noise_ratio)

	if morphology_opening > 0:
		data_object_mask = ski.morphology.opening(
			data_object_mask,
			ski.morphology.disk(morphology_opening, decomposition='sequence')
		)
	elif morphology_opening < 0:
		data_object_mask = ski.morphology.closing(
			data_object_mask,
			ski.morphology.disk(-1*morphology_opening, decomposition='sequence')
		)


	labels = ski.measure.label(data_object_mask)
	
	props = ski.measure.regionprops(labels)

	# only keep regions larger than `smallest_obj_area_px`
	large_region_labels = select_labels_from_props(props, lambda prop: prop.num_pixels >= smallest_obj_area_px)
	props = props_keep(props, large_region_labels)
	labels = labels_keep(labels, large_region_labels)

	props = sorted(props, key= lambda x: x.num_pixels, reverse=True)
	
	labels = labels_keep(labels, [p.label for p in props])

	labels, props = labels_and_props_relabel(labels, props, [p.label for p in props], tuple(range(1,len(props)+1)))
	
	bboxes = tuple(BoundingBox.from_min_max_tuple(prop.bbox).inflate(bbox_inflation_factor, bbox_inflation_px) for prop in props)

	return (
		labels, 
		bboxes,
		dict(
			name = name,
			signal_noise_ratio = signal_noise_ratio, 
			smallest_obj_area_px = smallest_obj_area_px,
			bbox_inflation_factor = bbox_inflation_factor,
			bbox_inflation_px = bbox_inflation_px,
			morphology_opening = morphology_opening,
		),
	)


def plot_source_regions(
		data : np.ndarray,
		labels,
		bboxes : tuple[BoundingBox,...],
		params : dict[str,Any],
		fig = None,
		ax = None,
		fig_kwargs=None,
	):
	noise_estimate = np.median(data[labels==0])
	nplots = len(bboxes)+2
	nr, nc = int(nplots // np.sqrt(nplots)), int(np.ceil(nplots/(nplots//np.sqrt(nplots))))
	
	f,ax = plot_helper.ensure_fig_and_ax(fig, ax, fig_kwargs={'figsize':(12,8)}, subplot_spec=(nr, nc))
	
	f.suptitle(f'Regions: S/N >= {params["signal_noise_ratio"]}; n_pixels >= {params["smallest_obj_area_px"]}')

	for a in ax:
		a.xaxis.set_visible(False)
		a.yaxis.set_visible(False)

	ax[0].set_title(f'{params["name"]}')
	ax[0].imshow(data)

	ax[1].set_title(f'Labels')
	ax[1].imshow(labels)

	for i, bbox in enumerate(bboxes):
		rect = mpl.patches.Rectangle(*bbox.to_mpl_rect(),edgecolor='r', facecolor='none', lw=1)
		text = f'region {i+1}'

		region_data = data[bbox.to_slices()]
		conservative_sig_noise = np.max(region_data)/np.median(region_data)
		region_sig_noise = np.max(region_data)/noise_estimate


		ax[0].add_patch(rect)
		ax[0].text(*bbox.mpl_min_corner, text, color='r', horizontalalignment='left', verticalalignment='top')
		ax[i+2].imshow(region_data)
		ax[i+2].set_title(f'{text}: S/N={region_sig_noise:0.2g} {"x".join(str(x) for x in bbox.extent)}')
	
	# Delete unused plot frames
	for i in range(len(bboxes)+2, len(ax)):
		ax[i].remove()

	return fig, ax

def run(
		fits_spec : aph.fits.specifier.FitsSpecifier,
		output_path_format : FPath,
		n_sources,
		signal_noise_ratio : float = 5, 
		smallest_obj_area_px : int = 5,
		bbox_inflation_factor : float = 1.1,
		bbox_inflation_px : int = 30,
		morphology_opening : int  = 3,
		plot : bool = False
	):
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data[fits_spec.slices]
		hdr = data_hdu.header
		original_data_type=data_hdu.data.dtype
		axes = fits_spec.axes['CELESTIAL']
		
		assert all([s==1 for i, s in enumerate(data.shape) if i not in axes]), f"For extracting sources, we can only operate on a single 2D image not one of shape {data.shape}."
		
		if data.ndim > 2:
			squeeze_select = tuple(0 if i not in axes else slice(None) for i in range(data.ndim))
			undo_squeeze = tuple(None if i not in axes else slice(None) for i in range(data.ndim))
		else:
			squeeze_select = None
			undo_squeeze = None
		
		if squeeze_select is not None:
			data = data[*squeeze_select]
		
		
		labels, bboxes, params = get_source_regions(
			data,
			Path(fits_spec.path).name,
			signal_noise_ratio,
			smallest_obj_area_px,
			bbox_inflation_factor,
			bbox_inflation_px,
			morphology_opening,
		)
		
		if n_sources is not None:
			labels[labels > n_sources] = 0
			bboxes = bboxes[:n_sources]
		else:
			n_sources = len(bboxes)
		
		
		if plot:
			fig, ax = plot_source_regions(data, labels, bboxes, params)
			plot_helper.output(show=True, figure=fig)
		
		
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'n_sources' : n_sources,
			'signal_noise_ratio' : signal_noise_ratio,
			'smallest_obj_area_px' : smallest_obj_area_px,
			'bbox_inflation_factor' : bbox_inflation_factor,
			'bbox_inflation_px' : bbox_inflation_px,
			'morphology_opening' : morphology_opening,
		}
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='extract_sources',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
	
	for i in range(0, n_sources):
		hdus = []
	
		output_data = data[bboxes[i].to_slices()].astype(original_data_type)
		if undo_squeeze:
			output_data = output_data[*undo_squeeze]
	
		hdus.append(fits.PrimaryHDU(
			header = hdr,
			data = output_data
		))
		
		
		# Want source 0 to be background, but do not save source 0, so start from source 1
		output_path = output_path_format.with_fields(i=i+1)
		
		hdul_output = fits.HDUList(hdus)
		hdul_output.writeto(output_path, overwrite=True)
		_lgr.info(f'Written source {i+1} to {output_path.relative_to(Path(),walk_up=True)}')
	
	


def parse_args(argv):
	parser = argparse.ArgumentParser(
		description=__doc__,
	)
	
	DEFAULT_OUTPUT_TAG = '_source'
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
			f'FITS Specifier of the data to operate upon . See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	
	
	parser.add_argument(
		'-o', 
		'--output_path_format', 
		type=FPath,
		metavar='str',
		default='{parent}/{stem}{tag}_{i}{suffix}',
		help = '\n    '.join((
			f'Output fits file path format, supports keyword substitution using parts of `fits_spec` path where:',
			'{parent}: containing folder',
			'{stem}  : filename (not including final extension)',
			f'{{tag}}   : script specific tag, "{DEFAULT_OUTPUT_TAG}" in this case',
			'{suffix}: final extension (everything after the last ".", including the ".")',
			'{i} : The source number from brightest to dimmest.',
			'\b'
		))
	)
	
	parser.add_argument(
		'-n',
		'--n_sources',
		type = int,
		default = None,
		help='Maximum number of sources that will be foundnd and made into their own file. Will return largest sources first.'
	)
	
	parser.add_argument(
		'--signal_noise_ratio',
		type=float,
		default = 5, 
		help='Minimum signal/noise for a source to be found'
	)
	parser.add_argument(
		'--smallest_obj_area_px',
		type=int,
		default = 5,
		help='Smallest number of pixels above S/N for a source to be found'
	)
	parser.add_argument(
		'--bbox_inflation_factor',
		type=float,
		default=1.1,
		help='Minimum factor by which to inflate the bounding box of a found source'
	)
	parser.add_argument(
		'--bbox_inflation_px',
		type= int,
		default = 30,
		help='Minimum number of pixels by which to inflate the bounding box of a found source.'
	)
	parser.add_argument(
		'--morphology_opening',
		type=int,
		default=3,
		help='Apply morphology opening to S/N ratio map of observation. Large value groups nearby sources together.'
	)
	
	
	parser.add_argument(
		'--plot',
		action='store_true',
		help='If present, will plot the regions found before writing them to disk.'
	)
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	other_file_path = Path(args.fits_spec.path)
	args.output_path_format = FPath(
		args.output_path_format.with_fields(
			parent=other_file_path.parent, 
			stem=other_file_path.stem, 
			suffix=other_file_path.suffix,
			tag=DEFAULT_OUTPUT_TAG,
			i='{i}'
		)
	)
	
	for k,v in vars(args).items():
		_lgr.debug(f'{k} = {v}')
	
	return args



def go(
		fits_spec,
		output_path_format=None,
		n_sources = None,
		signal_noise_ratio = None, 
		smallest_obj_area_px = None, 
		bbox_inflation_factor = None, 
		bbox_inflation_px = None, 
		morphology_opening = None,
		plot_FLAG=None,
		help_FLAG=None
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""	
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=1)
	
	exec_with_args(arglist)
	return

def exec_with_args(argv):
	args = parse_args(argv)
	
	run(
		args.fits_spec,
		output_path_format=args.output_path_format,
		n_sources = args.n_sources,
		signal_noise_ratio = args.signal_noise_ratio, 
		smallest_obj_area_px = args.smallest_obj_area_px, 
		bbox_inflation_factor = args.bbox_inflation_factor, 
		bbox_inflation_px = args.bbox_inflation_px, 
		morphology_opening = args.morphology_opening,
		plot = args.plot
	)

if __name__=='__main__':
	exec_with_args(sys.argv[1:])


