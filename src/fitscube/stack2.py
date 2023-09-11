#!/usr/bin/env python3
"""
Contains routines for stacking sections of a cube together
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')


import sys, os
from astropy.io import fits
import numpy as np

import utilities as ut
import utilities.args
import utilities.path
from utilities.classes import Slice

# DEBUGGING
#import matplotlib.pyplot as plt

def ignorenanmean(data, axis=None):
	nnm_sum = np.nansum(~np.isnan(data), axis)
	#print(nnm_sum)
	data_mean = np.nansum(data, axis)/nnm_sum
	#plt.imshow(nnm_sum)
	#plt.imshow(data_mean)
	#plt.show()
	return(data_mean)

stack_operation_choices = {
	'sum' : np.sum,
	'mean' : np.mean,
	'median': np.median,
	'nansum': np.nansum,
	'nanmean' : np.nanmean,
	'nanmedian' : np.nanmedian,
	'ignorenanmean' : ignorenanmean,
}


def parse_args(argv):
	parser = ut.args.DocStrArgParser(description=__doc__)

	## Add Positional Arguments ##
	parser.add_argument(
		'cubes', 
		type=str, 
		nargs='+',
		help='fitscubes to operate on',
	)

	## Add Optional Arguments ##
	parser.add_argument('--cube.ext', type = int, help = 'Extensions of fits cubes that holds the data we wish to stack', default = 0)
	#parser.add_argument('--output.fits.fmt', type=str, help='Format string that describes how to get the output file name from the original file.\n\n'+ut.path.fpath_from.__doc__, default='{fdir}/{fname}_stacked{fext}')
	#parser.add_argument('--output.fits.overwrite', action=ut.args.ActionTf, prefix='output.fits.', help='If present, will overwrite any existing file at output location')
	#parser.add_argument('--output.fits.copy', action=ut.args.ActiontF, prefix='output.fits', help='If present, will copy header data units from original cube to output cube.')
	parser = ut.path.add_args(parser, 'output.fits.', defaults={'fmt':'{fdir}/{fname}_stacked{fext}'})
	
	parser.add_argument('--stack.consolidate', action=ut.args.ActionTf, prefix='stack.', help='If True, will consolidate all adjacent stacks into one array if they have the same dimensions, operation, and axes.')
	parser.add_argument('--stack.axes', type=int, nargs='+', action='append', help='The axes we wish to stack over (i.e. the one that will get smaller), require one for each "--stack.slices" specified', default=None)
	parser.add_argument(
		'--stack.slices',
		type=str,
		nargs='+',
		action='extend',
		help = 'A set of slices that detail how to stack each cube. Each slice set will stack to one new image. I.e. "[20:30] [30:40]" will stack data from index 20->30 in the 0th axis and put them in one image, and from index 30->40 and put them in a second image. The axes stacked along is specified by "--cube.ext.axes"',
		default=None,
	)
	
	parser.add_argument(
		'--stack.slice_func',
		type=str,
		help="""\
			A piece of python code that returns a list of (axes, slice) tuples, where "slice" is
			something that could be passed to "utilities.classes.Slice()". Overwrites any values
			specified via "--stack.slices" and "--stack.axes".

			Format example:
				--stack.slice_func '[[[0,], f"[{i-20}:{i}]"] for j, i in enumerate(range(20,459,20))]'
			""",
		default =None,
	)

	parser.add_argument(
		'--stack.operation',
		type=str,
		nargs='+',
		action='extend',
		choices = tuple(stack_operation_choices.keys()),
		help=f'Operations to perform on each slice of the cubes. Choose from {tuple(stack_operation_choices.keys())}, if there are not enough operations specified for the number of stacks, the last one will be repeated',
		default=None,
	)

	parser.add_argument('--data.nan_cutoff', type=float, help='If the absolute value of a pixel is smaller than this value, bad data (NAN) is assumed', default=0)
	parser.add_argument('--data.nan_threshold', type=float, help='If the value of a pixel is smaller than this value, bad data (NAN) is assumed', default=-np.inf)

	args = vars(parser.parse_args()) # I prefer a dictionary interface

	# Filter arguments
	if args['stack.slices'] is None:
		args['stack.slices'] = ['[:]']
	if args['stack.axes'] is None:
		args['stack.axes'] = [[0]]
	if args['stack.operation'] is None:
		args['stack.operation'] = ['ignorenanmean']
	


	_msg = '#'*20 + ' ARGUMENTS ' + '#'*20
	for k, v in args.items():
		_msg += f'\n\t{k}\n\t\t{v}'
	_msg += '\n' + '#'*20 + '###########' + '#'*20
	_lgr.INFO(_msg)

	if args['stack.slice_func'] is None:
		# ensure we know how to stack the slices
		if len(args['stack.slices']) != len(args['stack.axes']):
			raise ValueError(f'Do not have the same number of slices as axes to slice over, have {args["stack.slices"]} and {args["cube.ext.axes"]} of each respectively.')

	else:
		at_tuple = eval(args['stack.slice_func'])
		_lgr.DEBUG(f'{at_tuple=}')
		args['stack.axes'], args['stack.slices'] = map(list, zip(*at_tuple))
		_lgr.DEBUG(f'{args["stack.axes"]=}')
		_lgr.DEBUG(f'{args["stack.slices"]=}')
		_lgr.DEBUG(f'{len(args["stack.axes"])=} {len(args["stack.slices"])=}')


	# ensure there are enough operations for the number of stack
	n = len(args['stack.slices']) - len(args['stack.operation'])
	if n > 0:
		args['stack.operation'] += [args['stack.operation'][-1]]*n

	return(args)
	

def main(argv):
	args = parse_args(argv)


	for cube in args['cubes']:
		with fits.open(cube) as hdul:
			fpath_output = ut.path.fpath_from(cube, args['output.fits.fmt'])

			hdu = hdul[args['cube.ext']]
			data = hdu.data
			bd_mask = (np.abs(data) < args['data.nan_cutoff']) | (data < args['data.nan_threshold'])
			data[bd_mask] = np.nan

			partitioned_data = [eval(f'Slice(data)["{x[1:-1]}"]', globals() , dict(x=x,data=data)) for x in args['stack.slices']]

			stacked_data = [stack_operation_choices[args['stack.operation'][i]](pd, axis=tuple(args['stack.axes'][i])) for i, pd in enumerate(partitioned_data)]

			consolidated_data = [[stacked_data[0]]]
			consolidated_data_name = [f'{args["stack.operation"][0]} {args["stack.axes"][0]} ({args["stack.slices"][0]}']
			j = 0
			for i in range(1,len(stacked_data)):
				if (args['stack.consolidate'] 
						and (stacked_data[i].shape == consolidated_data[-1][-1].shape)
						and (args['stack.operation'][i]==args['stack.operation'][i-1])
						and (args['stack.axes'][i] == args['stack.axes'][i-1])
						):
					consolidated_data[-1].append(stacked_data[i])
					consolidated_data_name[-1] += f', {args["stack.slices"][i]}'
				else:
					consolidated_data_name[-1] += ')'
					consolidated_data.append([stacked_data[i]])
					consolidated_data_name.append(f'{args["stack.operation"][i]} {args["stack.axes"][i]} ({args["stack.slices"][i]}')
					j+=1


			for i in range(len(consolidated_data)):
				consolidated_data[i] = np.stack(consolidated_data[i])

			output_hdul = fits.HDUList()
			_lgr.INFO('Adding calculated data to output FITS file...')
			for i, cd in enumerate(consolidated_data):
				output_hdul.append(fits.ImageHDU(header=hdu.header, data=cd, name=consolidated_data_name[i]))

			_lgr.INFO(f'Writing data to "{fpath_output}"...')
			ut.path.write_with_mode(fpath_output, output_hdul, writer=lambda fpath, data, append_flag: (([hdul.append(d) for d in data],hdul)[1] if append_flag else data).writeto(fpath, overwrite=True), mode=args['output.fits.mode'])

if __name__=='__main__':
	_lgr.setLevel('DEBUG')
	main(sys.argv[1:])


