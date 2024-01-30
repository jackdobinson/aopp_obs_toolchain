#!/usr/bin/env python3
"""
Contains utility function to help with plotting routines
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import utilities as ut
import utilities.args
import utilities.path
import utilities.text


def lowest_aspect_ratio_rectangle_of_at_least_area(x):
	sqrt_x = np.sqrt(x)
	b = int(sqrt_x)
	a = b
	while a*b < x:
		a += 1
	return(b,a)

def create_figure_with_subplots(nr, nc, nax=None ,size=6, squeeze=False, figure=None, fig_kwargs={}, sp_kwargs={}):
	"""
	Creates a figure and fills it with subplots
	
	# ARGUMENTS #
		nr
			<int> Number of rows
		nc
			<int> Number of columns
		nax
			<int> Number of axes to create
		size [2]
			<float> Size of figure to create x and y dimension, will be multipled by number of columns and rows
		fig_kwargs [dict]
			Figure keyword arguments. 'figsize' will overwrite passed values for 'size' if present.
		sp_kwargs [dict]
			Subplot keyword arguments. 'squeeze' will overwrite passed values if present. 

	# RETURNS #
		f
			Matplotlib figure created
		a [nax]
			List of axes contained in f
	"""
	# validate arguments
	if not hasattr(size, '__getitem__'):
		size = (size, size)
	if nax is None:
		nax = nr*nc

	# validate **kwargs
	if 'figsize' not in fig_kwargs:
		fig_kwargs['figsize'] = [nc*size[0], nr*size[1]]
	if 'squeeze' not in sp_kwargs:
		sp_kwargs['squeeze'] = squeeze
	
	if figure is None:
		f = plt.figure(**fig_kwargs)
	else:
		f = figure
	a = f.subplots(nr, nc, **sp_kwargs)
	
	for i, _ax in enumerate(a.flatten()):
		if i>=nax:
			_ax.remove()
	return(f, a)

def figure_n_subplots(n, figure=None, fig_kwargs={}, sp_kwargs={}):
	return(create_figure_with_subplots(
		*lowest_aspect_ratio_rectangle_of_at_least_area(n), 
		nax=n, size=6, squeeze=False, figure=figure,
		fig_kwargs=fig_kwargs,
		sp_kwargs=sp_kwargs)
	)

def lim_sym_around_value(data, value=0):
	farthest_from_value = np.nanmax(np.fabs(data-value))
	return(-farthest_from_value + value, farthest_from_value + value)

def lim_around_extrema(data, factor=0.1):
	dmin = np.nanmin(data)
	dmax = np.nanmax(data)
	return(dmin - np.fabs(dmin)*factor, dmax + np.fabs(dmax)*factor)

def remove_axes_ticks_and_labels(ax, state=False):
	ax.xaxis.set_visible(state)
	ax.yaxis.set_visible(state)
	return

def remove_axis_all(ax, state=False):
	remove_axes_ticks_and_labels(ax, state)
	for spine in ax.spines.values():
		spine.set_visible(state)
	return


def flip_x_axis(ax):
	ax.set_xlim(ax.get_xlim()[::-1])
	return

def flip_y_axis(ax):
	ax.set_ylim(ax.get_ylim()[::-1])
	return

def get_legend_hls(*args):
	# gets handles and labels for each axis in *args
	hs, ls = [], []
	for ax in args:
		h, l = ax.get_legend_handles_labels()
		hs += h
		ls += l
	return(hs, ls)

def set_legend(*args, a_or_f=None, **kwargs):
	# sets the legend for the passed axes or figure "a_or_f" to the combination
	# of all handles and labels in each axes via "args"
	# **kwargs is passed to the ".legend()" method
	hs, ls = get_legend_hls(*args)
	if a_or_f is None:
		a_or_f = args[0]
	a_or_f.legend(hs, ls, **kwargs)
	return


def add_plot_arguments(parser, prefix='plots.', defaults={}):
	plot_grp = parser.add_argument_group(title='Plotting Arguments', description=None)
	#plot_grp.add_argument(f'--{prefix}dir', type=str, help='Directory relative to target cubes where plots should be saved', default='./plots')
	plot_grp = ut.path.add_args(plot_grp, prefix, defaults.get('fmt', {'fmt':'{fdir}/plots/{fname}_{plot_name}{plot_ext}'}))
	#plot_grp.add_argument(f'--{prefix}save', action=ut.args.ActionTf, prefix=prefix, 
	#					   help='Should the plots be saved?')
	#plot_grp.add_argument(f'--{prefix}show', action=ut.args.ActiontF, prefix=prefix, 
	#					   help='Should the plots be shown interactively?')
	for _action in plot_grp._group_actions:
		print(_action.option_strings)
		if f'--{prefix}mode' in _action.option_strings:
			_action.choices = list(_action.choices)
			_action.choices.remove('append')
			_action.choices.append('show')
			_action.choices.append('save')
			_action.choices = tuple(_action.choices)
			_action.nargs='+'
			#_action.default = [_action.default]
			_action.default = [defaults.get('mode', 'no_output')]
	return


def get_plot_fpath(fmt_str, source_fpath=None, name=None, ext='.png'):
	return(ut.path.fpath_from(os.path.join(os.getcwd(),'plot') if source_fpath is None else source_fpath, fmt_str, plot_name='0' if name is None else name, plot_ext=ext))

def default_fig_writer(fig, fpath):
	fig.savefig(fpath, bbox_inches='tight')

def save_show_plt(fig, fpath, mode, make_dirs, writer=default_fig_writer, shower=lambda fig: fig.show(), add_path=True):
	"""
	# ARGUMENTS #
		fig
			figure to save
		fpath
			filename to use, will be combined with 'outfolder' to give the full path
		mode
			How should the plot be displayed?
		make_dirs
			Should intermediate directories be created?
		plotter
			Default fuction that plots "fig"
		add_path
			Should we add the file path to the figure?
	"""
	import os
	import utilities.text

	fpath_exists = os.path.exists(fpath)
	fpath_dir = os.path.dirname(fpath)
	fpath_dir_exists = os.path.exists(fpath_dir)

	write_flag = 'save' in mode
	show_flag = 'show' in mode

	if (fpath_exists and ('no_overwrite' in mode)):
		write_flag = False

	if ('no_output' in mode):
		write_flag = False,
		show_flag = False

		_lgr.INFO(f'No output written to "{fpath}", as {mode=}, and destination file {"does" if fpath_exists else "does not"} exist.')
		

	if not fpath_dir_exists:
		_lgr.INFO(f'Destination directory "{fpath_dir}" does not exist. {"Creating it..." if make_dirs else "Returning..."}')
		if make_dirs:
			os.makedirs(fpath_dir)

	if add_path:
		fig.text(0,0, '\n'.join(ut.text.Splitter().wrap(os.path.abspath(fpath), width=80)), fontsize=8, ha='left', va='top')

	if write_flag:
		_lgr.INFO(f'Writing figure to {fpath}...')
		writer(fig, fpath)

	if show_flag:
		_lgr.INFO(f'Showing plot...')
		shower(fig)
		input(f'Displaying plot, press [RETURN] to continue > ')

	plt.close(fig)
	return


def progress(ipos, imax, message='Progress', verbose=False):
	import sys
	msg = f'{message} {ipos} of {imax}'
	if ipos<imax-1:
		if verbose:
			print(msg)
		else:
			sys.stdout.write('\r'+msg)
			sys.stdout.flush()
	else:
		if not verbose:
			sys.stdout.write('\r'+msg+'\n')
			sys.stdout.flush()
	return


def default_ani_writer(ani, fpath):
	ani.save(fpath, progress_callback=progress)

def save_animate_plt(ani, fpath, mode, make_dirs, writer=default_ani_writer, shower=lambda fig: fig.show()):
	print(dir(ani))
	print(ani.__dict__)
	ani_writer = lambda f, p: writer(ani, p)
	save_show_plt(ani._fig, fpath, mode, make_dirs, ani_writer, shower, add_path=False)
	return

'''
def save_animate_plt(ani, fname, outfolder, animate_plot=False, save_plot=True, quiet=False, overwrite=True):
	"""
	Uses matplotlib.animation.FuncAnimation, see:
	https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html#matplotlib.animation.FuncAnimation

	It may be useful to use the "nonlocal" keyword if any values get reassigned during the update loop
	"""
	import os
	import utilities.text
	log = lambda msg: _lgr.INFO(msg) if not quiet else None
	if outfolder != None:
		os.makedirs(outfolder, exist_ok=True) # create folder if it does not exist
		filename = os.path.join(outfolder, fname)
		if (not filename.startswith('/')) and (not filename.startswith('./')):
			filename = './'+filename
		if save_plot and (not os.path.exists(filename) or overwrite):
			log(f'Saving animated plot {filename}')
			ani.save(filename, progress_callback=progress)
			log(f'Animated plot {filename} saved.')
	if animate_plot:
		log(f'Showing animated plot')
		plt.show()
		log(f'Animated plot closed...')
	plt.close()
	return
'''

