"""
Classes and routines that assist with plotting, specifically plotting the
progress of algorithms.
"""
import numpy as np
import matplotlib.pyplot as plt

def lowest_aspect_ratio_rectangle_of_at_least_area(x):
	sqrt_x = np.sqrt(x)
	b = int(sqrt_x)
	a = b
	while a*b < x:
		a += 1
	return(b,a)

def create_figure_with_subplots(nr, nc, nax=None ,size=6, squeeze=False, figure=None, flatten=True, fig_kwargs={}, sp_kwargs={}):
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
	a = f.subplots(nr, nc, **sp_kwargs)
	
	for i, _ax in enumerate(a.flatten()):
		if i>=nax:
			_ax.remove()
	return(f, a.flatten() if flatten else a)

def figure_n_subplots(n, figure=None, fig_kwargs={}, sp_kwargs={}):
	return(create_figure_with_subplots(
		*lowest_aspect_ratio_rectangle_of_at_least_area(n), 
		nax=n, size=6, squeeze=False, figure=figure,
		fig_kwargs={},
		sp_kwargs={})
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
