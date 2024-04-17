"""
Classes and routines that assist with plotting, specifically plotting the
progress of algorithms.
"""
from typing import Any, Callable
import dataclasses as dc
import copy
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.figure
import matplotlib.pyplot as plt

def output(
		show : bool | float = False, # Should we show the figure? If a number, then display figure for this number of seconds before moving onwards
		fname : str | Path | None = None, # If not None, save the figure to this file
		figure : mpl.figure.Figure | str | int | None = None, # The figure to operate on, default is current figure
		**kwargs # arguments to pass when saving figure, see plt.savefig(...)
	):
	"""
	Outputs the passed figure to a file `fname`, interactively `show` it, and close it afterwards
	"""
	if figure is not None:
		plt.figure(figure)
		
	if fname is not None:
		plt.savefig(fname, **kwargs)
	if type(show) is bool and show:
		plt.show()
	elif type(show) in (int,float):
		plt.pause(show)
		
	plt.close(figure)


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
	else:
		f = figure
	a = f.subplots(nr, nc, **sp_kwargs)
	
	for i, _ax in enumerate(a.flatten()):
		if i>=nax:
			_ax.remove()
	return(f, a.flatten() if flatten else a)

def figure_n_subplots(n, figure=None, fig_kwargs={}, sp_kwargs={}):
	"""
	Create a figure with `n` subplots arranged in a rectangle that has the lowest aspect ratio
	"""
	return(create_figure_with_subplots(
		*lowest_aspect_ratio_rectangle_of_at_least_area(n), 
		nax=n, size=6, squeeze=False, figure=figure, flatten=True,
		fig_kwargs=fig_kwargs,
		sp_kwargs=sp_kwargs)
	)


@dc.dataclass
class DiffClass:
	initial : Any = np.nan
	
	_stored : Any = dc.field(default=None, init=False, repr=False, compare=False)
	
	def set_initial(self, value):
		self.initial=value
		return self
	
	def __call__(self, value):
		prev = self._stored
		self._stored = copy.copy(value)
		return self.initial if prev is None else value - prev


def lim(data):
	a,b = np.nanmin(data), np.nanmax(data)
	return None if np.isnan(a) else a, None if np.isnan(b) else b
	

@dc.dataclass
class LimFixed:
	vmin : float | None = None
	vmax : float | None = None
	
	func : Callable[[Any],tuple[float,float]] = lim
	
	def __call__(self, data):
		a,b = self.func(data)
		if self.vmin is not None: a=self.vmin
		if self.vmax is not None: b=self.vmax
		return(a,b)


@dc.dataclass
class LimRememberExtremes:
	func : Callable[[Any],tuple[float,float]] = lim
	
	_vmin : float = None
	_vmax : float = None
	
	def __call__(self, data):
		a,b = self.func(data)
		if self._vmin is None or (a < self._vmin and a is not None): self._vmin = a
		if self._vmax is None or (b > self._vmax and b is not None): self._vmax = b
		return self._vmin, self._vmax

@dc.dataclass
class LimRememberNExtremes:
	n : int = 50
	func : Callable[[Any],tuple[float,float]] = lim
	
	_i : int = 0
	_vmin : tuple[int,float] = (0,None)
	_vmax : tuple[int,float] = (0,None)
	
	def __call__(self, data):
		#print(f'{data[-self.n:]=} {self._i=} {self.n=} {self._vmin=} {self._vmax=}')
		a,b = self.func(data[-self.n:])
		if a is not None and (self._vmin[0] < (self._i-self.n) or self._vmin[1] is None or a < self._vmin[1]): self._vmin = (self._i,a)
		if b is not None and (self._vmax[0] < (self._i-self.n) or self._vmax[1] is None or b > self._vmax[1]): self._vmax = (self._i,b)
		
		self._i += 1
		return self._vmin[1], self._vmax[1]

@dc.dataclass
class LimSymAroundCurrent:
	min_diff : float = 1.0
	func : Callable[[Any],tuple[float,float]] = lim
	
	def __call__(self, data):
		a, b = self.func(data)
		value = data[-1]
		farthest_from_value = np.nanmax(np.fabs([0 if a is None else a-value, 0  if b is None else b-value]))
		if farthest_from_value < self.min_diff:
			farthest_from_value = self.min_diff
		if np.isnan(farthest_from_value):
			return None,None
		return(None if a is None else -farthest_from_value + value, None if b is None else farthest_from_value + value)

@dc.dataclass
class LimSymAroundValue:
	value : float = 0
	pad : float = 0 # +ve numbers is a constant, -ve is a multiplier
	
	def __call__(self, data):
		farthest_from_value = np.nanmax(np.fabs(data-self.value))
		if np.isnan(farthest_from_value):
			return None,None
		if self.pad == 0: return(-farthest_from_value + self.value, farthest_from_value + self.value)
		elif self.pad > 0: return(-farthest_from_value + self.value - self.pad, farthest_from_value + self.value + self.pad)
		else: return((1+self.pad)*(-farthest_from_value + self.value), (1+self.pad)*(farthest_from_value + self.value))


@dc.dataclass
class LimAroundExtrema:
	factor : float = 0.1
	
	def __call__(self, data):
		dmin = np.nanmin(data)
		dmax = np.nanmax(data)
		return(dmin - np.fabs(dmin)*self.factor, dmax + np.fabs(dmax)*self.factor)	

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

def fig_add_patch(figure, *patches):
	figure.patches.extend(patches)

def fig_draw_bbox_of_artist(figure, *artists, color='red', linewidth=1, **kwargs):
	renderer = figure.canvas.get_renderer()
	for artist in artists:
		bbox = artist.get_window_extent(renderer)
		figure.lines.append(mpl.lines.Line2D(
			[bbox.x0, bbox.x0, bbox.x1, bbox.x1, bbox.x0], 
			[bbox.y0, bbox.y1, bbox.y1, bbox.y0, bbox.y0],
			linewidth=linewidth,
			color=color,
			**kwargs
		))
