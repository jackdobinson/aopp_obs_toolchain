"""
Graphical front end to SSA filtering of an image, should be usable for a cube and a 2d image
"""

import sys, os
from pathlib import Path
from typing import Callable, Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.widgets
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.patches
import matplotlib.lines

import PIL.Image

from aopp_deconv_tool.py_ssa import SSA

from aopp_deconv_tool.plot_helper import fig_draw_bbox_of_artist

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


default_scalar_formatter = mpl.ticker.ScalarFormatter().format_data


class CallbackMixin:
	def __init__(self, *args : tuple[str]):
		self.callbacks = {}
		for csn in args:
			self.callbacks[csn] = []
		
	def get_callback_sets(self):
		return self.callbacks.keys()
	
	def set_callback_sets(self, callback_set_names):
		for csn in callback_set_names:
			if csn not in self.callbacks:
				self.callbacks[csn] = []
	
	def remove_callback_sets(self, callback_set_names):
		for csn in callback_set_names:
			del self.callbacks[csn]
	
	def attach_callback(self, callback_set, acallable : Callable[[Any,...],bool]) -> int:
		try:
			i = self.callbacks[callback_set].index(None)
			self.callbacks[callback_set][i] = acallable
			return i
		except ValueError:
			i = len(self.callbacks[callback_set])
			self.callbacks[callback_set].append(acallable)
		return i
	
	def disconnect_callback_idx(self, callback_set, idx):
		self.callbacks[callback_set][idx] = None
	
	def call_callbacks(self, callback_set, *args, **kwargs):
		for acallback in self.callbacks.get(callback_set,[]):
			if acallback is not None: 
				acallback(*args, **kwargs)


class BasePlot:
	def __init__(self, ax, **kwargs):
		self.ax = ax
		self.data = None
		self.kwargs = kwargs
		# self.handle holds the things that are shown on the axes.
		self.handle = None
	
	def remove(self):
		if self.handle is not None:
			# if the handle is a list, then remove each element individually
			# otherwise, remove the handle itself
			if type(self.handle) in (list, tuple):
				for h in self.handle:
					h.remove()
			else:
				self.handle.remove()
		self.handle = None
	
	def set_title(self, title : None | str):
		self.ax.set_title(title)
	

class ImagePlot(BasePlot, CallbackMixin):
	def __init__(self, ax, **kwargs):
		CallbackMixin.__init__(self,'on_set_clim','on_set_data')
		BasePlot.__init__(self, ax, **kwargs)
	
	def set_clim(self, min=None, max=None):
		if min is None or max is None:
			image_data = self.handle.get_array()
			if min is None: min = np.nanmin(image_data)
			if max is None: max = np.nanmax(image_data)
		self.handle.set(clim=(min,max))
		self.call_callbacks('on_set_clim', min, max)
		
	
	def set_axes_limits(self, xlim=(None,None), ylim=(None,None)):
		self.ax.set_xlim(*xlim)
		self.ax.set_ylim(*ylim)
	
	
	def sanitise_data(self, new_data):
		if type(new_data) is not np.ndarray:
			raise RuntimeError(f'{self.__class__.__name__} accepts 2 dimensional numpy arrays')
		if new_data.ndim != 2:
			raise RuntimeError(f'{self.__class__.__name__} accepts 2 dimensional numpy arrays as input data, was passed data with {new_data.ndim} dimensions')
		return new_data
	
	def is_new_data_compatible(self, sanitised_data):
		if self.data is None:
			return False
		return all(s1 == s2 for s1,s2 in zip(self.data.shape, sanitised_data.shape))
	
	def remove(self):
		if self.handle is not None:
			self.handle.remove()
		self.handle = None
	
	def set_data(self, new_data):
		sanitised_new_data = self.sanitise_data(new_data)
		if not self.is_new_data_compatible(sanitised_new_data):
			self.remove()
			self.data = sanitised_new_data
			self.create()
		else:
			self.data = sanitised_new_data
			self.handle.set_data(self.data)
		self.set_axes_limits()
		self.set_clim()
		self.call_callbacks('on_set_data', self)
	
	
	def create(self):
		self.handle = self.ax.imshow(self.data, **self.kwargs)


class LinePlot(BasePlot, CallbackMixin):
	def __init__(self, ax, **kwargs):
		CallbackMixin.__init__(self,'on_set_data')
		BasePlot.__init__(self, ax, **kwargs)

	def is_new_data_compatible(self, sanitised_new_data):
		if (self.data is None
			or len(sanitised_new_data) > len(self.data)
		):
			return False
		return True

	

	def sanitise_data(self, new_data):
		# want to put data into a format where we have
		# three dimensions: "dataset", "x or y data", "values"
		# I.e. the shape is always (n_datasets, 2, n_points)
		ndim = 0
		_lgr.debug(f'{type(new_data)=}')
		if type(new_data) is np.ndarray:
			ndim = new_data.ndim
		else:
			# how many levels can we index
			test_value = new_data
			while hasattr(new_data, "__getitem__"):
				ndim += 1
				test_value = new_data[0]
		
		match (ndim):
			case 1:
				return ((np.arange(len(new_data)),new_data),)
			case 2:
				return (new_data,)
			case 3:
				return new_data
			case _:
				raise ValueError(f'{self.__class__.__name__} can only accept data with up to 3 dimensions of shape (n_datasets, 2, n_points). Passed data has {ndim} dimensions.')
			
		
	def set_axes_limits(self, 
			xlim : tuple[float | None, float | None] = (None,None), 
			ylim  : tuple[float | None, float | None] = (None,None)
		):
		self.ax.set_xlim(*xlim)
		self.ax.set_ylim(*ylim)
		


	def set_data(self, new_data):
		sanitised_new_data = self.sanitise_data(new_data)
		if not self.is_new_data_compatible(sanitised_new_data):
			self.remove()
			self.data = sanitised_new_data
			self.create()
		else:
			for i, line in enumerate(self.handle):
				if i < len(sanitised_new_data):
					line.set_data(sanitised_new_data[i])
				else:
					line.remove()
		self.set_axes_limits()
		self.call_callbacks('on_set_data', self)
	
	def create(self):
		self.handle = []
		for dataset in self.data:
			self.handle.extend(self.ax.plot(dataset[0], dataset[1], **self.kwargs))
				

class WidgetBase:
	def __init__(self, fig, ax_rect):
		self.ax = fig.add_axes(ax_rect)
		self.widget = None

	def set_active(self, active : bool):
		super(mpl.widgets.AxesWidget, self.widget).set_active(active)
	
	def get_active(self):
		return super(mpl.widgets.AxesWidget, self.widget).active
	
	def set_value(self, value):
		self.widget.set_val(value)
	
	def get_value(self):
		return self.widget.val
	
	def disconnect(self, cid):
		return self.widget.disconnect(cid)
	
	def remove(self):
		self.ax.remove()



class Slider(WidgetBase):
	def __init__(self, fig, ax_rect, min=0, max=0, step=None, label='slider', orientation='vertical', **kwargs):
		super().__init__(fig, ax_rect)
		
		self.widget = mpl.widgets.Slider(self.ax, valmin=min, valmax=max, valinit=min, valstep=step, label=label, orientation=orientation)
	
		self.set_labels_for_orientation(fig, orientation)
	
	def set_active(self, active : bool):
		if not active:
			self.set_limits(0,0)
		super().set_active(active)
	
	def set_labels_for_orientation(self, fig, orientation):
		renderer = fig.canvas.get_renderer()
		if orientation == 'vertical':
			h_pos = 0.2
			self.widget.label.set_ha('right')
			self.widget.label.set_va('bottom')
			self.widget.label.set_rotation('vertical')
			self.widget.label.set_position([h_pos,0.0])
			label_bb = self.ax.transAxes.inverted().transform_bbox(self.widget.label.get_window_extent(renderer))
			
			self.widget.valtext.set_ha('right')
			self.widget.valtext.set_va('bottom')
			self.widget.valtext.set_rotation('vertical')
			self.widget.valtext.set_position([h_pos,label_bb.height+0.01])
		
	
	def set_limits(self, min = None, max = None):
		if min is None: min = self.widget.valmin
		if max is None: max = self.widget.valmax
		assert min <= max, f'{self.__class__.__name__}.set_limits(...), min must be <= max, currently {min=} {max=}'
		self.widget.valmin = min
		self.widget.valmax = max
		
		value = self.get_value()
		if value < min:
			self.set_value(min)
		elif value > max:
			self.set_value(max)
		
		match self.widget.orientation:
			case 'vertical':
				self.ax.set_ylim(min,max)
			case 'horizontal':
				self.ax.set_xlim(min,max)
			case _:
				raise RuntimeError(f'Slider orientation "{self.widget.orientation}" is unknown, allowed values "vertical" or "horizontal"')
	
	def on_changed(self, acallable : Callable[[tuple[float,float]],bool]):
		def callback(x):
			if acallable(x):
				self.ax.figure.canvas.draw()
				
		return self.widget.on_changed(callback)


class CheckButtons(WidgetBase):
	def __init__(self, fig, ax_rect, labels=['checkbutton 1', 'checkbutton 2'],**kwargs):
		super().__init__(fig, ax_rect)
		self.labels = labels
		self.widget = mpl.widgets.CheckButtons(self.ax, labels=self.labels, **kwargs)
	
	def on_clicked(self, dict_of_callables : dict[str,Callable[[bool],None]]):
		def dispatch_check_events(label_toggled):
			callback = dict_of_callables.get(label_toggled,None)
			if callback is None: return

			state = self.get_value()[self.labels.index(label_toggled)]
			if callback(state):
				self.ax.figure.canvas.draw()
			
		return self.widget.on_clicked(dispatch_check_events)
	
	def get_value(self):
		return self.widget.get_status()
	
	def set_value(self, values):
		current_values = self.get_value()
		for i, (v, c) in enumerate(zip(values, current_values)):
			if v != c:
				self.widget.set_active(i)

class TextBox(WidgetBase):
	def __init__(self, fig, ax_rect, label='textbox', initial='', **kwargs):
		super().__init__(fig, ax_rect)
		self.widget = mpl.widgets.TextBox(self.ax, label=label, initial=initial, **kwargs)
		
	def get_value(self):
		return self.widget.text#.get_text()
	
	def on_submit(self, acallable : Callable[[str],bool]):
		def callback(text):
			if acallable(text.get_text()):
				self.ax.figure.canvas.draw()
		return self.widget.on_submit(callback)


class Range(Slider):
	def __init__(self, fig, ax_rect, label='range', min=-1E300, max=1E300, orientation='vertical', step=None, **kwargs):
		self.ax = fig.add_axes(ax_rect)
		self.widget = mpl.widgets.RangeSlider(self.ax, label, valmin=min, valmax=max, orientation=orientation, valstep=step, **kwargs)
		
		self.set_labels_for_orientation(fig, orientation)
			
	
	def set_limits(self, min = None, max = None, sticky_values=[False,False]):
		if min is None: min = self.widget.valmin
		if max is None: max = self.widget.valmax
		assert min < max, f'{self.__class__.__name__}.set_limits(...), min must be <= max, currently {min=} {max=}'
		self.widget.valmin = min
		self.widget.valmax = max
		minmax = (min,max)
		
		value = list(self.get_value())
		for i in range(len(value)):
			if sticky_values[i]:
				if value[i] < min:
					value[i] = min
				elif value[i] > max:
					value[i] = max
			else:
				value[i] = minmax[i]
		
		self.set_value(value)
		
		match self.widget.orientation:
			case 'vertical':
				self.ax.set_ylim(min,max)
			case 'horizontal':
				self.ax.set_xlim(min,max)
			case _:
				raise RuntimeError(f'Slider orientation "{self.widget.orientation}" is unknown, allowed values "vertical" or "horizontal"')


class RadioButtons(WidgetBase):
	def __init__(self, fig, ax_rect, labels, title='', **kwargs):
		super().__init__(fig, ax_rect)
		self.labels = labels
		self.widget = mpl.widgets.RadioButtons(self.ax, self.labels, **kwargs)
		self.title_text = self.ax.text(0.1,1.05,title, transform=self.ax.transAxes, va='top', ha='left')
		self.ax.axis('off')
		
	def set_value(self, idx):
		self.widget.set_active(idx)

	def get_value(self):
		return self.widget.active
	
	def on_clicked(self, callback : Callable[[int],bool]):
		def dispatch_click_event(selected_label):
			if callback(selected_label):
				self.ax.figure.canvas.draw()
		self.widget.on_clicked(dispatch_click_event)


class Button(WidgetBase):
	def __init__(self, fig, ax_rect, label, **kwargs):
		super().__init__(fig, ax_rect)
		self.widget = mpl.widgets.Button(self.ax, label, **kwargs)
	
	def on_clicked(self, callback : Callable[[],bool]):
		def dispatch_click_event(selected_label):
			if callback(selected_label):
				self.ax.figure.canvas.draw()
		self.widget.on_clicked(dispatch_click_event)




class BasePanel:
	def __init__(self, 
			parent_figure : None | mpl.figure.Figure | mpl.figure.SubFigure = None, 
			pf_gridspec : None | mpl.gridspec.GridSpec = None, 
			gridspec_index : int = 0, 
			window_title : None | str = None
		):
		# Information about how to draw the panel in the containing figure
		self.parent_figure = plt.figure(figsize=(12,8)) if parent_figure is None else parent_figure
		self.pf_gridspec = self.parent_figure.add_gridspec(1,1,left=0,right=0,top=1,bottom=1,wspace=0,hspace=0) if pf_gridspec is None else pf_gridspec
		self.window_title=window_title
		
		self.set_window_title()
		
		# Figure we are going to use to draw the panel
		self.figure = self.parent_figure.add_subfigure(self.pf_gridspec[gridspec_index], frameon=False)
		
		self.create_main_axes()
		self.create_main_handles()
	
	def create_main_axes(self):
		# Get the axes for the plot
		# (left, bottom, width, height) in fraction of figure area
		self.main_axes_rect = (0.2, 0.2, 0.7, 0.7)
		self.main_axes = self.figure.add_axes(self.main_axes_rect, projection=None, polar=False)
	
	def create_main_handles(self):
		self.main_data_handle = None
		self.main_plot_handle = None

	def set_window_title(self):
		if self.window_title is not None:
			self.figure.canvas.manager.set_window_title(self.window_title)
	
	def set_data(self, new_data):
		NotImplemented
	
	def show(self, data : None | Any = None, title : None | str = None):
		plt.figure(self.parent_figure)
		
		self.set_data(data)
		self.main_axes.set_title(title)
		
		plt.show()
	


class ImageViewer(CallbackMixin):
	def __init__(self, parent_figure=None, pf_gridspec = None, gridspec_index = 0, window_title='Image Viewer'):
		
		self.parent_figure = plt.figure(figsize=(12,8)) if parent_figure is None else parent_figure
		self.pf_gridspec = self.parent_figure.add_gridspec(1,1,left=0,right=0,top=1,bottom=1,wspace=0,hspace=0) if pf_gridspec is None else pf_gridspec
		self.figure = self.parent_figure.add_subfigure(self.pf_gridspec[gridspec_index], frameon=False)
		
		self.window_title=window_title
		self.set_window_title()
		
		self.main_axes_rect = (0.2, 0.2, 0.7, 0.7) # (left, bottom, width, height) in fraction of figure area
		self.main_axes = self.figure.add_axes(self.main_axes_rect, projection=None, polar=False)
		self.main_axes_image_data = None
		self.main_axes_im = ImagePlot(self.main_axes)
		
		self.image_plane_slider = Slider(self.figure, (0.02, 0.2, 0.02, 0.7), step=1, label='image plane', active=False)
		self.image_plane_slider.on_changed(self.set_image_plane)
		
		self.main_axes_visibility_controls = CheckButtons(self.figure, (0.08,0.8,0.06,0.1), ['xaxis','yaxis'], actives=[True,True])
		self.main_axes_visibility_controls.on_clicked(
			{	'xaxis' : lambda x: (self.main_axes.xaxis.set_visible(x),True)[1],
				'yaxis': lambda x: (self.main_axes.yaxis.set_visible(x), True)[1]
			}
		)
		self.main_axes_visibility_controls.set_value([False,False])
		
		self.image_clim_slider = Range(self.figure, (0.05, 0.2, 0.02, 0.7), label='clims')
		self.image_clim_slider.on_changed(lambda x: self.main_axes_im.set_clim(*x))
		self.image_clim_slider_modes = (
			'displayed data limits',
			'all data limits'
		)
		self.image_clim_slider_mode_callback_id = None
		
		self.image_clim_slider_mode_selector = RadioButtons(self.figure, (0.06,0.65,0.2,0.1), self.image_clim_slider_modes, title='clim slider mode')
		self.set_clim_slider_mode(self.image_clim_slider_modes[0])
		self.image_clim_slider_mode_selector.on_clicked(self.set_clim_slider_mode)
	
	def set_window_title(self):
		if self.window_title is not None:
			self.figure.canvas.manager.set_window_title(self.window_title)
	
	def set_clim_slider_mode(self, mode):
		if self.image_clim_slider_mode_callback_id is not None:
			self.main_axes_im.disconnect_callback_idx('on_set_data', self.image_clim_slider_mode_callback_id)
		
		match self.image_clim_slider_modes.index(mode):
			case 0:
				# Have clim slider always be min-max of new data plane
				self.image_clim_slider_mode_callback_id = self.main_axes_im.attach_callback(
					'on_set_data', 
					lambda other: self.image_clim_slider.set_limits(np.nanmin(other.data), np.nanmax(other.data))
				)
				if self.main_axes_image_data is not None:
					self.image_clim_slider.set_limits(
						np.nanmin(self.get_displayed_data()), 
						np.nanmax(self.get_displayed_data())
					)
			
			case 1:
				# Have clim slider holder the full range of data, and be min-max of new data plane when selected
				self.image_clim_slider_mode_callback_id = self.main_axes_im.attach_callback(
					'on_set_data', 
					lambda other: self.image_clim_slider.set_value((np.nanmin(other.data), np.nanmax(other.data)))
				)
				if self.main_axes_image_data is not None:
					self.image_clim_slider.set_limits(
						np.nanmin(self.main_axes_image_data), 
						np.nanmax(self.main_axes_image_data)
					)
			
			case _:
				ValueError(f'Unrecognised slider mode "{mode}", must be one of {self.image_clim_slider_modes}')
			
		if self.main_axes_image_data is not None:
			self.image_clim_slider.set_value(
				(np.nanmin(self.get_displayed_data()), np.nanmax(self.get_displayed_data()))
			)
		return True
		
	
	def get_displayed_data(self):
		return self.main_axes_image_data[self.image_plane_slider.get_value()] if self.image_plane_slider.get_active() else self.main_axes_image_data
	
	def get_data(self):
		return self.main_axes_image_data
	
	def set_data(self, data : np.ndarray = None):
		self.main_axes_image_data = data
		if self.main_axes_image_data is None: return
		
		
		if self.main_axes_image_data.ndim == 2:
			self.image_plane_slider.set_active(False)
		
			
		elif self.main_axes_image_data.ndim == 3:
			_lgr.debug(f'{self.main_axes_image_data.shape[0]=}')
			self.image_plane_slider.set_active(True)
			self.image_plane_slider.set_limits(0,self.main_axes_image_data.shape[0]-1)
		
		self.main_axes_im.set_data(self.get_displayed_data())
			
	def set_image_plane(self, x):
		self.main_axes_im.set_data(self.main_axes_image_data[int(x)])
		return True

	def show(self, data=None, title=None):
		plt.figure(self.parent_figure)
		
		self.set_data(data)
		if title is not None:
			self.main_axes.set_title(title)
		
		plt.show()



class ResidualViewer(ImageViewer):
	def __init__(self, parent_figure=None, pf_gridspec = None, gridspec_index = 0, window_title='Image Viewer'):
		super().__init__(parent_figure, pf_gridspec, gridspec_index, window_title)
		self.frozen_clims = (0,1)
	
	
	def set_clim_slider_mode(self, mode):
		if self.image_clim_slider_mode_callback_id is not None:
			self.main_axes_im.disconnect_callback_idx('on_set_data', self.image_clim_slider_mode_callback_id)
		
		match self.image_clim_slider_modes.index(mode):
			case 0:
				# Have clim slider always be min-max of new data plane
				self.image_clim_slider_mode_callback_id = self.main_axes_im.attach_callback(
					'on_set_data', 
					lambda other: self.image_clim_slider.set_limits(np.nanmin(other.data), np.nanmax(other.data))
				)
				if self.main_axes_image_data is not None:
					self.image_clim_slider.set_limits(
						np.nanmin(self.get_displayed_data()), 
						np.nanmax(self.get_displayed_data())
					)
			
			case 1:
				# Have clim slider hold the frozen limits but the image values value
				self.image_clim_slider_mode_callback_id = self.main_axes_im.attach_callback(
					'on_set_data', 
					lambda other: self.image_clim_slider.set_value((np.nanmin(other.data), np.nanmax(other.data)))
				)
				if self.main_axes_image_data is not None:
					self.image_clim_slider.set_limits(*self.frozen_clims)
			
			case _:
				ValueError(f'Unrecognised slider mode "{mode}", must be one of {self.image_clim_slider_modes}')
			
		if self.main_axes_image_data is not None:
			self.image_clim_slider.set_value(
				(np.nanmin(self.get_displayed_data()), np.nanmax(self.get_displayed_data()))
			)
		return True
	

class ImageAggregator(ImageViewer, CallbackMixin):
	def __init__(self, parent_figure=None, pf_gridspec = None, gridspec_index = 0, window_title='Image Aggregator'):
		CallbackMixin.__init__(self, 'on_set_image_agg_planes')
		ImageViewer.__init__(self, parent_figure, pf_gridspec , gridspec_index , window_title)
		
		self.image_plane_slider.remove()
		self.image_plane_slider = Range(self.figure, (0.02, 0.2, 0.02, 0.7), step=1, label='aggregate image planes')
		self.image_plane_slider.on_changed(self.set_image_agg_planes)
		
		
	
	def get_displayed_data(self):
		x = self.image_plane_slider.get_value()
		return np.nansum(self.main_axes_image_data[int(x[0]):int(x[1])], axis=0) if self.image_plane_slider.get_active() else self.main_axes_image_data
	
	
	def set_image_agg_planes(self, x):
		self.main_axes_im.set_data(np.nansum(self.main_axes_image_data[int(x[0]):int(x[1])], axis=0))
		self.call_callbacks('on_set_image_agg_planes', self)
		return True
	
	def set_data(self, data : np.ndarray = None):
		super().set_data(data)
		_lgr.debug(f'{data.shape=}')
		self.image_plane_slider.set_limits(0,data.shape[0])
		self.image_plane_slider.set_value((0,data.shape[0]))
		

class LineRegionPanel(BasePanel, CallbackMixin):
	class RemoveRegionException(Exception):
		pass

	def __init__(self, parent_figure=None, pf_gridspec = None, gridspec_index = 0, window_title='Line Region', **kwargs):
		CallbackMixin.__init__(self, 'on_region_changed')
		super().__init__(parent_figure, pf_gridspec, gridspec_index, window_title)
		
		self.main_plot_handle = LinePlot(self.main_axes, **kwargs)
		self.main_data_handle = None
		
		self.shown_span_hdls = None
		self.span_selector_wdgt = mpl.widgets.SpanSelector(
			self.main_axes,
			direction='horizontal',
			onselect = self.region_changed
		)
		self.set_region_limits()
	
	def set_data(self, new_data):
		self.main_data_handle = new_data
		self.main_plot_handle.set_data(self.main_data_handle)
	
	def set_region_limits(self, 
			min_value : None | float | int = None, 
			max_value : None | float | int = None, 
			snap_step_value : None | float | int = None, 
			min_size : None | float | int = None, 
			max_size : None | float | int = None,
			snap_action = 'round', 
			below_min_action='set', 
			above_max_action='set',
			region_too_small_action = 'remove_region',
			region_too_large_action = 'remove_region'
		):
		self.r_min_value = min_value
		self.r_max_value = max_value
		self.r_snap_step_value = snap_step_value
		self.r_min_size = min_size
		self.r_max_size = max_size
		self.r_snap_action = snap_action
		self.r_below_min_action = below_min_action
		self.r_above_max_action = above_max_action
		self.r_region_too_small_action = region_too_small_action
		self.r_region_too_large_action = region_too_large_action
	
	
	
	
	def apply_region_action(self, value: int | float, limit : int | float, action : str) -> int | float | None:
		match action:
			case 'round':
				remainder = value % limit
				return value - remainder if (remainder < 1/2*limit) else value - remainder + limit
			case 'floor':
				return value - value % limit
			case 'ceil':
				return (value + limit) % limit
			case 'set':
				return limit
			case 'remove_region':
				raise LineRegionPanel.RemoveRegionException
	
	def apply_region_limits(self, r_min, r_max):
		# Apply min/max values
		if self.r_min_value is not None:
			if r_min < self.r_min_value:
				r_min = self.apply_region_action(r_min, self.r_min_value, self.r_below_min_action)
			if r_max < self.r_min_value:
				r_max = self.apply_region_action(r_max, self.r_min_value, self.r_below_min_action)
		
		if self.r_max_value is not None:
			if r_min > self.r_max_value:
				r_min = self.apply_region_action(r_min, self.r_max_value, self.r_below_max_action)
			if r_max > self.r_max_value:
				r_max = self.apply_region_action(r_max, self.r_max_value, self.r_below_max_action)
		
		# Apply snap to step
		if self.r_snap_step_value is not None:
			r_min = self.apply_region_action(r_min, self.r_snap_step_value, self.r_snap_action)
			r_max = self.apply_region_action(r_max, self.r_snap_step_value, self.r_snap_action)
		
		if self.r_min_size is not None and (r_max - r_min) < self.r_min_size:
			region_size = self.apply_region_action((r_max - r_min), self.r_min_size, self.r_region_too_small_action)
			r_max = r_min + region_size
		
		if self.r_max_size is not None and (r_max - r_min) > self.r_max_size:
			region_size = self.apply_region_action((r_max - r_min), self.r_max_size, self.r_region_too_large_action)
			r_max = r_min + region_size
		return r_min, r_max
		
		
	
	def region_changed(self, r_min : float, r_max : float) -> bool:
		try:
			r_min, r_max = self.apply_region_limits(r_min, r_max)
		except LineRegionPanel.RemoveRegionException:
			self.remove_region()
			self.call_callbacks('on_region_changed', None, None)
			return
			
		self.set_region(r_min, r_max)
		self.call_callbacks('on_region_changed', r_min, r_max)
	
	def remove_region(self):
		if self.shown_span_hdls is not None:
			for h in self.shown_span_hdls:
				h.remove()
			self.shown_span_hdls = None
	
	def set_region(self, r_min, r_max):
		self.remove_region()
		self.shown_span_hdls = [
			self.main_axes.axvspan(r_min, r_max, alpha=0.3, color='tab:red'),
			self.main_axes.axvline(r_min, alpha=0.5, color='tab:red', ls='-'),
			self.main_axes.axvline(r_max, alpha=0.5, color='tab:red', ls='--')
		]
	

class SSAViewer:
	"""
	Graphical front-end to the singular spectrum analysis routines in the `aopp_deconv_tool.py_ssa` module.
	Shows the input data, the eigenvalues of the SSA components, an aggregate of SSA components, and a residual
	between the aggregate and the input data.
	
	Accepts numpy arrays to the `SSAViewer.show(image:np.ndarray)` method. Ouputs numpy arrays (`.npy` file)
	or TIFF (`.tif` file) images.
	"""
	def __init__(self, 
			parent_figure : mpl.figure.Figure | None = None, 
			pf_gridspec : mpl.gridspec.GridSpec | None = None, 
			gridspec_index : int = 0, 
			window_title :str = 'Image Viewer'
		):
		
		
		self.parent_figure = plt.figure(figsize=(15,10), layout='constrained') if parent_figure is None else parent_figure
		self.pf_gridspec = self.parent_figure.add_gridspec(1,1,left=0,right=0,top=1,bottom=1,wspace=0,hspace=0) if pf_gridspec is None else pf_gridspec
		self.figure = self.parent_figure.add_subfigure(self.pf_gridspec[gridspec_index], frameon=False)
		self.set_window_title(window_title)
		
		self.f_gridspec = self.figure.add_gridspec(2,2,left=0.4,right=0.6,top=0.7,bottom=0.3,wspace=0,hspace=0)
		
		
		self.input_image_viewer = ImageViewer(self.figure, pf_gridspec=self.f_gridspec, gridspec_index=0, window_title=None)
		self.ssa_evalue_panel = LineRegionPanel(self.figure, self.f_gridspec, 1, None)
		self.ssa_aggregate_viewer = ImageAggregator(self.figure, pf_gridspec=self.f_gridspec, gridspec_index=2, window_title=None)
		self.ssa_residual_viewer = ResidualViewer(self.figure, pf_gridspec=self.f_gridspec, gridspec_index=3, window_title=None)
		
		
		self.input_image_viewer.main_axes_im.set_title('Input Data')
		
		
		self.ssa_evalue_panel.main_axes.set_yscale('log')
		self.ssa_evalue_panel.main_axes.set_xlabel('Singular Value Index')
		self.ssa_evalue_panel.main_axes.set_ylabel('Singular Value')
		self.ssa_evalue_panel.main_plot_handle.set_title('SSA Singular Values of Components')
		self.ssa_evalue_panel.set_region_limits(0,None,1,1,None)
		self.ssa_evalue_panel.attach_callback('on_region_changed',
			self.set_image_plane_range_from_evalue_panel_region
		)
		
		self.ssa_aggregate_viewer.main_axes_im.set_title('Sum of SSA Components')
		self.ssa_aggregate_viewer.attach_callback('on_set_image_agg_planes',
			self.set_image_plane_range_from_aggregator
		)
		
		self.ssa_residual_viewer.main_axes_im.set_title('Residual (Input Data - Sum of SSA Components)')
		
		
		self.save_button = Button(self.figure, (0.05, 0.001, 0.15, 0.03), 'Save Aggregated SSA Components')
		self.save_button.on_clicked(self.save_aggregated_ssa_components)		
		self.save_path_text = TextBox(self.figure, (0.05, 0.031, 0.3, 0.02), label='path')
		self.save_checks = CheckButtons(self.figure, (0.2,0.001, 0.15,0.03), ['overwrite', 'create parent folders'])
		
		
		
		self.save_feedback_text = self.figure.text(0.05, 0.052, 'Formats: ".npy", ".tiff"')
		
		
		
		
		self.input_data = None
		self.ssa_data = None


	def save_aggregated_ssa_components(self, event):
		save_path = Path(self.save_path_text.get_value())
		_lgr.debug(f'{save_path=}')
		can_overwrite, can_create_parents = self.save_checks.get_value()
		
		success_flag = False
		feedback = [f'Saving to "{save_path.absolute()}"']
		fail_feedback = []
		success_feedback = []
		
		_lgr.debug(f'Collecting failure feedback')
		if save_path.exists():
			fail_feedback.append('DESTINATION ALREADY EXISTS')
			if save_path.is_dir():
				fail_feedback.append('AND IS A DIRECTORY') 
			elif save_path.is_file():
				fail_feedback.append('AND IS A FILE')
		elif not save_path.parent.exists():
			fail_feedback.append('PARENT DIRECTORY NOT PRESENT')
		
		ext = save_path.suffix
		
		
		if (not save_path.exists() or (can_overwrite and not save_path.is_dir())) and (save_path.parent.exists() or can_create_parents):
			_lgr.debug(f'Starting save attempt...')
			try:
				if not save_path.parent.exists() and can_create_parents:
					save_path.parent.mkdir(parents=True, exist_ok=True)
				
				data = self.ssa_aggregate_viewer.get_displayed_data()
				match ext:
					case '.npy':
						np.save(save_path, data)
					case '.tiff':
						PIL.Image.fromarray(data).save(save_path)
					case _:
						raise RuntimeError(f'Unknown extension "{ext}"')
			except Exception as e:
				fail_feedback.append(str(e))
				success_flag=False
			else:
				success_feedback.append('successful')
				success_flag = True
			_lgr.debug(f'Completed save attempt.')
		
		_lgr.debug(f'Writing feedback...')
		if success_flag:
			self.save_feedback_text.set_text(' '.join(feedback+success_feedback))
			self.save_feedback_text.set_color('green')
		else:
			self.save_feedback_text.set_text(' '.join(feedback+fail_feedback))
			self.save_feedback_text.set_color('red')
		_lgr.debug(f'Feedback written.')
		return True

	def set_image_plane_range_from_evalue_panel_region(self, r_min : int, r_max : int):
		if r_min is None:
			r_min = 0
		if r_max is None:
			r_max = len(self.ssa_evalue_panel.main_data_handle)
		self.ssa_aggregate_viewer.image_plane_slider.set_value((r_min,r_max))
		self.ssa_aggregate_viewer.main_axes_im.set_data(np.nansum(self.ssa_aggregate_viewer.main_axes_image_data[int(r_min):int(r_max)], axis=0))
		self.set_residual_data(self.input_image_viewer.get_displayed_data(),self.ssa_aggregate_viewer.get_displayed_data())

	def set_image_plane_range_from_aggregator(self, agg : tuple[int,int]):
		x = agg.image_plane_slider.get_value()
		self.set_residual_data(self.input_image_viewer.get_displayed_data(),agg.get_displayed_data())
		self.ssa_evalue_panel.set_region(x[0],x[1]+1)

	def set_residual_data(self, obs_data : np.ndarray, model_data : np.ndarray):
		residual = obs_data - model_data
		self.ssa_residual_viewer.frozen_clims = (
			min(np.nanmin(obs_data), np.nanmin(residual)),
			max(np.nanmax(obs_data), np.nanmin(residual))
		)
		self.ssa_residual_viewer.set_data(residual)
		

	def set_window_title(self, title : str):
		if title is not None:
			self.figure.canvas.manager.set_window_title(title)

	def show(self, data : np.ndarray = None, title : str = None):
		"""
		Show the SSA decomposition of `data` with `title`
		"""
		plt.figure(self.parent_figure)
		
		self.set_data(data)
		
		plt.show()
	
	def set_data(self, data : np.ndarray):
		self.input_data = data
		
		self.input_image_viewer.set_data(self.input_data)
		
		self.ssa = SSA(
			self.input_image_viewer.get_displayed_data(),
			(5,5),
			rev_mapping='fft',
			svd_strategy='eigval', # uses less memory and is faster
			#grouping={'mode':'elementary'}
			grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
		)
		_lgr.debug(f'{self.ssa.X_ssa.shape=} {self.ssa.m=} {np.diag(self.ssa.s_g)=}')
		self.ssa_evalue_panel.set_data(np.diag(self.ssa.s_g))
		self.ssa_aggregate_viewer.set_data(self.ssa.X_ssa)
		self.ssa_residual_viewer.set_data(self.input_image_viewer.get_displayed_data() - self.ssa_aggregate_viewer.get_displayed_data())

if __name__ == '__main__':
	#image = sys.argv[1]
	
	if len(sys.argv) < 2:
		image = np.sqrt(np.sum(((np.indices((20, 100, 128)).T - np.array([10,50,64])).T)**2, axis=0))
		for i in range(image.shape[0]):
			image[i,...] = image[i]**(2*((i+1)/image.shape[0]))
	else:
		image = np.array(PIL.Image.open(sys.argv[1]))
		
		# Can only use greyscale images, so sum along colour axis
		if image.ndim == 3:
			image = np.sum(image, axis=-1)
	
	
	
	
	imviewer = SSAViewer()
	imviewer.show(image, 'image')