"""
Graphical front end to SSA filtering of an image, should be usable for a cube and a 2d image
"""

import sys, os
import pathlib
from typing import Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.widgets
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.patches


from py_ssa import SSA

from plot_helper import fig_draw_bbox_of_artist

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


default_scalar_formatter = mpl.ticker.ScalarFormatter().format_data


class ImagePlot:
	def __init__(self, ax, **kwargs):
		self.ax = ax
		self.data_shape = (0,0)
		self.kwargs = kwargs
		self.im = None
		self.callbacks = {}
	
	def set_clim(self, min=None, max=None):
		
		if min is None or max is None:
			image_data = self.im.get_array()
			if min is None: min = np.nanmin(image_data)
			if max is None: max = np.nanmax(image_data)
		self.im.set(clim=(min,max))
		self.call_callbacks('on_set_clim', min, max)
		
	
	def set_axes_limits(self, xlim=(None,None), ylim=(None,None)):
		self.ax.set_xlim(*xlim)
		self.ax.set_ylim(*ylim)
	
	def set_data(self, data):
		if self.im is None or any(s1 != s2 for s1,s2 in zip(self.data_shape,data.shape)):
			if self.im is not None: 
				self.im.remove()
			self.im = self.ax.imshow(data, **self.kwargs)
			self.data_shape = data.shape
	
		self.im.set_data(data)
		self.set_axes_limits()
		self.set_clim()
		self.call_callbacks('on_set_data', data)
		
	def attach_callback(self, callback_set, acallable : Callable[[float,float],None]):
		possible_callback_sets = (
			'on_set_clim',
			'on_set_data'
		)
		if callback_set not in self.callbacks:
			if callback_set in possible_callback_sets:
				self.callbacks[callback_set] = []
			else:
				raise ValueError(f'callback set name "{callback_set}" not permitted, must be one of {possible_callback_sets}')
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
			acallback(*args, **kwargs)


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
		return self.widget.text.get_text()
	
	def on_submit(self, acallable : Callable[[str],bool]):
		def callback(text):
			if acallable(text.get_text()):
				self.ax.figure.canvas.draw()
		return self.widget.on_submit(callback)


class Range(Slider):
	def __init__(self, fig, ax_rect, label='range', min=-1E300, max=1E300, orientation='vertical', **kwargs):
		self.ax = fig.add_axes(ax_rect)
		self.widget = mpl.widgets.RangeSlider(self.ax, label, valmin=min, valmax=max, orientation=orientation, **kwargs)
		
		self.set_labels_for_orientation(fig, orientation)
			
	
	def set_limits(self, min = None, max = None, sticky_values=[False,False]):
		if min is None: min = self.widget.valmin
		if max is None: max = self.widget.valmax
		assert min <= max, f'{self.__class__.__name__}.set_limits(...), min must be <= max, currently {min=} {max=}'
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



class ImageViewer:
	def __init__(self, parent_figure=None, pf_gridspec = None, gridspec_index = 0, window_title='Image Viewer'):
		self.parent_figure = plt.figure(figsize=(12,8)) if parent_figure is None else parent_figure
		self.pf_gridspec = self.parent_figure.add_gridspec(1,1,left=0,right=0,top=1,bottom=1,wspace=0,hspace=0) if pf_gridspec is None else pf_gridspec
		self.figure = self.parent_figure.add_subfigure(self.pf_gridspec[gridspec_index])
		
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
		self.figure.canvas.manager.set_window_title(self.window_title)
	
	def set_clim_slider_mode(self, mode):
		if self.image_clim_slider_mode_callback_id is not None:
			self.main_axes_im.disconnect_callback_idx('on_set_data', self.image_clim_slider_mode_callback_id)
		
		match self.image_clim_slider_modes.index(mode):
			case 0:
				# Have clim slider always be min-max of new data plane
				self.image_clim_slider_mode_callback_id = self.main_axes_im.attach_callback(
					'on_set_data', 
					lambda x: self.image_clim_slider.set_limits(np.nanmin(x), np.nanmax(x))
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
					lambda x: self.image_clim_slider.set_value((np.nanmin(x), np.nanmax(x)))
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
		
		
		_lgr.debug(f'{self.main_axes_image_data.ndim=}')
		if self.main_axes_image_data.ndim == 2:
			self.image_plane_slider.set_active(False)
		
			
		elif self.main_axes_image_data.ndim == 3:
			_lgr.debug(f'{self.main_axes_image_data.shape[0]=}')
			self.image_plane_slider.set_active(True)
			self.image_plane_slider.set_limits(0,self.main_axes_image_data.shape[0]-1)
		
		self.main_axes_im.set_data(self.main_axes_image_data[self.image_plane_slider.get_value()] if self.image_plane_slider.get_active() else self.main_axes_image_data)
	
	
	def set_image_plane(self, x):
		self.main_axes_im.set_data(self.main_axes_image_data[int(x)])
		return True

	def show(self, data=None, title=None):
		plt.figure(self.parent_figure)
		
		self.set_data(data)
		if title is not None:
			self.main_axes.set_title(title)
		
		plt.show()



class SSAViewer(ImageViewer):
	def __init__(self, *args, **kwargs):
		if 'window_title' not in kwargs: kwargs['window_title'] = 'SSA Viewer'
		super().__init__(*args, **kwargs)



if __name__ == '__main__':
	#image = sys.argv[1]
	
	test_image = np.sqrt(np.sum(((np.indices((20, 100, 128)).T - np.array([10,50,64])).T)**2, axis=0))
	
	for i in range(test_image.shape[0]):
		test_image[i,...] = test_image[i]**(2*((i+1)/test_image.shape[0]))
	
	
	imviewer = SSAViewer()
	imviewer.show(test_image, 'test_image')