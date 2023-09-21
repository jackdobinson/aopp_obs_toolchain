"""
Defines the public facing interface for plot classes
"""
import dataclasses as dc
from collections import namedtuple
from typing import Any, Callable
from functools import wraps

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


AxisDataMapping = namedtuple('AxisDataMapping', ('label','attribute'))



class Base:
	
	# independent and dependent variables in the format
	# {'label1':'attribute1', 'label2':'attribute2',...}
	# a specific plot type has predefined labels and attributes, these can be
	# overwritten in the __init__ of the class
	title : str = 'Base Plot Helper Title'
	datasource_name : str = 'datasource name'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_data_mappings : tuple[AxisDataMapping] = tuple() # (('ax1_label','ax1_attr'), ('ax2_label','ax2_attr'), ...)
	ax : mpl.axes.Axes = None # matplotlib axis
	hdl : Any = None # handle to visualisation of the plot
	
	
	def __init__(self, ax=None, datasource=None, datasource_data_getter=None, static_frame=True, **plt_kwargs) -> None:
		"""
		Plot initialistion happens here.
		
		Arguments:
			**kwargs
				dictionary containing any overwrites to class parameters
		"""
		self.plt_kwargs = plt_kwargs
		self.static_frame = static_frame # Does the frame of the plot (including axis labels) change?
		
		self.datasource = None
		self.datasource_data_getter = None
		self.ax = None
		if datasource is not None and datasource_data_getter is not None:
			self.attach_datasource(datasource, datasource_data_getter)
			if ax is not None:
				self.attach_ax(ax)
		
		
		self.n_updates = 0
		
	
	
	def is_datasource_attached(self):
		return hasattr(self, 'datasource') and self.datasource is not None
	
	def on_attach_datasource(self):
		"""
		Performs any initial setup that is required when a datasource is attached.
		Should be overwritten by subclass.
		"""
		if not self.is_datasource_attached():
			raise RuntimeError(f'{self} does not have a "datasource" attribute')
	
	def attach_datasource(self, datasource, datasource_data_getter):
		self.datasource = datasource
		self.datasource_data_getter = datasource_data_getter
		self.on_attach_datasource()
	
	def is_ax_attached(self):
		return hasattr(self, 'ax') and self.ax is not None
	
	def on_attach_ax(self):
		"""
		Performs any initial setup that is required when an axes is attached.
		Should be overwritten by subclass
		"""
		if not self.is_ax_attached():
			raise RuntimeError(f'{self} does not have a "axes" attribute')
		
	
	def attach_ax(self, ax):
		self.ax = ax
		if self.title is not None:
			self.ax.set_title(self.title)
		for set_axis_label, adm in zip((
					self.ax.set_xlabel, 
					self.ax.set_ylabel
				), 
				self.axis_data_mappings
			):
			if adm.label is not None:
				set_axis_label(adm.label)
		self.on_attach_ax()
	
	
	def attach(self, ax : mpl.axes.Axes, datasource : Any, datasource_data_getter : Callable[[Any],tuple[Any]] ):
		"""
		Attach a `datasource` (input) and an axes `ax` (output) to the plot.
		"""
		self.attach_datasource(datasource, datasource_data_getter)
		self.attach_ax(ax)

	def detach_datasource(self):
		self.datasource = None
	
	def detach_ax(self):
		self.ax = None
		
	def detach(self):
		self.detatch_datasource()
		self.detatch_ax()

	def update_plot_data(self, data):
		raise NotImplementedError
	
	
	def update_plot_visual(self):
		for set_lims, adm in zip((
					self.ax.set_xlim,
					self.ax.set_ylim,
					self.hdl.set_clim if hasattr(self.hdl,'set_clim') else lambda *a, **k: None
				), 
				self.axes_data_mappings
			):
			if adm.attribute is not None:
				d = getattr(self, adm.attribute)
				set_lims(np.min(d),np.max(d))
		
		
		if not self.static_frame and not self.ax.drawn:
			for x in self.ax.get_children():
				self.ax.draw_artist(x)
			self.ax.drawn = True
		self.ax.draw_artist(self.hdl)
		#plt.pause(0.001)
		
	def update(self):
		assert self.is_datasource_attached(), "requires datasource attribute"
		
		self.update_plot_data(self.datasource_data_getter(self.datasource))
		self.update_plot_visual()
		self.n_updates += 1
	
	def iter_ax(self, attrs):
		assert self.is_ax_attached(), "require ax attribute"
		for attr in attrs:
			yield getattr(self.ax, attr)
	
	
	


			
