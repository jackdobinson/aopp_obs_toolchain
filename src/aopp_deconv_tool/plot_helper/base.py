"""
Defines the public facing interface for plot classes, used to interface with hooks in aopp_deconv_tool.algorithm.deconv classes
"""
import dataclasses as dc
from collections import namedtuple
from typing import Any, Callable, ClassVar
from functools import wraps

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import aopp_deconv_tool.plot_helper as plot_helper
#AxisDataMapping = namedtuple('AxisDataMapping', ('label','attribute','limit_getter'))

@dc.dataclass(repr=False, eq=False, slots=True)
class AxisDataMapping:
	"""
	Defines the relationship between a "datasource" attribute and the axis that it is plotted along
	"""
	label : str
	attribute : str | None = None
	limit_getter : Callable[[Any],tuple[float,float]] = plot_helper.lim
	
	def get_limits(self, d):
		return self.limit_getter(d)

@dc.dataclass(repr=False, eq=False, slots=True)
class Base:
	
	# independent and dependent variables in the format
	# {'label1':'attribute1', 'label2':'attribute2',...}
	# a specific plot type has predefined labels and attributes, these can be
	# overwritten in the __init__ of the class
	title : str = 'Base Plot Helper Title'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_labels : tuple[str,...] = tuple()
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda :
		tuple() # (('ax1_label','ax1_attr'), ('ax2_label','ax2_attr'), ...)
	)
	
	static_frame : bool = True
	plt_kwargs : dict = dc.field(default_factory=dict)
	ax_funcs : list = dc.field(default_factory=list)
	show_limits_in_title : dict = dc.field(default_factory=dict) #{0:False, 1:False, 2:False}, 0,1,2 are axis numbers (x,y,z)
	
	
	# Internal attributes
	# Will be assigned after initialisation
	datasource_name : str = dc.field(default='datasource name', init=False)
	ax : mpl.axes.Axes = dc.field(default=None, init=False) # matplotlib axis
	datasource : Any = dc.field(default = None, init=False)
	datasource_data_getter : Callable | None = dc.field(default=None, init=False)
	
	hdl : Any = dc.field(default=None,init=False) # handle to visualisation of the plot
	n_updates : int = dc.field(default=0, init=False)

	
	def get_axis_limits(self,ax_num):
		match ax_num:
			case 0:
				return self.ax.get_xlim() if self.ax is not None else None
			case 1:
				return self.ax.get_ylim() if self.ax is not None else None
			case 2:
				return self.hdl.get_clim() if self.hdl is not None else None
			case _:
				raise RuntimeError(f'Plot {self.title=} does not have an axis of number {ax_num}')
	
	def get_axis_limits_str(self, ax_num):
		al = self.get_axis_limits(ax_num)
		if al is None:
			return 'None'
		return "{0:5.3g},{1:5.3g}".format(*al)
	
	def set_ax_title(self):
		if self.title is not None:
			axis_limit_strs = []
			for ax_num, show_limits_flag in self.show_limits_in_title.items():
				if show_limits_flag:
					axis_limit_strs.append(f'{self.axis_data_mappings[ax_num].label}:[{self.get_axis_limits_str(ax_num)}]')
			return self.ax.set_title(self.title
				+ (('\n' + ' '.join(axis_limit_strs)) if len(axis_limit_strs)>0 else '')
			)
		return None
	
	def is_datasource_attached(self):
		return hasattr(self, 'datasource') and self.datasource is not None
	
	def on_attach_datasource(self):
		"""
		Performs any initial setup that is required when a datasource is attached.
		Should be overwritten by subclass.
		"""
		self.n_updates=0
		if not self.is_datasource_attached():
			raise RuntimeError(f'{self} does not have a "datasource" attribute')
	
	def attach_datasource(self, datasource, datasource_data_getter):
		self.datasource = datasource
		self.datasource_data_getter = datasource_data_getter
		self.on_attach_datasource()
		return(self)
	
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
		self.set_ax_title()
		
		for i, (set_axis_label, adm) in enumerate(zip((
					self.ax.set_xlabel, 
					self.ax.set_ylabel
				), 
				self.axis_data_mappings
			)):
			if adm.label is not None:
				if len(self.axis_labels) > i and self.axis_labels[i] is not None:
					set_axis_label(self.axis_labels[i])
				else:
					set_axis_label(adm.label)
		for ax_func in self.ax_funcs:
			ax_func(self.ax)
		self.on_attach_ax()
		return(self)
	
	
	def attach(self, ax : mpl.axes.Axes, datasource : Any, datasource_data_getter : Callable[[Any],tuple[Any]] ):
		"""
		Attach a `datasource` (input) and an axes `ax` (output) to the plot.
		"""
		self.attach_datasource(datasource, datasource_data_getter)
		self.attach_ax(ax)
		return(self)

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
				self.axis_data_mappings
			):
			if adm.attribute is not None:
				d = getattr(self, adm.attribute)
				set_lims(*adm.get_limits(d))
		
		if not self.static_frame:
			self.set_ax_title()
			if not self.ax.drawn:
				for x in self.ax.get_children():
					self.ax.draw_artist(x)
				self.ax.drawn = True
					
		self.ax.draw_artist(self.hdl)
		
	def update(self):
		assert self.is_datasource_attached(), "requires datasource attribute is not None"
		
		self.update_plot_data(self.datasource_data_getter(self.datasource))
		self.update_plot_visual()
		self.n_updates += 1
	
	def iter_ax(self, attrs):
		assert self.is_ax_attached(), "require ax attribute"
		for attr in attrs:
			yield getattr(self.ax, attr)
	
	
	


			
