"""
Contains classes that assist with plotting histograms.
"""

import sys
from time import sleep
from typing import Any
import dataclasses as dc


import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

import aopp_deconv_tool.plot_helper as plot_helper
from aopp_deconv_tool.plot_helper.base import Base, AxisDataMapping


@dc.dataclass(repr=False, eq=False, slots=True)
class PlotSet:
	"""
	Defines a set of plots that can be animated
	"""
	fig : mpl.figure.Figure
	title : str = 'Plot Set'
	plots : list[Base] = dc.field(default_factory=list)
	blit : bool = True
	cadence : int = 1
	
	# Internal attributes
	n_frames : int = dc.field(default=0, init=False)
	blit_regions : Any = dc.field(default=None, init=False)
	blit_regions_data : Any = dc.field(default=None, init=False)
	title_hdl : Any = dc.field(default=None, init=False)
	title_blit_region : Any = dc.field(default=None, init=False)
	title_blit_region_data : Any = dc.field(default=None, init=False)
		
	def attach_datasource(self, datasource : Any):
		"""
		loop through plots, any with no attached datasource will use this one
		"""
		for p in self.plots:
			if p.is_datasource_attached():
				continue
			p.attach_datasource(datasource, p.datasource_data_getter)
		return self
		
	def show(self):
		self.title_hdl = self.fig.suptitle(self.title.format(self=self))
		self.n_frames = 0
		self.fig.canvas.draw()
		plt.show(block=False)
		
		#if self.blit:
			#plt.pause(0.4)
			#self.blit_regions = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.fig.axes]
			#self.blit_regions = [self.fig.canvas.copy_from_bbox(ax.get_tightbbox()) for ax in self.fig.axes]
	
	
	def get_ax_bb(self, ax):
		return ax.get_tightbbox().expanded(1.2,1.2)
	
	def update(self):
		
		if self.blit:
			# do one full draw then animate with blit
			if self.n_frames == 0:
				
				for ax in self.fig.axes:
					ax.drawn = False
				
				for plot in self.plots:
					plot.update()
				plt.pause(0.1)
					
				# set axes invisible that will change
				self.title_hdl.set_visible(False)
				for p in self.plots:
					p.ax.set_visible(True if p.static_frame else False)
				
					
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				
				# set axes visible again
				self.title_hdl.set_visible(True)
				for p in self.plots:
					p.ax.set_visible(True)
					
				# need axes visible to get their extent
				self.title_blit_region = self.title_hdl.get_window_extent().expanded(1.2,1.2)
				self.blit_regions = tuple(self.get_ax_bb(ax) for ax in self.fig.axes)
				
				# set axes invisible that will change
				self.title_hdl.set_visible(False)
				for p in self.plots:
					p.ax.set_visible(True if p.static_frame else False)
				
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
						# need axes invisible to get the blank background
				self.title_blit_region_data = self.fig.canvas.copy_from_bbox(self.title_blit_region)
				self.blit_regions_data = tuple(self.fig.canvas.copy_from_bbox(r) for r in self.blit_regions)
						
				# set axes visible again
				self.title_hdl.set_visible(True)
				for p in self.plots:
					p.ax.set_visible(True)
				
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				
				
				self.n_frames += 1
				return
				
			if self.n_frames % self.cadence == 0 :	
				# remove last frame plots
				self.fig.canvas.restore_region(self.title_blit_region_data)
				for ax, blit_region_data in zip(self.fig.axes, self.blit_regions_data):
					self.fig.canvas.restore_region(blit_region_data)
					ax.drawn=False
			
				# blit new frame plots
			self.title_hdl.set_text(self.title.format(self=self))
			
			if self.n_frames % self.cadence == 0 :
				self.fig.draw_artist(self.title_hdl)
				self.fig.canvas.blit(self.title_blit_region)
			
			for plot in self.plots:
				plot.update()
				if self.n_frames % self.cadence == 0 :
					self.fig.canvas.blit(self.get_ax_bb(plot.ax))
		else:
			self.title_hdl = self.fig.suptitle(self.title.format(self))
			for plot in self.plots:
				plot.update()
			
			if self.n_frames % self.cadence == 0 :
				self.fig.canvas.draw()
		
		if self.n_frames % self.cadence == 0 :	
			self.fig.canvas.flush_events()
		
	
		self.n_frames += 1


@dc.dataclass(repr=False, eq=False, slots=True)
class Histogram(Base):
	title : str = 'Histogram'
	datasource_name : str = 'histogram datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda :(
		AxisDataMapping('value','bins',limit_getter=plot_helper.LimRememberExtremes()),
		AxisDataMapping('count','_hist',limit_getter=plot_helper.LimRememberExtremes()))
	)
	nbins : int = 100
	
	# Internal
	_hist : np.ndarray = None
	_bins : np.ndarray = None
	
	
	@property
	def bins(self):
		return self._bins
	
	@bins.setter
	def bins(self, value):
		self._bins[:-1] = value
		self._bins[-1] = self._bins[-2] + (self._bins[-2] - self.bins[-3])
	
	@property
	def hist(self):
		return self._hist
	
	@hist.setter
	def hist(self, value):
		self._hist[1:-1] = value
	
	def update_plot_data(self, data):
		self.hist[1:-1], self.bins = np.histogram(data, bins=self.nbins, **self.plt_kwargs)
		self.hdl.set_data(self.bins, self.hist)
	
	def on_attach_datasource(self):
		super(Histogram, self).on_attach_datasource()
		# initialise data holders, need to have zeros at either end due to how `self.ax.step` works
		self._hist = np.zeros((self.nbins+2,))
		self._bins = np.linspace(0,1,self.nbins+2,endpoint=True)

	def on_attach_ax(self):
		super(Histogram, self).on_attach_ax()
		self.hdl = self.ax.step([],[], label=self.datasource_name)[0]
	

@dc.dataclass(repr=False, eq=False, slots=True)
class Image(Base):
	title : str = 'Image'
	datasource_name : str = 'image datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda :(
		AxisDataMapping('x',None), 
		AxisDataMapping('y',None), 
		AxisDataMapping('brightness', '_z_data'))
	)
	
	_z_data : np.ndarray = None
	
	def update_plot_data(self, data):
		
		self._z_data = data
		if self.hdl is None:
			#self.x_mesh, self.y_mesh = np.indices(self._z_data.shape) #np.arange(0,data.shape[0],1)
			#self.y_mesh = np.arange(0,data.shape[1],1)
			#self.hdl = self.ax.pcolormesh(self.x_mesh, self.y_mesh, self._z_data, label=self.datasource_name, shading='nearest', **self.plt_kwargs)
			kwargs = dict(origin='lower', interpolation='none')
			kwargs.update(self.plt_kwargs)
			self.hdl = self.ax.imshow(self._z_data, label=self.datasource_name, **kwargs)
			#self.ax.legend()
		else:
			self.hdl.set_data(self._z_data)
	
	def on_attach_datasource(self):
		super(Image, self).on_attach_datasource()

	def on_attach_ax(self):
		super(Image, self).on_attach_ax()


@dc.dataclass(repr=False, eq=False, slots=True)
class VerticalLine(Base):
	title : str = 'Vertical Line'
	datasource_name : str = 'vertical line datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda :(AxisDataMapping(None,None),))
	
	
	# Internal
	_x_pos : float = None
	
	def update_plot_data(self, data):
		self._x_pos = data
		if self.hdl is not None:
			self.hdl.remove()
		self.hdl = self.ax.axvline(self._x_pos, label=self.datasource_name, **self.plt_kwargs)
	
	def on_attach_datasource(self):
		super(VerticalLine, self).on_attach_datasource()

	def on_attach_ax(self):
		super(VerticalLine, self).on_attach_ax()


@dc.dataclass(repr=False, eq=False, slots=True)
class HorizontalLine(Base):
	title : str = 'Horizontal Line'
	datasource_name : str = 'horizontal line datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda :(AxisDataMapping(None,None),))
	
	
	# Internal
	_y_pos : float = None
	
	def update_plot_data(self, data):
		
		self._y_pos = data
		if self.hdl is not None:
			self.hdl.remove()
		self.hdl = self.ax.axhline(self._y_pos, label=self.datasource_name, **self.plt_kwargs)
	
	def on_attach_datasource(self):
		super(HorizontalLine, self).on_attach_datasource()

	def on_attach_ax(self):
		super(HorizontalLine, self).on_attach_ax()


@dc.dataclass(repr=False, eq=False, slots=True)
class IterativeLineGraph(Base):
	title : str = 'Iterative Line Graph'
	datasource_name : str = 'iterative line datasource'
	
	axis_data_mappings : tuple[AxisDataMapping] = dc.field(default_factory= lambda : (
			AxisDataMapping(
				'iteration',
				'_x',
				limit_getter=plot_helper.LimRememberExtremes(plot_helper.LimFixed(0))
			),
			AxisDataMapping(
				'value',
				'_y',
				limit_getter=plot_helper.LimRememberExtremes(plot_helper.LimFixed(0))
			)
		)
	)
	
	_x : list = dc.field(default_factory=list)
	_y : list = dc.field(default_factory=list)
	
	def update_plot_data(self, data):
		#print(f'{self.datasource_name} {self._y}')
		self._x.append(len(self._x))
		self._y.append(data)
		self.hdl.set_data(self._x,self._y)
	
	def on_attach_datasource(self):
		super(IterativeLineGraph, self).on_attach_datasource()
		self._x = []
		self._y = []

	def on_attach_ax(self):
		super(IterativeLineGraph, self).on_attach_ax()
		self.hdl = self.ax.plot(self._x, self._y, label=self.datasource_name, **self.plt_kwargs)[0]
