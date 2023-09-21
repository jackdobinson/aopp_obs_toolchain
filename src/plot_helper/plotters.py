"""
Contains classes that assist with plotting histograms.
"""

"""
Contains classes that assist with plotting histograms.
"""
import sys
from time import sleep
from plot_helper.base import Base, AxisDataMapping

import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

class PlotSet:
	def __init__(self,
			fig : mpl.figure.Figure,
			title : str = 'Plot Set',
			plots : list[Base] = [],
			blit=True
		):
		self.fig = fig
		self.title = title
		self.plots= plots
		self.blit = blit
		self.n_frames = 0
		
		
		
	def show(self):
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
				for p in self.plots:
					p.ax.set_visible(True if p.static_frame else False)
				
					
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				
				# set axes visible again
				for p in self.plots:
					p.ax.set_visible(True)
					
				# need axes visible to get their extent
				self.blit_regions = tuple(self.get_ax_bb(ax) for ax in self.fig.axes)
				
				# set axes invisible that will change
				for p in self.plots:
					p.ax.set_visible(True if p.static_frame else False)
				
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()

				# need axes invisible to get the blank background
				self.blit_regions_data = tuple(self.fig.canvas.copy_from_bbox(r) for r in self.blit_regions)

				
				# set axes visible again
				for p in self.plots:
					p.ax.set_visible(True)
				
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				
				
				self.n_frames += 1
				return
				
				
				
			for ax, blit_region_data in zip(self.fig.axes, self.blit_regions_data):
				self.fig.canvas.restore_region(blit_region_data)
				ax.drawn=False
				
			for plot in self.plots:
				plot.update()
				#self.fig.canvas.blit(plot.ax.bbox)
				self.fig.canvas.blit(self.get_ax_bb(plot.ax))

		else:
			for plot in self.plots:
				plot.update()
			self.fig.canvas.draw()
			#self.fig.canvas.
		self.fig.canvas.flush_events()
		self.n_frames += 1


class Histogram(Base):
	title : str = 'Histogram'
	datasource_name : str = 'histogram datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axes_data_mappings : tuple[AxisDataMapping] = (AxisDataMapping('value','bins'), AxisDataMapping('count','hist'))
	nbins : int = 100
	
	@property
	def bins(self):
		return self._bins[1:]
	
	@bins.setter
	def bins(self, value):
		self._bins = value
	
	def update_plot_data(self, data):
		self.hist, self.bins = np.histogram(data, bins=self.nbins, **self.plt_kwargs)
		self.hdl.set_data(self.bins, self.hist)
	
	def on_attach_datasource(self):
		super().on_attach_datasource()
		self.hist = np.array([])
		self.bins = np.array([])

	def on_attach_ax(self):
		super().on_attach_ax()
		self.hdl = self.ax.step([],[], label=self.datasource_name)[0]
		self.ax.legend()
	


class Image(Base):
	title : str = 'Image'
	datasource_name : str = 'image datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axes_data_mappings : tuple[AxisDataMapping] = (AxisDataMapping('x',None), AxisDataMapping('y',None), AxisDataMapping('brightness', 'z_data'))
	
	def update_plot_data(self, data):
		
		self.z_data = data
		if self.hdl is None:
			#self.x_mesh, self.y_mesh = np.indices(self.z_data.shape) #np.arange(0,data.shape[0],1)
			#self.y_mesh = np.arange(0,data.shape[1],1)
			#self.hdl = self.ax.pcolormesh(self.x_mesh, self.y_mesh, self.z_data, label=self.datasource_name, shading='nearest', **self.plt_kwargs)
			kwargs = dict(origin='lower', interpolation='none')
			kwargs.update(self.plt_kwargs)
			self.hdl = self.ax.imshow(self.z_data, label=self.datasource_name, **kwargs)
			#self.ax.legend()
		else:
			self.hdl.set_data(self.z_data)
	
	def on_attach_datasource(self):
		print('on_attach_datasource')
		super().on_attach_datasource()

	def on_attach_ax(self):
		print('on_attach_ax')
		super().on_attach_ax()


class VerticalLine(Base):
	title : str = 'Vertical Line'
	datasource_name : str = 'vertical line datasource'
	
	# Order corresponds to which axes they are associated with (0,1,2,..) -> (x,y,z,...)
	axes_data_mappings : tuple[AxisDataMapping] = (AxisDataMapping('value',None),)
	
	def update_plot_data(self, data):
		
		self.x_pos = data
		if self.hdl is not None:
			self.hdl.remove()
		self.hdl = self.ax.axvline(self.x_pos, label=self.datasource_name, **self.plt_kwargs)
	
	def on_attach_datasource(self):
		super().on_attach_datasource()

	def on_attach_ax(self):
		super().on_attach_ax()
