"""
Contains classes that assist with plotting histograms.
"""

"""
Contains classes that assist with plotting histograms.
"""
import sys
from plot_helper.base import Base, AxisDataMapping

import numpy as np

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
		print('on_attach_datasource')
		super().on_attach_datasource()

	def on_attach_ax(self):
		print('on_attach_ax')
		super().on_attach_ax()
