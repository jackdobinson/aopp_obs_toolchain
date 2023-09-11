#!/usr/bin/env python3
"""
Contains classes for fitting regions to fitscubes
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')


import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
import utils as ut # used for convenience functions
import nemesis.read
import nemesis.write
import numpy as np
import numpy.ma
import scipy as sp
import scipy.optimize
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_wdgt
import matplotlib.patches
import matplotlib.cm
import matplotlib.collections
import copy
from astropy.io import fits
import logging
import logging_setup
import fitscube
import fitscube.header

# base class for displaying a fitscube
class FitscubeViewerBase():
	def __init__(self, hdul, hdul_extension=0, title='Fitscube viewer', figsize=(40,30), colourmap='viridis', 
				icmapmin=0, icmapmax=1, ifreqmin=0, ifreqmax=None, nancolour='white'):
		self.hdul = hdul
		self.hdul_extension=hdul_extension
		self.title = title
		self.figsize = figsize
		self.colourmap = matplotlib.cm.get_cmap(colourmap).copy()
		self.colourmap.set_bad(color=nancolour)
		self.icmapmin = icmapmin*100
		self.icmapmax = icmapmax*100
		self.ifreqmin = ifreqmin
		self.ifreqmax = self.hdul[self.hdul_extension].shape[0]-1 if type(ifreqmax)==type(None) else ifreqmax

		self.func_dict = {'mean':np.nanmean, 'median':np.nanmedian, 'sum':np.nansum}
		self.checkbutton_dict = {'log scale':False}
		self.checkbutton_freq_slicer_dict = {'Slice[x]/Single[ ]':True}

		self.create_plot()
		self.create_widgets()
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.update_ax_title()
		return
	
	def run(self):
		plt.show() # matplotlib handles event passing etc.
		return
	
	def create_plot(self, a11_axis=[0.15,0.15,0.65,0.75]):
		self.f = plt.figure(figsize=[x/2.54 for x in self.figsize])
		self.f.canvas.set_window_title(self.title)
		self.a11 = self.f.add_axes(a11_axis) # [left, bottom, width, height]
		self.im11 = self.a11.imshow(np.full(self.hdul[self.hdul_extension].data.shape[1:], np.nan), 
									origin='lower',
									cmap=self.colourmap)
		return
	
	def update_data(self):
		func = self.func_dict[self.radioFuncPicker.value_selected]
		freq_min = int(self.sliderSetFreqMin.val)
		freq_max = int(self.sliderSetFreqMax.val)
		if not self.sliderSetFreqSingle.active:
			self.data = func(self.hdul[self.hdul_extension].data[freq_min:freq_max,:,:], axis=0)
		else:
			self.data = self.hdul[self.hdul_extension].data[int(self.sliderSetFreqSingle.val), :,:] # don't pass through a function
		if self.checkbutton_dict['log scale']:
			self.data[np.where(self.data<=0)]=np.nan
			self.data = np.log(self.data)
			self.data_min = np.nanmin(self.data)
			self.data_max = np.nanmax(self.data)
		else:
			self.data_min = np.nanmin(self.data)
			self.data_max = np.nanmax(self.data)
		self.data_range = self.data_max - self.data_min
	
	def update_plot(self):
		cmap_min = self.sliderSetCmapMin.val/100*self.data_range + self.data_min
		cmap_max = self.sliderSetCmapMax.val/100*self.data_range + self.data_min
		#self.im11 = self.a11.imshow(self.data, origin='lower',
		#							vmin=cmap_min, vmax=cmap_max, cmap=self.colourmap)
		self.im11.set_data(self.data)
		self.im11.set_clim(cmap_min, cmap_max)
		return
	
	def update_ax_title(self):
		return
	
	def create_widgets(self):
		print(f'DEBUGGING: in "create_widgets()" at {self}')
		self.buttonResetViewerAx = self.f.add_axes([0.0,0.0,0.1,0.05])
		self.buttonResetViewer = mpl_wdgt.Button(self.buttonResetViewerAx, 'Reset', hovercolor='0.9')
		
		self.sliderSetCmapMinAx = self.f.add_axes([0.85,0.15, 0.02, 0.75])
		self.sliderSetCmapMin = mpl_wdgt.Slider(self.sliderSetCmapMinAx, 'Cmap\nMin',
												0, 100,
												valinit = self.icmapmin,
												orientation = 'vertical',
												valfmt = '%1.1e')
		
		self.sliderSetCmapMaxAx = self.f.add_axes([0.89,0.15, 0.02, 0.75])
		self.sliderSetCmapMax = mpl_wdgt.Slider(self.sliderSetCmapMaxAx, 'Cmap\nMax',
												0, 100,
												self.icmapmax,
												orientation = 'vertical',
												valfmt = '%1.1e')
		
		self.sliderSetFreqMinAx = self.f.add_axes([0.93,0.15,0.02,0.75])
		self.sliderSetFreqMin = mpl_wdgt.Slider(self.sliderSetFreqMinAx, 'Freq\nMin',
												0, self.hdul[self.hdul_extension].shape[0]-1,
												valinit = self.ifreqmin,
												orientation='vertical', valfmt='%d', valstep=1)
		
		self.sliderSetFreqMaxAx = self.f.add_axes([0.97,0.15,0.02,0.75])
		self.sliderSetFreqMax = mpl_wdgt.Slider(self.sliderSetFreqMaxAx, 'Freq\nMax',
												0, self.hdul[self.hdul_extension].shape[0]-1,
												valinit = self.ifreqmax,
												orientation='vertical', valfmt='%d', valstep=1)
		
		self.radioFuncPickerAx = self.f.add_axes([0.0, 0.85, 0.1,0.15])
		self.radioFuncPicker = mpl_wdgt.RadioButtons(self.radioFuncPickerAx, 
													sorted(self.func_dict.keys()), active=1)
		
		self.checkLogScaleAx = self.f.add_axes([0.0, 0.8, 0.1, 0.05])
		self.checkLogScale = mpl_wdgt.CheckButtons(
									self.checkLogScaleAx,
									sorted(self.checkbutton_dict.keys()), 
									actives=[self.checkbutton_dict[k] for k in sorted(self.checkbutton_dict.keys())]
							)

		self.checkFreqSliceOrSingleAx = self.f.add_axes([0.85, 0.95, 0.15, 0.05])
		self.checkFreqSliceOrSingle = mpl_wdgt.CheckButtons(
											self.checkFreqSliceOrSingleAx, 
											sorted(self.checkbutton_freq_slicer_dict.keys()),
											actives=[self.checkbutton_freq_slicer_dict[k] 
														for k in sorted(self.checkbutton_freq_slicer_dict.keys())] 
									)

		self.sliderSetFreqSingleAx = self.f.add_axes([0.95, 0.15,0.02,0.75])
		self.sliderSetFreqSingle = mpl_wdgt.Slider(self.sliderSetFreqSingleAx,
													'Freq Bin',
													0, self.hdul[self.hdul_extension].shape[0]-1,
													valinit = int((self.ifreqmin+self.ifreqmax)/2),
													orientation='vertical', valfmt='%d', valstep=1)
	
		self.sliderSetFreqSingleAx.set_visible(False)
		self.sliderSetFreqSingle.set_active(False)

		# connect buttons to callback functions and store connection ids (CID's)
		self.buttonResetViewerCID = self.buttonResetViewer.on_clicked(self.buttonResetViewerClicked)
		self.sliderSetCmapMinCID = self.sliderSetCmapMin.on_changed(self.sliderSetCmapMinChanged)
		self.sliderSetCmapMaxCID = self.sliderSetCmapMax.on_changed(self.sliderSetCmapMaxChanged)
		self.sliderSetFreqMinCID = self.sliderSetFreqMin.on_changed(self.sliderSetFreqMinChanged)
		self.sliderSetFreqMaxCID = self.sliderSetFreqMax.on_changed(self.sliderSetFreqMaxChanged)
		self.radioFuncPickerCID = self.radioFuncPicker.on_clicked(self.radioFuncPickerClicked)
		self.checkLogScaleCID = self.checkLogScale.on_clicked(self.checkLogScaleClicked)
		self.checkFreqSliceOrSingleCID = self.checkFreqSliceOrSingle.on_clicked(self.checkFreqSliceOrSingleClicked)
		self.sliderSetFreqSingleCID = self.sliderSetFreqSingle.on_changed(self.sliderSetFreqSingleChanged)
		return		
	
	def buttonResetViewerClicked(self, event):
		self.sliderSetCmapMin.reset()
		self.sliderSetCmapMax.reset()
		self.sliderSetFreqMin.reset()
		self.sliderSetFreqMax.reset()
		self.radioFuncPicker.set_active(1) # set 'median' active as that is the original state
		for i, k in enumerate(sorted(self.checkbutton_dict.keys())):
			if self.checkbutton_dict[k]:
				self.checkLogScale.set_active(i) # deactivate everything as that is the original state
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def sliderSetCmapMinChanged(self, val):
		cmap_min = val
		cmap_max = self.sliderSetCmapMax.val
		if cmap_min > cmap_max:
			self.sliderSetCmapMax.set_val(cmap_min)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def sliderSetCmapMaxChanged(self, val):
		cmap_max = val
		cmap_min = self.sliderSetCmapMin.val
		if cmap_min > cmap_max:
			self.sliderSetCmapMin.set_val(cmap_max)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def sliderSetFreqMinChanged(self, val):
		freq_min = val
		freq_max = self.sliderSetFreqMax.val
		if freq_min > freq_max:
			self.sliderSetFreqMax.set_val(freq_min+1)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def sliderSetFreqMaxChanged(self, val):
		#print(f'DEBUGGING: sliderSetFreqMaxChanged to {val}')
		freq_max = val
		freq_min = self.sliderSetFreqMin.val
		if freq_min > freq_max:
			self.sliderSetFreqMin.set_val(freq_max-1)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def radioFuncPickerClicked(self, label):
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
	
	def checkLogScaleClicked(self, label):
		if self.checkbutton_dict[label]:
			self.checkbutton_dict[label] = False
		else:
			self.checkbutton_dict[label] = True
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return()

	def checkFreqSliceOrSingleClicked(self, label):
		#print(f'DEBUGGING: checkFreqSliceOrSingle is {self.checkbutton_freq_slicer_dict[label]}')
		if self.checkbutton_freq_slicer_dict[label]:
			# if this was true, then we were using slices so change to single value
			self.checkbutton_freq_slicer_dict[label] = False
			self.sliderSetFreqMaxAx.set_visible(False)
			self.sliderSetFreqMax.set_active(False)
			self.sliderSetFreqMinAx.set_visible(False)
			self.sliderSetFreqMin.set_active(False)
			self.sliderSetFreqSingleAx.set_visible(True)
			self.sliderSetFreqSingle.set_active(True)
			self.sliderSetFreqSingle.set_val(int((self.sliderSetFreqMax.val+self.sliderSetFreqMin.val)/2))
		else:
			# we were using single values so change to slices
			self.checkbutton_freq_slicer_dict[label] = True
			self.sliderSetFreqMaxAx.set_visible(True)
			self.sliderSetFreqMax.set_active(True)
			self.sliderSetFreqMinAx.set_visible(True)
			self.sliderSetFreqMin.set_active(True)
			self.sliderSetFreqSingleAx.set_visible(False)
			self.sliderSetFreqSingle.set_active(False)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return

	def sliderSetFreqSingleChanged(self, val):
		self.sliderSetFreqMin.set_val(val)
		self.sliderSetFreqMax.set_val(val+1)
		self.update_data()
		self.updateCmapText()
		self.update_plot()
		self.f.canvas.draw()
		return
		
	
	def updateCmapText(self):
		cmin = self.sliderSetCmapMin.val
		cmax = self.sliderSetCmapMax.val
		cmin_text = self.sliderSetCmapMin.valfmt % (cmin/100*self.data_range + self.data_min)
		cmax_text = self.sliderSetCmapMax.valfmt % (cmax/100*self.data_range + self.data_min)
		self.sliderSetCmapMin.valtext.set_text(cmin_text)
		self.sliderSetCmapMax.valtext.set_text(cmax_text)
		return

# extend base class to give functionality for moving an ellipse to encircle the planet's disk
class FitscubeDiskFinder(FitscubeViewerBase):
	def __init__(self, hdul, hdul_extension=0, title='fitscube disk finder', figsize=(40,30), colourmap='viridis',
					ecx=0, ecy=0, a=1, b=1, theta=0, icmapmin=0, icmapmax=100, ifreqmin=0, ifreqmax=None):
		self.ecx = ecx
		self.ecy = ecy
		self.a = a
		self.b = b
		self.theta = theta
		super().__init__(hdul, hdul_extension, title, figsize, colourmap, icmapmin, icmapmax, ifreqmin, ifreqmax)
		return

	def create_plot(self):
		super().create_plot()
		self.ellipse = matplotlib.patches.Ellipse((self.ecx, self.ecy), 
													self.a*2, self.b*2, self.theta, 
													edgecolor='tab:green', facecolor='none')
		return
	def update_plot(self):
		super().update_plot()
		self.a11.add_artist(self.ellipse)
		return

	def create_widgets(self):
		super().create_widgets()
		self.textboxEllipseCenterXAx = self.f.add_axes([0.2, 0.0, 0.06, 0.03])
		self.textboxEllipseCenterX = mpl_wdgt.TextBox(self.textboxEllipseCenterXAx,
														label = 'x_0', initial=f'{self.ecx:.5G}', hovercolor='0.9')

		self.textboxEllipseCenterYAx = self.f.add_axes([0.3, 0.0, 0.06, 0.03])
		self.textboxEllipseCenterY = mpl_wdgt.TextBox(self.textboxEllipseCenterYAx,
														label = 'y_0', initial=f'{self.ecy:.5G}', hovercolor='0.9')

		self.textboxEllipseEqRadAx = self.f.add_axes([0.4, 0.0, 0.06, 0.03])
		self.textboxEllipseEqRad = mpl_wdgt.TextBox(self.textboxEllipseEqRadAx,
														label = 'a', initial=f'{self.a:.5G}', hovercolor='0.9')


		self.textboxEllipsePolRadAx = self.f.add_axes([0.5, 0.0, 0.06, 0.03])
		self.textboxEllipsePolRad = mpl_wdgt.TextBox(self.textboxEllipsePolRadAx,
														label = 'b', initial=f'{self.b:.5G}', hovercolor='0.9')

		self.textboxEllipseAngleAx = self.f.add_axes([0.6, 0, 0.06, 0.03])
		self.textboxEllipseAngle = mpl_wdgt.TextBox(self.textboxEllipseAngleAx,
														label = 'theta', initial=f'{self.theta:.5f}', hovercolor='0.9')


		self.textboxEllipseCenterXCID = self.textboxEllipseCenterX.on_submit(self.textboxEllipseCenterXSubmitted)
		self.textboxEllipseCenterYCID = self.textboxEllipseCenterY.on_submit(self.textboxEllipseCenterYSubmitted)
		self.textboxEllipseEqRadCID = self.textboxEllipseEqRad.on_submit(self.textboxEllipseEqRadSubmitted)
		self.textboxEllipsePolRadCID = self.textboxEllipsePolRad.on_submit(self.textboxEllipsePolRadSubmitted)
		self.textboxEllipseAngleCID = self.textboxEllipseAngle.on_submit(self.textboxEllipseAngleSubmitted)

		return		
	def textboxEllipseCenterXSubmitted(self, text):
		self.ecx = float(text)
		self.updateEllipseCenter()
		#self.update_plot()
		self.f.canvas.draw()
		return

	def textboxEllipseCenterYSubmitted(self, text):
		self.ecy = float(text)
		self.updateEllipseCenter()
		#self.update_plot()
		self.f.canvas.draw()
		return

	def textboxEllipseEqRadSubmitted(self, text):
		self.a = float(text)
		self.updateEllipseRadii()
		self.f.canvas.draw()
		return

	def textboxEllipsePolRadSubmitted(self, text):
		self.b = float(text)
		self.updateEllipseRadii()
		self.f.canvas.draw()
		return

	def textboxEllipseAngleSubmitted(self, text):
		self.theta = float(text)
		self.updateEllipseAngle()
		self.f.canvas.draw()
		return

	def updateEllipseAngle(self):
		self.ellipse.angle = self.theta

	def updateEllipseRadii(self):
		self.ellipse.width = 2*self.a
		self.ellipse.height = 2*self.b

	def updateEllipseCenter(self):
		self.ellipse.set_center((self.ecx, self.ecy))
		return

	def getEllipseCenter(self):
		return(self.ecx, self.ecy)
	def getEllipseCenterFits(self):
		return(self.ecx+1, self.ecy+1)

	def getEllipseParams(self):
		return(self.ecx, self.ecy, self.a, self.b, self.theta)

# extend base class to give functionality for picking latitide region
class FitscubeLatitudeRegionPicker(FitscubeViewerBase):
	def __init__(self, hdul, hdul_extension=0, title='fitscube disk finder', figsize=(40,30), colourmap='viridis',
					icmapmin=0, icmapmax=100, ifreqmin=0, ifreqmax=None, sizes=(20,), latmin=0, latmax=0):
		self.sizes=sizes
		self.latmin=latmin
		self.latmax=latmax
		self.pix_region= np.zeros((0,2))	
		super().__init__(hdul, hdul_extension, title, figsize, colourmap, icmapmin, icmapmax, ifreqmin, ifreqmax)
		return

	def create_plot(self):
		super().create_plot()
		self.pix_region_patch = matplotlib.collections.RegularPolyCollection(4, sizes=self.sizes,
																			facecolors='none',
																			offsets=self.pix_region,
																			transOffset=self.a11.transData,
																			edgecolor='red',
																			lw=1)
		self.a11.add_collection(self.pix_region_patch)
		return

	def update_plot(self):
		super().update_plot()	
		self.pix_region_patch.remove()
		self.pix_region_patch = matplotlib.collections.RegularPolyCollection(4, sizes=self.sizes,
																			facecolors='none',
																			offsets=self.pix_region,
																			transOffset=self.a11.transData,
																			edgecolor='red',
																			lw=1)
		self.a11.add_collection(self.pix_region_patch)
		return

	def create_widgets(self):
		super().create_widgets()
		self.textboxLatMinAx = self.f.add_axes([0.2, 0.0, 0.1, 0.05])
		self.textboxLatMin = mpl_wdgt.TextBox(self.textboxLatMinAx,
														label = 'Lat Min', initial=f'{self.latmin}', hovercolor='0.9')

		self.textboxLatMaxAx = self.f.add_axes([0.4, 0.0, 0.1, 0.05])
		self.textboxLatMax = mpl_wdgt.TextBox(self.textboxLatMaxAx,
														label = 'Lat Max', initial=f'{self.latmax}', hovercolor='0.9')

		self.textboxLatMinCID = self.textboxLatMin.on_submit(self.textboxLatMinSubmitted)
		self.textboxLatMaxCID = self.textboxLatMax.on_submit(self.textboxLatMaxSubmitted)
		return		
	def textboxLatMinSubmitted(self, text):
		self.latmin = float(text)
		self.updateLatRange()
		#self.update_plot()
		self.f.canvas.draw()
		return

	def textboxLatMaxSubmitted(self, text):
		self.latmax = float(text)
		self.updateLatRange()
		#self.update_plot()
		self.f.canvas.draw()
		return
	
	def updateLatRange(self):
		chosen_idxs = 	np.argwhere(
							np.logical_and(
								np.logical_and(
										self.hdul['LATITUDE'].data > self.latmin,
										self.hdul['LATITUDE'].data < self.latmax),
								~np.isnan(self.hdul['LATITUDE'].data)
							)
						)
		self.pix_region = chosen_idxs[:,[1,0]]
		self.update_plot()
		return

	def getRegionData(self):
		return(self.pix_region[:,0], self.pix_region[:,1])
	def getRegionIdxs(self):
		return(np.flip(np.transpose(self.pix_region),axis=0))

class FitscubeCircleDriftFinder(FitscubeViewerBase):
	"""
	Subclasses "FitscubeViewerBase()" to let the user click to define a circle that will be associated with a
	specific wavelength of the fitscube. The user can then go to a different wavelength and make another circle.
	Many circles can be defined in this way, giving a circle vs. wavelength dependence. This is then turned into
	a circle center vs wavelength dependence, and eventually a drift with wavelength. The drift linearly interpolated
	and subtracted to create a non-drifting image. Will store the circle center, radius, and drift with wavelength
	relation in the fitscube's header.

	TODO:
		* Implement clicking on plot area to define a circle (3 points minimum, see "../fitscube_fit_circle.py"
		* Add quick buttons for going up and down to frequencies that already have a defined circle
		* Display already defined circles on freq-bin picker axis (prob. hard)
		* Make a button to purge all circles
		* Change reset button to reset to the defaults for this mode
	"""
	def __init__(self, hdul, hdul_extension=0, title='Fitscube Circle Drift Finder', figsize=(40,30), colourmap='viridis',
				icmapmin=0, icmapmax=1, ifreqmin=0, ifreqmax=None, nancolour='white',
				cx=0, cy=0, cr=1):
		self.cx, self.cy, self.cr = cx, cy, cr # circle x, y (center) and r (radius)
		super().__init__(hdul, hdul_extension, title, figsize, colourmap, icmapmin, icmapmax, ifreqmin, ifreqmax)
		return
	
	def create_plot(self):
		super().create_plot()
		self.f.canvas.mpl_connect('button_press_event', self.plotOnClick)
		self.circles = {}#mpl.patches.Circle((self.cx, self.cy), radius=self.cr, 
						#					edgecolor='tab:green', 
						#					facecolor='none')
		self.circle_artist = None
		self.scatter_graphs = {} #[None]*self.hdul[0].data.shape[0] # create holder for scatter graphs
		self.optimise_results = {} # create a holder for circle center finding results (including error etc.)
		return
		
	def create_widgets(self):
		print(f'DEBUGGING: in "create_widgets()" at {self}')
		super().create_widgets()
		self.sliderSetFreqSingleAx.set_visible(True)
		self.sliderSetFreqSingle.set_active(True)
		self.sliderSetFreqMaxAx.set_visible(False)
		self.sliderSetFreqMax.set_active(False)
		self.sliderSetFreqMinAx.set_visible(False)
		self.sliderSetFreqMin.set_active(False)
		self.checkFreqSliceOrSingle.set_active(0) # de-activate slice mode

		self.buttonNextDriftCircleAx = self.f.add_axes([0.98, 0.6, 0.2, 0.2])
		self.buttonNextDriftCircle = mpl_wdgt.Button(self.buttonNextDriftCircleAx, '^\n|', hovercolor='0.9')

		self.buttonPrevDriftCircleAx = self.f.add_axes([0.98, 0.4, 0.2, 0.2])
		self.buttonPrevDriftCircle = mpl_wdgt.Button(self.buttonPrevDriftCircleAx, '|\nv', hovercolor='0.9')

		self.buttonShowHideCircleAx = self.f.add_axes([0.1, 0.0, 0.1, 0.05])
		self.buttonShowHideCircle = mpl_wdgt.Button(self.buttonShowHideCircleAx, 'Show/Hide Circle', hovercolor='0.9')

		self.buttonRemoveCircleDataAx = self.f.add_axes([0.3, 0.0, 0.1, 0.05])
		self.buttonRemoveCircleData = mpl_wdgt.Button(self.buttonRemoveCircleDataAx, 'Remove Current Circle',
														hovercolor='0.9')

		self.buttonNextDriftCircleCID = self.buttonNextDriftCircle.on_clicked(self.buttonNextDriftCircleClicked)
		self.buttonPrevDriftCircleCID = self.buttonPrevDriftCircle.on_clicked(self.buttonPrevDriftCircleClicked)
		self.buttonShowHideCircleCID = self.buttonShowHideCircle.on_clicked(self.buttonShowHideCircleClicked)
		self.buttonRemoveCircleDataCID = self.buttonRemoveCircleData.on_clicked(self.buttonRemoveCircleDataClicked)

		self.buttonShowHideCircleState = False # default state of showing circle

		self.checkLogScale.set_active(0) # make log scale be default

		return

	def buttonRemoveCircleDataClicked(self, event):
		print(f'DEBUGGING: In "buttonRemoveCircleDataClicked()" event {event}')
		idx = int(self.sliderSetFreqSingle.val)
		if idx in self.scatter_graphs.keys():
			self.scatter_graphs[idx].remove()
			del self.scatter_graphs[idx]
		if idx in self.circles.keys():
			self.circles[idx].remove()
			del self.circles[idx]
		self.update_plot()
		self.f.canvas.draw()
		return
		

	def buttonShowHideCircleClicked(self, event):
		print(f'DEBUGGING: In "buttonShowHideCircleClicked()" event {event}')
		self.buttonShowHideCircleState = not self.buttonShowHideCircleState
		"""
		idx = int(self.sliderSetFreqSingle.val)
		if self.circles.get(idx, None) is not None:
			if self.circle_artist is None:
				self.circle_artist.remove()
				self.circle_artist = None
			else:
				self.circle_artist = self.a11.add_artist(self.circles[idx])
		"""
		self.update_plot()
		self.f.canvas.draw()
		return
		

	def buttonNextDriftCircleClicked(self, event):
		# if i've clicked this we want to go to the next highest frequency that has a circle
		idxs = sorted(self.scatter_graphs.keys())
		idx = int(self.sliderSetFreqSingle.val)
		for i in idxs:
			if i > idx:
				self.sliderSetFreqSingle.set_val(i)
				return
		print('DEBUGGINGL Could not find a set of points at a larger frequecy')
		return

	def buttonPrevDriftCircleClicked(self, event):
		idxs = sorted(self.scatter_graphs.keys())
		idx = int(self.sliderSetFreqSingle.val)
		for i in idxs[::-1]: # reverse order
			if i < idx:
				self.sliderSetFreqSingle.set_val(i)
				return
		print('DEBUGGING: Could not find a set of points at a smaller frequency')
		return

	def buttonResetViewerClicked(self, event):
		self.sliderSetCmapMax.reset()
		self.sliderSetCmapMin.reset()
		self.sliderSetFreqSingle.reset()
		self.sliderSetFreqSingleAx.set_visible(True)
		self.sliderSetFreqSingle.set_active(True)
		self.sliderSetFreqMaxAx.set_visible(False)
		self.sliderSetFreqMax.set_active(False)
		self.sliderSetFreqMinAx.set_visible(False)
		self.sliderSetFreqMin.set_active(False)
		return
		
	def update_plot(self):
		super().update_plot()
		if self.sliderSetFreqSingle.active:
			idx = int(self.sliderSetFreqSingle.val)
			this_idx_circle = self.circles.get(idx,None)
			if self.circle_artist is not None:
				self.circle_artist.remove()
				self.circle_artist = None
			if (this_idx_circle is not None) and (self.buttonShowHideCircleState):
				self.circle_artist = self.a11.add_artist(this_idx_circle)
			for sc in self.scatter_graphs.values():
				sc.set_visible(False)
			if self.scatter_graphs.get(idx,None) is not None:
				self.scatter_graphs[idx].set_visible(True)
				
		else:
			# if we are not using single frequency mode then set circles to invisible
			if self.circle_artist is not None:
				self.circle_artist.remove()
		return
	
	def calc_R(self, xc, yc, xs, ys):
		"""
		Calculates the distance from a point (xc,yc) to a set of points with the x-components xs and y components ys
		"""
		return(np.sqrt((xs-xc)**2 + (ys-yc)**2))

	def calc_R2(self, xc, yc, xs, ys):
		return((xs-xc)**2 + (ys-yc)**2)


	def circle_optimise_func(self, c, xs, ys):
		"""
		If all points (xs,ys) are on the same circle with center c, then their distance from c will all be the same.
		In that case Ri - Ri.mean() will be zero.
		"""
		Ri = self.calc_R2(c[0], c[1], xs, ys)
		return((Ri - Ri.mean())**2)
		#return(Ri - Ri.mean())

	def plotOnClick(self, event):
		"""
		Processes events created when a user clicks on the plot
		"""
		print(f'DEBUGGING: in "plotOnClick()", event {event}')
		print(event.inaxes != self.a11, not self.sliderSetFreqSingle.active)
		# we don't want to bother doing this if we are in slice mode
		if (event.inaxes != self.a11) or (not self.sliderSetFreqSingle.active):
			print(f'Outside axis or not in single mode, returning...')
			return
		idx = int(self.sliderSetFreqSingle.val)
		if self.scatter_graphs.get(idx, None) is None:
			# we have to create the scatter graph for this frequency
			self.scatter_graphs[idx] = self.a11.scatter([], [], color='red', s=1) # create an empty scatter graph
		apoint = np.array(((event.xdata, event.ydata),))
		offsets = np.concatenate([self.scatter_graphs[idx].get_offsets(), apoint], axis=0)
		if offsets.shape[0] > 2:
			# if we have more than two points we can draw a circle
			xs, ys = offsets[:,0], offsets[:,1]
			#c2, ier = sp.optimize.leastsq(self.circle_optimise_func, (np.mean(xs), np.mean(ys)), args=(xs,ys))
			opt_result = sp.optimize.least_squares(		self.circle_optimise_func, 
														(np.mean(xs), np.mean(ys)), 
														args=(xs,ys)
													)
			self.optimise_results[idx] = opt_result # store result for later reference
			c2 = opt_result.x # store the result of the least squares fit
			Ri2 = self.calc_R(c2[0], c2[1],xs, ys)
			R_2 = Ri2.mean()
			residual = sum((Ri2 - R_2)**2)
			if self.circles.get(idx,None) is not None:
				self.circles[idx].set_center(c2)
				self.circles[idx].set_radius(R_2)
			else:
				# we don't have a circle created for this wavelength yet so create one
				self.circles[idx] = mpl.patches.Circle(c2, radius=R_2, facecolor='none', edgecolor='red', lw=1)
		self.scatter_graphs[idx].set_offsets(offsets)
		#self.update_data() # we don't change data so we can ignore this
		# self.updateCmapText() # we don't change the color map so we can ignore this
		self.update_plot()
		self.f.canvas.draw()
		return

	def getDrift(self):
		# TODO
		# * Get linear interpolation of circle center drift
		# * return an offset from the 0th frequency bin w.r.t frequency bin
		# * Change over to np.polynomial.Polynomial.fit()
		# * Make a comparison between straight line P(1) and constant P(0) if no substantial improvement use P(0)
		idxs = np.array(sorted(self.circles.keys()))
		centers = np.array([self.circles[i].get_center() for i in idxs])
		centers_x = centers[:,0]
		centers_y = centers[:,1]
		print(f'DEBUGGING: centers_x {centers_x}')
		freq_bins = np.arange(0,self.hdul[self.hdul_extension].data.shape[0])
		print(f'DEBUGGING: freq_bins {freq_bins}')

		w = np.array([1/self.optimise_results[i].cost for i in idxs]) # cost = how far we are from a perfect circle

		p0_fit_x, p0_fit_x_rrsr = np.polynomial.Polynomial.fit(idxs, centers_x, 0, full=True, w=w) # fit a constant
		p0_fit_y, p0_fit_y_rrsr = np.polynomial.Polynomial.fit(idxs, centers_y, 0, full=True, w=w) # fit a constant
		p1_fit_x, p1_fit_x_rrsr = np.polynomial.Polynomial.fit(idxs, centers_x, 1, full=True, w=w) # fit a line
		p1_fit_y, p1_fit_y_rrsr = np.polynomial.Polynomial.fit(idxs, centers_y, 1, full=True, w=w) # fit a line

		p0_centers_x = p0_fit_x(freq_bins)
		p0_centers_y = p0_fit_y(freq_bins)
		p1_centers_x = p1_fit_x(freq_bins)
		p1_centers_y = p1_fit_y(freq_bins)
		drift_adj_x = -(p1_centers_x - p1_centers_x[0])
		drift_adj_y = -(p1_centers_y - p1_centers_y[0])

		p0_chisq_x = np.sum(((p0_fit_x(idxs) - centers_x)**2/centers_x))
		p0_chisq_y = np.sum(((p0_fit_y(idxs) - centers_y)**2/centers_y))
		p1_chisq_x = np.sum(((p1_fit_x(idxs) - centers_x)**2/centers_x))
		p1_chisq_y = np.sum(((p1_fit_y(idxs) - centers_y)**2/centers_y))

		return(
			idxs, # indices of the user defined circles
			np.stack([centers_x, centers_y], axis=-1), # centers of the user defined circles
			freq_bins, # indices of the whole frequency range (from 0 to self.hdul[0],data,shape[0])
			np.stack([drift_adj_x, drift_adj_y], axis=-1), # drift adjustments for linear (p1) fit
			np.stack([p0_centers_x, p0_centers_y], axis=-1), # centers when using a constant (p0) fit
			np.stack([p1_centers_x, p1_centers_y], axis=-1), # centers when using a linear (p1) fit
			#np.array([p0_fit_x_rrsr[0], p0_fit_y_rrsr[0]]), # (data[i] - model[i])**2 for constant (p0) fit
			#np.array([p1_fit_x_rrsr[0], p1_fit_y_rrsr[0]]), # (data[i] - model[i])**2 for linear (p1) fit
			np.array([p0_chisq_x, p0_chisq_y]), # chisq
			np.array([p1_chisq_x, p1_chisq_y]), # chisq
			w
		)

	def applyDriftCorrection(self):
		"""
		Gets the drift correction from self.getDrift() and applies it to the hdul under consideration.
		By default creates a new fits file.
		TODO:
		* Write a way of saving and loading the drift corrections so I can alter them if needed
		"""
		(	idxs, centers, freq_bins, drift_adj, 
			p0_centers, p1_centers, p0_sq_resid, 
			p1_sq_resid, weights
		) = self.getDrift()

		for i in freq_bins:
			# have to reverse order of drift adjustment because we fits files uses zyx ordering
			self.hdul[self.hdul_extension].data[i,:,:] = sp.ndimage.shift(self.hdul[self.hdul_extension].data[i,:,:], drift_adj[i,::-1], 
														order=0, 
														mode='constant', 
														cval=np.nan, 
														prefilter=False)

		self.hdul.writeto('drift_adjusted.fits', overwrite=True)
		return(idxs, centers, freq_bins, drift_adj, p0_centers, p1_centers, p0_sq_resid, p1_sq_resid, weights)
				
class FitscubeMaskPainter(FitscubeViewerBase):			
	def __init__(self, hdul, hdul_extension=0, title='fitscube mask painter', figsize=(40,30), colourmap='viridis',
					icmapmin=0, icmapmax=100, ifreqmin=0, ifreqmax=None, mask=None):
		self.paint_mask_flag = False
		self.remove_mask_flag = False
		self.in_plot_axis_flag = False
		self.mask = mask
		super().__init__(hdul, hdul_extension, title, figsize, colourmap, icmapmin, icmapmax, ifreqmin, ifreqmax)
		return

	def create_plot(self):
		super().create_plot(a11_axis=[0.25, 0.15, 0.65, 0.75])
		self.a11.set_title('Click (and drag) to set mask')
		self.f.canvas.mpl_connect('button_press_event', self.plotOnClick)
		self.f.canvas.mpl_connect('button_release_event', self.plotOnRelease)
		self.f.canvas.mpl_connect('axes_enter_event', self.plotOnEnterAxis)
		self.f.canvas.mpl_connect('axes_leave_event', self.plotOnLeaveAxis)
		self.f.canvas.mpl_connect('motion_notify_event', self.plotOnMouseMove)
		if self.mask is None:
			self.mask = np.zeros(self.hdul[self.hdul_extension].data.shape[1:], dtype=bool) # create a holder for various named masks	
		self.mask_scatter = self.a11.scatter([],[],color='red', s=3, marker='x')
		self.mask_ax = self.f.add_axes([0.0, 0.15, 0.4,0.4])
		self.mask_ax.set_title('Masked Data')
		self.masked_data = np.ma.array(np.nanmedian(self.hdul[self.hdul_extension].data, axis=0), mask=self.mask)
		self.mask_im = self.mask_ax.imshow(self.masked_data, origin='lower', cmap=self.colourmap)
		return

	def update_masked_data(self):
		self.masked_data.mask = self.mask
		self.mask_im.set_data(self.masked_data)

	def plotOnMouseMove(self, event):
		if self.in_plot_axis_flag:
			#print(event)
			x, y = (int(event.xdata), int(event.ydata))
			if self.paint_mask_flag:
				self.mask[y,x] = True
				#print('paint')
			if self.remove_mask_flag:
				self.mask[y,x] = False
				#print('remove')
			if self.paint_mask_flag or self.remove_mask_flag:
				offsets = np.stack(np.nonzero(self.mask)[::-1], axis=-1)
				self.mask_scatter.set_offsets(offsets)
				#self.update_plot() # don't need to do this I think as data does not change
				self.update_masked_data()
				self.f.canvas.draw()
		return

	def plotOnLeaveAxis(self, event):
		#print(event.inaxes)
		if event.inaxes == self.a11: # i.e. we are leaving this axis, a bit odd naming
			self.in_plot_axis_flag = False
		return

	def plotOnEnterAxis(self, event):
		#print(event.inaxes)
		if event.inaxes == self.a11:
			self.in_plot_axis_flag = True
		return

	def plotOnRelease(self, event):
		print(f'Mouse Clicked {event}')
		if event.button == 1:
			self.paint_mask_flag = False
		if event.button == 3:
			self.remove_mask_flag = False
		self.update_masked_data()
		self.f.canvas.draw()
		return

	def plotOnClick(self, event):
		#print(f'Mouse Clicked {event}')
		if not self.in_plot_axis_flag:
			#print(f'Button clicked outside axis, returning...')
			return
		x, y = (int(event.xdata), int(event.ydata))
		if event.button == 1: #  if we left-click, add pixel to mask
			self.mask[y,x] = True
			self.paint_mask_flag=True
		elif event.button == 3: # if we right-click, remove pixel from mask
			self.mask[y,x] = False
			self.remove_mask_flag=True
		# show masked region
		# get masked pixels
		offsets = np.stack(np.nonzero(self.mask)[::-1], axis=-1)
		#print(offsets)
		self.mask_scatter.set_offsets(offsets)
		#self.update_plot()
		self.update_masked_data()
		self.f.canvas.draw()
		return

	def getMask2d(self):
		return(self.mask)
		
	def getMask3d(self):
		mask3d = np.zeros_like(self.hdul[self.hdul_extension].data, dtype=bool)
		mask3d[:,:,:] = self.mask[None,:,:]
		return(mask3d)

	def getMaskedData(self):
		return(np.ma.array(self.hdul[self.hdul_extension].data, mask=self.getMask3d(), fill_value=np.nan))

	def getNanData(self):
		nandata = np.zeros_like(self.hdul[self.hdul_extension].data)
		nandata[...] = self.hdul[self.hdul_extension].data[...]
		nandata[np.nonzero(self.getMask3d())] = np.nan
		return(nandata)

class FitscubeSpectrumExtractor(FitscubeMaskPainter):
	def __init__(self, hdul, hdul_extension=0, title='fitscube spectrum extractor', figsize=(40,30), colourmap='viridis',
					icmapmin=0, icmapmax=100, ifreqmin=0, ifreqmax=None, mask=None):
		super().__init__(hdul, hdul_extension, title, figsize, colourmap, icmapmin, icmapmax, ifreqmin, ifreqmax, mask)
		return
	
	def getSelectedRegionSpectrum(self):
		mask_3d = self.getMask3d()
		wavs = fitscube.header.get_wavelength_grid(self.hdul[self.hdul_extension])
		return(wavs, self.hdul[self.hdul_extension].data[mask_3d].reshape(mask_3d.shape[0], -1))
	
def main(argv):
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))


	for ff in args['fits_files']:
		ff_dir = os.path.dirname(ff)

		# these have different interpretations depending upon the mode, therefore open the actual files etc.
		# inside the "if 'XXXX' in args['mode']:" blocks
		output_savef = os.path.join(ff_dir, args['output.save']) if args['output.save'] is not None else None
		output_modifyf = os.path.join(ff_dir, args['output.modify']) if args['output.modify'] is not None else None
		
		with fits.open(ff) as hdul:
			if 'viewer' in args['mode']:
				viewer = FitscubeViewerBase(hdul, args['extension'])
				viewer.run()
			if 'disk_finder' in args['mode']:
				ecx = hdul[args['extension']].header.get('ELL_CX', hdul[args['extension']].data.shape[-1]//2)
				ecy = hdul[args['extension']].header.get('ELL_CY', hdul[args['extension']].data.shape[-2]//2)
				a = hdul[args['extension']].header.get('RAD_EQ', hdul[args['extension']].data.shape[-1]//4)/(np.abs(hdul[args['extension']].header.get('CDELT1',1))*3600)
				b = hdul[args['extension']].header.get('RAD_PROJ', hdul[args['extension']].data.shape[-2]//4)/(np.abs(hdul[args['extension']].header.get('CDELT2',1))*3600)
				print(a,b)
				viewer = FitscubeDiskFinder(hdul, args['extension'], ecx=ecx, ecy=ecy, a=a, b=b)
				viewer.run()
				(ecx, ecy) = viewer.getEllipseCenter()
				print('python center coords', ecx, ecy)
				print('fits/fortran center coords', ecx+1, ecy+1)
			if 'latitude_region_picker' in args['mode']:
				viewer = FitscubeLatitudeRegionPicker(hdul, args['extension'])
				viewer.run()
				region_data = viewer.getRegionData()
				region_idxs = viewer.getRegionIdxs()
				print(region_data)
				print(region_idxs)
			if 'circle_drift_finder' in args['mode']:
				viewer = FitscubeCircleDriftFinder(hdul, args['extension'])
				viewer.run()
				(	idxs, centers, freq_bins, drift_adj, 
					p0_centers, p1_centers, p0_sq_resid, 
					p1_sq_resid, opt_results
				) = viewer.applyDriftCorrection()
				print(f'DEBUGGING: idxs {idxs}')
				print(f'DEBUGGING: centers {centers}')
				print(f'DEBUGGING: freq_bins {freq_bins}')
				print(f'DEBUGGING: drift_adj {drift_adj}')
				print(f'DEBUGGING: p0_centers {p0_centers}')
				print(f'DEBUGGING: p1_centers {p1_centers}')
				print(f'DEBUGGING: p0_sq_resid {p0_sq_resid}')
				print(f'DEBUGGING: p1_sq_resid {p1_sq_resid}')
				print(f'DEBUGGING: w {opt_results}')
				#print(opt_results.values()[0].grad)

				f1 = plt.figure(figsize=[_x/2.54 for _x in (24,12)])
				a1 = f1.add_axes([0.1, 0.05, 0.4, 0.9])
				a2 = f1.add_axes([0.6, 0.05, 0.4, 0.9])
				s11 = a1.scatter(idxs, centers[:,0], label='center_pos_x', marker='x', color='tab:blue')
				l11 = a1.plot(freq_bins, p1_centers[:,0], 'b-', label=f'linear fit x {p1_sq_resid[0]}')
				l12 = a1.plot(freq_bins, p0_centers[:,0], 'b--', label=f'constant fit x {p0_sq_resid[0]}')
				#a1.scatter(freq_bins, drift_adj[:,0], label='center_x drift', s=1, marker='o', color='tab:blue')
				a1.set_xlabel('frequency bin')
				a1.set_ylabel('pixel number of center x component')
				a1.legend()
		
				s21 = a2.scatter(idxs, centers[:,1], label='center_pos_y', marker='x', color='tab:red')
				l21 = a2.plot(freq_bins, p1_centers[:,1], 'r-', label=f'linear fit y {p1_sq_resid[1]}')
				l22 = a2.plot(freq_bins, p0_centers[:,1], 'r--', label=f'constant fit y {p0_sq_resid[1]}')
				#a2.scatter(freq_bins, drift_adj[:,1], label='center_y drift', s=1, marker='o', color='tab:red')
				a2.set_ylabel('pixel number of center y component')
				a2.set_xlabel('frequency bin')
				a2.legend()

				#hdls1, lbls1 = a1.get_legend_handles_labels()
				#hdls2, lbls2 = a2.get_legend_handles_labels()
				#a1.legend(hdls1+hdls2, lbls1+lbls2, loc='lower right')
				f1.suptitle('Fitted and interpolated cirlce center positions')

				plt.show()
			if 'mask_painter' in args['mode']:
				if output_modifyf is not None:
					logging.info('Opening file "{output_modifyf}" for modification.')
					mask_hdul = fits.open(save_modifyf)
					mask_data = np.array(mask_hdul.data[0,:,:], dtype=bool)
				else:
					mask_data = None

				viewer = FitscubeMaskPainter(hdul, args['extension'], mask=mask_data)
				viewer.run()

				#md = np.ma.array(hdul[0].data, mask=viewer.getMask3d())
				md = viewer.getMaskedData()
				#md[np.nonzero(md.mask)] = np.nan
				#md = viewer.getNanData()

				print(md.mask)

				if output_savef is not None:
					logging.info(f'Saving mask to file "{output_savef}"')
					maskarr = np.array(np.ma.getmaskarray(md), dtype=float)
					#print(maskarr)
					#print(type(maskarr))
					out_hdu = fits.PrimaryHDU(maskarr)
					out_hdul = fits.HDUList([out_hdu])
					out_hdul.writeto(output_savef, overwrite=args['output.overwrite'])

				if output_modifyf is not None:
					mask_hdul.close()

				f1 = plt.figure(figsize=[_x/2.54 for _x in (12,12)])
				a1 = f1.add_axes([0.1, 0.1, 0.9, 0.9])
				cmap = mpl.cm.get_cmap('viridis')
				#cmap.set_bad('red') # why is this not setting the masked region?
				im1 = a1.imshow(np.nanmedian(md, axis=0), origin='lower', cmap=cmap)

				plt.show()
			if 'spectrum_extractor' in args['mode']:
				viewer = FitscubeSpectrumExtractor(hdul, args['extension'], mask=None)
				viewer.run()
				
				wavs, region_spec = viewer.getSelectedRegionSpectrum()
				#print(wavs.shape)
				#print(region_spec.shape)
				region_spec_mean = np.nanmean(region_spec, axis=tuple(range(1,len(region_spec.shape))))
				#print(region_spec_mean.shape)
				spec = np.stack([wavs, region_spec_mean]).T
				#print(spec.shape)
				#for i in range(len(wavs)):
				#	print(spec[i])
				
				np.savetxt(output_savef, spec)
				
				
				
				
				
				
				
				

def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	# A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	# see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	# of them do.
	# Quick reference:
	# ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	# ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	# ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	# ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	class RawDefaultTypeFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class RawDefaultFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap(__doc__), wrapsize=79),
									formatter_class = RawDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('fits_files', type=str, nargs='+', help='List of fits files to fit regions to', default=[])

	parser.add_argument('--extension', type=int, help='extension of fits file to operate on', default=0)

	modes_avail = ('viewer', 'disk_finder', 'latitude_region_picker', 'circle_drift_finder', 'mask_painter', 'spectrum_extractor')
	parser.add_argument('--mode', type=str, nargs='*', 
									help=f'Mode to open fits file in, if multiple passed will do all of them. Available modes are {modes_avail}.', 
									choices=modes_avail,
									default=['viewer'])
	parser.add_argument('--output.save', type=str, nargs='?', default=None, const='output.fits', help='If we want to save data, this is the file we will save it to. If given a relative path (e.g. "./some/file") the program will assume the path is relative to the current "fits_files", if given an absolute path (e.g. "/root/some/file") the program will use the same file for all "fits_files"')
	parser.add_argument('--output.no_overwrite', action='store_false', dest='output.overwrite', help='If present, do not overwrite output files. Relative and absolute paths behave the same as "--output.save"')
	parser.add_argument('--output.modify', type=str, nargs='?', default=None, const='output.fits', help='If present, or specified, will load values from this file as starting values for parameters to find. Relative and absolute paths behave the same as "--output.save".')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface

	return(parsed_args)


if __name__=='__main__':
	main(sys.argv[1:])
