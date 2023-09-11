#!/usr/bin/env python3
"""
Operates on nemesis input files to modify them, modifications can be controlled with a GUI, CLI,
or arguments depending upon the mode chosen. Currently only works with *.spx files.

TODO:
* Change 'file_modifiers' dictionary to 'modes', and use prefixes to denote what the modes should
  operate on. E.g. 'mode["spx.viewer"]' instead of 'file_modifiers[".spx"]["spx_viewer"]'
* Change autodection of filetype and associating file and operation based on filename extension
  to associating operation with file by position, i.e. the first file gets the first operation
  applied to it.


The `if __name__=='__main__':` statement allows execution of code if the script is called directly.
eveything else not in that block will be executed when a script is imported. 
Import statements that the rest of the code relies upon should not be in the if statement, python
is quite clever and will only import a given package once, but will give it multiple names if it
has been imported under different names.

Standard library documentation can be found at https://docs.python.org/3/library/

Packages used in this program are:
	sys
	os 
"""

import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
import utils as ut # used for convenience functions
import nemesis.read
import nemesis.write
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mpl_wdgt
import copy


class SpxViewerBase():
	def __init__(self, data_dict, title='*.spx viewer'):
		self.original_data_dict = data_dict # keep original in case we want to revert changes
		self.dd = copy.deepcopy(data_dict)
		self.ymin = None
		self.ymin_auto = None
		self.ymax = None
		self.ymax_auto = None
		self.title = title
		self.create_plot()
		self.create_widgets()
		self.update_plot()
		self.update_ax_title()
		return

	def run(self):
		plt.show() # matplotlib handles event passing etc.
		return

	def create_plot(self):
		self.f = plt.figure(figsize=[_x/2.54 for _x in (40,30)])
		self.f.canvas.set_window_title(self.title)
		#self.a11 = self.f.add_subplot(1,1,1)
		self.a11 = self.f.add_axes([0.15, 0.15, 0.75, 0.75])
		self.line11_1, = self.a11.plot(self.dd['spec_record'][0][:,0], self.dd['spec_record'][0][:,1], lw=1, color='tab:blue')
		self.err_fill = self.a11.fill_between(self.dd['spec_record'][0][:,0],
							self.dd['spec_record'][0][:,1] + self.dd['spec_record'][0][:,2],
							self.dd['spec_record'][0][:,1] - self.dd['spec_record'][0][:,2],
							color='tab:blue')

		dmaxs = []
		dmins = []
		for i in range(len(self.dd['spec_record'])):
			#dmax = np.max(self.dd['spec_record'][i][:,1] + self.dd['spec_record'][i][:,2])
			dmax = np.max(self.dd['spec_record'][i][:,1])
			#dmin = np.min(self.dd['spec_record'][i][:,1] - self.dd['spec_record'][i][:,2])
			dmaxs.append(dmax)
			#dmins.append(dmin)
		self.ymin = 0 #min(dmins)
		self.ymax = max(dmaxs)
		self.ymin_auto = self.ymin
		self.ymax_auto = self.ymax
		return

	def update_plot(self):
		ngeom = int(self.sliderSetGeom.val)
		xyz = self.dd['spec_record'][ngeom]
		self.line11_1.set_data(xyz[:,0], xyz[:,1])
		self.err_fill.remove() # remove old error fill
		self.err_fill = self.a11.fill_between(xyz[:,0],
							xyz[:,1] + xyz[:,2],
							xyz[:,1] - xyz[:,2],
							color='tab:blue', alpha=0.5)
		#print(np.max(xyz[:,1]))
		#self.a11.set_ylim([0.0, 1.3*np.max(xyz[:,1])])
		self.a11.set_ylim([self.ymin, self.ymax])
		return

	def update_ax_title(self):
		ngeom = int(self.sliderSetGeom.val)
		nav = int(self.sliderSetFov.val)
		self.a11.set_title('fwhm {} lat {} lon {}\nflat {} flon {} sol_ang {} emiss_ang {} azi_ang {} wgeom {}'.format( self.dd['fwhm'], self.dd['latitude'], self.dd['longitude'],
			self.dd['fov_averaging_record'][ngeom][nav,0],
			self.dd['fov_averaging_record'][ngeom][nav,1],
			self.dd['fov_averaging_record'][ngeom][nav,2],
			self.dd['fov_averaging_record'][ngeom][nav,3],
			self.dd['fov_averaging_record'][ngeom][nav,4],
			self.dd['fov_averaging_record'][ngeom][nav,5])
		)
		return

	def create_widgets(self):
		self.buttonResetViewerAx = self.f.add_axes([0.0, 0.0, 0.1, 0.05])# [left, bottom, width, height]
		self.buttonResetViewer = mpl_wdgt.Button(self.buttonResetViewerAx, 'Reset', hovercolor='0.9')
		
		self.sliderSetGeomAx = self.f.add_axes([0.91, 0.15, 0.03, 0.6])
		self.sliderSetGeom = mpl_wdgt.Slider(self.sliderSetGeomAx, 'ngeom', 0, self.dd['ngeom']-1, valinit=0,
											orientation='vertical', valfmt='%d', valstep=1)
			
		self.sliderSetFovAx = self.f.add_axes([0.95, 0.15, 0.03, 0.6])
		self.sliderSetFov = mpl_wdgt.Slider(self.sliderSetFovAx, 'fov', 0, 
											self.dd['navs'][int(self.sliderSetGeom.val)]-1,
											valinit = 0,
											orientation='vertical',
											valfmt='%d',
											valstep=1)

		self.textboxYAxisMinAx = self.f.add_axes([0.05, 0.9, 0.05, 0.05])
		self.textboxYAxisMin = mpl_wdgt.TextBox(self.textboxYAxisMinAx, label='y-axis min',
												initial='auto',
												hovercolor=0.9)
	
		self.textboxYAxisMaxAx = self.f.add_axes([0.05, 0.95, 0.05, 0.05])
		self.textboxYAxisMax = mpl_wdgt.TextBox(self.textboxYAxisMaxAx, label='y-axis max',
												initial='auto',
												hovercolor=0.9)	

		self.buttonResetViewerCID = self.buttonResetViewer.on_clicked(self.buttonResetViewerClicked)
		self.sliderSetGeomCID = self.sliderSetGeom.on_changed(self.sliderSetGeomChanged)
		self.sliderSetFovCID = self.sliderSetFov.on_changed(self.sliderSetFovChanged)
		self.textboxYAxisMinAxCID = self.textboxYAxisMin.on_submit(self.textboxYAxisMinSubmitted)
		self.textboxYAxisMaxAxCID = self.textboxYAxisMax.on_submit(self.textboxYAxisMaxSubmitted)
		return

	def buttonResetViewerClicked(self, event):
		self.dd = copy.deepcopy(self.original_data_dict)
		self.sliderSetGeom.reset()
		self.sliderSetFov.reset()
		self.update_plot()
		self.update_ax_title()
		self.f.canvas.draw()
		return

	def sliderSetGeomChanged(self, val):
		self.update_sliderSetFovLimits()
		self.update_plot()
		self.update_ax_title()
		self.f.canvas.draw()
		return
	
	def sliderSetFovChanged(self, val):
		self.update_ax_title()
		self.f.canvas.draw()
		return

	def textboxYAxisMinSubmitted(self, text):
		if text.lower() == 'auto':
			self.ymin = self.ymin_auto
		else:
			try:
				self.ymin = float(text)
			except ValueError:
				self.textboxYAxisMin.set_val('auto')
				return
		self.update_plot()
		self.f.canvas.draw()
		return

	def textboxYAxisMaxSubmitted(self, text):
		if text.lower() == 'auto':
			self.ymax = self.ymax_auto
		else:
			try:
				self.ymax = float(text)
			except ValueError:
				self.textboxYAxisMax.set_val('auto')
				return
		self.update_plot()
		self.f.canvas.draw()
		return

	def update_sliderSetFovLimits(self):
		ngeom = int(self.sliderSetGeom.val)
		self.sliderSetFov.disconnect(self.sliderSetFovCID)
		self.sliderSetFov =  mpl_wdgt.Slider(self.sliderSetFovAx, 'fov', 0, 
											self.dd['navs'][ngeom]-1,
											valinit = 0,
											orientation='vertical',
											valfmt='%d',
											valstep=1)
		self.sliderSetFovCID = self.sliderSetFov.on_changed(self.sliderSetFovChanged)
		return

class SpxModifier(SpxViewerBase):
	def	create_widgets(self):
		super().create_widgets()
		xyz_ends = np.array([self.dd['spec_record'][0][0], self.dd['spec_record'][0][-1]])
		#self.clickedPoints = self.a11.scatter(xyz_ends[:,0],xyz_ends[:,1],color='tab:red',s=2)
		self.clickedPoints = self.a11.scatter([],[],color='tab:red',s=2)
		self.previewInterpPoints, = self.a11.plot(xyz_ends[:,0],xyz_ends[:,1],color='tab:red', lw=1)
		self.f.canvas.mpl_connect('button_press_event', self.windowClicked)
		
		self.buttonApplySmoothAx = self.f.add_axes([0.1, 0.0, 0.25, 0.05])# [left, bottom, width, height]
		self.buttonApplySmooth = mpl_wdgt.Button(self.buttonApplySmoothAx, 'Apply Smoothing', hovercolor='0.9')
		self.buttonApplySmoothCID = self.buttonApplySmooth.on_clicked(self.buttonApplySmoothClicked)

		self.scaledErrFac = 1.0
		self.scaledErr = self.a11.fill_between([],[],[],color='tab:red',alpha=0.5)
		self.textErrFacAx = self.f.add_axes([0.5, 0.05, 0.15, 0.05])# [left, bottom, width, height]
		self.textErrFac = mpl_wdgt.TextBox(self.textErrFacAx, label='Error Factor', initial='1.0',
						hovercolor='0.9')
		self.textErrFacCID = self.textErrFac.on_submit(self.textErrFacSubmitted)

		self.buttonApplyErrFacAx = self.f.add_axes([0.5, 0.0, 0.15, 0.05])# [left, bottom, width, height]
		self.buttonApplyErrFac = mpl_wdgt.Button(self.buttonApplyErrFacAx, 'Apply Err Fac', hovercolor='0.9')
		self.buttonApplyErrFacCID = self.buttonApplyErrFac.on_clicked(self.buttonApplyErrFacClicked)

		self.textSmoothChisqAx = self.f.add_axes([0.85, 0.05, 0.15, 0.05])# [left, bottom, width, height]
		self.textSmoothChisq = mpl_wdgt.TextBox(self.textSmoothChisqAx, label='Smooth Chisq/n', initial='0.0',
						hovercolor='0.9')
		
		self.textNPointsAx = self.f.add_axes([0.85, 0.0, 0.15, 0.05])# [left, bottom, width, height]
		self.textNPoints = mpl_wdgt.TextBox(self.textNPointsAx, label='Number of points', initial='0.0',
						hovercolor='0.9')

		self.buttonScaleErrOpacAx = self.f.add_axes([0.0, 0.05, 0.3, 0.05])# [left, bottom, width, height]
		self.buttonScaleErrOpac = mpl_wdgt.Button(self.buttonScaleErrOpacAx, 'Scale Error by Opacity', 
												hovercolor='0.9')
		self.buttonScaleErrOpacCID = self.buttonScaleErrOpac.on_clicked(self.buttonScaleErrOpacClicked)

		return

	def buttonScaleErrOpacClicked(self, event):
		opac_data = np.loadtxt(os.path.expanduser('~/Documents/reference_data/telluric_features/mauna_kea/sky_transmission/mktrans_zm_30_15.dat'))
		ngeom = int(self.sliderSetGeom.val)
		xyz = self.dd['spec_record'][ngeom]
		opac_data_interp = np.interp(xyz[:,0], opac_data[:,0], opac_data[:,1])
		opac_data_interp[np.where(opac_data_interp < 1E-10)] = 1E-10
		self.dd['spec_record'][ngeom][:,2] *= 1.0 / opac_data_interp
		self.update_plot()
		self.f.canvas.draw()
		return

	def get_modified_data(self):
		for i in range(self.dd['ngeom']):
			xyz = self.dd['spec_record'][i]
			self.dd['nconvs'][i] = xyz.shape[0]
		return(self.dd)

	def buttonApplyErrFacClicked(self,event):
		ngeom = int(self.sliderSetGeom.val)
		self.dd['spec_record'][ngeom][:,2] *= self.scaledErrFac
		self.scaledErrFac = 1.0
		self.textErrFac.set_val('1.0')
		self.update_scaledErr()
		self.update_plot()
		self.f.canvas.draw()
		return

	def update_scaledErr(self):
		self.scaledErr.remove()
		if self.scaledErrFac==1:
			self.scaledErr = self.a11.fill_between([],[],[],color='tab:red',alpha=0.5)
		else:
			ngeom = int(self.sliderSetGeom.val)
			xyz = self.dd['spec_record'][ngeom]
			self.scaledErr = self.a11.fill_between(
								xyz[:,0],
								xyz[:,1] + self.scaledErrFac*xyz[:,2], 
								xyz[:,1] - self.scaledErrFac*xyz[:,2],
								color='tab:red',
								alpha=0.5)
		return

	def textErrFacSubmitted(self, text):
		self.scaledErrFac = float(text)
		self.update_scaledErr()
		self.f.canvas.draw()
		return

	def update_plot(self):
		super().update_plot()
		ngeom = int(self.sliderSetGeom.val)
		xyz = self.dd['spec_record'][ngeom]
		offsets = self.clickedPoints.get_offsets()
		if offsets.shape[0] > 0:
			offsets[:,1] = np.interp(offsets[:,0], xyz[:,0], xyz[:,1])
			offsets = offsets[np.argsort(offsets[:,0])]
			self.clickedPoints.set_offsets(offsets)
			self.previewInterpPoints.set_data(offsets[:,0], offsets[:,1])
			back_interp = np.interp(xyz[:,0], offsets[:,0], offsets[:,1])
			#print(back_interp)
			#print(xyz[:,1])
			chisq_per_dof = np.sum(((xyz[:,1] - back_interp)**2/back_interp))
			#chisq_per_dof = np.sum((xyz[:,1] - back_interp)**2)/np.max(xyz[:,1])
			#print(chisq_per_dof)
			self.textSmoothChisq.set_val('{:0.2e}'.format(chisq_per_dof))
			self.textNPoints.set_val('{}'.format(offsets.shape[0]))
		return
		

	def windowClicked(self, event):
		if event.inaxes != self.a11:
			return
		offsets = self.clickedPoints.get_offsets()
		apoint = np.array(((event.xdata, event.ydata),))
		offsets = np.concatenate([offsets, apoint],axis=0)
		self.clickedPoints.set_offsets(offsets)
		self.update_plot()
		self.f.canvas.draw()
		return

	def buttonApplySmoothClicked(self,event):
		offsets = self.clickedPoints.get_offsets()
		ngeom = int(self.sliderSetGeom.val)
		xyz = self.dd['spec_record'][ngeom]
		vye = np.zeros((offsets.shape[0],3))
		vye[:,0] = offsets[:,0]
		vye[:,1] = offsets[:,1]
		vye[:,2] = np.interp(vye[:,0], xyz[:,0], xyz[:,2])
		self.dd['spec_record'][ngeom] = vye
		self.update_plot()
		self.f.canvas.draw()
		return

	def buttonResetViewerClicked(self, event):
		self.scaledErrFac = 1
		self.update_scaledErr()
		xyz_ends = np.array([self.dd['spec_record'][0][0], self.dd['spec_record'][0][-1]])
		self.clickedPoints.set_offsets(xyz_ends[:,(0,1)])
		self.previewInterpPoints.set_data(xyz_ends[:,0], xyz_ends[:,1])
		
		super().buttonResetViewerClicked(event)
		return

def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	
	for nf in args['nem_files']:
		ut.pINFO('file {}'.format(nf))
		fname, fext = os.path.splitext(nf) # splits filename '.../something.ext' into '.../something' and '.ext'
		if fext == '.spx':
			ftn_list, ftn_list_desc = file_modifiers[fext][args['spx_modifier']]
			ut.pINFO('Operating on file {} with modifier functions {}. This modifer: {}'.format(nf, args['spx_modifier'], ftn_list_desc))
			data_dict = nemesis.read.spx(nf) # read file information into dictionary
			ut.pINFO('Data read from file {}'.format(nf))
			kwargs = {} # does nothing for now but can build in some forwards compatibility
			for ftn in ftn_list:
				data_dict, kwargs = ftn(data_dict, **kwargs)

			if not args['do_not_save']:
				if args['overwrite']:
					outfname = nf
				else:
					outfname = ''.join([fname,args['suffix'],fext])
				ut.pINFO('Writing data to file {} ...'.format(outfname))
				nemesis.write.spx(data_dict, outfname) 
				ut.pINFO('... data written.')
			else:
				ut.pINFO('Argument "do_not_save" passed, no data modified...')
	

	return


def spx_viewer(spx_dict, **kwargs):
	ut.pINFO('Inside function "spx_viewer"')
	spx_viewer = SpxViewerBase(spx_dict)
	spx_viewer.run()
	return(spx_dict, kwargs)

def spx_manual_subsample(spx_dict, **kwargs):
	ut.pINFO('Inside function "spx_manual_subsample')
	spx_viewer = SpxModifier(spx_dict)
	spx_viewer.run()
	# here we should modify spx_dict depending on what happened in spx_viewer
	# probably just need to do
	mod_spx_dict = spx_viewer.get_modified_data()
	return(mod_spx_dict, kwargs)


file_modifiers = {
	'.spx':{
		'spx_manual_subsample':(
			[spx_manual_subsample], 
			'Opens a dialog so the user can manually subsample the spectra in the *.spx file'
		),
		'spx_viewer':(
			[spx_viewer],
			'Opens and views an *.spx file, does not modify it in any way'
		)
	}
}

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

	parser.add_argument('nem_files', type=str, nargs='+', help='List of Nemesis files to modify, how they are interpreted is based upon their extension', default=[])
	parser.add_argument('--spx_modifier', type=str, choices=file_modifiers['.spx'].keys(), help='The function list used to modify the *.spx file, each function in the list will be applied sequentially.')
	parser.add_argument('--overwrite', action='store_true', help='If present will overwrite the modified files')
	parser.add_argument('--do_not_save', action='store_true', help='If present, no changes will be written anywhere')
	parser.add_argument('--suffix', type=str, help='Suffix to append to the modified files (ignored if "--overwrite" is passed)', default='_mod')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
