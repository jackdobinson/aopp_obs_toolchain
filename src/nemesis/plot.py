#!/usr/bin/python3
"""
Provides routines for plotting NEMESIS files, when run as a standalone script, enables showing and/or saving of plots.
When imported gives access to plotting functions for each type of data file.
"""
import os, sys
import copy
import utils as ut
import numpy as np	
import numpy.ma as ma
import math as m
import nemesis.common as nc
import nemesis.read
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.animation
import nemesis.cfg
import nemesis.exceptions
import astropy.wcs as wcs
import astropy.io.fits as fits
import regions
from regions import PixCoord, CirclePixelRegion, RectanglePixelRegion
import textwrap
import plotutils

def main(argv):
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	args_pruned = args.copy()
	for k in ('runnames', 'show_plots'):
		args_pruned.pop(k, None)

	if len(args['plot']) + len(args['ensemble_plot']) == 0:
		ut.pINFO("You have not passed any plots groups to '--plot' or '--ensemble_plot'. Therefore nothing will happen. To plot all graphs, just pass 'all' to both of the arguments")

	# Plotting data from multiple single runs
	for runname in args['runnames']:
		# Find output directory name and create it if it does not exits.
		if not args['no_save_plots']:
			runname_plot_output_folder = os.path.normpath(os.path.join(os.path.dirname(runname),args['plot_output_folder']))
			if os.path.exists(runname_plot_output_folder):
				if not os.path.isdir(runname_plot_output_folder):
					ut.pERROR('Path {} exists, but is not a directory, exiting...'.format(runname_plot_output_folder))
					sys.exit()
			else:
				ut.pINFO('Path {} does not exist, creating...'.format(runname_plot_output_folder))
				os.makedirs(runname_plot_output_folder)
		else:
			runname_plot_output_folder = None

		# Set a load of defaults because some of matplotlib's are strange.
		ut.plot_defaults()

		# Create function call list
		plot_func_call_list = []
		for _x in args['plot']:
			plot_func_call_list += plot_arg_choice_groups[_x]
		plot_func_call_list = list(set(plot_func_call_list)) # get unique members of list, order does not matter

		figures = []
		for plot_func in plot_func_call_list:
			figs = plot_func(runname, outfolder=runname_plot_output_folder, show_plot=args['show_plots'], **args_pruned)
			figures += figs
		# Do something with the figures here if we need to


		# Close all figures as we don't need them any more
		plt.close('all')

	# plotting collated data from multiple runs
	if not args['no_save_plots']:
		ensemble_output_folder = os.path.normpath(args['ensemble_output_folder'])
		if os.path.exists(ensemble_output_folder):
			if not os.path.isdir(ensemble_output_folder):
				ut.pERROR("Path {} exists, but is not a directory, exiting...".format(ensemble_output_folder))
				sys.exit()
		else:
			ut.pINFO('Path {} does not exists, creating...'.format(ensemble_output_folder))
			os.makedirs(ensemble_output_folder)
	else:
		ensemble_output_folder = None

	ensemble_func_call_list = []
	for _x in args['ensemble_plot']:
		ensemble_func_call_list += ensemble_arg_choice_groups[_x]
	ensemble_func_call_list = list(set(ensemble_func_call_list))

	efigs = []
	for efunc in ensemble_func_call_list:
		figs = efunc(args['runnames'], outfolder=ensemble_output_folder, show_plot=args['show_plots'], **args_pruned)
		efigs += figs

	# Do something with figures here if needed

	plt.close('all')
	

	# plot region files
	rfigs = ensemble_region(args['regionfiles'], args['runnames'], outfolder=ensemble_output_folder, show_plot=args['show_plots'])
	# do something with figures if needed here
	plt.close('all')
	

	return
		



def mre_spectral(runname, outfolder=None, show_plot=False, log_scale=True, **kwargs):
	"""
	Creates plots based on the data in the <runname>.mre file of a NEMESIS run
	runname - The name of the nemesis run
	outfolder - folder to output plot to
	show_plots - should we show an interactive version of the plot?
	"""
	# DO STUFF
	ut.pINFO('Plotting spectral data from *.mre file...')

	try:
		mre_data = nemesis.read.mre(runname)
	except nemesis.exceptions.NemesisReadError as nre:
		print(nre)
		ut.pWARN('Could not read *.mre file from {}, skipping'.format(runname))
		return([None])

	figs = []
	
	for i, (wl, rm, rr, re, rpd) in enumerate(zip(	mre_data['waveln'], 
													mre_data['radiance_meas'], 
													mre_data['radiance_retr'],
													mre_data['radiance_err'], 
													mre_data['radiance_perc_diff'])):

		fig1 = plt.figure(figsize=[x/2.54 for x in (16,24)])
		# create spectral plot
		a11 = fig1.add_subplot(2,1,1)
		"""
		a11.step(wl, rm, color='tab:blue', label='measurement', where='mid')
		a11_lim1 = a11.get_ylim()
		a11.step(wl, rr, color='tab:orange', label='retrieval', where='mid')
		a11_lim2 = a11.get_ylim()
		# create error bars for spectral plot
		a11.fill_between(wl, rm+re, rm-re, color='tab:blue', label='measurement error', step='mid', alpha=0.5)
		"""
		#spec_plt_x = np.array([wl,wl])
		#spec_plt_y = np.array([rm,rr])
		if log_scale:
			plot_spec_err(np.array([wl,wl]), np.array([rm,rr]), e=[re,None], fig=fig1, ax=a11, ylabel='log(Radiance)\nlog(uW cm^{-2} sr^{-1} cm)', data_label=['Measurement', 'Retrieved'], yscale='log')
		else:
			plot_spec_err(np.array([wl,wl]), np.array([rm,rr]), e=[re,None], fig=fig1, ax=a11, ylabel='Radiance\nuW cm^{-2} sr^{-1} cm', data_label=['Measurement', 'Retrieved'])

		a21 = fig1.add_subplot(2,1,2)
		a22 = a21.twinx()
		l21 = a21.step(wl,rm-rr,color='tab:blue', label='Residual', where='mid')
		l22 = a22.step(wl, rpd, color='tab:orange', label='% Difference', where='mid')
		l2 = l21+l22
		a21.set_xlabel('Wavelength (um)')
		a21.set_ylabel('Measured-Retrieved Radiance')
		a22.set_ylabel('Measured-Retrieved Percent')
		lbls = [mkr.get_label() for mkr in l2]
		a22.legend(l2,lbls,loc=0)
		fig1.suptitle(f'Geometry {i}')
		fig1.tight_layout()
		save_show_plt(fig1, f'mre_{i}.png', outfolder=outfolder, show_plot=show_plot)
		figs.append(fig1)

	return(figs)

def plot_spec_err(x, y, e=None, fig=None, ax=None, title=None, xlabel='Wavelength (um)', ylabel='Flux (W m^-2)', idx_order='C', data_label=None, yscale='linear'):
	"""Assume all data is passed as numpy arrays or can operate in a similar way. See the end of the file for an example of use"""
	if not(idx_order.upper() in ('C', 'FORTRAN', 'ROW', 'COLUMN', 'F')):
		pERROR('Must specify either row-mayor (C-style) or column-major (Fortran-style) index ordering')
		sys.exit()
	if idx_order.upper() in ('FORTRAN', 'COLUMN', 'F'):
		# we want to use ROW-major ordering so just transpose everything
		x = np.transpose(x)
		y = np.transpose(y)
		e = np.transpose(e)
	
	if x.shape != y.shape:
		pERROR('x and y data must have the same shape. Currently they have shapes {} and {} respectively'.format(x.shape, y.shape))
		sys.exit()
	
	if fig==None:
		fig = plt.figure()
	if ax == None:
		ax = fig.gca()

	if len(x.shape) ==1:
		x = np.array([x])
		y = np.array([y])
		e = np.array([e])
		data_label = np.array([data_label])
	elif len(x.shape) == 2:
		if type(e)==type(None):
			e = np.full([x.shape[0],1], np.nan)
		if type(data_label)==type(None):
			data_label = np.full([x.shape[0]], data_label)
	else:
		pERROR('Data must have 1 or 2 dimensions')
		sys.exit()

	colours = list(mplc.TABLEAU_COLORS.keys())
	#print(x.shape)
	#print(y.shape)
	artists = []
	for i in range(x.shape[0]):
		if type(e[i]) != type(None):
			if len(e[i].shape)==1:
				ei = np.array([e[i],-e[i]])
			else:
				ei = e[i]
			ei0, ei1 = y[i]+ei[0], y[0]+ei[1]
			artists.append(ax.fill_between(x[i], ei0, ei1, color=colours[i], alpha=0.5, step='mid'))
		artists.append(ax.step(x[i], y[i], color=colours[i], where='mid', label=data_label[i]))

	if yscale in ('log'):
		ymin, ymax = np.exp(np.nanmin(np.log(y[np.isfinite(y)]))), np.exp(np.nanmax(np.log(y[np.isfinite(y)]))) # stop silly limits when using log scale
	else:	
		ymin, ymax = np.nanmin(y[np.isfinite(y)]), np.nanmax(y[np.isfinite(y)])
	ax.set_ylim(ymin,ymax)

	ax.set_yscale(yscale)

	if title!=None:	ax.set_title(title)
	if xlabel!=None: ax.set_xlabel(xlabel)
	if ylabel!=None: ax.set_ylabel(ylabel)
	if not all([dl==None for dl in data_label]): ax.legend()
	return(fig, ax, artists)

def mre_parameter(runname, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots the a-priori and retrieved parameters from data in the <runname>.mre file.

	Relies upon functions defined in "nemeis/cfg.py" to go from a parametric representation of a variable
	to a profile of the variable in pressure, height, or wavelength.
	Also relies upon "varident_to_labels_and_profiles" to hide all the details of what type of variable is
	represented by a 3 digit ID code, "varident_to_labels_and_profiles" calls the functions in "nemesis/cfg.py"
	"""
	
	ut.pINFO('Plotting parameter data from *.mre file...')
	try:
		mre_data = nemesis.read.mre(runname)
	except nemesis.exceptions.NemesisReadError as nre:
		print(nre)
		ut.pWARN('Could not read *.mre file from {}, skipping'.format(runname))
		return([None])

	# gives the smallest square that 'nvar' (the total number of variables) will fit into
	n = int(m.ceil(m.sqrt(mre_data['nvar'])))
	#print(f'INFO: n {n}')
	
	fig2 = plt.figure(figsize=[x/2.54 for x in (12*n,12*n)])
	a_arr = fig2.subplots(nrows=n, ncols=n, squeeze=False, gridspec_kw={'wspace':0.25, 'hspace':0.25})
	#print(f'INFO: a_arr {a_arr}')	

	for i, (vid, vpar, ap, ae, rp, re, nv) in enumerate(zip(mre_data['varident'],
															mre_data['varparam'],
															mre_data['aprprof'],
															mre_data['aprerr'],
															mre_data['retprof'],
															mre_data['reterr'],
															mre_data['nxvar'])):
		ax = a_arr[i//n][i%n]
		vpar_str = '[{:.2} {:.2} {:.2} {:.2} {:.2}]'.format(*vpar)


		# get the information for apriori and retrieved values
		ap_var_info = varident_to_labels_and_profiles(runname, vid, vpar, ap[:nv], ae[:nv])
		re_var_info = varident_to_labels_and_profiles(runname, vid, vpar, rp[:nv], re[:nv])
	
		#print('### {} Vs. {} ###'.format(ap_var_info['independent_label'], ap_var_info['dependent_label']))
		#print(vid)
		#print(vpar_str)
		#print(nv)
		#print(ap_var_info)
		#print(ap_var_info['independent_var'].shape)
		#print(ap_var_info['dependent_var'].shape)
		#print(ap_var_info['dependent_err'].shape)

		# we want to use a line with a filled region to represent value and error, but some of our variables
		# are best plotted with the independent variable on the y axis not the x axis, so swap the plotting
		# function if that is the case.
		if ap_var_info['independent_label'] in ['Pressure', 'Height']:
			fill_between_ftn = ax.fill_betweenx
		else:
			fill_between_ftn = ax.fill_between
		fill_between_ftn(ap_var_info['independent_var'], 
						ap_var_info['dependent_var']+ap_var_info['dependent_err'],
						ap_var_info['dependent_var']-ap_var_info['dependent_err'],
						color='tab:blue',
						label='Apriori Error', step='mid', alpha=0.5)
		#print(re_var_info)
		#print(re_var_info['independent_var'].shape)
		#print(re_var_info['dependent_var'].shape)
		#print(re_var_info['dependent_err'].shape)

		fill_between_ftn(re_var_info['independent_var'], 
						re_var_info['dependent_var']+re_var_info['dependent_err'],
						re_var_info['dependent_var']-re_var_info['dependent_err'],
						color='tab:orange',
						label='Retrieved Error', step='mid', alpha=0.5)
	


		# again, some of our variables are best plotted with the dependent variable on the x axis, so make two cases for this
		if ap_var_info['independent_label'] in ['Pressure', 'Height']:
			ystep_plot(ax, ap_var_info['dependent_var'], ap_var_info['independent_var'], color='tab:blue', label='Apriori Value', where='pre')
			ystep_plot(ax, re_var_info['dependent_var'], re_var_info['independent_var'], color='tab:orange', label='Retrieved Value', where='pre')

			ax.invert_yaxis()
			ax.set_ylabel(re_var_info['independent_label'])
			ax.set_xlabel(re_var_info['dependent_label'])
			#if ap_var_info['independent_label'] in ['Pressure']:
			#	ax.set_xscale('log')
			#	ax.set_yscale('log')
		else:
			ax.step(ap_var_info['independent_var'], ap_var_info['dependent_var'],
					color='tab:blue', label='Apriori Value', where='mid')
			ax.step(re_var_info['independent_var'], re_var_info['dependent_var'],
					color='tab:orange', label='Retrieved Value', where='mid')
			ax.set_xlabel(re_var_info['independent_label'])
			ax.set_ylabel(re_var_info['dependent_label'])
	
		# some of our variables should be plotted with log scales and others with linear scales, so make a few special cases
		# do all of our axis scaling here because otherwise we would have to re-write a lot of the code above.
		if ap_var_info['independent_label'] in ['Pressure']:
			# plot pressure in log-log scaling
			ax.set_xscale('log')
			ax.set_yscale('log')
		elif ('Imaginary part of' in ap_var_info['dependent_label']) and ('refractive index' in ap_var_info['dependent_label']):
			# Imaginary refractive index should be plotted on a log-linear scale vs wavelength
			ax.set_xscale('linear')
			ax.set_yscale('log')
		else: # fall back on linear scales
			ax.set_xscale('linear')
			ax.set_yscale('linear')

		ax.legend()
		
	# Remove unused plot axes
	for j in range(i+1, n*n):
		a_arr[j//n][j%n].remove()

	fig2.tight_layout

	save_show_plt(fig2, 'mre_ret_v_apr.png', outfolder=outfolder, show_plot=show_plot)

	return([fig2]) # return figure so we can use this programatically later

def chi_sq(runname, outfolder=None, show_plot=False, **kwargs):
	ut.pINFO('Plotting chi-squared values from *.itr file')
	try:
		itrd = nemesis.read.itr(runname) # may as well use the cache, if we ever put this in a loop will have to disable/pass as a variable
	except nemesis.exceptions.NemesisReadError:
		ut.pWARN("Could not read *.itr file from {}. Skipping...".format(runname))
		return([None])
	
	f1 = plt.figure(figsize=[x/2.54 for x in (12,12)])
	a11 = f1.add_subplot(1,1,1)

	itr_idx = np.array(list(range(itrd['niter'])))
	a11.plot(itr_idx, itrd['chisq_arr']/itrd['nx'], color='tab:blue', label='chi-squared/dof')
	a11.plot(itr_idx, itrd['phi_arr']/itrd['nx'], color='tab:orange', label='phi-squared/dof')

	a11.set_yscale('log')
	a11.set_xlabel('Iteration')
	a11.set_ylabel('Cost Function/Degrees of Freedom')
	a11.legend()

	save_show_plt(f1, 'chi_phi.png', outfolder=outfolder, show_plot=show_plot)
	return([f1])

def improvement_factor(runname, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots the "improvement factor" 

		I_f = 1 - (re/r)/(ae/a)

	where
		re = retrieved error
		r = retrieved value
		ae = apriori error
		a = apriori value

	Of all the parameters in a *.mre file.
	"""
	try:
		mre_data = nemesis.read.mre(runname)
	except nemesis.exceptions.NemesisReadError as nre:
		print(nre)
		ut.pWARN('Could not read *.mre file from {}, skipping'.format(runname))
		return([None])

	n = int(m.ceil(m.sqrt(mre_data['nvar'])))
	f1 = plt.figure(figsize=[x/2.54 for x in (12*n,12*n)])

	a_arr = f1.subplots(nrows=n, ncols=n, squeeze=False, gridspec_kw={'wspace':0.25, 'hspace':0.25})
	for i, (vid, vpar, ap, ae, rp, re, nv) in enumerate(zip(mre_data['varident'],
															mre_data['varparam'],
															mre_data['aprprof'],
															mre_data['aprerr'],
															mre_data['retprof'],
															mre_data['reterr'],
															mre_data['nxvar'])):
		ax = a_arr[i//n][i%n]
		vpar_str = '[{:.2} {:.2} {:.2} {:.2} {:.2}]'.format(*vpar)


		# get the information for apriori and retrieved values
		ap_var_info = varident_to_labels_and_profiles(runname, vid, vpar, ap[:nv], ae[:nv])
		re_var_info = varident_to_labels_and_profiles(runname, vid, vpar, rp[:nv], re[:nv])

		imp_fac = get_improvement_factor(re_var_info, ap_var_info) #1.0 - (re_var_info['dependent_err']/re_var_info['dependent_var'])/(ap_var_info['dependent_err']/ap_var_info['dependent_var'])

		if ap_var_info['independent_label'] in ['Pressure', 'Height']:
			ystep_plot(ax, imp_fac, ap_var_info['independent_var'], label='Improvement Factor')
			ax.invert_yaxis()
			ax.set_ylabel(re_var_info['independent_label'])
			ax.set_xlabel(re_var_info['dependent_label'])
			if ap_var_info['independent_label'] in ['Pressure']:
				ax.set_xscale('log')
				ax.set_yscale('log')
		else:
			ax.step(re_var_info['independent_var'], imp_fac, label='Improvement Factor', where='mid')
			ax.set_xlabel(re_var_info['independent_label'])
			ax.set_ylabel(re_var_info['dependent_label'])

		ax.legend()

	f1.suptitle('I_f = 1 - (re/r)/(ae/a)')

	# Remove unused plot axes
	for j in range(i+1, n*n):
		a_arr[j//n][j%n].remove()

	f1.tight_layout

	save_show_plt(f1, "improvement_factor.png", outfolder=outfolder, show_plot=show_plot)

	return([f1])

def limb_darkening(runname, outfolder=None, show_plot=False, **kwargs):
	"""
	Compares limb darkening at continuum (~1.55um) with absorbtion (>~1.6um)
	See *.mre file, use the *.spx file to get the logitudes, plot cos(zenith) vs radiance for each wavelength
	Check it against minneart limb darkening
	pcm_flag
		If True will use "pcolormesh" to create a map with non-regualar pixel sizes to account for a
		non regular spacing in the x and y directions. If False, will use regularly sized pixels to
		show data but will label them non-linearly to account for non-regular spacing of data.
	"""
	ut.pINFO('Plotting limb darkening, using *.mre and *.spx file...')
	try:
		mre_data = nemesis.read.mre(runname)
		spx_data = nemesis.read.spx(runname)
	except nemesis.exceptions.NemesisReadError as nre:
		print(nre)
		ut.pWARN('Something went wrong when reading *.mre or *.spx file from {}, skipping...'.format(runname))
		return([None])


	# for each nav (of which there are ngeom of them) in the spx file, there is nconv wavelengths to plot
	# want to get wavs, radiance vs cos(zenith)

	data = np.zeros((spx_data['ngeom'], spx_data['nconvs'][0], 5)) # need room for wavelength, measured, measured error, model, model error,
	for i in range(spx_data['ngeom']):
		data[i,:,:3] = np.array([1,1E6,1E6])*spx_data['spec_record'][i] # *.mre files are in uW cm-2 sr-1 um-1, and *.spx are in W cm-2 sr-1 um-1. So multiply by 1E6 to keep in same units
		data[i,:,3] = mre_data['radiance_retr'][i]
		data[i,:,4] = mre_data['radiance_err'][i]
	cossol = np.zeros((spx_data['ngeom']))
	coszen = np.zeros((spx_data['ngeom']))
	for i, far in enumerate(spx_data['fov_averaging_record']):
		coszen[i] = np.cos(far[0][3]*np.pi/180) # zenith angle
		cossol[i] = np.cos(far[0][2]*np.pi/180) # solar angle

	#print(data)
	#print(coszen)
	#print(cossol)
	
	# ORGANIZE DATA
	#datslice=np.s_[:,:,3]
	#errslice=np.s_[:,:,4]

	nx, ny = (4,2)
	def make_fa(nx,ny):
		f1 = plt.figure(figsize=[x/2.54 for x in (12*nx, 12*ny)])
		a1 = f1.subplots(nrows=ny, ncols=nx, squeeze=False, gridspec_kw={'wspace':0.25, 'hspace':0.25})
		return(f1,a1)

	f1,a1 = make_fa(nx,ny)
	f2,a2 = make_fa(nx,ny)

	wavs = data[0,:,0]
	model_fit = doplot_limbdark_set(f1, a1, cossol, coszen, wavs, data, np.s_[:,:,3], np.s_[:,:,4], 
									errorbar_flag=True, pcm_flag=True) # model
	meas_fit = doplot_limbdark_set(f2, a2, cossol, coszen, wavs, data, np.s_[:,:,1], np.s_[:,:,2],
									errorbar_flag=True, pcm_flag=True) # measurement
	#doplot_limbdark_set(f2, a2, wavs, data, np.s_[:,:,1], np.s_[:,:,2])

	f1.suptitle('Retrieved Model Data')
	f2.suptitle('Measured Data')

	save_show_plt(f1, 'limb_darkening_model.png', outfolder=outfolder, show_plot=show_plot)
	save_show_plt(f2, 'limb_darkening_meas.png', outfolder=outfolder, show_plot=show_plot)


	return([f1, f2])

def doplot_limbdark_set(f1, a1, cossol, coszen, wavs, data, datslice, errslice, errorbar_flag=True, pcm_flag=True):
	IperF = data[datslice]
	IperFabserr = data[errslice]
	IperFpererr = IperFabserr/IperF
	u0 = cossol
	u = coszen
	log_u0u = np.log(u0*u)
	log_IperF = np.log(IperF)
	log_IperFabserr = IperFpererr*log_IperF
	log_uIperF = np.log(u[:,None]*IperF)
	log_uIperFabserr = IperFpererr*log_uIperF

	fit = np.zeros_like(log_uIperF)
	xs = np.zeros((log_uIperF.shape[1],2))
	rs = [None]*log_uIperF.shape[1] # np.zeros(log_uIperF.shape[1])
	for i in range(log_uIperF.shape[1]):
		x,r,rank,s,a,aa,aaa = fit_minnaert_limb_darkening(u0,u,data[:,i,datslice[-1]])
		fit[:,i] = np.matmul(np.vstack([np.ones_like(log_u0u), log_u0u]).T, x)
		xs[i] = x
		rs[i] = r
	#rs = np.vstack(rs)
	#print(xs)
	#print(rs)
	# plot raw data
	a1[0,0].plot(u, IperF) # plot all of the curves on the same axis
	#normdata = copy.copy(IperF)
	#meas_rad_max = normdata[0,:,0]
	#model_rad_max = normdata[0,:,2]
	#print(model_rad_max)
	#print(normdata.shape, model_rad_max.shape, meas_rad_max.shape)
	#normdata[:,:,2] /= model_rad_max[None,:]
	#normdata[:,:,0] /= meas_rad_max[None,:]
	a1[0,0].set_prop_cycle(None)
	a1[0,0].plot(coszen, IperF)
	plotutils.flip_x_axis(a1[0,0])
	a1[0,0].set_xlabel('cos(zenith)')
	a1[0,0].set_title('raw radiance')


	datasets = [np.s_[:,0], np.s_[:,-1]] # plot first and last
	#datasets = [np.s_[:,i] for i in range(IperF.shape[1])]
	a1[0,1].set_prop_cycle(None)
	#a1[0,1].plot(log_u0u, np.vstack([log_uIperF[dataset], log_IperF[dataset]]).T, ms=1, lw=0.5, marker='.')
	a1[0,1].set_prop_cycle(None)
	for ds in datasets:
		a1[0,1].plot(log_u0u, log_uIperF[ds], ms=1, lw=0.5, marker='.', ls='-')
	a1[0,1].set_prop_cycle(None)
	for ds in datasets:
		a1[0,1].plot(log_u0u, log_IperF[ds], ms=1, lw=0.5, marker='.', ls='--')
	plotutils.flip_x_axis(a1[0,1])
	a1_01_lims = plotutils.save_lims(a1[0,1])
	if errorbar_flag:
		a1[0,1].set_prop_cycle(None)
		for ds in datasets:
			eb11=a1[0,1].errorbar(log_u0u, log_uIperF[ds], yerr=log_uIperFabserr[ds], 
									ls='None', marker='None', elinewidth=0.5)
			#print(eb11)
			#print([i for i in eb11])
			eb11[-1][0].set_linestyle('-') # this is the line collection object of the plot
			erralpha = min([1,0.1*np.sum(1.0/IperFpererr[ds])/len(IperFpererr[ds])])
			eb11[-1][0].set_alpha(erralpha**3)
		a1[0,1].set_prop_cycle(None)
		for ds in datasets:
			eb12=a1[0,1].errorbar(log_u0u, log_IperF[ds], yerr=log_IperFabserr[ds], 
									ls='None', marker='None', elinewidth=0.5)
			eb12[-1][0].set_linestyle('--')
			erralpha = min([1.0,0.1*np.sum(1.0/IperFpererr[ds])/len(IperFpererr[ds])])
			eb12[-1][0].set_alpha(erralpha**3)
			#print(f'erralpha {erralpha}')
	a1[0,1].set_prop_cycle(None)
	for ds in datasets:
		a1[0,1].plot(log_u0u, fit[ds], lw=1)
	a1[0,1].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[0,1].set_ylabel('log(cos(zenith)*I/F)')
	a1[0,1].set_title(f'wavelengths = {[wavs[ds[-1]] for ds in datasets]}')
	plotutils.set_lims(a1[0,1], a1_01_lims)
	#print(log_u0u)
	#print(u)
	#print(IperF[:,0])
	#print(log_IperF[:,0])
	#print(log_uIperF[:,0])
	
	c = np.ones_like(log_uIperF)*np.array(range(log_uIperF.shape[1]))[None,:]
	#print(c)
	#print(c.shape)
	#a1[0,2].scatter(np.ones_like(log_uIperF)*log_u0u[:,None], log_uIperF, s=1, c=c)
	a1[0,2].set_prop_cycle(None)
	a1[0,2].plot(log_u0u, log_uIperF, ms=1, lw=0.5, marker='.')
	a1[0,2].set_prop_cycle(None)
	a1[0,2].plot(log_u0u, fit, lw=1)
	plotutils.flip_x_axis(a1[0,2])
	a1_02_lims = plotutils.save_lims(a1[0,2])
	#plot error bars
	if errorbar_flag:
		a1[0,2].set_prop_cycle(None)
		for i in range(log_uIperF.shape[1]):
			eb21=a1[0,2].errorbar(log_u0u, log_uIperF[:,i], yerr=log_uIperFabserr[:,i], 
									ls='None', marker='None', elinewidth=0.5)
			eb21[-1][0].set_linestyle('-')
			erralpha = min([1.0,0.1*np.sum(1.0/IperFpererr[:,i])/len(IperFpererr[:,i])])
			eb21[-1][0].set_alpha(erralpha**3)
	plotutils.set_lims(a1[0,2], a1_02_lims)
	a1[0,2].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[0,2].set_ylabel('log(cos(zenith)*I/F)')
	a1[0,2].set_title('data with minnaert fit')

	#a1[0,3].scatter(np.ones_like(log_uIperF)*log_u0u[:,None], log_uIperF-fit, s=1, c=c)
	a1[0,3].set_prop_cycle(None)
	resid = log_uIperF - fit
	a1[0,3].plot(log_u0u, resid, ms=1, lw=0.5, marker='.')
	a1_03_lims = plotutils.save_lims(a1[0,3])
	#plotutils.flip_x_axis(a1[0,3])
	# plot error bars
	if errorbar_flag:
		a1[0,3].set_prop_cycle(None)
		for i in range(resid.shape[1]):
			eb21=a1[0,3].errorbar(log_u0u, resid[:,i], yerr=log_uIperFabserr[:,i], 
									ls='None', marker='None', elinewidth=0.5)
			eb21[-1][0].set_linestyle('-')
			erralpha = min([1.0,0.1*np.sum(1.0/IperFpererr[:,i])/len(IperFpererr[:,i])])
			eb21[-1][0].set_alpha(erralpha**3)
	plotutils.set_lims(a1[0,3], a1_03_lims)
	a1[0,3].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[0,3].set_ylabel('log(cos(zenith)*I/F)')
	a1[0,3].set_title('residual (data - fit)')

	xlim_raw = a1[0,0].get_xlim()
	xlim = a1[0,3].get_xlim()

	#print('HERE')
	
	if not pcm_flag:
		im0 = a1[1,0].imshow(np.swapaxes(IperF, 0,1), origin='lower', 
							extent=(u[0], u[-1], wavs[0],wavs[0]),
							aspect='auto') 
		plotutils.set_imshow_ticks_and_labels(im0, a1[1,0], IperF.shape, (u, wavs))
	else:
		pc0 = plotutils.plot_pcm(a1[1,0], u, wavs, np.swapaxes(IperF,0,1), lims=[xlim_raw,None])
		plotutils.flip_x_axis(a1[1,0])
	a1[1,0].set_xlabel('cos(zenith)')
	a1[1,0].set_ylabel('wavelength (um)')
	a1[1,0].set_title('raw radiance')


	if not pcm_flag:	
		im1 = a1[1,1].imshow(np.swapaxes(log_uIperF, 0,1), origin='lower', 
						extent=(log_u0u[0], log_u0u[-1], wavs[0], wavs[-1]), aspect='auto')
		plotutils.set_imshow_ticks_and_labels(im1, a1[1,1], log_uIperF.shape, (log_u0u, wavs))
	else:
		pcm1 = plotutils.plot_pcm(a1[1,1], log_u0u, wavs, np.swapaxes(log_uIperF,0,1), lims=[xlim,None])
		plotutils.flip_x_axis(a1[1,1])
	a1[1,1].set_title('log(cos(zenith)*I/F)')
	a1[1,1].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[1,1].set_ylabel('wavelength (um)')
	#print('HERE')
	#print(log_uIperF)
	#print(log_u0u)
	#print(wavs)
	#plt.show()
	#sys.exit()

	# NOTE: wavelength axis is incorrect as wavelengths are not evenly spaced!
	#       log_u0u axis is incorrect as they are not evenly space either
	if not pcm_flag:
		im2 = a1[1,2].imshow(np.swapaxes(fit, 0,1), origin='lower',
						extent=(log_u0u[0], log_u0u[-1], wavs[0], wavs[-1]), aspect='auto')
		plotutils.set_imshow_ticks_and_labels(im2, a1[1,2], fit.shape, (log_u0u, wavs))
	else:
		pcm2 = plotutils.plot_pcm(a1[1,2], log_u0u, wavs, np.swapaxes(fit,0,1), lims=[xlim,None])
		plotutils.flip_x_axis(a1[1,2])
	a1[1,2].set_title('Fitted log(cos(zenith)*I/F)')
	a1[1,2].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[1,2].set_ylabel('wavelength (um)')

	if not pcm_flag:
		im3 = a1[1,3].imshow(np.swapaxes(log_uIperF-fit, 0,1), origin='lower',
							extent=(log_u0u[0], log_u0u[-1], wavs[0], wavs[-1]), aspect='auto')
		plotutils.set_imshow_ticks_and_labels(im3, a1[1,3], log_uIperF.shape, (log_u0u, wavs))
	else:
		pcm3 = plotutils.plot_pcm(a1[1,3], log_u0u, wavs, np.swapaxes(log_uIperF-fit,0,1), lims=[xlim,None])
		plotutils.flip_x_axis(a1[1,3])
	a1[1,3].set_title('Residual (data-fit) log(cos(zenith)*I/F)')
	a1[1,3].set_xlabel('log(cos(zenith)*cos(solar_zenith))')
	a1[1,3].set_ylabel('wavelength (um)')
	
	# should I put the residuals on the same scale?
	if not pcm_flag:
		imgs = (im1,im2)
	else:
		imgs = (pcm1,pcm2)
	vls = []
	vhs = []
	for img in imgs:
		l,h = img.get_clim()
		vls.append(l)
		vhs.append(h)
	vl = min(vls)
	vh = max(vhs)
	for img in imgs:
		img.set_clim(vl, vh)
	return(fit)

def retrieval_setup(runname, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots the initial setup of the retrieval, uses *.spx and *.apr files
	"""
	print('INFO: Plotting intial setup of retrieval using *.apr and *.spx files')
	try:
		apr_data = nemesis.read.apr(runname)
		spx_data = nemesis.read.spx(runname)
	except nemesis.exceptions.NemesisReadError:
		print(f'WARNING: Could not read *.apr or *.spx file from {runname}. Skipping...')
		return([None])

	f1 = None
	return([f1])

def ensemble_minnaert_limb_darkening(runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Creates a synthetic image using a set of nemesis runs that cover a latitude range. Needs the parent fits file
	to correctly create image. See ../fitscube/limb_darkening_synthetic_img.py.

	# ARGUMENTS #
		runnames
			<str> name of nemesis runs to plot
		outfolder
			<str> directory to save plots to
		show_plot
			<bool> If True will show plots interactively
		minnaert.enesemble.fits
			<str> Parent fits file that the nemesis input data was originally compiled from

	# RETURNS #
		figure_list [n]
			<figures, array> A list of figures created in this function.
	
	"""

	lat_ranges = np.zeros((len(runnames), 2))
	initial_mlds = [None]*len(runnames) #np.zeros((len(runnames), 3))
	retrieved_mlds = [None]*len(runnames) #np.zeros((len(runnames), 3))
	for i, runname in enumerate(runnames):
		mre_data = nemesis.read.mre(runname)
		spx_data = nemesis.read.spx(runname)
		# all wavelengths in each *.mre file have same zenith and solar-zenith angles
		print(spx_data['fov_averaging_record'])

		u = np.cos(np.ones_like(mre_data['waveln'])*np.array(spx_data['fov_averaging_record'])[:,0,3,None]*np.pi/180) # this is a bit hacky as it doesn't take into account different ngeoms
		u0 = np.cos(np.ones_like(mre_data['waveln'])*np.array(spx_data['fov_averaging_record'])[:,0,2,None]*np.pi/180)
		initial_radiances = mre_data['radiance_meas']
		retrieved_radiances = mre_data['radiance_retr']		

		print(f'u.shape {u.shape} u0.shape {u0.shape} retrieved_radiances.shape {retrieved_radiances.shape}')
		print(u)

		mld_initial = np.array([fit_minnaert_limb_darkening(u[:,j], u0[:,j], initial_radiances[:,j]) for j in range(u.shape[1])])
		mld_retrieved = np.array([fit_minnaert_limb_darkening(u[:,j], u0[:,j], retrieved_radiances[:,j]) for j in range(u.shape[1])])

		print(f'mld_retrieved.shape {mld_retrieved.shape} mld_retrieved[0].shape {mld_retrieved[0].shape} mld_retrieved[0][0].shape {mld_retrieved[0][0].shape}')

		x_initial = np.array([_x[0] for _x in mld_initial])
		x_retrieved = np.array([_x[0] for _x in mld_retrieved])
		#print(x_retrieved)

		# wavelengths, ln((I/F)_0), k
		initial_mlds[i] = np.array((mre_data['waveln'][0], x_initial[:,0], x_initial[:,1]))
		retrieved_mlds[i] = np.array((mre_data['waveln'][0], x_retrieved[:,0], x_retrieved[:,1]))

		#print(retrieved_mlds[i])

		# get latitude range from file path
		abspath = os.path.abspath(runname)
		containing_folder = abspath.split(os.sep)[-2]
		lat_min, lat_max = float(containing_folder.split('_')[1]), float(containing_folder.split('_')[3])
		lat_ranges[i,:] = (lat_min, lat_max)

	initial_mlds = np.array(initial_mlds)
	retrieved_mlds = np.array(retrieved_mlds)
	print(f'initial_mlds.shape {initial_mlds.shape} retrieved_mlds.shape {retrieved_mlds.shape}, lat_ranges.shape {lat_ranges.shape}')
	print(lat_ranges)

	### Create plot of wavelength vs ln((I/F)_0) and k for each latitide slice ###
	f1 = plt.figure(figsize=[_x/2.54 for _x in (12*2,12*2)])

	xmin = np.nanmin(np.concatenate((initial_mlds[0,0,:], retrieved_mlds[0,0,:]), axis=None))
	xmax = np.nanmax(np.concatenate((initial_mlds[0,0,:], retrieved_mlds[0,0,:]), axis=None))

	ln_IperF0_min = np.nanmin(np.concatenate((initial_mlds[0,1,:], retrieved_mlds[0,1,:]), axis=None))
	ln_IperF0_max = np.nanmax(np.concatenate((initial_mlds[0,1,:], retrieved_mlds[0,1,:]), axis=None))

	k_min = np.nanmin(np.concatenate((initial_mlds[0,2,:], retrieved_mlds[0,2,:]), axis=None))
	k_max = np.nanmax(np.concatenate((initial_mlds[0,2,:], retrieved_mlds[0,2,:]), axis=None))

	a1 = f1.add_subplot(2,1,1)
	p11 = a1.plot([],[], label='measured')
	p12 = a1.plot([],[], label='retrieved')
	a1.set_ylabel('ln((I/F)_0)')
	a1.set_xlabel('wavelength (um)')
	a1.legend(loc='lower left')
	a1.set_xlim(xmin, xmax)
	a1.set_ylim(ln_IperF0_min, ln_IperF0_max)
	
	a2 = f1.add_subplot(2,1,2)
	p21 = a2.plot([],[], label='measured')
	p22 = a2.plot([],[], label='retrieved')
	a2.set_ylabel('k')
	a2.set_xlabel('wavelength (um)')
	a2.legend(loc='upper left')
	a2.set_xlim(xmin, xmax)
	a2.set_ylim(k_min, k_max)


	print(p11)
	def update(i):
		f1.suptitle(f'lat {lat_ranges[i,0]} to {lat_ranges[i,1]}')
		p11[0].set_data(initial_mlds[i,0,:], initial_mlds[i,1,:])
		p12[0].set_data(retrieved_mlds[i,0,:], retrieved_mlds[i,1,:])
		p21[0].set_data(initial_mlds[i,0,:], initial_mlds[i,2,:])
		p22[0].set_data(retrieved_mlds[i,0,:], retrieved_mlds[i,2,:])
		return

	ordered_lats = np.sort(lat_ranges[:,0])
	ordered_idxs = [np.argmin(np.abs(lat_ranges[:,0] - l)) for l in ordered_lats]
	print(ordered_idxs)
	ani = matplotlib.animation.FuncAnimation(f1, update, ordered_idxs, interval=300)
	outmovname = 'minnaert_params.mov'
	print(outfolder)
	plotutils.save_animate_plt(ani, outmovname, outfolder, save_plot=True, animate_plot=show_plot)
	### ---------------------------------------------------------------- ###

	### Create a plot showing the retrieved vs measured radiances across planet disk ###

	from astropy.io import fits
	import fitscube.process.sinfoni
	hdul = fits.open(kwargs['minnaert.ensemble.fits'])

	# get data into correct format
	# TODO: try to regrid hdul[0].data to same smoothed wavelength grid as retrieved data, make it an option?
	# z, y, x
	if kwargs['minnaert.ensemble.regrid_wavelength_to_fits']:
		wavgrid = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])
	else:
		import subsample
		wavgrid = retrieved_mlds[0,0,:]
		wg = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])
		nz, ny, nx = hdul[0].data.shape
		newdat = np.zeros((len(wavgrid), ny, nx))
		dw = wavgrid[1]-wavgrid[0]
		print(wavgrid)
		print(dw, nz, ny, nx)
		for i in range(hdul[0].data.shape[1]):
			for j in range(hdul[0].data.shape[2]):
				ignore, newdat[:,i,j] = subsample.conv(wg, hdul[0].data[:,i,j], dw, outgrid=wavgrid)
		hdul[0].data = newdat

	retrieved_radiances_from_minnaert = np.zeros_like(hdul[0].data)
	ln_IperF_0 = np.ones_like(hdul[0].data)
	k = np.ones_like(hdul[0].data)
	for i, (lat_min, lat_max) in enumerate(lat_ranges):
		print(f'lat_min {lat_min} lat_max {lat_max}')
		lat_idxs = np.nonzero((hdul['LATITUDE'].data > lat_min) & (hdul['LATITUDE'].data <= lat_max))
		print(f'lat_idxs {lat_idxs}')
		print(retrieved_mlds[i])
		retrieved_ln_IperF_0 = np.interp(wavgrid, retrieved_mlds[i][0,:], retrieved_mlds[i][1,:])
		retrieved_k = np.interp(wavgrid, retrieved_mlds[i][0,:], retrieved_mlds[i][2,:])
		print(f'retrieved_ln_IperF_0.shape {retrieved_ln_IperF_0.shape} ln_IperF_0.shape {ln_IperF_0.shape}')
		print(f'ln_IperF_0[:,lat_idxs[0],lat_idxs[1]].shape {ln_IperF_0[:,lat_idxs[0],lat_idxs[1]].shape}')
		ln_IperF_0[:,lat_idxs[0],lat_idxs[1]] = ln_IperF_0[:,lat_idxs[0],lat_idxs[1]]*retrieved_ln_IperF_0[:,None]
		k[:,lat_idxs[0],lat_idxs[1]] = k[:,lat_idxs[0],lat_idxs[1]]*retrieved_k[:,None]

	fits_u_and_u0 =  np.cos((np.pi/180)*hdul['ZENITH'].data[None,:,:])

	# alter these to change what is displayed
	retrieved_radiances_from_minnaert = 1E-6*radiance_from_minnaert_params(ln_IperF_0, k, fits_u_and_u0, fits_u_and_u0) # change from uW to W
	raw_data = hdul[0].data
	chi_sq = (raw_data - retrieved_radiances_from_minnaert)**2/raw_data
	residual = raw_data - retrieved_radiances_from_minnaert
	chi_sq = chi_sq
	residual = residual

	nr, nc = 2,4
	f2 = plt.figure(figsize=[_x/2.54 for _x in (12*nc,12*nr)])

	a21 = f2.add_subplot(nr,nc,1)
	im21 = a21.imshow(np.nanmedian(retrieved_radiances_from_minnaert, axis=0)) # dummy data overwritten later
	a21.set_title('retrieved radiances from minneart')
	
	a22 = f2.add_subplot(nr,nc,2)
	im22 = a22.imshow(np.nanmedian(raw_data, axis=0)) # dummy data overwritten later
	a22.set_title('raw data')
	
	a23 = f2.add_subplot(nr,nc,3)
	im23 = a23.imshow(np.nanmedian(residual, axis=0)) # dummy data overwritten later
	a23.set_title('Residual')

	a26 = f2.add_subplot(nr,nc,4)
	im26 = a26.imshow(np.nanmedian(chi_sq, axis=0)) # dummy data overwritten later
	a26.set_title('Chi Square')

	disk_idxs = np.nonzero(hdul['DISK_MASK'].data)
	print(f'INFO: disk_idxs {disk_idxs}')
	#a = hdul[0].data[:,disk_idxs[0],disk_idxs[1]]
	#print(f'a.shape {a.shape}')
	#ref_dat_radiances = np.nanmedian(hdul[0].data[:,disk_idxs[0],disk_idxs[1]], axis=1)
	ref_dat_radiances = np.nanmean(hdul[0].data[:,disk_idxs[0],disk_idxs[1]], axis=1)
	#ref_ret_radiances = np.nanmedian(retrieved_radiances_from_minnaert, axis=(1,2)) # *.mre works in uW not W
	ref_ret_radiances = np.nanmean(retrieved_radiances_from_minnaert, axis=(1,2)) # *.mre works in uW not W

	a24 = f2.add_axes([0.1, 0.3, 0.8, 0.15])
	p241 = a24.plot(wavgrid, ref_dat_radiances, label='mean raw radiance')
	p242 = a24.plot(wavgrid, ref_ret_radiances, label='mean retrieved radiance')
	a24_vline = a24.axvline(wavgrid[0], color='red')
	a24.set_yscale('log')
	a24.set_xlabel('Wavelength (um)')
	a24.set_ylabel('Radiance (W ...)')
	a24.set_ylim(0, 1.1*np.nanmax(ref_ret_radiances))
	a24.set_xlim(np.nanmin(wavgrid), np.nanmax(wavgrid))
	a24.legend()

	a25 = f2.add_axes([0.1, 0.1, 0.8, 0.15])
	p251 = a25.plot(wavgrid, np.nanmean(residual, axis=(1,2)), label='mean residual', color='tab:blue')
	a25_x2 = a25.twinx()
	p252 = a25_x2.plot(wavgrid, np.nanmean(chi_sq, axis=(1,2)), label='mean chi squared', color='tab:orange')
	a25.set_yscale('log')
	a25_x2.set_yscale('log')
	a25.set_xlabel('Wavelength (um)')
	a25.set_ylabel('Residual')
	a25_x2.set_ylabel('chi squared')
	a25.set_xlim(np.nanmin(wavgrid), np.nanmax(wavgrid))
	lines1, labels1 = a25.get_legend_handles_labels()
	lines2, labels2 = a25_x2.get_legend_handles_labels()
	a25.legend(lines1+lines2, labels1+labels2)
	a25_vline = a25.axvline(wavgrid[0], color='red')

	def update(i):
		nonlocal a24_vline, a25_vline
		im21.set_data(retrieved_radiances_from_minnaert[i,:,:])
		im22.set_data(raw_data[i,:,:])
		im23.set_data(residual[i,:,:])
		im26.set_data(chi_sq[i,:,:])
		im21.set_clim((np.nanmin(retrieved_radiances_from_minnaert[i]), np.nanmax(retrieved_radiances_from_minnaert[i])))
		im22.set_clim((np.nanmin(raw_data[i]), np.nanmax(raw_data[i])))
		im23.set_clim((np.nanmin(residual[i]), np.nanmax(residual[i])))
		im26.set_clim((np.nanmin(chi_sq[i]), np.nanmax(chi_sq[i])))
		a24_vline.remove()
		a24_vline = a24.axvline(wavgrid[i], color='red')
		a25_vline.remove()
		a25_vline = a25.axvline(wavgrid[i], color='red')
		return
	
	ani = matplotlib.animation.FuncAnimation(f2, update, hdul[0].data.shape[0], interval=100)
	outmovname_radiances = 'minnaert_radiances.mov'
	plotutils.save_animate_plt(ani, outmovname_radiances, outfolder, save_plot=True, animate_plot=show_plot)
	### ------------------------------------------------------------------------------- ###
	
	return([])
	

def ensemble_improvement_factor(runnames, outfolder=None, show_plot=False, **kwargs):
	imp_factors = combined_improvement_factor(runnames)

	n = int(m.ceil(m.sqrt(len(imp_factors))))
	f1 = plt.figure(figsize=[x/2.54 for x in (12*n,12*n)])
	a_arr = f1.subplots(nrows=n, ncols=n, squeeze=False, gridspec_kw={'wspace':0.25, 'hspace':0.25})

	i = 0
	for indep_lbl in imp_factors.keys():
		ax = a_arr[i//n][i%n]
		if indep_lbl in ['Pressure', 'Height']:
			ystep_plot(ax, imp_factors[indep_lbl]['improvement_factor']/imp_factors[indep_lbl]['n_vars'], imp_factors[indep_lbl]['independent_var'], label='Mean Improvement Factor')
			ax.invert_yaxis()
			ax.set_ylabel(indep_lbl)
			ax.set_xlabel("Mean Improvement Factor")
			if indep_lbl in ['Pressure']:
				ax.set_xscale('log')
				ax.set_yscale('log')
		else:
			ax.step(imp_factors[indep_lbl]['independent_var']/imp_factors[indep_lbl]['n_vars'], imp_factors[indep_lbl]['improvement_factor'], label='Mean Improvement Factor', where='mid')
			ax.set_xlabel(indep_lbl)
			ax.set_ylabel("Mean Improvement Factor")

		i+=1
	if i < n*n:
		for j in range(i+1, n*n):
			a_arr[i//n][j%n].remove()
	f1.tight_layout

	save_show_plt(f1, "ensemble_improvement_factor.png", outfolder=outfolder, show_plot=show_plot)

	return([f1])

def ensemble_parameters_vs_lat_max_imp_fac(runnames, outfolder=None, show_plot=False, prop='lat', **kwargs):
	"""
	Plots the value of all variables at the value of "most improvement" as defined by the "improvement factor". If >1 variable share the same
	independent variable (e.g. Height, Pressure, Wavelength) then the "most improved" value is taken as the largest value of the sum of their
	improvement factors.
	"""
	if prop == 'lat':
		xlabel = 'Latitude (degrees)'
		outfname = 'ensemble_parameters_vs_latitude.png'
	elif prop == 'lon':
		xlabel = 'Longitude (degrees)'
		outfname = 'ensemble_parameters_vs_longitude.png'
	else:
		ut.pERROR('Unknown property "{}" to plot against, returning None...'.format(prop))
		return([None])

	imp_factors = combined_improvement_factor(runnames)
	# get the index of the maximum improvement for each independent variable accross all runs
	for indep_lbl in imp_factors.keys():
		imp_factors[indep_lbl]['max_idx'] = np.argmax(imp_factors[indep_lbl]['improvement_factor'])
		if indep_lbl in ['Pressure']:
			imp_factors[indep_lbl]['max_idx'] = 17
	# Now plot the values of variables at this index
	# NOTE: this depends on knowing the latitude of the data, we can find this from the *.mre file
	xdata = []
	re_ydata = {}
	ap_ydata = {}
	dep_info = {}
	for runname in runnames:
		try:
			mre_data = nemesis.read.mre(runname)
		except nemesis.exceptions.NemesisReadError as nre:
			print(nre)
			ut.pWARN('Could not read *.mre file for run {}, skipping...'.format(runname))
		xdata.append(mre_data[prop])
		print(mre_data[prop])
		for i, (vid, vpar, ap, ae, rp, re, nv) in enumerate(zip(mre_data['varident'],
																mre_data['varparam'],
																mre_data['aprprof'],
																mre_data['aprerr'],
																mre_data['retprof'],
																mre_data['reterr'],
																mre_data['nxvar'])):
			ap_var_info = varident_to_labels_and_profiles(runname, vid, vpar, ap[:nv], ae[:nv])
			re_var_info = varident_to_labels_and_profiles(runname, vid, vpar, rp[:nv], re[:nv])
	
			dep_lbl = re_var_info['dependent_label']
			indep_lbl = re_var_info['independent_label']
			if len(re_ydata.get(dep_lbl, [])) == 0:
				re_ydata[dep_lbl] = {}
				re_ydata[dep_lbl]['var'] = [re_var_info['dependent_var'][imp_factors[indep_lbl]['max_idx']]]
				re_ydata[dep_lbl]['err'] = [re_var_info['dependent_err'][imp_factors[indep_lbl]['max_idx']]]
				dep_info[dep_lbl] = {'indep_lbl':indep_lbl, 'indep_val':re_var_info['independent_var'][imp_factors[indep_lbl]['max_idx']]}
			else:
				re_ydata[dep_lbl]['var'] += [re_var_info['dependent_var'][imp_factors[indep_lbl]['max_idx']]]
				re_ydata[dep_lbl]['err'] += [re_var_info['dependent_err'][imp_factors[indep_lbl]['max_idx']]]
			if len(ap_ydata.get(dep_lbl, [])) == 0:
				ap_ydata[dep_lbl] = {}
				ap_ydata[dep_lbl]['var'] = [ap_var_info['dependent_var'][imp_factors[indep_lbl]['max_idx']]]
				ap_ydata[dep_lbl]['err'] = [ap_var_info['dependent_err'][imp_factors[indep_lbl]['max_idx']]]
			else:
				ap_ydata[dep_lbl]['var'] += [ap_var_info['dependent_var'][imp_factors[indep_lbl]['max_idx']]]
				ap_ydata[dep_lbl]['err'] += [ap_var_info['dependent_err'][imp_factors[indep_lbl]['max_idx']]]
	# turn everything into numpy arrays
	xdata = np.array(xdata)
	sorted_idxs = np.argsort(xdata)
	xdata = xdata[sorted_idxs]
	for dep_lbl in re_ydata.keys():
		re_ydata[dep_lbl]['var'] = np.array(re_ydata[dep_lbl]['var'])[sorted_idxs]
		re_ydata[dep_lbl]['err'] = np.array(re_ydata[dep_lbl]['err'])[sorted_idxs]
		ap_ydata[dep_lbl]['var'] = np.array(ap_ydata[dep_lbl]['var'])[sorted_idxs]
		ap_ydata[dep_lbl]['err'] = np.array(ap_ydata[dep_lbl]['err'])[sorted_idxs]
	n = int(m.ceil(m.sqrt(len(imp_factors))))
	f1 = plt.figure(figsize=[x/2.54 for x in (12*n,12*n)])
	a_arr = f1.subplots(nrows=n, ncols=n, squeeze=False, gridspec_kw={'wspace':0.25, 'hspace':0.25})
	i = 0
	for dep_lbl in re_ydata.keys():
		ax = a_arr[i//n][i%n]
		#print(dep_lbl)
		#print(xdata)
		#print(re_ydata[dep_lbl]['var'])
		ax.fill_between(xdata, 
						ap_ydata[dep_lbl]['var']+ap_ydata[dep_lbl]['err'],
						ap_ydata[dep_lbl]['var']-ap_ydata[dep_lbl]['err'],
						color='tab:blue',
						label='Apriori Error', step='mid', alpha=0.5)
		ax.fill_between(xdata, 
						re_ydata[dep_lbl]['var']+re_ydata[dep_lbl]['err'],
						re_ydata[dep_lbl]['var']-re_ydata[dep_lbl]['err'],
						color='tab:orange',
						label='Retrieved Error', step='mid', alpha=0.5)
		ax.step(xdata, re_ydata[dep_lbl]['var'], label='Retrieved Value', where='mid', color='tab:orange')
		ax.step(xdata, ap_ydata[dep_lbl]['var'], label='Apriori Value', where='mid', color='tab:blue')
		ax.set_xlabel(xlabel)
		ax.set_ylabel(dep_lbl)
		ax.set_title('{}={}'.format(dep_info[dep_lbl]['indep_lbl'], dep_info[dep_lbl]['indep_val']))
		i+=1
	if i < n*n:
		for j in range(i+1, n*n):
			a_arr[i//n][j%n].remove()
	f1.tight_layout
	save_show_plt(f1, outfname, outfolder=outfolder, show_plot=show_plot)
	return([f1])

def ensemble_parameters_vs_lon_max_imp_fac(runnames, outfolder=None, show_plot=False, **kwargs):
	return(ensemble_parameters_vs_lat_max_imp_fac(runnames, outfolder=outfolder, show_plot=show_plot, prop='lon'))

def ensemble_params_v_lat(runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots a large number of plots of all parameters vs latitude at a constant value of the parameters independent variable.
	Then uses ffmpeg to assemble the frames into a movie
	x-axis = latitude
	y-axis = parameter dependent variable value
	time-axis = parameter independent variable value
	"""
	indep_lbls, dep_lbls, re_dep_arr, re_dep_err, ap_dep_arr, ap_dep_err, sorted_lat_arr, nindep_each_param, indep_arr= get_params_vs_lat(runnames)
		
	ndigits = int(np.ceil(np.log10(len(dep_lbls))) + 1)
	fig_list = []
	# May need to fiddle with this to get the plotting working really nicely. Ideally would like to have things NOT MOVE ABOUT when running the movie
	for i, indep_lbl in enumerate(indep_lbls):
		for j in range(nindep_each_param[i]):
			f1 = plt.figure(figsize=[x/2.54 for x in (12,12)])
			#fig_list.append(f1)
			ax = f1.add_subplot(1,1,1)


			ax.fill_between(sorted_lat_arr, ap_dep_arr[i][:,j]+ap_dep_err[i][:,j],
											ap_dep_arr[i][:,j]-ap_dep_err[i][:,j],
											color='tab:blue',
											label='Apriori Error',
											step='mid', alpha=0.5)
	
			ax.fill_between(sorted_lat_arr, re_dep_arr[i][:,j]+re_dep_err[i][:,j],
											re_dep_arr[i][:,j]-re_dep_err[i][:,j],
											color='tab:orange',
											label='Retrieved Error',
											step='mid', alpha=0.5)
			ax.step(sorted_lat_arr, ap_dep_arr[i][:,j], label='Apriori Value', color='tab:blue', where='mid')
			ax.step(sorted_lat_arr, re_dep_arr[i][:,j], label='Retrieved Value', color='tab:orange', where='mid')

			ax.set_xlabel('Latitude (deg North)')
			ax.set_ylabel(dep_lbls[i])
			ax.set_title('{} = {}'.format(indep_lbl, indep_arr[i][j]))
			f1.tight_layout
	
			fmt_str = '{}_{:0'+'{}'.format(ndigits)+'}.png'
			print(fmt_str)
			savefig_name = fmt_str.format(dep_lbls[i].replace(' ', '_'), j)
			print(savefig_name)
			if type(outfolder)!=type(None):
				frames_folder = os.path.join(outfolder,'./frames')
				if os.path.exists(frames_folder):
					if not os.path.isdir(frames_folder):
						ut.pERROR('Path {} exists, but is not a directory, exiting...'.format(frames_folder))
						sys.exit()
				else:
					ut.pINFO('Path {} does not exist, creating...'.format(frames_folder))
					os.makedirs(frames_folder)
			else:
				frames_folder = None

			save_show_plt(f1, savefig_name, outfolder=frames_folder, show_plot=show_plot)
			plt.close('all') # close all figures to avoid using up all the memory

		# by now we have created all the frames for the movie
		ffmpeg_str = '{}_%0{}d.png'.format(dep_lbls[i].replace(' ','_'), ndigits)
		print(ffmpeg_str)
		if type(outfolder)!=type(None):
			import subprocess as sp
			sp.run(['ffmpeg', '-framerate', '5', '-f', 'image2', '-i', os.path.join(frames_folder,ffmpeg_str), '-y', os.path.join(outfolder,'{}_lat.mov'.format(dep_lbls[i].replace(' ','_')))])
			

	return(fig_list)

def ensemble_params_v_lon(runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots a large number of plots of all parameters vs longitude at a constant value of the parameters independent variable.
	Then uses ffmpeg to assemble the frames into a movie
	x-axis = longitude
	y-axis = parameter dependent variable value
	time-axis = parameter independent variable value
	"""
	indep_lbls, dep_lbls, re_dep_arr, re_dep_err, ap_dep_arr, ap_dep_err, sorted_lon_arr, nindep_each_param, indep_arr= get_params_vs_lat(runnames, prop='lon')
		
	ndigits = int(np.ceil(np.log10(len(dep_lbls))) + 1)
	fig_list = []
	# May need to fiddle with this to get the plotting working really nicely. Ideally would like to have things NOT MOVE ABOUT when running the movie
	for i, indep_lbl in enumerate(indep_lbls):
		for j in range(nindep_each_param[i]):
			f1 = plt.figure(figsize=[x/2.54 for x in (12,12)])
			#fig_list.append(f1)
			ax = f1.add_subplot(1,1,1)


			ax.fill_between(sorted_lon_arr, ap_dep_arr[i][:,j]+ap_dep_err[i][:,j],
											ap_dep_arr[i][:,j]-ap_dep_err[i][:,j],
											color='tab:blue',
											label='Apriori Error',
											step='mid', alpha=0.5)
	
			ax.fill_between(sorted_lon_arr, re_dep_arr[i][:,j]+re_dep_err[i][:,j],
											re_dep_arr[i][:,j]-re_dep_err[i][:,j],
											color='tab:orange',
											label='Retrieved Error',
											step='mid', alpha=0.5)
			ax.step(sorted_lon_arr, ap_dep_arr[i][:,j], label='Apriori Value', color='tab:blue', where='mid')
			ax.step(sorted_lon_arr, re_dep_arr[i][:,j], label='Retrieved Value', color='tab:orange', where='mid')

			ax.set_xlabel('Longitude (deg North)')
			ax.set_ylabel(dep_lbls[i])
			ax.set_title('{} = {}'.format(indep_lbl, indep_arr[i][j]))
			f1.tight_layout
	
			fmt_str = '{}_{:0'+'{}'.format(ndigits)+'}.png'
			print(fmt_str)
			savefig_name = fmt_str.format(dep_lbls[i].replace(' ', '_'), j)
			print(savefig_name)
			if type(outfolder)!=type(None):
				frames_folder = os.path.join(outfolder,'./frames')
				if os.path.exists(frames_folder):
					if not os.path.isdir(frames_folder):
						ut.pERROR('Path {} exists, but is not a directory, exiting...'.format(frames_folder))
						sys.exit()
				else:
					ut.pINFO('Path {} does not exist, creating...'.format(frames_folder))
					os.makedirs(frames_folder)
			else:
				frames_folder = None

			save_show_plt(f1, savefig_name, outfolder=frames_folder, show_plot=show_plot)
			plt.close('all') # close all figures to avoid using up all the memory

		# by now we have created all the frames for the movie
		ffmpeg_str = '{}_%0{}d.png'.format(dep_lbls[i].replace(' ','_'), ndigits)
		print(ffmpeg_str)
		if type(outfolder)!=type(None):
			import subprocess as sp
			sp.run(['ffmpeg', '-framerate', '5', '-f', 'image2', '-i', os.path.join(frames_folder,ffmpeg_str), '-y', os.path.join(outfolder,'{}_lon.mov'.format(dep_lbls[i].replace(' ','_')))])
			

	return(fig_list)

def ensemble_chi_phi_v_lat(runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots the final values of chi-sq/dof and phi-sq/dof for a collection of nemesis runs against latitude
	"""
	dat_arr = np.zeros((2,len(runnames)))
	lat_arr = np.zeros((len(runnames)))
	for i, runname in enumerate(runnames):
		try:
			itrd = nemesis.read.itr(runname)
			mred = nemesis.read.mre(runname)
			lat_arr[i] = mred['lat']
			dat_arr[0,i] = itrd['chisq_arr'][-1]/itrd['nx']
			dat_arr[1,i] = itrd['phi_arr'][-1]/itrd['nx']
		except nemesis.exceptions.NemesisReadError:
			ut.pWARN("Could not read either *.itr or *.mre file for runname {}, setting value to NAN for missing data.".format(runname))
			lat_arr[i] = np.nan
			dat_arr[0,i] = np.nan
			dat_arr[1,i] = np.nan
	
	# sort arrays by latitude
	sort_idxs = np.argsort(lat_arr)
	lat_arr = lat_arr[sort_idxs]
	dat_arr[0,:] = dat_arr[0,sort_idxs]
	dat_arr[1,:] = dat_arr[1,sort_idxs]
	#print(lat_arr)
	f1 = plt.figure(figsize=[_x/2.54 for _x in (10,10)])
	a11 = f1.add_subplot(1,1,1)
	a11.plot(lat_arr, dat_arr[0,:], color='tab:blue', label='chisq per dof')
	a11.plot(lat_arr, dat_arr[1,:], color='tab:orange', label='phisq per dof')

	a11.set_yscale('log')
	a11.set_xlabel('Latitude (deg)')
	a11.set_ylabel('Cost Function/Degrees of Freedom')
	a11.legend()

	save_show_plt(f1, 'chi_phi_v_lat.png', outfolder=outfolder, show_plot=show_plot)
	return([f1])	

def ensemble_chi_phi_v_lon(runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Plots the final values of chi-sq/dof and phi-sq/dof for a collection of nemesis runs against longitude
	"""
	dat_arr = np.zeros((2,len(runnames)))
	lon_arr = np.zeros((len(runnames)))
	for i, runname in enumerate(runnames):
		try:
			itrd = nemesis.read.itr(runname)
			mred = nemesis.read.mre(runname)
			lon_arr[i] = mred['lon']
			dat_arr[0,i] = itrd['chisq_arr'][-1]/itrd['nx']
			dat_arr[1,i] = itrd['phi_arr'][-1]/itrd['nx']
		except nemesis.exceptions.NemesisReadError:
			ut.pWARN("Could not read either *.itr or *.mre file for runname {}, setting value to NAN for missing data.".format(runname))
			lon_arr[i] = np.nan
			dat_arr[0,i] = np.nan
			dat_arr[1,i] = np.nan
	
	# sort arrays by latitude
	sort_idxs = np.argsort(lon_arr)
	lon_arr = lon_arr[sort_idxs]
	dat_arr[0,:] = dat_arr[0,sort_idxs]
	dat_arr[1,:] = dat_arr[1,sort_idxs]
	#print(lat_arr)
	f1 = plt.figure(figsize=[_x/2.54 for _x in (10,10)])
	a11 = f1.add_subplot(1,1,1)
	a11.plot(lon_arr, dat_arr[0,:], color='tab:blue', label='chisq per dof')
	a11.plot(lon_arr, dat_arr[1,:], color='tab:orange', label='phisq per dof')

	a11.set_yscale('log')
	a11.set_xlabel('Longitude (deg)')
	a11.set_ylabel('Cost Function/Degrees of Freedom')
	a11.legend()

	save_show_plt(f1, 'chi_phi_v_lon.png', outfolder=outfolder, show_plot=show_plot)
	return([f1])	
	
def ensemble_region(regionfiles, runnames, outfolder=None, show_plot=False, **kwargs):
	"""
	Generally an ensemble is a collection of NEMESIS retrievals over some area.
	Want to plot a representation of the region covered by the retrievals
	* Plot the disk of the planet and the pointing
	* Plot the disk of the planet and the region
	* Plot the lat-lon of the retrievals (will not be 1:1 with the region in case of coadding)
	"""
	figlist = []
	for regionfile in regionfiles:
		rcont = nc.RegionContainer(regionfile)
		plotfile = regionfile.rsplit('.',1)[0]+'.png'
		if rcont.get_region_type() == 'circle':
			ctr = PixCoord(rcont.data['cxp']['value'],rcont.data['cyp']['value'])
			print(type(ctr))
			print(ctr)
			rgn = CirclePixelRegion(center=ctr, radius=rcont.data['rp']['value'])
		elif rcont.get_region_type() == 'rect':
			rgn = RectanglePixelRegion(PixCoord(x=rcont.data['cxp']['value'],y=rcont.data['cyp']['value']),
										rcont.data['wp']['value'], rcont.data['hp']['value'])
		else:
			ut.pERROR('UNKNOWN REGION TYPE. Cannot display region')
			return([])

		with fits.open(rcont.data['fits_file']['value']) as hdul:
			w1 = wcs.WCS(hdul[0].header)
			"""
			# need some extra time to implement this bit
			lats = np.zeros((len(runnames)))
			lons = np.zeros((len(runnames)))
			for i, runname in enumerate(runnames):
				try:
					mred = nemesis.read.mre(runnames)
					lats[i] = mred['lat']
					lons[i] = mred['lon']
				except nemesis.exceptions.NemesisReadError:
					ut.pWARN('Could not read *.mre file for runname {}'.format(runname))
			"""
			f1 = plt.figure(figsize=[_x/2.54 for _x in (12,12)])
			a11 = f1.add_subplot(1,1,1)
			a11.imshow(np.nanmean(hdul[0].data[300:310,:,:],axis=0),origin='lower')
			a11.add_patch(rgn.as_artist(facecolor='none',edgecolor='red',lw=1))
			#a11.scatter(lats, lons, color='red')
			a11.tick_params(axis='both', which='both', top=False,bottom=False,left=False,right=False,
					labeltop=False,labelbottom=False,labelleft=False,labelright=False)
			a11.set_title(regionfile.rsplit('.',1)[0])


		save_show_plt(f1, plotfile, outfolder=outfolder, show_plot=show_plot)
		figlist.append(f1)
	return(figlist)
		

#
# PUT NON PLOTTING FUNCTIONS (except main) BELOW THIS LINE
#

def radiance_from_minnaert_params(ln_IperF0, k, u, u0):
	"""
	Creates radiances from minnaert parameters

	ln(u*I/F) = ln((I/F)_0) + k*ln(u_0*u)

	I/F = (I/F)_0 * u_0^k * u^(k-1)
	
	"""
	IperF0 = np.exp(ln_IperF0)
	return(IperF0 * u0**(k) * u**(k-1))

def fit_minnaert_limb_darkening(u0, u, rads):
	"""
	we are fitting
	log(u*I/F) = log((I/F)_0) + k*log(u_0*u)

	let u = cos(zenith)
	u_0 = cos(solar_zenith)
	I/F = radiance
	(I/F)_0 = radiance at zenith=0

	re-write as a matrix equation of the form y=Ax
	y is our result, scatt_logcoszenIperF
	x are our parameters, ln((I/F)_0) and k <--- these are the thing we want to find
	A is the coefficients of our parameters [1, ln(u_0*u)]
	"""
	#print('INFO: in "fit_minnaert_limb_darkening()"')
	#print(u0)
	#print(u)
	#print(rads)
	# masked arrays don't work nicely with this function so
	# make sure we just have normal numpy arrays
	if type(rads) == np.ma.MaskedArray:
		rads = rads.filled(fill_value=np.nan) # any values that should be masked are set to nan and will be ignored
	log_u0u = np.log(u0*u)
	log_uIperF = np.log(u*rads)
	# filter out nan values
	nandata = np.isnan(log_uIperF)
	#print('type(nandata)', type(nandata))
	#print(nandata)
	#print(nandata.data)
	#print(nandata.mask)
	#print(nandata.filled())
	log_uIperF = log_uIperF[~nandata]
	log_u0u = log_u0u[~nandata]
	#print('log_u0u.shape', log_u0u.shape)
	#print(log_u0u)
	#print('log_uIperF.shape', log_uIperF.shape)
	#print(log_uIperF)
	#print('type(log_u0u)', type(log_u0u))
	A = np.vstack([np.ones_like(log_u0u), log_u0u]).T
	
	#print('A.shape', A.shape)	
	#print(A)

	x, residual, rank, s = np.linalg.lstsq(A, log_uIperF, rcond=None)
	#print('x.shape', x.shape)
	#print(x)
	#print('residual.shape', residual.shape)
	#print(residual)
	return(x, residual, rank, s, log_u0u, log_uIperF, nandata)

def get_params_vs_lat(runnames, prop='lat'):
	"""
	For a set of runs, returns data in an arrangement where parameters vs latitude is easy to plot

	ARGUMENTS:
		runnames
			<str,array> A list of runnames to operate on
		prop
			<str> The property to get params agains, originally hard-coded as 'lat' (latitude) but 
			made more adaptable.

	RETURNS:
		indep_lbls [nparam]
			<str,array> The labels of the independent variables for each parameter
		dep_lbls [nparams]
			<str,array> The labels of the dependent variable for each parameter
		re_dep_arr [nparams,[nruns,nvar[i]]]
			<list,<float,array>> Retrieved values for each parameter. Consists of 'nparam' numpy arrays
			each of which has a size of 'nruns' by 'nvar[i]' where 'nruns' are the number of OPENABLE runs,
			and 'nvar[i]' is the number of values in the i^th parameter
		re_dep_err [nparams,[nruns,nvar[i]]]
			<list,<float,array>> Error on the retrieved values for each parameter. Arranged in the same way as
			're_dep_arr'
		ap_dep_arr [nparams,[nruns,nvar[i]]]
			<list,<float,array>> Apriori values for each parameter. Arranged in the same way as 're_dep_arr'
		ap_dep_err [nparams,[nruns,nvar[i]]]
			<list,<float,array> Error on the apriori values for each parameter. Arranged in the same wat as
			're_dep_arr'
		sorted_lat_arr [nruns]
			<float,array> The latitude of each OPENABLE run
		nindep_each_param [nparam]
			<int,array> The number of values (specifically independent values, but this is the same as the
			dependent values) for each parameter.
		indep_arr [nparam,nvar[i]]
			<list,<float,array>> The values of the independent variable for each parameter.

	NOTES:
		nruns
			The number of OPENABLE runs, by 'openable' we mean runs for which nemesis.read.mre() succeeded with no errors
		nvar[i]
			The number of (x,y) pairs needed to describe the i^th parameter. E.g. a volume mixing ratio may have 20 pressure levels
			where it is defined.

	EXAMPLE:
		indep_lbls, dep_lbls, re_dep_arr, re_dep_err, ap_dep_arr, ap_dep_err, sorted_lat_arr, nindep_each_param, indep_arr = get_params_vs_lat(runnames)
	"""
	xdata = []
	re_ydata = {}
	ap_ydata = {}
	dep_info = {}
	lat_var_info = {}
	runname_var_info = [None]*len(runnames)
	lat_arr = np.zeros((len(runnames)))
	for i, runname in enumerate(runnames):
		try:
			mre_data = nemesis.read.mre(runname)
		except nemesis.exceptions.NemesisReadError as nre:
			print(nre)
			ut.pWARN('Could not read *.mre file for run {}, skipping...'.format(runname))
			continue
		xdata.append(mre_data[prop])
		#print(mre_data['lat'])
		#print(mre_data['lon'])
		lat_arr[i] = mre_data[prop]
		runname_var_info[i] = get_var_info(runname)
		if type(runname_var_info[i]) == type(None):
			continue

	# filter runnames by runs that have finished and not been interrupted
	runnames = [runname for i, runname in enumerate(runnames) if type(runname_var_info[i])!=type(None)]
	lat_arr = np.array([l for i, l in enumerate(lat_arr) if type(runname_var_info[i])!=type(None)])
	# this one has to go last as the others rely upon it for filtering
	runname_var_info = [vi for i, vi in enumerate(runname_var_info) if type(vi)!=type(None)]

	# now have all var_info and latitudes associated with runnames in order, 
	# want to look along the pressure/height/wavelength levels for each parameter, and have value of pressure level [i] vs latitude

	# going to need a list of numpy arrays because they are all different sizes
	# assume all runnames have the same parameters

	nruns = len(runnames)
	nparams = len(runname_var_info[0])
	vid_hashes = runname_var_info[0].keys()
	ndep_each_param = [len(runname_var_info[0][some_vid]['retrieved']['dependent_var']) for some_vid in vid_hashes]
	nindep_each_param = [len(runname_var_info[0][some_vid]['retrieved']['independent_var']) for some_vid in vid_hashes]

	#print('nruns', nruns)
	#print('nparams', nparams)
	#print('vid_hashes', vid_hashes)
	#print('ndep_each_param', ndep_each_param)
	#print('nindep_each_param', nindep_each_param)

	indep_arr = [np.zeros((nindep_each_param[j])) for j in range(nparams)] # only need one as each run has the same parameter structure
	indep_lbls = [None]*nparams
	dep_lbls = [None]*nparams


	#ap_dep_arr = [[np.zeros((ndep_each_param[j])) for j in range(nparams)] for i in range(nruns)]
	#re_dep_arr = [[np.zeros((ndep_each_param[j])) for j in range(nparams)] for i in range(nruns)]
	ap_dep_arr = [np.zeros((nruns, ndep_each_param[i])) for i in range(nparams)]
	re_dep_arr = [np.zeros((nruns, ndep_each_param[i])) for i in range(nparams)]
	ap_dep_err = [np.zeros((nruns, ndep_each_param[i])) for i in range(nparams)]
	re_dep_err = [np.zeros((nruns, ndep_each_param[i])) for i in range(nparams)]

	#print('shape indep_arr', [ia.shape for ia in indep_arr])
	#print('shape indep_lbls', len(indep_lbls))
	#print('shape dep_lbls', len(dep_lbls))
	#print('shape ap_dep_arr', len(ap_dep_arr), ap_dep_arr[0].shape)
	#print('shape re_dep_arr', len(re_dep_arr), re_dep_arr[0].shape)

	#print('shape runname_var_info', len(runname_var_info))

	for i, runname in enumerate(runnames):
		for j, vid_hash in enumerate(vid_hashes):
			#print(i,j)
			vdat_type = 'retrieved'
			if i==0:
				indep_arr[j][:] = runname_var_info[i][vid_hash][vdat_type]['independent_var']
				indep_lbls[j] = runname_var_info[i][vid_hash][vdat_type]['independent_label']
				dep_lbls[j] = runname_var_info[i][vid_hash][vdat_type]['dependent_label']
			re_dep_arr[j][i,:] = runname_var_info[i][vid_hash][vdat_type]['dependent_var']
			re_dep_err[j][i,:] = runname_var_info[i][vid_hash][vdat_type]['dependent_err']
			vdat_type = 'apriori'
			ap_dep_arr[j][i,:] = runname_var_info[i][vid_hash][vdat_type]['dependent_var']
			ap_dep_err[j][i,:] = runname_var_info[i][vid_hash][vdat_type]['dependent_err']
	
	#print('indep_lbls', indep_lbls)
	#print('dep_lbls', dep_lbls)			

	sorted_idxs = np.argsort(lat_arr)
	sorted_lat_arr = lat_arr[sorted_idxs]
	#print(sorted_idxs)
	# sort everything else
	for j in range(nparams):	
		re_dep_arr[j][:,:] = re_dep_arr[j][sorted_idxs,:]
		re_dep_err[j][:,:] = re_dep_err[j][sorted_idxs,:]
		ap_dep_arr[j][:,:] = ap_dep_arr[j][sorted_idxs,:]
		ap_dep_err[j][:,:] = ap_dep_err[j][sorted_idxs,:]

	return(indep_lbls, dep_lbls, re_dep_arr, re_dep_err, ap_dep_arr, ap_dep_err, sorted_lat_arr, nindep_each_param, indep_arr)

def get_var_info(runname):
	ut.pINFO('Getting variable info from *.mre file for run {}'.format(runname))
	try:
		mre_data = nemesis.read.mre(runname)
	except nemesis.exceptions.NemesisReadError as nre:
		print(nre)
		ut.pWARN('Could not read *.mre file for run {}, skipping...'.format(runname))
		return(None)
	var_info = {}
	for i, (vid, vpar, ap, ae, rp, re, nv) in enumerate(zip(mre_data['varident'],
															mre_data['varparam'],
															mre_data['aprprof'],
															mre_data['aprerr'],
															mre_data['retprof'],
															mre_data['reterr'],
															mre_data['nxvar'])):
		vid_hash = '{} {} {}'.format(int(vid[0]),int(vid[1]),int(vid[2]))
		var_info[vid_hash] = {}
		var_info[vid_hash]['apriori'] = varident_to_labels_and_profiles(runname, vid, vpar, ap[:nv], ae[:nv])
		var_info[vid_hash]['retrieved'] = varident_to_labels_and_profiles(runname, vid, vpar, rp[:nv], re[:nv])
	return(var_info)

def combined_improvement_factor(runnames):
	"""
	Calculates the combined improvement factor for variables that share an independent variable (e.g. height, pressure, wavelength)
	accross all runnames. 

	ARGUMENTS:
		runnames
			<str,array> A list of strings that contain runnames to operate on

	RETURNS:
		A dictionary with the following structure:
			{ 'indep_lbl': { 	'improvement_factor':<float,array>,
								'independent_var':<float,array>,
								'n_vars':<int>
							}
			}
			
			indep_lbl
				<str> The label of the independent variable that the improvement fractor is for.
				is a dictionary that contains the following keys:

				improvement_factor
					<float,array> The summed improvement factor for the specified independent variable
				independent_var
					<float,array> The value of the specified independent variable
				n_vars
					<int> The number of variables that had the same independent variable and were summed together.
	"""
	imp_factors = {}
	example_type_info = {}
	for runname in runnames:
		try:
			mre_data = nemesis.read.mre(runname)
		except nemesis.exceptions.NemesisReadError as nre:
			print(nre)
			ut.pWARN("Could not read mre file for run {}, skipping".format(runname))

		for i, (vid, vpar, ap, ae, rp, re, nv) in enumerate(zip(mre_data['varident'],
																mre_data['varparam'],
																mre_data['aprprof'],
																mre_data['aprerr'],
																mre_data['retprof'],
																mre_data['reterr'],
																mre_data['nxvar'])):
			ap_var_info = varident_to_labels_and_profiles(runname, vid, vpar, ap[:nv], ae[:nv])
			re_var_info = varident_to_labels_and_profiles(runname, vid, vpar, rp[:nv], re[:nv])
			imp_fac = get_improvement_factor(re_var_info, ap_var_info)
			# assume all the runs we are operating on have the same variables.
			if len(example_type_info.get(re_var_info['independent_label'], []))==0:
				example_type_info[re_var_info['independent_label']] = re_var_info
			if len(imp_factors.get(re_var_info['independent_label'], []))==0:
				imp_factors[re_var_info['independent_label']] = {}
				imp_factors[re_var_info['independent_label']]['improvement_factor'] = imp_fac
				imp_factors[re_var_info['independent_label']]['independent_var'] = re_var_info['independent_var']
				imp_factors[re_var_info['independent_label']]['n_vars'] = 1
			else:
				imp_factors[re_var_info['independent_label']]['improvement_factor'] += imp_fac
				imp_factors[re_var_info['independent_label']]['n_vars'] += 1
	return(imp_factors)

def get_improvement_factor(re_var_info, ap_var_info):
	"""
	Calculates the "improvment factor"

		I_f = 1 - (re/r)/(ae/a)

	where
		re = retrieved error
		r = retrieved value
		ae = apriori error
		a = apriori value

	ARGUMENTS:
		re_var_info
			<dict> Dictionary of retrieved variables. Must have 'dependent_err', 'dependent_var' keys
		ap_var_info
			<dict> Dictionary of apriori variables. Must have 'dependent_err', 'dependent_var' keys

	RETURNS:
		imp_fac
			<float,array> Calculated improvement factor as a numpy array
	"""
	imp_fac = 1.0 - (re_var_info['dependent_err']/re_var_info['dependent_var'])/(ap_var_info['dependent_err']/ap_var_info['dependent_var'])
	return(imp_fac)

def ystep_plot(ax, xdata, ydata, *args, **kwargs):
	"""
	There's not a way of doing 'step' plots on the y-axis (or of swapping the axes) so I have to do
	this hacky stuff to make them work. I am using the matplotlib "step" plot in the background to
	make this work. See <https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.step.html> for the
	documentation of the function matplotlib.axes.Axes.step().

	ARGUMENTS:
		ax
			<MatplotlibAxesHandle> The axes object you wish to draw the graph in
		xdata [ndata]
			<float, array> The data that goes on the x-axis, as this is a non-standard plot it is usually the
			dependent variable.
		ydata [ndata]
			<float, array> The data that goes on the y-axis, as this is a non-standard plot it is usually the
			independent variable.
		args
			<list> A list of other positional arguments, will be passed through to the function "matplotlib.axes.Axes.step"
		kwargs
			<dict> A dictionary of all keword arguments, will be passed through to the function "matplotlib.axes.Axes.step"
	"""
	# The dependent variable will be along the x-axis, double up all the datapoints so x=[x1,x2,x3,...]
	# becomes x=[x1,x1,x2,x2,x3,x3,...]. These will be the x-values for the star and end of each line segment
	x_ap_dat = np.stack((xdata,xdata)).flatten(order='F')

	# now we need to generate the y-axis values, these will be the independent variable.
	# For the first data point, we want to draw a vertical line that has the datapoint (x1,y1) as it's mid point
	# so the y=[y1,y2,y3,...] vector must become y=[y0.5,y1.5,y1.5,y2.5,y2.5,y3.5,...], combining this with the
	# x vector, we get lines running between points (x1,y0.5)->(x1,y1.5)->(x2,y1.5)->(x2,y2.5)->(x3,y2.5)->(x3,y3.5)...
	# to get the correct y-vector we need the midpoints (y1.5, y2.5, ...)
	midpoints = (ydata[1:]+ydata[:-1])/2
	midpoints = np.stack((midpoints,midpoints)).flatten(order='F')
	# then create an array that can hold the mid-points as well as the two end-points
	y_ap_dat = np.zeros(midpoints.shape[0]+2)
	# copy the midpoints into the middle part of the array
	y_ap_dat[1:-1] = midpoints
	# set the endpoints to have the same gap from endpoint to datapoint as from datapoint to midpoint, 
	# i,e, y1-y0.5 = y1.5-y1
	# so y0.5 = 2*y1 - y1.5
	y_ap_dat[0] = 2*ydata[0] - midpoints[0]
	# similarly, yN - y(N-1).5 = yN.5 - yN
	# therefore, yN.5 = 2*yN - y(N-1).5
	y_ap_dat[-1] = 2*ydata[-1] - midpoints[-1]
	#print(x_ap_dat)
	#print(y_ap_dat)
	#print('x_ap_dat.shape {}'.format(x_ap_dat.shape))
	#print('y_ap_dat.shape {}'.format(y_ap_dat.shape))
	plot_hdl = ax.step(x_ap_dat, y_ap_dat, *args, **kwargs)
	return(plot_hdl)

def save_show_plt(fig, fname, outfolder=None, show_plot=False, quiet=False, writetext=True, save_plot=True):
	if outfolder!=None:
		if writetext: fig.text(0,0, '\n'.join(ut.str_wrap(os.path.abspath(outfolder),wrapsize=80)), fontsize=8, ha='left', va='top')
		if save_plot:
			fig.savefig(os.path.join(outfolder, fname), bbox_inches='tight')
		if not quiet: ut.pINFO('Plot saved as {}'.format(os.path.join(outfolder,fname)))
	if show_plot:
		if not quiet: ut.pINFO('Showing plot...')
		plt.show()
		if not quiet: ut.pINFO('Interactive plot closed.')
	return()

def varident_to_labels_and_profiles(runname, varident, varparams, state_vector, state_vector_err):
	"""
	Gets the correct axis labels and computed profiles for a passed parameter

	ARGUMENTS:
		runname
			<str> The name of the run to operate on
		varident [3]
			<int, array> The identity of the variable under consideration
		varparam [*]
			<float,array> Any extra parameters to the variable
		state_vector [nv]
			<float, array> The section of the state vector that corresponds to the variable under consideration
		state_vector_err [nv]
			<float, array> The section of the error on the state vector that corresponds to the variable under consideration

	RETURNS:
		A dictionay with the following keys:
			independent_var
				<float, array> The independent variable that the variable under consideration is computed with respect to (e.g. pressure grid)
			dependent_var
				<float, array> The value of the variable under consideration at each point of the independent variable
			dependent_err
				<float, array> The error on the variable under consideration
			independent_label
				<str> The name of the idependent variable (e.g. "Pressure")
			dependent_label
				<str> The name of the variable under consideration
	"""
	ref_data = nemesis.read.ref(runname)
	ngas = ref_data['ngas']
	height_grid = ref_data['height']
	pressure_grid = ref_data['press']
	aerosol_data = nemesis.read.aerosol(os.path.join(os.path.dirname(runname),'aerosol.ref'))
	ncont = aerosol_data['ncont']
	xsc_data = nemesis.read.xsc(runname)
	wavelength_grid = xsc_data['wavs']

	# Name of quantity is described by varident[0] and varident[1]
	# How the quantity is represented is described by varident[2]

	xlabel = None
	if varident[0]==0:
		xlabel='Temperature'
	elif -ncont<=varident[0]<0:
		xlabel='Cloud Density (Type {:d})'.format(int(-varident[0]))
	elif varident[0]==-(ncont+1):
		xlabel='para-H2 Fraction'
	elif varident[0]==-(ncont+2):
		xlabel='Fractional Cloud Cover'
	elif varident[0]==999:
		xlabel = 'Surface Temperature'
	elif varident[0]==888:
		xlabel = 'Surface Albedo Spectrum'
	elif varident[0]==889:
		xlabel='Surface Albedo Scaling Factor'
	elif varident[0]==887:
		xlabel='Cross Section Spectrum of Cloud Type'
	elif varident[0]==777:
		xlabel='Correction to the tangent height of limb observations'
	elif varident[0]==666:
		xlabel='Pressure at a defined altitude used for Mars MCS limb observations'
	elif varident[0]==555:
		xlabel='Planetary Radius'
	elif varident[0]==444:
		xlabel='Imaginary part of cloud type {:d}\'s refractive index'.format(int(varident[1]))
	elif varident[0]==445:
		xlabel='Imaginary part of cloud type {:d}\'s refractive index using Mie coated sphere model'.format(int(varident[1]))
	elif varident[0]==333:
		xlabel='Planetary surface gravity'
	elif varident[0]<100:
		# assume it's a gas type
		gas_data = nemesis.read.gasinforef_raddata()
		this_gas_data = gas_data[varident[0]]
		xlabel='{} isotopologue {:d} volume mixing ratio'.format(this_gas_data['name'], int(varident[1]))
	else:
		ut.pERROR('I couldn\'t be arsed to input all the "Magic Number" cases of varident so if you are using one of these you may need to edit the file ".../nemesis/plot.py", otherwise something went wrong when reading varident {}'.format(varident))

	var_repr_func = nemesis.cfg.var_repr_funcs.get(varident[2], nemesis.cfg.var_repr_not_implemented)
	var_info = var_repr_func(varparams, state_vector, state_vector_err, height_grid, pressure_grid, wavelength_grid)
	var_info.update({'dependent_label':xlabel})
	
	return(var_info)

# Have to put this after function declarations, otherwise they don't exists (duh...)
# Holds lists of functions that can be called when specifying a choice at command line
# To be part of this list, a function must accept the following interface
# 	[<matplotlib_figure>] = function_name(<str>, outfolder=<str>, show_plot=<bool>)
#
# 
plot_arg_choice_groups = 	{	'all':[mre_spectral, mre_parameter, chi_sq, improvement_factor, limb_darkening], # this shoud plot everything it can
								'mre':[mre_spectral, mre_parameter],
								'itr':[chi_sq],
								'mre_spectral':[mre_spectral],
								'mre_parameter':[mre_parameter],
								'improvement_factor':[improvement_factor],
								'limb_darkening':[limb_darkening]
							}
plot_arg_choice_help = """\
	Tells the program to plot certain groups of plots for individual nemesis runs. 
	Choices are:
		all
			mre_spectral
			mre_parameter
			chi_sq
			improvement_factor
		mre
			mre_spectral
			mre_parameter
		itr
			chi_sq
		mre_spectral
			mre_spectral
		mre_parameter
			mre_parameter
		improvement_factor
			improvement_factor
		limb_darkening
			limb_darkening
	Plot types are:
		mre_spectral
			Plots wavelegth vs radiance for measured and retrieved values. Also plots absolute and relative error.
		mre_parameter
			Plots a-priori and retrieved values for each parameter input to NEMESIS.
		chi_sq
			Plots chi-squared per degrees of freedom, and phi-sqaured per degrees of freedom for each iteration of the NEMESIS retrieval.
		improvement_factor
			Plots the improvement factor for each parameter, 
			Impfac = 1 - (re/r)(ae/a), 
			r = retrieved, 
			a=apriori, 
			e=error.
		limb_darkening
			Plots radiance vs cos(zenith) for each wavelength.\
	"""


ensemble_arg_choice_groups = 	{	'all':[ensemble_improvement_factor, ensemble_params_v_lat, ensemble_params_v_lon, ensemble_chi_phi_v_lat, ensemble_chi_phi_v_lon],
									'improvement_factor':[ensemble_improvement_factor],
									'params_at_max_imp_fac':[ensemble_parameters_vs_lat_max_imp_fac, ensemble_parameters_vs_lon_max_imp_fac],
									'params':[ensemble_params_v_lat, ensemble_params_v_lon],
									'chisq':[ensemble_chi_phi_v_lat, ensemble_chi_phi_v_lon],
									'minnaert':[ensemble_minnaert_limb_darkening]
								}
ensemble_arg_choice_help="""\
	Tells the program to plot cetain groups of plots for the entire ensemble of NEMESIS runs.
	Choices are:
		all
			ensemble_improvement_factor
			ensemble_params_v_lat
			ensemble_params_v_lon
			ensemble_chi_phi_v_lat
			ensemble_chi_phi_v_lon
		improvement_factor
			ensemble_improvment_factor
		params_at_max_imp_fac
			ensemble_parameters_vs_lat_max_imp_fac
			ensemble_parameters_vs_lon_max_imp_fac
		params
			ensemble_params_v_lat
			ensemble_params_v_lon
		chisq
			ensemble_chi_phi_v_lat
			ensemble_chi_phi_v_lon
		minnaert
			ensemble_minnaert_limb_darkening
	Plot types are:
		ensemble_improvement_factor
			Plots the mean improvement factor for each parameter
			ImpFac = 1 - (re/r)(ae/a),
		ensemble_params_v_lat
			Plots a large number of parameter plots at all different latitudes and combines them into a movie using ffmpeg
		ensemble_params_v_lon
			Plots a large number of parameter plots at all different longitudes and combines them into a movie using ffmpeg
		ensemble_chi_phi_v_lat
			Plots the final value of chi-sq and phi-sq per dof for each retrieval vs latitude
		ensemble_chi_phi_v_lon
			Plots the final value of chi-sq and phi-sq per dof for each retrieval vs longitude
		ensemble_parameters_vs_lat_max_imp_fac
			Plots a single plot of parameters vs latitude at the maximum value of that parameter's mean improvement factor
		ensemble_parameters_vs_lon_max_imp_fac
			Plots a single plot of parameters vs longitude at the maximum value of that parameter's mean improvement factor
		ensemble_minnaert_limb_darkening
			Plots a set of images designed to compare minnaert limb darkening before and after retrieveal. Combines them 
			into a movie using ffmpeg
		\
	"""

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
	class RawDefaultTypeHelpFormatter(ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter, ap.RawTextHelpFormatter):
		pass	
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
									formatter_class = RawDefaultTypeHelpFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('runnames', type=str, nargs='+', help=ut.str_block_indent_raw('The name of the nemesis run to plot (or the path to the <runname>.inp file to help with globbing)'))
	
	parser.add_argument('--plot', type=str, nargs='*', choices = plot_arg_choice_groups.keys(), 
						help=ut.str_block_indent_raw(textwrap.dedent(plot_arg_choice_help)),
						#help='File groups to plot, different groups correspond to different collections of plots. Choices corresponding to a <runname>.* file extension will plot data from that file',
						default=[])
	parser.add_argument('--ensemble_plot', type=str, nargs='*', choices=ensemble_arg_choice_groups.keys(),
						help=ut.str_block_indent_raw(textwrap.dedent(ensemble_arg_choice_help)),
						default=[])

	parser.add_argument('--plot_output_folder', type=str, 
						help=ut.str_block_indent_raw('Folder to output plots to. If a relative folder is passed it is relative to the directory containing the nemesis run. If an absolute folder is passed, all plots will be saved to that single folder.'), 
						default='./plots')

	parser.add_argument('--ensemble_output_folder', type=str,
						help=ut.str_block_indent_raw('Folder to output ensemble plots to. If passed, all ensemble plots will be placed in the folder "$PWD/plots" or another folder if one is specified'),
						default=os.path.abspath("./plots"))

	parser.add_argument('--regionfiles', type=str, nargs='*', help=ut.str_block_indent_raw('List of files that describe a region, will make a representation of them to aid with analysis'), default=[])

	parser.add_argument('--show_plots', action='store_true', help=ut.str_block_indent_raw('Should we show the plots interactively?'))
	parser.add_argument('--no_save_plots', action='store_true', help=ut.str_block_indent_raw('If present, we will not save the plots to disk'))

	parser.add_argument('--minnaert.ensemble.fits', type=str, help=ut.str_block_indent_raw('If plotting minnaert limb darkening, to create a synthetic image the fits file the data was grabbed from is needed'), default=None)
	parser.add_argument('--minnaert.ensemble.regrid_wavelength_to_fits', action='store_true', help=ut.str_block_indent_raw('If present, will regrid the nemesis results to the fits wavelength grid'))

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface

	# Filter Inputs Here
	parsed_args['runnames'] = [os.path.expanduser(x)[:-4] if os.path.expanduser(x).endswith('.inp') else os.path.expanduser(x) for x in parsed_args['runnames']]

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
