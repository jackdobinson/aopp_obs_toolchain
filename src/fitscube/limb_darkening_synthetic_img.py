#!/usr/bin/env python3
"""
This docstring should describe the program

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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
#import numpy.ma as ma
import utils as ut # used for convenience functions
from astropy.io import fits
import fitscube.fit_region
import fitscube.process.sinfoni
import subsample
import plotutils

def minnaert_limb_darkening(hdul, 
							check_region_flag=False, 
							zenmax=60, 
							latmin=-15, 
							latmax=10, 
							outmovname_prefix='limb_darkening', 
							wavgrid=None, 
							show_plot=True, 
							show_animated_plot=True, 
							save_plot=True, 
							save_animated_plot=True, 
							plot_dir=None, 
							frame_dir=None, 
							overwrite_existing=False):
	import nemesis.plot
	#print('IN minnaert_limb_darkening')
	
	# get main region indicies 'rids' and choked region indicies 'r2ids'
	rids = np.nonzero(np.logical_and.reduce([hdul['LATITUDE'].data>latmin, 
												hdul['LATITUDE'].data<latmax, 
												~np.isnan(hdul['LATITUDE'].data)]))
	r2ids = np.nonzero(np.logical_and.reduce([hdul['LATITUDE'].data>latmin, 
													hdul['LATITUDE'].data<latmax, 
													~np.isnan(hdul['LATITUDE'].data), 
													hdul['ZENITH'].data<zenmax]))
	#print('rids.shape', [x.shape for x in rids])
	#print('rids',rids)

	# turn regions into slices
	rs = np.s_[rids]
	rs2 = np.s_[r2ids]
	#print('rs',rs)

	# use this to check we are operating on the region we think we should be
	if check_region_flag and (save_plot or show_plot):
		# check that we are choosing the correct region
		f2 = plt.figure(figsize=[x/2.54 for x in (24,24)])
		a1 = f2.subplots(ncols=1,nrows=1,squeeze=False,gridspec_kw={'hspace':0.25,'wspace':0.25})
		# choose the data to display
		dat = np.nanmedian(hdul[0].data, axis=0)
		#dat = np.cos(hdul['ZENITH'].data[:,:]*np.pi/180)
		# set the chosen region to zeros, or another value that will stand out
		dat[rs] = 0.0 
		# plot the data
		im1 = a1[0,0].imshow(dat, origin='lower',cmap='viridis')
		# as fits files are ordered as [z,y,x] rids[1] should be the x-axis indices 
		# and rids[0] should be the y-axis indices

		sc1 = a1[0,0].scatter(rids[1],rids[0],s=1, c='red')
		sc2 = a1[0,0].scatter(r2ids[1],r2ids[0],s=0.5, c='green')
		# the red dots should line up with the region set to zeros (or another value 
		# that will stand out) if my understanding of
		# how the region is defined is correct

		plotutils.save_show_plt(f2, 'minnaert_limb_darkening_region.png', plot_dir, show_plot=show_plot, save_plot=save_plot)
		plt.close(f2)

	# get properties needed by minnaert limb darkening calculations for both regions
	u = np.cos((np.pi/180)*hdul['ZENITH'].data[rs])
	u0 = np.cos((np.pi/180)*hdul['ZENITH'].data[rs]) # solar zenith if almost identical to observer zenith for neptune and other outer planets
	u2 = np.cos((np.pi/180)*hdul['ZENITH'].data[rs2])
	u02 = np.cos((np.pi/180)*hdul['ZENITH'].data[rs2]) # solar zenith is almost identical to observer zenith for neptune and other outer planets

	# if we were't passed the wavelength grid to use, then assume hdul has 
	# enough information in it to get the wavelength grid via our standard method
	if type(wavgrid) == type(None):
		wavgrid = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])

	# create holders for minnaert limb darkening parameters. We will need to 
	# have space for each wavelength to have unique parameters
	xs, resids, ranks, ss, log_u0us, log_uIperFs, nanidxs = [],[],[],[],[],[],[]
	x2s, resid2s, rank2s, s2s, log_u0u2s, log_uIperF2s, nanidx2s = [],[],[],[],[],[],[]
	npx = np.prod(u.shape)
	npx2 = np.prod(u2.shape)


	# can I median filter the u, u0, radiances to try and reduce signal from clouds? 
	# I'm not sure there's a simple way of doing that.
	# I would have to make a histogram of u and u0, bin them together, see where the 
	# bins overlap, then median the radiances that correspond to the overlaping u and u0 bins.


	# loop over wavelengths and find limb darkening parameters, store them in the holders we just made
	for i in range(len(wavgrid)):
		radiances = hdul[0].data[i,:,:][rs]
		region_2_radiances = hdul[0].data[i,:,:][rs2]
		# DEBUGGING
		#print(f'i {i}')
		#print(f'u0 {u0}')
		#print(f'u {u}')
		#print(f'radiances {radiances}')
		#print('u.shape',u.shape)
		#print('radiances.shape',radiances.shape)
		#print(radiances)
		#if i > 50: sys.exit('DEBUG EXIT')
		# END DEBUGGING
		x, r, rank, s, log_u0u, log_uIperF, nanidx = nemesis.plot.fit_minnaert_limb_darkening(u0, u, radiances)	
		x_2, r_2, rank_2, s_2, log_u0u_2, log_uIperF_2, nanidx_2 = nemesis.plot.fit_minnaert_limb_darkening(u02, u2, region_2_radiances)
		#print('log_uIperF.shape', log_uIperF.shape)
		#print(log_uIperF)
		#print('nanidx.shape', nanidx.shape)
		#print(nanidx)
		#print(type(nanidx))
		x2s.append(x_2)
		xs.append(x)
		resids.append(r)
		resid2s.append(r_2)
		ranks.append(rank)
		rank2s.append(rank_2)
		ss.append(s)
		s2s.append(s_2)
		log_u0us.append(log_u0u)
		log_u0u2s.append(log_u0u_2)
		log_uIperFs.append(log_uIperF)
		log_uIperF2s.append(log_uIperF_2)
		nanidxs.append(nanidx)
		nanidx2s.append(nanidx_2)


	# if we should make a plot, create all the neccesary parts
	if save_animated_plot or show_animated_plot:
		f1 = plt.figure(figsize=[x/2.54 for x in (24,24)])
		#a1 = f1.subplots(nrows=2,ncols=1,squeeze=False,gridspec_kw={'hspace':0.25, 'wspace':0.25})
		a1=[]
		a1.append(f1.add_axes([0.1, 0.35, 0.8, 0.55])) # [left bottom width height]
		cax = f1.add_axes([0.925, 0.35, 0.025, 0.55]) # [left bottom width height]
	
		a1.append(f1.add_axes([0.1, 0.05, 0.6, 0.2])) # [left bottom width height]
		a1.append(f1.add_axes([0.75, 0.05, 0.2, 0.2]))		


		sca1 = a1[0].scatter([], [], c=[], s=10, marker='o', lw=0, label='datapoints')
		#sca1 = a1[0].plot([], [], markersize=1, marker='o', lw=0, label='datapoints')
		#sca1 = a1[0].scatter(log_u0u, log_uIperF, c=zens[~nanidxs], s=10, marker='o', lw=0, label='datapoints')
		cbar1 = f1.colorbar(sca1, cax=cax)
		cbar1.set_label('Zenith')
		plt1 = a1[0].plot([], [], ms=0, lw=1, color='red', label='whole region best fit line')
		plt2 = a1[0].plot([], [], ms=0, lw=1, color='green', label='reduced region best fit line')
		#plt1 = a1[0].plot(log_u0u, x[0] + x[1]*log_u0u, ms=0, lw=1, label='best fit line')
		a1[0].axvline(np.log(np.cos((np.pi/180)*zenmax)**2))
		a1[0].set_xlabel('log(u_0*u)')
		a1[0].set_ylabel('log(u*IperF)')
		a1[0].legend(loc='lower right')
			
		a1[1].plot(wavgrid, np.nanmean(hdul[0].data[:,rs[0],rs[1]], axis=(1)), label='Radiance')
		a1[1].set_xlabel('wavelength (um)')
		a1[1].set_ylabel('radiance')
		a1[1].set_title('Wavelength of limb darkening calculation')
		
		#dat = np.nanmean(hdul[0].data,axis=0)
		#dat[rs]=0
		dat = np.zeros_like(hdul[0].data[0])
		im1 = a1[2].imshow(dat, origin='lower', cmap='viridis')
		a1[2].scatter(rids[1],rids[0],s=1,c='red', label='full region')
		a1[2].scatter(r2ids[1], r2ids[0], s=0.5, c='green', label='zenith restricted region')
		a1[2].set_title('Region')

		#print(plt1)


		# define the limits of the graph, can't use NAN or INF for default limits so just use something sensible
		xmin = min([np.nanmin(a) if len(a)>0 else -10 for a in log_u0us])
		xmax = max([np.nanmax(a) if len(a)>0 else 0 for a in log_u0us])
		ymin = min([np.nanmin(a) if len(a)>0 else -10 for a in log_uIperFs])
		ymax = max([np.nanmax(a) if len(a)>0 else 0 for a in log_uIperFs])

		# if we get the limits the wrong way around, swap them
		def swap(a,b):
			return(b,a)
		if xmin > xmax:
			xmin,xmax = swap(xmin,xmax)
		if ymin > ymax:
			ymin, ymax = swap(ymin, ymax)
		#print('xlimits', xmin, xmax)
		#print('ylimits', ymin, ymax)
		a1[0].set_xlim([xmin,xmax])
		a1[0].set_ylim([ymin,ymax])

		
		sca1.set_offset_position('screen') 	# this means we can use datavalues to change the offsets, 
											# this seems to be the wrong way around according to the docs but it works

		# define the colormap for the scatter points, use zenith angle as a sanity check
		zencolours = sca1.get_cmap()((hdul['ZENITH'].data[rs]-0)/(90)) 	# use 'rs' because we want to 
																		# plot the WHOLE region as a scatter graph
		sca1.set_clim(0, 90)

		# get number of frames and the number of digits needed to have a fixed-width representation
		n = len(wavgrid)
		ndigits = int(np.ceil(np.log10(n)))+1

		#f1.tight_layout
		#if show_plot:
		#	plt.ion()
		#	plt.show()
		name=f'{outmovname_prefix}_latmin_{latmin}_latmax_{latmax}_zenmax_{zenmax}'
		fmt_str = '{:0'+f'{ndigits}'+'}.png'
		this_frame_dir = os.path.join(frame_dir, name)
		os.makedirs(this_frame_dir, exist_ok=True)

		wavmarkerline = None	
		dat=None
		#for i in range(n):
		def update(i):
			nonlocal wavmarkerline
			nonlocal dat
			#print(f'log_uIperFs[{i}].shape', log_uIperFs[i].shape)
			#print(log_uIperFs[i])
			#print(f'log_uIperFs[{i}].get_fill_value()', log_uIperFs[i].get_fill_value())
			titlestr = (f'frame {i}, minnaert limb darkening: lambda={wavgrid[i]:#0.4G}um\nzenmax={zenmax} latmin={latmin} latmax={latmax}')
			if type(wavmarkerline) != type(None):
				wavmarkerline.remove()
			wavmarkerline = a1[1].axvline(wavgrid[i], 0, 1, c='red') # create vertical line at position of displayed wavelength
			if len(log_uIperFs[i])>0:
				#print('HERE')
				points = np.stack([log_u0us[i], log_uIperFs[i]], axis=-1)
				#print('npoints', points.shape)
				#print(points)
				#print(type(points))
				#print(points.get_fill_value())
				sca1.set_offsets(points)
				sca1.set_facecolor(zencolours[~nanidxs[i]])
				#print(f'log_uIperFs[{i}].shape', log_uIperFs[i].shape)
				#print(log_uIperFs[i])
				#print(f'log_uIperFs[{i}].get_fill_value()', log_uIperFs[i].get_fill_value())
				#a1[0].scatter(log_u0us[i], log_uIperFs[i], c=zencolours[~nanidxs[i]], s=10, lw=0, marker='o')
				plt1[0].set_data(log_u0us[i], xs[i][0]+xs[i][1]*log_u0us[i])
				plt2[0].set_data(log_u0us[i], x2s[i][0] + x2s[i][1]*log_u0us[i])
				titlestr = titlestr + f'\nWhole Region:   k_w={xs[i][1]:#0.3G} ln(I/F_0)_w={xs[i][0]:#0.3G}\nReduced Region: k_r={x2s[i][1]:#0.3G} ln(I/F_0)_r={x2s[i][0]:#0.3G}'
			else:
				#print('THERE')
				sca1.set_offsets(np.zeros((0,2)))
				plt1[0].set_data(np.array([]),np.array([]))
				plt2[0].set_data(np.array([]),np.array([]))
				titlestr = titlestr + '\nWhole Region:   No data\nReduced Region: No data'
			a1[0].set_title(titlestr, loc='left')
			dat = hdul[0].data[i,:,:]
			im1.set_data(dat)
			im1.set_clim([np.nanmin(dat), np.nanmax(dat)])
		ani = matplotlib.animation.FuncAnimation(f1, update, range(n), interval=100)
		outmovname = f'{name}.mov'
		plotutils.save_animate_plt(ani, outmovname, plot_dir, animate_plot=show_animated_plot, save_plot=(save_animated_plot and (os.path.exists(outmovname) and overwrite_existing)))

	# minnaert limb darkening for the full range of zenith angles
	mld_full = {'ks':np.array([_x[1] for _x in xs]),
				'log_IperF0s':np.array([_x[0] for _x in xs]),
				'residuals':np.array(resids), 
				'ranks':ranks,
				'singulars':ss,
				'log_u0us':log_u0us, 
				'log_uIperFs':log_uIperFs,
				'nanidxs':nanidxs,
				'nwavs':len(nanidxs),
				'wavgrid':wavgrid,
				'zenmax':90,
				'latmin':latmin,
				'latmax':latmax,
				'npx':npx
				}

	# minnaert limb darkening for the range of zenith angles up to zenmax
	mld_small = {'ks':np.array([_x[1] for _x in x2s]),
				'log_IperF0s':np.array([_x[0] for _x in x2s]),
				'residuals':np.array(resid2s), 
				'ranks':rank2s,
				'singulars':s2s,
				'log_u0us':log_u0u2s, 
				'log_uIperFs':log_uIperF2s,
				'nanidxs':nanidx2s,
				'nwavs':len(nanidx2s),
				'wavgrid':wavgrid,
				'zenmax':zenmax,
				'latmin':latmin,
				'latmax':latmax,
				'npx':npx2
				}

	return(mld_full, mld_small)

def progress(ipos, imax, message='Progress', verbose=False):
	if ipos<imax-1:
		if verbose:
			print(f'{message} {ipos} of {imax}')
		else:
			sys.stdout.write(f'\r{message} {ipos} of {imax}')
			sys.stdout.flush()
	else:
		if not verbose:
			sys.stdout.write(f'\r{message} {ipos} of {imax}\n')	
			sys.stdout.flush()
	return

def smooth_wav(data, n=200, nstart=0, nend=-1, wavgrid=None, mode='conv', show_plot=True):
	"""
	Smooth a data cube along wavelength axis so it has a size of 'n' along that axis evenly spaced, assume 3 axes and that wavelength is the 0th one
	"""
	print('IN smooth_wav')
	shape = data.shape
	n_orig = shape[0]
	if nend <0:
		nend = n_orig +1+ nend
	nrange = nend - nstart
	n_conv_size = int(np.around(nrange/n))
	nvals = np.linspace(nstart, nend, n)

	n_shape = list(shape)
	n_shape[0] = n

	sdata = np.zeros(n_shape)

	#print(shape)
	#print(nvals.shape)
	#print(n_conv_size)
	if 'conv' in mode:
		conv_funcs = {'conv_top_hat':subsample.norm_top_hat, 'conv_triangle':subsample.norm_triangle}
		if mode in conv_funcs.keys():
			kernel=conv_funcs[mode](n_conv_size)
		else:
			print(f'ERROR: Unrecognised mode "{mode}", recognised "conv" modes are {tuple(conv_funcs.keys())}, exiting...')
			sys.exit()
		for i in range(shape[1]):
			for j in range(shape[2]):
				conv_data = np.convolve(data[nstart:nend,i,j], kernel, mode='same')
				#print(conv_data.shape)	
				sdata[:,i,j] = np.interp(nvals, np.arange(0,n_orig), conv_data, left=np.nan, right=np.nan)
	elif mode=='median':
		import scipy.signal
		median_data = np.zeros_like(data)
		if n_conv_size%2==0:
			n_conv_size += 1 # must be odd for this to work
		median_data = scipy.signal.medfilt(data, (n_conv_size,1,1))
		for i in range(shape[1]):
			for j in range(shape[2]):
				sdata[:,i,j] = np.interp(nvals, np.arange(0,n_orig), median_data[:,i,j], left=np.nan, right=np.nan)
	else:
		print(f'ERROR: Unrecognised mode {mode}, exiting...')
		sys.exit()


	if type(wavgrid)!=type(None):
		outwavgrid = np.interp(nvals, np.arange(0,n_orig), wavgrid, left=np.nan, right=np.nan)

	if show_plot:
		f1 = plt.figure(figsize=[x/2.54 for x in (24,16)])	
		a1 = f1.subplots(nrows=1, ncols=1, squeeze=False, gridspec_kw={'hspace':0.25, 'wspace':0.25})

		yvals = np.nanmean(data, axis=(1,2))
		ysmoothvals = np.nanmean(sdata, axis=(1,2))
		if type(wavgrid)==type(None):
			xvals = range(len(yvals))
			xsmoothvals = nvals
		else:
			xvals = wavgrid
			xsmoothvals = outwavgrid

		a1[0,0].plot(xvals, yvals)
		a1[0,0].plot(xsmoothvals, ysmoothvals)
		a1[0,0].set_ylim((0, 1.1*np.nanmax(ysmoothvals)))

		plt.show()

	if type(wavgrid) == type(None):
		return(sdata)
	else:
		return(sdata, outwavgrid)
	

def synthetic_image_minnaert(hdul, mlds, mld_smalls, outmovname='minneart_synth_img', show_plot=True, show_animated_plot=True, save_plot=True, save_animated_plot=True, plot_dir=None, frame_dir=None):
	"""
	Using: log(u*(I/F)) = log((I/F)_0) + k*log(u_0*u)
	(I/F) = (I/F)_0 * u0^k * u*(k-1)
	"""
	import copy
	print('IN synthetic_image_minnaert')
	cmap = copy.copy(plt.cm.viridis)
	cmap.set_bad(alpha=0.0)
	cmap.set_under('red')
	cmap.set_over('cyan')

	# 256 colors in entire range, therefore join between colorbars should be at position 256*x_join_frac
	residual_cmap = lambda x_join_frac: mpl.colors.LinearSegmentedColormap.from_list('residual_cmap',
						np.vstack([
							mpl.cm.viridis(np.linspace(0,1,int(np.ceil(x_join_frac*256)))), 
							mpl.cm.autumn_r(np.linspace(0,1,int(np.floor(1.0-x_join_frac)*256)))
						]))

	u = np.cos((np.pi/180)*hdul['ZENITH'].data)
	u0 =  np.cos((np.pi/180)*hdul['ZENITH'].data)
	IperF = np.full((mlds[0]['nwavs'], u.shape[0], u.shape[1]), np.nan)
	IperF_s = np.full((mld_smalls[0]['nwavs'], u.shape[0], u.shape[1]), np.nan)
	#print('number of mlds', len(mlds))

	for i, (mld, mld_small) in enumerate(zip(mlds, mld_smalls)):
		#print(f'\tconstructing data for latitude section {i}')
		#for k, v in mld.items():
		#	print(f'### {k} ###')
		#	print(v)
		if all([_x==0 for _x in mld['ranks']]):
			continue
		#print(f'using mld {i}')
		log_IperF0 = np.array(mld['log_IperF0s'])
		IperF0 = np.exp(log_IperF0)
		k = np.array(mld['ks'])
		aslice = np.nonzero(np.logical_and.reduce([hdul['LATITUDE'].data>mld['latmin'], hdul['LATITUDE'].data<mld['latmax'], ~np.isnan(hdul['LATITUDE'].data)]))
		IperF[:,aslice[0],aslice[1]] = (IperF0[:,None,None] * u0[None,:,:]**k[:,None,None] * u[None,:,:]**(k[:,None,None]-1))[:,aslice[0],aslice[1]]
		#IperF[:,aslice[0],aslice[1]] = (IperF0[:,None,None]*np.ones_like(u)[None,:,:])[:,aslice[0],aslice[1]]
		#IperF[:,aslice[0], aslice[1]] = i

		#print(f"latmin {mld['latmin']} latmax {mld['latmax']}")
		#print('ks', k)
		#print(f'ks[30] {k[30]}')
		#print('IperF0s', IperF0)
		#print(f'log_IperF0s[30] {log_IperF0[30]}')
		#print(f'IperF0s[30] {IperF0[30]}')

		log_IperF0_s = np.array(mld_small['log_IperF0s'])
		IperF0_s = np.exp(log_IperF0_s)
		k_s = np.array(mld_small['ks'])
		aslice_s = np.nonzero(np.logical_and.reduce([hdul['LATITUDE'].data>mld['latmin'], hdul['LATITUDE'].data<mld['latmax'], ~np.isnan(hdul['LATITUDE'].data)]))
		IperF_s[:,aslice_s[0],aslice_s[1]] = (IperF0_s[:,None,None] * u0[None,:,:]**k_s[:,None,None] * u[None,:,:]**(k_s[:,None,None]-1))[:,aslice_s[0],aslice_s[1]]
		

	if save_animated_plot or show_animated_plot:
		#print('\tsetting up plot environment')
		static_plot_idx = int(np.around(hdul[0].data.shape[0]/2))	
		f1 = plt.figure(figsize=[x/2.54 for x in (3*16,2*16+8)])

		a1 = f1.add_axes([0.05, 0.6, 0.25, 0.35]) # [left bottom width height]
		cax1 = f1.add_axes([0.305, 0.6, 0.01, 0.35])
		a2 = f1.add_axes([0.35, 0.6, 0.25, 0.35]) # [left bottom width height]
		cax2 = f1.add_axes([0.605, 0.6, 0.01, 0.35])
		a3 = f1.add_axes([0.65, 0.6, 0.25, 0.35]) # [left bottom width height]
		cax3 = f1.add_axes([0.91,0.6,0.01, 0.35])

		a4 = f1.add_axes([0.05, 0.2, 0.25, 0.35]) # [left bottom width height]
		cax4 = f1.add_axes([0.305, 0.2, 0.01, 0.35])
		a5 = f1.add_axes([0.35, 0.2, 0.25, 0.35]) # [left bottom width height]
		cax5 = f1.add_axes([0.605, 0.2, 0.01, 0.35])
		a6 = f1.add_axes([0.65, 0.2, 0.25, 0.35]) # [left bottom width height]
		cax6 = f1.add_axes([0.91,0.2,0.01, 0.35])

		a7 = f1.add_axes([0.05, 0.05, 0.9, 0.1])
	
		#full_region_residual = (hdul[0].data[static_plot_idx]/IperF[static_plot_idx])**2
		#restricted_region_residual = (hdul[0].data[static_plot_idx]/IperF_s[static_plot_idx])**2
		full_region_residual = np.log(((hdul[0].data.data - IperF)/hdul[0].data.data)**2)
		restricted_region_residual = np.log(((hdul[0].data.data - IperF_s)/hdul[0].data.data)**2)
		full_residual_image_title = 'Residual log[((comparison - synthetic)/comparison)^2]'
		restricted_residual_image_title = 'Residual log[((comparison - synthetic)/comparison)^2]'

		#comparison_data = np.log(hdul[0].data)
		comparison_data = np.log(hdul[0].data.data)
		comparison_data_title = 'raw data for comparison'
		
		#full_synthetic_image = np.log(IperF)
		#restricted_synthetic_image = np.log(IperF_s)
		full_synthetic_image = np.log(IperF)
		restricted_synthetic_image = np.log(IperF_s)

		mask = [np.stack(np.nonzero(x)[::-1], axis=-1) for x in hdul[0].data.mask]
		#print(mask)


		cmap1 = mpl.cm.Spectral_r

		vmin,vmax = (np.nanmin(hdul[0].data[static_plot_idx]), np.nanmax(hdul[0].data[static_plot_idx]))
		im1 = a1.imshow(full_synthetic_image[static_plot_idx], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap) # full region synthetic image
		cbar1 = f1.colorbar(im1, cax=cax1, extend='both')
		im2 = a2.imshow(comparison_data[static_plot_idx], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap) # data for comparison
		sca1 = a2.scatter(*mask[static_plot_idx].T, s=1, c='red', label='masked data')
		cbar2 = f1.colorbar(im2, cax=cax2, extend='both')
		im3 = a3.imshow(full_region_residual[static_plot_idx], origin='lower', vmin=-6, vmax=6, cmap=cmap1) #residual of full image and data
		cbar3 = f1.colorbar(im3, cax=cax3, extend='both')

		im4 = a4.imshow(restricted_synthetic_image[static_plot_idx], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap) # restricted region synthetic image
		cbar4 = f1.colorbar(im4, cax=cax4, extend='both')
		im5 = a5.imshow(comparison_data[static_plot_idx], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap) # data for comparison
		sca2 = a5.scatter(*mask[static_plot_idx].T, s=1, c='red', label='masked data')
		cbar5 = f1.colorbar(im5, cax=cax5, extend='both')
		im6 = a6.imshow(restricted_region_residual[static_plot_idx], origin='lower', vmin=-6, vmax=6, cmap=cmap1) # residual of full image and data
		cbar6 = f1.colorbar(im6, cax=cax6, extend='both')

		l1 = a7.plot(mlds[0]['wavgrid'], np.nanmean(hdul[0].data, axis=(1,2)), lw=1)

		a1.set_title('Full region minnaert limb darkened synthetic image')
		a2.set_title(comparison_data_title)
		a3.set_title(full_residual_image_title)

		a4.set_title('Restricted region minnaert limb darkened synthetic image')
		a5.set_title(comparison_data_title)
		a6.set_title(restricted_residual_image_title)
		wline = None
		a7.set_xlabel('Wavelength (um)')

		if False: # put every frame on the same colour scale
			vmin, vmax = (np.nanmin(comparison_data), np.nanmax(comparison_data))
			for im in (im1, im2, im4, im5):
				im.set_clim((vmin,vmax))

			vmin, vmax = (np.nanmin(full_region_residual), np.nanmax(full_region_residual))
			im3.set_cmap(residual_cmap((1-vmin)/(vmax-vmin)))
			im3.set_clim((vmin, vmax))

			vmin, vmax = (np.nanmin(restricted_region_residual), np.nanmax(restricted_region_residual))
			im6.set_cmap(residual_cmap((1-vmin)/(vmax-vmin)))
			im6.set_clim((np.nanmin(restricted_region_residual), np.nanmax(restricted_region_residual)))

		def update(i):
			nonlocal wline
			im1.set_data(full_synthetic_image[i])
			im2.set_data(comparison_data[i])
			sca1.set_offsets(mask[i])
			im3.set_data(full_region_residual[i])		
			im4.set_data(restricted_synthetic_image[i])
			im5.set_data(comparison_data[i])
			sca2.set_offsets(mask[i])
			im6.set_data(restricted_region_residual[i])
			if True: # put each frame on it's own colour scale, but make comparison data and synthetic images share a colourscale
				vmin,vmax = (np.nanmin(comparison_data[i]), np.nanmax(comparison_data[i]))
				for im in (im1,im2,im4,im5):
					im.set_clim((vmin,vmax))
				#cmin1, cmax1 = (np.nanmin(full_region_residual[i]), np.nanmax(full_region_residual[i]))
				#cmin2, cmax2 = (np.nanmin(restricted_region_residual[i]), np.nanmax(restricted_region_residual[i]))
				#im3.set_cmap(residual_cmap((1-cmin1)/(cmax1-cmin1)))
				#im6.set_cmap(residual_cmap((1-cmin2)/(cmax2-cmin2)))
				#im3.set_clim((cmin1, cmax1))
				#im6.set_clim((cmin2, cmax2))
			if type(wline)!=type(None):
				wline.remove()
			wline = a7.axvline(mlds[0]['wavgrid'][i])
			a7.set_title(f'Wavelentgh bin {i}')
			return
		#print(mlds[0]['nwavs'])
		ani = matplotlib.animation.FuncAnimation(f1, update, range(mlds[0]['nwavs']))
		#print(plot_dir)
		#print(plot_dir.replace(':','\:'))
		#ani_dir = os.path.join(plot_dir, frame_dir, outmovname)
		#os.makedirs(ani_dir, exist_ok=True)
		#ani_file_tmp = os.path.abspath(os.path.join(ani_dir, f'{outmovname}.mp4'))
		#ani_file = os.path.abspath(os.path.join(plot_dir, f'{outmovname}.mov'))
		#ani.save(ani_file, progress_callback=progress)
		plotutils.save_animate_plt(ani, f'{outmovname}.mov', plot_dir, animate_plot=show_animated_plot, save_plot=save_animated_plot)
		#os.rename(ani_file_tmp, ani_file)
	return
	
def mask_bright_spots(hdul, mask_frac=0.3, mask_type='additive', wavgrid=None, wavlims=(-np.inf, 2.3), static_mask_wav=1.67, show_plot=False, show_animated_plot=False, save_plot=True, save_animated_plot=True,
					 plot_dir=None, frame_dir=None, overwrite=False):
	"""
	Identifies overly bright regions and masks them out.
	
	# ARGUMENTS #
		hdul
			<header data unit list> List of header data units of a fits file, we will be operating on the 0th memeber
		mask_frac
			<float> The fraction of difference between the median and the maximum that will be considered 'bright'
		mask_type
			<str> How the mask is calculated, ('adaptive', 'static', 'additive').
				'adaptive' 
					uses a different mask for each wavelength
				'static' 
					uses the same mask for each wavelength based off of the mask for a single wavelength
				'additive' 
					uses the same mask for each wavelength, where a pixel is masked if it would be masked in ANY
					of the 'adaptive' masks
		wavgrid
			<array, float> An array of floats that describes the wavelengths (usually in um, but as long as it's the
			same unit as 'static_mask_wav' it doesn't matter) that each slice of the hdul[0].data[i,:,:] attribute
			corresponds to. If None, will use the hdul[0].header to calculate, but if the data has been smoothed
			or rebinned along the spectral axis then this method will yield incorrect results so you must pass the
			correct wavelength grid.
		wavlims
			<2, float> 2 numbers which describe the minimum and maximum wavelengths to consider in the mask
		static_mask_wav
			<float> The wavelength to create a mask of bright spots for (if 'adaptive_mask_flag=False'). Should be in
			the same units as 'wavgrid' (usually um). If 'static_mask_wav' is not in 'wavgrid' then the closest
			wavelength will be used.
		show_plot
			<bool> If true will show plots for debugging and visualisation
		show_animated_plot
			<bool> If True will show animated plots (movies) interactively
		save_plot
			<bool> If True will save plots to disk
		save_animated_plot
			<bool> If True will save animated plots (movies) to disk
		plot_dir
			<str> Directory to save plots to
		frame_dir
			<str> Directory to save movie frames to if needed
		overwrite
			<bool> If present will overwrite existing plots and movies

	"""
	print('IN mask_bright_spots')


	mask3d = np.zeros_like(hdul[0].data, dtype='bool') # False = not masked out (i.e. data is good), True = masked out (i.e. data is bad)

	astr = ''
	mask_type_choices = ('adaptive', 'additive', 'static', 'manual')
	if not any([mask_type==choice for choice in mask_type_choices]):
		print(f'ERROR: Must set a mask type to one of {mask_type_choices}, passed value is {mask_type}')
		raise ValueError
	if mask_type in ('adaptive',):
		for i in range(hdul[0].data.shape[0]):
			if wavgrid[i] < wavlims[0] or wavgrid[i] > wavlims[1]:
				continue
			spec_slice = hdul[0].data[i,:,:]
			mask2d = np.zeros_like(spec_slice, dtype='bool')
			mask2d = spec_slice > mask_frac*(np.nanmax(spec_slice) - np.nanmedian(spec_slice, axis=(0,1))) + np.nanmedian(spec_slice, axis=(0,1))
			mask3d[i] = mask2d
		astr = f'adaptive mask\n'

	if mask_type in ('additive',):
		mask2d = np.zeros(hdul[0].data.shape[1:], dtype='bool')
		for i in range(hdul[0].data.shape[0]):
			if wavgrid[i] < wavlims[0] or wavgrid[i] > wavlims[1]:
				continue
			spec_slice = hdul[0].data[i,:,:]
			mask2d = np.logical_or(mask2d, spec_slice > mask_frac*(np.nanmax(spec_slice) - np.nanmedian(spec_slice, axis=(0,1))) + np.nanmedian(spec_slice, axis=(0,1)))
		mask3d[:,:,:] = mask2d[None,:,:]
		astr = f'additive mask\n'

	if mask_type in ('static',):
		if type(wavgrid)==type(None):
			# if we've not been passed a wavelength grid then grab it using our normal method
			wavgrid = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])
		#print(wavgrid)
		idx = np.nanargmin((np.abs(wavgrid - static_mask_wav))) # find the index of 'wavgrid' closest to 'static_mask_wav'
		spec_slice =  hdul[0].data[idx, :, :] 
		spec_slice_max = np.nanmax(spec_slice)
		spec_slice_median = np.nanmedian(spec_slice)
		mask2d = spec_slice > (mask_frac*(spec_slice_max - spec_slice_median) + spec_slice_median)
		mask3d[:,:,:] = mask2d[None,:,:]
		astr = f'static mask at wavelength {static_mask_wav} idx {idx}\n'

	if mask_type in ('manual',):
		viewer = fitscube.fit_region.FitscubeMaskPainter(hdul)
		viewer.run()
		mask3d = viewer.getMask3d()
		astr = 'manually specified mask'

	# PLOTTING
	# set up and initialise plot
	f1 = plt.figure(figsize=[x/2.54 for x in (16,5/4*16)])
	a1 = f1.subplots(nrows=2,ncols=1,squeeze=False,gridspec_kw={'hspace':0.25, 'wspace':0.25, 'height_ratios':(4,1)})
	spec_bin_list = range(hdul[0].data.shape[0])
	data_median = np.nanmedian(hdul[0].data, axis=(1,2))

	im1 = a1[0,0].imshow(np.full(hdul[0].data.shape[1:], np.nan), origin='lower')
	sca1 = a1[0,0].scatter([], [], s=1, c='red', label='masked data')
	l1 = a1[1,0].plot(spec_bin_list, data_median)
	wline = a1[1,0].axvline(0, color='red', label='current wavelength bin')
	print(f'INFO: wline {wline}')
	a1[0,0].legend()
		
	a1[1,0].set_xlabel('Wavelength bin')
	a1[1,0].set_ylabel('Radiance')
	a1[1,0].legend()
	
	def update(i):
		nonlocal wline
		spec_slice =  hdul[0].data[i, :, :] 
		idxs = np.nonzero(mask3d[i,:,:])
		title = f"{astr}{i}th slice with values > {mask_frac}*(max-median)+median masked out"
		a1[0,0].set_title(title)
		im1.set_data(spec_slice)
		im1.set_clim((np.nanmin(spec_slice), np.nanmax(spec_slice)))
		sca1.set_offsets(np.stack(idxs[::-1], axis=-1))
		wline.remove()
		wline = a1[1,0].axvline(i, color='red')
		a1[1,0].set_title(f'Wavelentgh bin {i}')
		plt.pause(0.1)
	ani = matplotlib.animation.FuncAnimation(f1, update, range(hdul[0].data.shape[0]), interval=100)
	plotutils.save_animate_plt(ani, 'cloud_mask.mov', plot_dir, animate_plot=show_animated_plot, save_plot=save_animated_plot, overwrite=overwrite)
	#plt.ioff()
	#plt.close()

	# hdul[0].data[mask3d] = np.nan
	# return(hdul[0].data)
	return(np.ma.array(hdul[0].data, mask=mask3d, fill_value=np.nan))	


def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	FFWriter = matplotlib.animation.writers['ffmpeg']()
	plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

	for tc in args['target_cubes']:
		# create relative plot and frame directories
		tc_dir = os.path.dirname(tc)
		plot_dir = os.path.join(os.path.dirname(tc), args['plot_dir'])
		frame_dir = os.path.join(os.path.dirname(tc), args['frame_dir'])
		os.makedirs(plot_dir, exist_ok=True)
		os.makedirs(frame_dir, exist_ok=True)
		print(f'INFO: Saving plots to directory {plot_dir}')
		print(f'INFO: Saving frames to directory {frame_dir}')

		with fits.open(tc) as hdul:
			wavgrid = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])
			hdul[0].data, wavgrid = smooth_wav(hdul[0].data, wavgrid=wavgrid, mode='conv_top_hat', show_plot=False)
			# DEBUGGING
			#hdul[0].data = hdul[0].data[34:35]
			#wavgrid = wavgrid[34:35]
			#print(wavgrid)
			# ---------
			if args['mask.load'] is not None:
				with fits.open(os.path.join([tc_dir,args['mask.load']])) as mask_hdul:
					hdul[0].data = np.ma.array(hdul[0].data, mask=mask_hdul[0].data, fill_value=np.nan)

			else:
				hdul[0].data = mask_bright_spots(hdul, wavgrid=wavgrid, mask_type=args['mask.mode'],
												show_plot=args['show_plots'], show_animated_plot=args['show_animated_plots'], 
												save_plot=args['save_plots'], save_animated_plot=args['save_animated_plots'], 
												plot_dir=plot_dir, frame_dir=frame_dir, overwrite=args['overwrite_mask_plots'])
			if args['mask.save'] is not None:
				mask_hdul = fits.HDUList([fits.PrimaryHDU(hdul[0].data.mask)])
				mask_hdul.writeto(os.path.join([tc_dir, args['mask.save']]))

			#print('hdul[0].data.get_fill_value()', hdul[0].data.get_fill_value())
			zenmax = 60
			latvals = np.linspace(-90, 90, 37)
			#latvals = np.linspace(-90, 90, 37)[13:15]
			#latvals = np.linspace(-10, -5, 2) # FOR DEBUGGING
			mld_fulls = []
			mld_smalls = []
			for i in range(len(latvals)-1):
				print(f'INFO: Creating minnaert limb darkening parameters for latitude {latvals[i]}')
				mld_full, mld_small = minnaert_limb_darkening(hdul, show_plot=args['show_plots'], plot_dir=plot_dir, frame_dir=frame_dir, show_animated_plot=args['show_animated_plots'], 
																save_plot=args['save_plots'], save_animated_plot=args['save_animated_plots'], overwrite_existing=args['overwrite_existing_limb_darkening_plots'],
																zenmax=zenmax, latmin=latvals[i], latmax=latvals[i+1], outmovname_prefix='limb_darkening_smoothed_masked', wavgrid=wavgrid)
				#print(mld_full)
				#print(mld_small)
				mld_fulls.append(mld_full)
				mld_smalls.append(mld_small)
			# disabled for DEBUGGING
			synthetic_image_minnaert(hdul, mld_fulls, mld_smalls, outmovname='minnaert_synth_img_top_hat_masked', show_plot=args['show_plots'], 
																plot_dir=plot_dir, frame_dir=frame_dir, show_animated_plot=args['show_animated_plots'], 
																save_plot=args['save_plots'], save_animated_plot=args['save_animated_plots'])
	return

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

	parser.add_argument('target_cubes', type=str, nargs='+',  help='fitscubes to operate on')

	# General Plotting
	parser.add_argument('--plot_dir', type=str, help='Directory (relative to the current member of "target_cubes" being operated on) to save plots to (including movies)', default='./plots')
	parser.add_argument('--frame_dir', type=str, help='Directory (relative to the current member of "target_cubes" being operated on) to save frames of movies to, not all movies need to have their frames saved', default='./plots/frames')
	parser.add_argument('--show_plots', action='store_true', help='If present, will show static plots interactively')
	parser.add_argument('--show_animated_plots', action='store_true', help='If present will show animated plots interactively')
	parser.add_argument('--no_save_plots', action='store_false', dest='save_plots', help='If present will not save any plots to disk')
	parser.add_argument('--no_save_animated_plots', action='store_false', dest='save_animated_plots', help='If not present will not save any animated plots (movies) to disk')

	# Specific plotting
	parser.add_argument('--overwrite_existing_limb_darkening_plots', action='store_true', help='If present will re-compute intermediate plots that show log(I/F) vs log(u_0*u) for each latitude slice (takes a while).')
	parser.add_argument('--overwrite_mask_plots', action='store_true', help='If present, will overwrite movies and plots of the cloud mask (recomputation always happens as this is fast).')

	parser.add_argument('--mask.save', type=str, nargs='?', default=None, const='mask.fits', help='If present will save the created mask to the relative file "./mask.fits", or a filename can be passed')
	parser.add_argument('--mask.load', type=str, nargs='?', default=None, const='mask.fits', help='If present, will load a mask from the relative file "./mask.fits", or a different filename can be passed')
	parser.add_argument('--mask.mode', type=str, default='manual', help='How the mask is calculated or manually chosen')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
