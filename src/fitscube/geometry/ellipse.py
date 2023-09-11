#!/usr/bin/env python3

import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import sys, os
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import copy
import types

import utilities as ut
import webservices
import webservices.jplhorizons
import astro_resources
import fitscube.fit_region
import utilities.flow_control

class SimpleNamespace:
	def __init__(self):
		return
const = SimpleNamespace()
const.deg2rad = np.pi/180

def project_lat_lon(hdul, ecx_est, ecy_est, a_est, b_est, t_est, sub_obs_lat, show_plots=True, find_disk_manually=True, ext=0):
	"""
	Fits an ellipsoid to the passed data

	ARGUMENTS:
		hdul [nz,ny,nx]
			<header data unit list> A list of header data units, the 0th hdu will have an ellipse fitted to it
		ecx_est
			<float> estimate of x-axis ellipse center
		ecy_est
			<float> estimate of y-axis ellipse center
		a_est
			<float> estimate of semi-major axis
		b_est
			<float> estimate of semi-minor axis
		t_est
			<float> estimate of position angle

	RETURNS:
		ecx
			<int> The center of the fitted ellipse in the x-direction, is an int due to fitting algorithm
		ecy
			<int> The center of the fitted ellipse in the y-direction, is an int due to fitting algorithm
		Re
			<float> (arcsec) The equatorial radius of the fitted ellipsoid
		Rp
			<float> (arcsec) The polar radius of the fitted ellipsoid
		Rpp
			<float> The projected polar radius of the fitted ellipsoid (as observed in the image)
		ellipse [ny,nx]
			<array, <array, float>> The ellipse that is fitted to the planetary disk
		lats [ny,nx]
			<array,<array,float>> A map of latitudes across the disk of the planet, locations off-disk are filled with NANs
		lons [ny,nx]
			<array, <array, float>> A map of longitudes across the disk of the planet, locations off-disk are filled with NANs
		zens [ny,nx]
			<array, <array, float>> A map of 'observer-to-zenith' angles across the disk of the planet, locations off-disk are filled with NANs
		xx [ny,nx]
			<array, <array, float>> A map of angular distance (arcsec) in the x-direction from the center of the fitted ellipse
		yy [ny,nx]
			<array,<array,float>> A map of angular distance (arcsec) in the y-direction from the center of the fitted ellipse
	"""
	print(f'{hdul=}')
	print(f'{ecx_est=}')
	print(f'{ecy_est=}')
	print(f'{a_est=}')
	print(f'{b_est=}')
	print(f'{t_est=}')
	print(f'{sub_obs_lat=}')
	print(f'{show_plots=}')
	print(f'{find_disk_manually=}')
	print(f'{ext=}')
	Re = a_est # equatorial radius
	Rp = b_est
	oblateness = 1 - Rp/Re

	# define an ellipsoid with same oblateness, but major axis = 1
	a = 0.5 # major axis of 1, sma of 0.5
	b = a*(1.0 - oblateness)
	m = np.tan(sub_obs_lat)
	if m != 0.0:
		xdiff_sq = 4.0/((1.0/a**2) + (b**2/(m**2*a**4)))
		ydiff_sq = (4.0*b**4)/((m*a)**2 + b**2)
		proj_dist = np.sqrt(xdiff_sq + ydiff_sq)
		Rpp = proj_dist*Re # major axis = 1, so just scale by equatorial diameter
	else:
		Rpp = Rp

	print(f'Equatorial diameter {Re} px')
	print(f'Polar diameter {Rp} px')
	print(f'Projected polar diameter {Rpp} px')

	# auto-find pointing by finding where the greatest signal enclosed by
	# an ellipse with a=Re/2, b=Rpp/2 ?	

	"""
	a = Re/px_scale
	b = Rpp/px_scale
	#e_idx, ellipse = find_max_ellipse_top_hat(hdul, a, b, t=0, find_disk_manually=find_disk_manually, ext=ext)
	(ecx, ecy, a, b, t), ellipse = find_max_ellipse_top_hat(hdul, a, b, t=north_pole_position_angle, find_disk_manually=find_disk_manually, ext=ext)
	e_idx = (ecx,ecy)
	#e_idx, ellipse = find_ellipse_gradient(np.nanmedian(hdul[ext].data, axis=(0)), a, b, t=0)
	"""
	
	if find_disk_manually:
		diskFitter = fitscube.fit_region.FitscubeDiskFinder(hdul, hdul_extension=ext, ecx=ecx_est, ecy=ecy_est, a=a_est, b=Rpp, theta=t_est, icmapmin=0, 
														icmapmax=1, ifreqmin=0, ifreqmax=hdul[ext].data.shape[0])
		diskFitter.run()
		e_params = diskFitter.getEllipseParams()
	else:
		# assume our estimates are correct
		e_params = (ecx_est, ecy_est, a_est, Rpp, t_est)
	e_idx = e_params[:2]
	a, b, t = e_params[2:]
	print(e_params)

	ellipse = ellipse_top_hat_as(hdul[ext].data.shape[1:], e_idx[0], e_idx[1],a,b,const.deg2rad*(360-t),error=1E-3)
	

	print(f'ellipse a {a} b {b} t {t}')
	print(f'ellipse center indices {e_idx}')

	#### Plot found ellipse	
	if show_plots:
		f1 = plt.figure(figsize=[_x/2.54 for _x in (36,12)])

		a11 = f1.add_subplot(1,3,1)
		# create an ellipse patch, will need to copy it to reuse it because matplotlib is weird
		disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, t, facecolor='none', edgecolor='tab:green')
		data = np.nanmean(hdul[ext].data, axis=0)
		#data = np.abs(np.diff(data, axis=0)[:,1:] + np.diff(data, axis=1)[1:,:])
		#data = np.log(np.sum(np.abs(np.gradient(data)), axis=0))
		#data = np.log(np.nanmean(hdul[ext].data, axis=0))
		cmin = np.nanmin(data)
		cmax = np.nanmax(data)
		img1 = a11.imshow(data, origin='lower', vmin=cmin, vmax=cmax)
		a11.set_title('Median of data')
		a11.add_patch(copy.copy(disk_ellipse))

		print(data.shape)
		print(ellipse.shape)
		#ec = np.array([e_idx[1],e_idx[0]])
		ec = e_idx
		b_dir = np.array([np.cos(const.deg2rad*(t-270)),np.sin(const.deg2rad*(t-270))])
		a_dir = np.array([b_dir[1], -b_dir[0]])

		aa = a*a_dir	
		bb = b*b_dir
		#aa = a*np.array([np.sin(t-270),-np.cos(t-270)])
		#bb = b*np.array([np.cos(t-270),np.sin(t-270)])
		print(ec, aa, bb)
		print(ec+aa, ec+bb)

		al = np.stack([ec,ec+aa], axis=0)
		bl = np.stack([ec,ec+bb], axis=0)
		print(al)
		print(bl)

		#DEBUGGING
		#plt.show()

		a12 = f1.add_subplot(1,3,2)
		img2 = a12.imshow(ma.array(data,mask=np.logical_not(np.array(ellipse, dtype='bool'))), origin='lower', vmin=cmin, vmax=cmax)
		#img2 = a12.imshow(ellipse, origin='lower', zorder=0)
		a12.scatter(e_idx[0], e_idx[1], edgecolor='red', facecolor='none', zorder=1, label='ellipse center')
		a12.plot(al[:,0], al[:,1], color='red', zorder=2, label='semi-major axis')
		a12.plot(bl[:,0], bl[:,1], color='green', zorder=3, label='semi-minor axis')
		a12.legend()
		a12.set_title('Ellipse fitted to planet disc')
		#disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, 0, facecolor='none', edgecolor='tab:green')
		a12.add_patch(copy.copy(disk_ellipse))

		a13 = f1.add_subplot(1,3,3)
		img3 = a13.imshow(data - ellipse*data, origin='lower', vmin=cmin, vmax=cmax)
		a13.set_title('data median minus fitted ellipse')
		#disk_ellipse = ptc.Ellipse(e_idx, 2*a, 2*b, 0, facecolor='none', edgecolor='tab:green')
		a13.add_patch(copy.copy(disk_ellipse))

		plt.show()
	##### end plotting

	# t should be the anti-clockwise angle from vertical that the north pole is at
	# so I should adjust eoffA and poffA to account for non-vertical north pole
	b_dir = np.array([np.cos(const.deg2rad*(t-270)),np.sin(const.deg2rad*(t-270))])
	a_dir = np.array([b_dir[1], -b_dir[0]])
	print(a_dir)
	print(b_dir)
	cob_mat = np.stack([a_dir,b_dir]) # change of basis to equatorial-polar direction instead of x-y
	ec_vec = np.zeros((2,1))
	eq_pol_vec = np.zeros((2,1))


	ny, nx = hdul[ext].data.shape[1:]
	lats = np.ones((ny,nx))*np.nan # fill with junk value
	lons = np.ones((ny,nx))*np.nan # fill with junk value
	zens = np.ones((ny,nx))*np.nan # fill with junk value
	ecx, ecy = e_idx
	print(f'ellipse center x {ecx} y {ecy}')
	for i in range(nx):
		for j in range(ny):
			#eoffA = (i-ecx)
			#poffA = (j-ecy)
			#print(f'i {i} j {j} eoffA {eoffA} poffA {poffA} nx {nx} ny {ny}')
			#iflag, xlat, xlon, zen = projpos_ellipse(Re, Rp, sub_obs_lat, eoffA, poffA) # Should this use Rpp?
			#print(f'Re {Re} Rp {Rp} ecx {ecx} ecy {ecy} iflag {iflag} xlat {xlat} xlon {xlon} zen {zen}')

			ec_vec[0] = i-e_idx[0]
			ec_vec[1] = j-e_idx[1]
			eq_pol_vec = cob_mat @ ec_vec
			#print(f'{i=} {j=} {eq_pol_vec=} {nx=} {ny=}')
			iflag, xlat, xlon, zen = projpos_ellipse(a, b, sub_obs_lat, eq_pol_vec[0], eq_pol_vec[1]) # Should this use Rpp?
			#print(f'{a=}{b=} {e_idx=} {iflag=} {xlat=} {xlon=} {zen=}')
			if iflag:
				lats[j,i] = xlat
				lons[j,i] = xlon
				zens[j,i] = zen

	# calculate row-column pixel scale offsets from center of ellipse
	yy, xx = np.mgrid[:ny, :nx]
	yy = (yy-ecy)
	xx = (xx-ecx)
	xy = np.stack([xx,yy], axis=-1)[...,None]
	xy = cob_mat @ xy
	xx, yy = np.moveaxis(xy.reshape(xy.shape[:-1]), -1, 0)

	### plot for debugging
	if show_plots:
		print(f'lats.shape {lats.shape}')
		print(f'lons.shape {lons.shape}')
		print(f'zens.shape {zens.shape}')

		nx, ny = (4,1)
		f2 = plt.figure(figsize=[_x/2.54 for _x in (12*nx, 12*ny)])

		a21 = f2.add_subplot(ny,nx,1)
		im21 = a21.imshow(data, origin='lower')
		a21.set_title('Median of data')	
		a21.add_patch(copy.copy(disk_ellipse))

		a22 = f2.add_subplot(ny,nx,2)
		#im22 = a22.imshow(lats, origin='lower')
		im22 = a22.imshow(ma.array(lats, mask=np.isnan(lats)), origin='lower')
		a22.set_title('Latitude')
		a22.add_patch(copy.copy(disk_ellipse))

		a23 = f2.add_subplot(ny,nx,3)
		#im23 = a23.imshow(lons, origin='lower')
		im23 = a23.imshow(ma.array(lons, mask=np.isnan(lons)), origin='lower')
		a23.set_title('longitude')
		a23.add_patch(copy.copy(disk_ellipse))

		a24 = f2.add_subplot(ny,nx,4)
		#im24 = a24.imshow(zens, origin='lower')
		im24 = a24.imshow(ma.array(zens, mask=np.isnan(zens)), origin='lower')
		a24.set_title('angle of LOS w.r.t zenith')
		a24.add_patch(copy.copy(disk_ellipse))

		plt.show()
	### end plotting

	# store relavent data in class
	#self.ellipse_center_x = ecx
	#self.ellipse_center_y = ecy
	#self.Re = Re
	#self.Rp = Rp
	#self.Rpp = Rpp
	#self.disk = ellipse
	#self.lats = lats
	#self.lons = lons
	#self.zens = zens
	#self.xx = xx
	#self.yy = yy
	return(e_params, lats, lons, zens, xx, yy)

def find_ellipse_gradient(data,a,b,t):
	graddata = np.log(np.sum(np.abs(np.gradient(data)),axis=0))
	a_vec = fitEllipse(graddata)
	ecx,ecy = ellipse_center(a_vec)
	a,b = ellipse_axis_length(a_vec)
	t = ellipse_angle_of_rotation2(a_vec)

	print(ecx, ecy, a, b, t)

	nx, ny = (1,1)
	f1 = plt.figure(figsize=[_x/2.54 for _x in (12*nx, 12*ny)])

	a11 = f1.add_subplot(ny,nx,1)
	im11 = a11.imshow(graddata, origin='lower')
	a11.set_title('Median of data')	
	disk_ellipse = ptc.Ellipse((ecx,ecy), 2*a, 2*b, t, facecolor='none', edgecolor='tab:green')
	a11.add_patch(disk_ellipse)

	plt.show()

def find_max_ellipse_top_hat(hdul, a=1, b=1, t=0, find_disk_manually=True, ext=0):
	cmin, cmax, chmin, chmax = (0, 1, 0, hdul[ext].data.shape[0])
	data = np.nanmedian(hdul[ext].data[chmin:chmax,:,:], axis=0)
	data_range = np.nanmax(data) - np.nanmin(data)
	data_min = np.nanmin(data)
	data = np.clip(data, cmin*data_range+data_min, cmax*data_range+data_min)
	top_hat_sum = np.zeros_like(data)
	print('data.shape {}'.format(data.shape))
	y, x = np.mgrid[:data.shape[0], :data.shape[1]] # fortran y-x ordering from data
	# simple grid search, make better if I need to
	"""
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			eth = ellipse_top_hat_as(data.shape, x[i,j], y[i,j], a, b, t)
			top_hat_sum[i,j] = np.nansum(eth*data)
	
	max_top_hat_sum_idx = np.unravel_index(np.nanargmax(top_hat_sum), data.shape)
	ecx, ecy = (x[max_top_hat_sum_idx], y[max_top_hat_sum_idx])
	"""
	ecx, ecy = tuple(_x//2 for _x in hdul[ext].shape[1:])
	"""
	ny, nx = data.shape
	popt, pconv = fit_ellipse(data, params=[nx/2, ny/2, a, b, t], fixed=[False, False, True, True, True])
	ecx, ecy, a, b, t = popt
	"""
	if find_disk_manually:
		diskFitter = fitscube.fit_region.FitscubeDiskFinder(hdul, hdul_extension=ext, ecx=ecx, ecy=ecy, a=a, b=b, theta=t, icmapmin=cmin, 
															icmapmax=cmax, ifreqmin=chmin, ifreqmax=chmax)
		diskFitter.run()
		ecx, ecy, a, b, t = diskFitter.getEllipseParams()
	return((ecx,ecy,a,b,t), ellipse_top_hat_as(data.shape, ecx, ecy,a,b,const.deg2rad*(360-t),error=1E-3))
	
def projpos_ellipse(Re, Rp, eps, eoff, poff):
	"""
	Finds the longitude, latitude, and zenith angles of the intercept point of an ellipse and a
	line fired offset from an observers line of sight.

	ARGUMENTS:
		Re
			<float> Equatorial radius of ellipse (unit A)
		Rp
			<float> Polar radius of ellipse (unit A)
		eps
			<float> Sub-observer (planetocentric) latitude (degrees)
		eoff
			<float> equatorial offset of beam (unit A)
		poff
			<float> polar offset of beam (unit A)

	RETURNS:
		iflag
			<bool> True if line intercepts ellipse, False otherwise
		xlat
			<float> Latitude of intercept point
		xlon
			<float> Longitude of intercept point
		zen
			<float> Angle of line w.r.t Zenith of intercept point
	"""
	# using line in symmetric form (x-x0)/alpha = (y-y0)/beta = (z-z0)/gamma
	# line goes through the point (x0,y0,z0)
	# line has the direction vector d = (alpha, beta, gamma), d should be a unit vector

	x0 = 0.0 # x-point on the beam-line (zero x-coord means the point the line goes through is in the yz-plane
	y0 = eoff # y-point on the beam-line, equatorial offset of beam (arcsec)
	z0 = poff/np.cos(eps*const.deg2rad) # z-point on the beam-line

	alpha = np.sin(np.pi/2.0 - eps*const.deg2rad) # x-dimension of unit direction vector
	beta = 0.0 # y-dimension of unit direction vector, no change in y-direction so y = eoff always, so the line is contained within the xz-plane at y=eoff
	gamma = np.cos(np.pi/2 - eps*const.deg2rad)# z-dimension of unit direction vector

	iflag, x, y, z = intercept_ellipse(Re, Rp, alpha, beta, gamma, x0, y0,z0)
	#; iflag - does our line intercept (True) or not (False) the ellipsoid? (tangents count as not intercepting)
	#; x - the x-values of the two intercepting points
	#; y - the y-values of the two intercepting points
	#; z - the z-values of the two intercepting points

	# initialise variables with junk
	xlat = -999
	xlon = -999
	zen = -999
	if iflag:
		# find closest point to observer. LOS is not neccesarily along the x-axis but if we assume we 
		# are on the "right hand side" of the graph, then we just want the point with the larger x-value
		dx = (x-x0)/alpha
		inear = np.argmax(dx) # find closest point
		x1 = x[inear]
		y1 = y[inear]
		z1 = z[inear]
		
		# (x1,y1,z1) now contains the "intercept closest to observer" or ICO_point
		r = np.sqrt(x1**2 + y1**2 + z1**2) # find distance of ICOpoint from origin

		# theta is the angle between the x-axis and the origin->ICOpoint line
		theta = np.arccos(z1/r)
		xlat = 90 - theta/const.deg2rad
		# planetocentric latiude is the angle between the equatorial plane and a line joining
		# the ICOpoint to the origin

		# the component of orign->ICOpoint in the xy-plane is r*sin(theta), so the angle between the
		# x-axis and the ICOpoint is arccos(x1/r*sin(theta)) (c.f. spherical polars)
		cphi = x1/(r*np.sin(theta))
		cphi = np.clip(cphi, -1, 1)
		phi = np.arccos(cphi)
		if y1 < 0:
			phi*=-1
		xlon = phi/const.deg2rad

		v1 = np.array([x1/r, y1/r, z1/r]) # unit vector of ICOpoint
		#v1 = np.array([x1/Re, y1/Re, z1/Rp]) # unit vector of ICOpoint
		v2 = np.array([alpha, beta, gamma]) # unit vector defining direction of LOS

		zen = np.arccos(np.dot(v1,v2))/const.deg2rad
		# however, the zenith is the normal to the surface, v1 dot v2 != angle normal to the surface
		# possibly this is good enough when oblateness is small, but when it is large this will be wrong
		# shoud use v1 = 2*(x1/a^2, y1/a^2, z1/b^2) for correct answer, I think...
		# see https://math.stackexchange.com/questions/2931909/normal-of-a-point-on-the-surface-of-an-ellipsoid

	return(iflag, xlat, xlon, zen) 

def intercept_ellipse( a, b, alpha, beta, gamma, x0, y0, z0):
	"""
	procedure to find the intercepts (if any) between the line
	
	 (x-x0)       (y-y0)      (z-z0)
	 ------  =    ------   =  ------     [1]
	 alpha         beta       gamma
	
	and the ellipsoid
	
	
	 x^2    y^2    z^2   
	 --- +  --- +  ---  = 1              [2]
	 a^2    a^2    b^2
	
	Input variables
		a
			<real> ellipsoid semi-major axis
		b    	
			<real> ellipsoid semi-minor axis
		alpha	
			<real> line x-gradient
		beta 	
			<real> line y-gradient
		gamma	
			<real> line z-gradient
		x0	
			<real> line x-intercept
		y0	
			<real> line y-intercept
		z0	
			<real> line z-intercept
	
	Output variables
	  iflag	integer	Set to 1 if line intercepts, set to -1  otherwise
	  x(2)	real	x-intercepts
	  y(2)	real	y-intercepts
	  z(2)	real	z-intercepts
	
	Pat Irwin	11/2/07
	
	**********************************************************
	
	
	Breaking [1] into two equations gives
	
	(x-x0)/alpha = (y-y0)/beta          [3]
	
	(x-x0)/alpha = (z-z0)/gamma         [4]
	
	
	rearranging to make y and z in terms of x
	
	y = (beta/alpha)(x-x0) + y0         [5]
	
	z = (gamma/alpha)(x-x0) + z0        [6]
	
	
	substituting [5] and [6] into [2] gives a big
	long equation that can eventually be simplified
	down to a quadratic in x of the form
	
	a1 x^2 + b1 x + c1 = 0              [7]
	
	and can be solved by the quadratic formula, giving
	                _______________  
	               /               |
	    -b1 +/-  \/ b1^2 - 4 a1 c1
	x = ----------------------------    [8]
	               2 a1
	"""
	# The equations below give the values of the coefficients a1, b1, c1

	a1 = 1.0/a**2 + (beta/(a*alpha))**2 + (gamma/(b*alpha))**2

	b1 = (-2*x0*beta**2/alpha**2 + 2*beta*y0/alpha)/a**2
	b1 = b1 + (-2*x0*gamma**2/alpha**2 + 2*gamma*z0/alpha)/b**2

	c1 = ((beta*x0/alpha)**2 - 2*beta*y0*x0/alpha + y0**2)/a**2
	c1 = c1+ ((gamma*x0/alpha)**2 - 2*gamma*x0*z0/alpha + z0**2)/b**2 -1


	xtest = b1**2 - 4*a1*c1 # determinant of [7]

	# in the case of the determinant > 0 we can have two
	# real roots, therefore the line [1] intercepts the
	# ellipsoid [2] at two places
	x = np.zeros([2])
	y = np.zeros([2])
	z = np.zeros([2])

	# if the determinant of [7] is > 0, we have two real roots
	# therefore calculate the value of x using [8], and then
	# find y and z using [5] and [6]
	if(xtest >= 0.0):
		iflag=True
		x[0]=(-b1 + np.sqrt(xtest))/(2*a1) # positive root
		x[1]=(-b1 - np.sqrt(xtest))/(2*a1) # negative root

		y = y0 + (beta/alpha)*(x-x0) # [5]
		z = z0 + (gamma/alpha)*(x-x0)# [6]

		# here, x, y, z have two components each.
		# each of them has a value for the positive root (in index 0)
		# and the negative root (in index 1)

		test = (x/a)**2 + (y/a)**2 + (z/b)**2 # calcuate [2] with our found numbers
		# [2] should always equal 1, so subtract 1 to find the error
		xtest1 = np.abs(test[0]-1.0)
		xtest2 = np.abs(test[1]-1.0)

		err = 1e-5 # deterimine our error threshold
		if(xtest1 > err or xtest2 > err):
			# if the error is larger than our threshold, inform the user
			print('Problem in interceptellip - solution not on ellipsoid')
			print(f'test = {test}')
	else:
		iflag=False # if we don't have two real roots, we don't intercept the ellipsoid

	return(iflag, x,y,z)

def ellipse_top_hat_as(shape, x0, y0, a, b, t, error=1E-3):
	y, x = np.mgrid[:shape[0], :shape[1]]
	return(ellipse_top_hat_2D((y,x), y0, x0, b, a, t, error).reshape(shape))
		
def ellipse_top_hat_2D(pos, x0=0, y0=0, a=1, b=1, t=0, error=1E-3):
	"""
	pos - x, y positions on a 2d grid to evaluate ellipse at
	x0 - x position of center of ellipse
	y0 - y position of center of ellipse
	a - semi-major axis
	b - semi-minor axis
	t - angle of rotation (+ve horizontal axis to major axis of ellipse)
	see https://en.wikipedia.org/wiki/Ellipse#General_ellipse
	"""
	x, y = pos
	A = (a*np.sin(t))**2 + (b*np.cos(t))**2
	B = 2*(b**2-a**2)*np.sin(t)*np.cos(t)
	C = (a*np.cos(t))**2 + (b*np.sin(t))**2
	D = -2*A*x0 - B*y0
	E = -B*x0 - 2*C*y0
	F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
	
	bg = np.zeros_like(x)
	center_value = A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F 
	#print(f'center_value {center_value}')
	#print(f'F {F}')
	ellipse_idxs = np.where(A*x**2 + B*x*y + C*y**2 + D*x + E*y <= -(1-error)*F)
	bg[ellipse_idxs] = 1.0
	return(bg.flatten())

def fit_ellipse(data, params=[0,0,1,1,0], fixed=[False]*5):
	"""
	DOES NOT WORK
	"""
	nx, ny = data.shape
	x, y = np.mgrid[:nx, :ny]
	variables = [p  for p,f in zip(params, fixed) if not f]
	constants = [p for p,f in zip(params, fixed) if f]
	#print(f'variables {variables}')
	#print(f'constants {constants}')
	popt, pconv = spo.curve_fit(ellipse_top_hat_2D, (x,y), np.nan_to_num(data, copy=True).ravel(), p0=variables)
	popt_full = []
	i=0
	j=0
	for f in fixed: 
		if f:
			popt_full.append(constants[i])
			i+=1
		else:
			popt_full.append(popt[j])
			j+=1
	#print(f'popt_full {popt_full}')
	return(popt_full, pconv)


def parse_args(argv):
	import argparse
	import utilities.args
	
	parser = ut.args.DocStrArgParser(
		description=__doc__,
		fromfile_prefix_chars = '@',
		allow_abbrev=False,
	)


	## Add Positional Arguments ##
	parser.add_argument(
		'cubes', 
		type=str, 
		nargs='+',
		help='fitscubes to operate on',
	)

	## Add Optional Arguments ##
	parser.add_argument('--cube.ext.sci', type = int, help = 'Extensions of fits cubes that holds science data', default = 0)
	parser.add_argument('--cube.ext.info', type = int, help = 'Extensions of fits cubes that holds observation information', default = 0)
	parser.add_argument('--cube.ext.sci.image_axes', type=int, nargs='+', help='The axes numbers that correspond to RA-DEC axes', default=[1,2])
	parser.add_argument('--output.fits', type=str, help='Output file for the updated FITS file', default=None)
	parser.add_argument('--output.fits.overwrite', action='store_true', help='If present, will overwrite any existing file at output location')
	parser.add_argument('--cube.auto_fit_disk', action=ut.args.ActiontF, help='Should we automatically center and fit the disk?')

	
	args = vars(parser.parse_args()) # I prefer a dictionary interface

	_msg = '#'*20 + ' ARGUMENTS ' + '#'*20
	for k, v in args.items():
		_msg += f'\n\t{k}\n\t\t{v}'
	_msg += '\n' + '#'*20 + '###########' + '#'*20
	_lgr.INFO(_msg)

	return(args)
	

def get_ellipse_inputs_from_fits(hdul, info_ext=0, sci_ext=0, use_horizons_pos=False):
	from astropy.wcs import WCS
	from astropy.coordinates import SkyCoord
	from astropy import units as u

	import utilities.np


	info_hdu = hdul[info_ext]
	sci_hdu = hdul[sci_ext]

	print(sci_hdu.header.get('ORIGIN',None))
	observer_location = info_hdu.header.get('ORIGIN',None).lower()
	observation_datetime = info_hdu.header.get('HIERARCH ESO OBS START', info_hdu.header.get('DATE-OBS', None))
	
	observation_target = ut.flow_control.mapping_lookup_with_keys(
		info_hdu.header,  
		('HIERARCH ESO OBS TARG NAME', 'OBJECT', 'HIERARCH ESO OBS NAME'), 
		(None, 'no name', 'unknown', ''), 
		lambda x: x.lower()
	)

	print(f'{observer_location = }')
	print(f'{observation_datetime = }')
	print(f'{observation_target = }')

	target_oblateness = astro_resources.planet_physical_parameters[observation_target]['equatorial_radius']/astro_resources.planet_physical_parameters[observation_target]['polar_radius']

	print(f'{target_oblateness = }')

	observation_eph = webservices.jplhorizons.query_eph_to_ids(observation_target, observer_location, observation_datetime, ('ang_width', 'PDObsLat', 'NPole_ang', 'NPole_dist', 'RA', 'DEC'))
	north_pole_position_angle = observation_eph['NPole_ang']
	sub_observer_latitude = observation_eph['PDObsLat']

	for k, v in observation_eph.items():
		print(f'{k}\n\t{v}')

	wcs = WCS(sci_hdu.header)

	if use_horizons_pos:
		sky_center = SkyCoord(ra=observation_eph['RA'], dec=observation_eph['DEC'], unit='deg')
	else:
		sci_com = ut.np.get_com(np.nansum(sci_hdu.data, axis=(0,)))
		sky_center = SkyCoord.from_pixel(*sci_com, wcs)

	sky_center_px = sky_center.to_pixel(wcs)
	sky_limb_equatorial_point = sky_center.directional_offset_by((observation_eph['NPole_ang']+90)*u.deg, (observation_eph['ang_width']/2)*u.arcsec)
	sky_limb_equatorial_point_px = sky_limb_equatorial_point.to_pixel(wcs)

	get_pixel_separations = lambda s1_p, s2_p:  np.sqrt((s1_p[0]-s2_p[0])**2+(s1_p[1]-s2_p[1])**2)

	if use_horizons_pos:
		x = sky_center_px[1]
		y = sky_center_px[0]
	else:
		x = sky_center_px[1]
		y = sky_center_px[0]
	a = get_pixel_separations(sky_center_px, sky_limb_equatorial_point_px)
	b = a/target_oblateness

	return(x, y, a, b, north_pole_position_angle, sub_observer_latitude)
	


def main(argv):
	args = parse_args(argv)

	print(args)

	for cube in args['cubes']:
		with fits.open(cube) as hdul:
			hdul.info()
			ellipse_inputs = get_ellipse_inputs_from_fits(hdul, args['cube.ext.info'], args['cube.ext.sci'], use_horizons_pos=False)
			ellipse_params, lats, lons, zens, xx, yy = project_lat_lon(hdul, *ellipse_inputs, show_plots=True, find_disk_manually=not args['cube.auto_fit_disk'], ext=args['cube.ext.sci'])
			for k, v in (('Latitude', lats), ('Longitude', lons), ('Zenith', zens), ('equatorial_offset',xx), ('polar_offset', yy)):
				hdul.append(fits.ImageHDU(header = hdul[args['cube.ext.sci']].header, data = v, name=k))
			hdul.writeto(args['output.fits'], overwrite=args['output.fits.overwrite'])
			

			


if __name__=='__main__':
	main(sys.argv[1:])
