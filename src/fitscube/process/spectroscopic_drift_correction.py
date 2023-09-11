#!/usr/bin/env python3
"""
Corrects for spectroscopic drift of an observed object.
Based on http://articles.adsabs.harvard.edu/pdf/1982PASP...94..715F

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
import numpy as np
import scipy as sp
import scipy.signal
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.time as t
import astropy.units as u
import plotutils
import utils as ut # used for convenience functions
import fitscube.process.sinfoni

# global variable
DEBUG = True

def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	#print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	
	# TESTING DRIFT DIRECTION -------------------------------------------------
	if False:
		palomar_obs_lat = np.deg2rad(33+21/60) # degrees to radians
		forty_five_obs_lat = np.deg2rad(45)
		observer_latitude = palomar_obs_lat

		obj_ha_range = np.deg2rad(15*np.linspace(-12,12,401))

		obj_dec_range_north= np.deg2rad(np.append(np.linspace(0,60, 13), np.array([-40,-20,-10,70,80])))
		obj_dec_range_south= np.deg2rad(np.append(np.linspace(-60,0, 13), np.array([40,20,10,-70,-80])))
		
		obj_dec_grid_north, obj_ha_grid = np.meshgrid(obj_dec_range_north, obj_ha_range)
		obj_dec_grid_south, obj_ha_grid = np.meshgrid(obj_dec_range_south, obj_ha_range)
		
		ddd_angle_north = np.rad2deg(eta(observer_latitude, obj_dec_grid_north, obj_ha_grid))
		ddd_observer_angle_north = np.rad2deg(eta(observer_latitude, np.array([observer_latitude]), obj_ha_grid))
		
		ddd_angle_south = np.rad2deg(eta(-observer_latitude, obj_dec_grid_south, obj_ha_grid))
		ddd_observer_angle_south = np.rad2deg(eta(-observer_latitude, np.array([-observer_latitude]), obj_ha_grid))
		
		nr, nc = 2, 1
		f1 = plt.figure(figsize=[_x/2.54 for _x in (24*nr,24*nc)])
		a = f1.subplots(nrows=nr, ncols=nc, gridspec_kw={'hspace':0.2})
		f1.suptitle('Direction of Spectroscopic Drift W.R.T Direction of Celestial Pole')
		
		a[0].plot(np.rad2deg(obj_ha_range)/15, ddd_angle_north, '-', lw=1)
		a[0].plot(np.rad2deg(obj_ha_range)/15, ddd_observer_angle_north, 'k:', lw=1)
		#a[0].set_xlabel('Hour angle (hours)')
		a[0].set_xticks([])
		a[0].set_ylabel('Parallactic angle (deg)')
		a[0].set_title(f'Northern Hemisphere obs_lat={np.rad2deg(observer_latitude)} deg')
		
		
		a[1].plot(np.rad2deg(obj_ha_range)/15, ddd_angle_south, '-', lw=1)
		a[1].plot(np.rad2deg(obj_ha_range)/15, ddd_observer_angle_south, 'k:', lw=1)
		a[1].set_xlabel('Hour angle (hours)')
		a[1].set_ylabel('Parallactic angle (deg)')
		a[1].set_title(f'Southern Hemisphere obs_lat={np.rad2deg(-observer_latitude)} deg')
		
		#plotutils.save_show_plt(f1, 'parallactic_angle_test.png', outfolder='./plots', show_plot=True, save_plot=False)
		plt.show()
		
	# END TESTING DRIFT DIRECTION ---------------------------------------------
	
	#target_cubes = ['/home/dobinsonl/scratch/telescope_data/sinfoni/neptune_single_obs_1_reduced/SINFO.2018-08-19T10:02:42/Hip038133_tpl/out_objnod.fits']
	#target_cubes = ['/home/dobinsonl/scratch/telescope_data/sinfoni/neptune_single_obs_1_reduced/SINFO.2018-08-19T09:43:00/MOV_Neptune---H+K_0.1_3_tpl/out_objnod.fits']
	
	for tc in args['target_cubes']:
		print('='*79)
		print(f'INFO: Operating on target cube fits file "{tc}"')
		with fits.open(tc) as hdul:
			lst = float(hdul[0].header['LST']) #seconds
			utc = float(hdul[0].header['UTC']) # seconds
			dec = float(hdul[0].header['DEC']) #deg
			ra = float(hdul[0].header['RA']) #deg
			lat = float(hdul[0].header['HIERARCH ESO TEL GEOLAT']) #deg
			lon = float(hdul[0].header['HIERARCH ESO TEL GEOLON']) #deg
			wavgrid = fitscube.process.sinfoni.datacube_wavelength_grid(hdul[0])
			temp = float(hdul[0].header['HIERARCH ESO TEL AMBI TEMP']) # deg-celcius
			press = float(hdul[0].header['HIERARCH ESO TEL AMBI PRES START']) # torr (mm Hg), probably
			relh = float(hdul[0].header['HIERARCH ESO TEL AMBI RHUM']) # relative humidity
			alt = float(hdul[0].header['HIERARCH ESO TEL ALT'])
			date = hdul[0].header['DATE-OBS']
			file_parang = float(hdul[0].header['HIERARCH ESO TEL PARANG START'])
			rec1_raw1_name = hdul[0].header['HIERARCH ESO PRO REC1 RAW1 NAME']
			guide_ra = float(hdul[0].header['HIERARCH ESO ADA GUID RA'])
			guide_dec = float(hdul[0].header['HIERARCH ESO ADA GUID DEC'])
			print('Printing data from header:')
			print(f'\tlst {lst}')
			print(f'\tutc {utc}')
			print(f'\tdec {dec}')
			print(f'\tra {ra}')
			print(f'\tlat {lat}')
			print(f'\tlon {lon}')
			print(f'\twavgrid[::100] {wavgrid[::100]}')
			print(f'\ttemp {temp}')
			print(f'\tpress {press}')
			print(f'\trelh {relh}')
			print(f'\talt {alt}')
			print(f'\tdate {date}')
			print(f'\tfile_parang {file_parang}')
			print(f'\trec1_raw1_name {rec1_raw1_name}')
			print(f'\tguide_ra {guide_ra}')
			print(f'\tguide_dec {guide_dec}')

							

			# DEBUGGING
			s2d = lambda ra_dec: np.sum(np.array(ra_dec)*np.array([[15,15/60,15/3600],[1,1/60,1/3600]]), axis=1)
			#ra, dec = s2d([[7,48,51.515],[-43,-18,-40.607]])
			#ra, dec = guide_ra, guide_dec
			#ra, dec = 346.88811, -6.6954334 #  for neptune test file
			#ra, dec = 117.21461, -43.311323 # for std star test

			date = rec1_raw1_name.rsplit('.',1)[0].split('.')[1]
			obs_date = t.Time(date, format='fits', scale='utc', location=(lat*u.deg, lon*u.deg))
			obs_date_sidereal_time = obs_date.sidereal_time('mean', longitude=lon*u.deg).value
			print(f'INFO: RA {ra} DEC {dec}')
			print(f'INFO: lat {lat} lon {lon}')	
			print(f'INFO: date {date}')	

			print(obs_date_sidereal_time, lst/3600)
	
			pp_water = (relh/100)*saturated_wvp_buck(temp, unit='mm Hg') # should convert to torr (or mm Hg)
			print(f'INFO: obs_date_sidereal_time {obs_date_sidereal_time} ra {ra} ra/15 {ra/15}')
			hour_angle = obs_date_sidereal_time - ra/15
			zenith_angle = np.pi/2 - np.deg2rad(alt)
			print(f'INFO: temp {temp}')
			print(f'INFO: press {press}')
			print(f'INFO: pp_water {pp_water}')
			print(f'INFO: hour_angle {hour_angle}')
			print(f'INFO: zenith_angle {zenith_angle}')

			#hour_angle_rad= np.array([np.deg2rad(lst*(15/3600) - ra) ]) # should be equivalent to ust
			#hour_angle_rad = np.array([np.deg2rad(utc*(15/360) + lon - ra)]) # should be equivalent to lst
			hour_angle_rad = np.array([np.deg2rad(hour_angle*15)])
			dec_rad = np.array([np.deg2rad(dec)])
			lat_rad = np.deg2rad(lat)
			dec_grid, ha_grid = np.meshgrid(dec_rad, hour_angle_rad)
			# calculate drift angle
			drift_ang = np.rad2deg(eta(lat_rad, dec_grid, ha_grid))
			print(f'INFO: drift_ang {drift_ang} (deg)')
			print(f'INFO: file_parang {file_parang} (deg)')

			# display fits file data

			# work out direction of drift_ang on field

			# display direction of drift_ang and magnitude of drift_mag on plot

			tc_wcs = WCS(hdul[0])
			tc_ra_dec_coord = SkyCoord(ra, dec, unit='deg')
			tc_ra_dec_coord_px = tc_ra_dec_coord.to_pixel(tc_wcs)
			m, theta, dxdy, drift_dir_xy = get_field_drift_gradient(drift_ang[0,0], tc_wcs, ref_px=tc_ra_dec_coord_px)
			print('DEBUGGING: tc_ra_dec_coord_px', tc_ra_dec_coord_px)
			print('DEBUGGING: drift_dir_xy', drift_dir_xy)
			print('DEBUGGING: dxdy', dxdy)
			print('DEBUGGING: theta', np.rad2deg(theta))
			print('DEBUGGING: m', m)
				
			if args['show_plots'] and False:
				# Plot drift direction for inspection
				nr, nc = 1, 1
				f2 = plt.figure(figsize=[_x/2.54 for _x in (24*nr,24*nc)])
				a2 = f2.subplots(nrows=nr, ncols=nc, gridspec_kw={'hspace':0.2}, squeeze=False)
				print(f'DEBUGGING: tc_ra_dec_coord_px {tc_ra_dec_coord_px}')
				a2[0,0].imshow(np.nanmedian(hdul[0].data, axis=0), origin='lower')
				a2[0,0].scatter(xs, ys, color='black')
				a2[0,0].scatter(*tc_ra_dec_coord_px, color='red')
				a2[0,0].scatter(*posang_zero.to_pixel(tc_wcs), color='blue', label='N')
				a2[0,0].scatter(*posang_pi.to_pixel(tc_wcs), color='green', label='S')
				a2[0,0].scatter(*posang_pi_over_2.to_pixel(tc_wcs), color='yellow', label='E')
				a2[0,0].scatter(*posang_3pi_over_2.to_pixel(tc_wcs), color='orange', label='W')
				#a2[0,0].scatter(*scp_coord.to_pixel(tc_wcs), color='lightgreen', label='SCP')
				#a2[0,0].scatter(*ncp_coord.to_pixel(tc_wcs), color='magenta', label='NCP')

				
				a2[0,0].arrow(xs[0], ys[0], xs[-1]-xs[0], ys[-1]-ys[0], color='red', length_includes_head=True, head_starts_at_zero=True, label='eta drift direction')
				a2[0,0].arrow(fp_xs[0], fp_ys[0], fp_xs[-1]-fp_xs[0], fp_ys[-1]-fp_ys[0], color='green', length_includes_head=True, head_starts_at_zero=True, label='file_parang drift direction')

				a2[0,0].scatter(com_px, com_py)
				a2[0,0].arrow(com_px_start, com_py_start, com_px_end-com_px_start, com_px_end-com_px_start, color='red', label='com drift vector')

				a2[0,0].arrow(*np.array(tc_ra_dec_coord_px), *dxdy, color='orange', label='field drift direction')

				a2[0,0].legend()
				a2[0,0].set_title('if drift is -ve then green line will be 180 deg from orange line')
				plt.show()

			if args['drift_mag_mode'] == 'refractive_index':
				print('INFO: Using refractive index to calculate magnitude of drift')
				drift_mag = dR(wavgrid, temp, press, pp_water, zenith_angle, wavgrid[0], refractive_index=lambda w, T, P, f: n_tpf_mathar07(w, T, P, f/saturated_wvp_buck(T, unit='mm Hg')))
				print(f'INFO: drift_mag {drift_mag}')
				x_shift = np.linspace(0, dxdy[0]*drift_mag[-1], wavgrid.shape[0])
				y_shift = np.linspace(0, dxdy[1]*drift_mag[-1], wavgrid.shape[0])
			elif args['drift_mag_mode'] == 'empirical':
				print('INFO: Using correlation of image accross wavelengths to calculate an empirical fit to drift')
				# try correlation along drift direction
				ref_freq_idx = 1000		# frequency index to use as a reference, will use the image slice [ref_freq_idx,:,:] as the image to correlate the others with
				n_freq_steps = 500		# Number of frequency slices to correlate to the image slice [ref_freq_idx,:,:]. If you want to correlate to every frequency set this to hdul[0].data.shape[0]

				freq_idxs = np.linspace(0, hdul[0].data.shape[0], n_freq_steps, dtype=int, endpoint=False) # use a subsample of frequencies to reduce computation time
				corr_w_f = np.zeros((n_freq_steps, hdul[0].data.shape[1], hdul[0].data.shape[2])) # create a holder for correlation results
				max_w = np.zeros((n_freq_steps,2)) # create a holder for the coordinates of the correlation maximum

				if args['show_plots'] and False:
					# plot image slice at either end of wavelength range
					nr, nc = 1, 2
					f4 = plt.figure(figsize=[_x/2.54 for _x in (24*nr,24*nc)])
					a4 = f4.subplots(nrows=nr, ncols=nc, gridspec_kw={'hspace':0.2}, squeeze=False)

					a4[0,0].imshow(hdul[0].data[300,:,:], origin='lower')
					a4[0,1].imshow(hdul[0].data[2000,:,:], origin='lower')
					plt.show()


				print('INFO: Starting loop over freq_idxs')
				ref_freq_data = np.nan_to_num(hdul[0].data[ref_freq_idx,:,:]) # reference data that we will correlate everything else to
				for i, f in enumerate(freq_idxs):
					sys.stdout.write(f'\tINFO: idx {i} freq {f}\r')

					# get the data to correlate with
					corr_freq_data = np.nan_to_num(hdul[0].data[f,:,:]) 
					#corr_freq_data = np.log(hdul[0].data[f,:,:])

					# perform correlation and store result
					corr_w_f[i,:,:] = sp.signal.correlate2d(corr_freq_data, ref_freq_data, mode='same', boundary='fill', fillvalue=0)
		
					# get the maximum of the correlation, this tells us where the images align with eachother
					max_w[i,:] = np.array(np.unravel_index(np.nanargmax(np.nan_to_num(corr_w_f[i,:,:])), ref_freq_data.shape))
					#max_w[i,:] = np.nan_to_num(centroid2(corr_w_f[i,:,:]))

			
				# Do we want to smooth the result?
				if False:
					import subsample
					max_w[:,0] = np.convolve(max_w[:,0], subsample.norm_top_hat(20), mode='same')
					max_w[:,1] = np.convolve(max_w[:,1], subsample.norm_top_hat(20), mode='same')

				# Use difference not absolute value?
				if True:
					# minus off x and y at reference wavelength 
					max_w = max_w - max_w[300][None,:]	
				print(f'DEBUGGING: max_w.shape {max_w.shape}')
				print(f'max_w[0] {max_w[0]} max_w[-1] {max_w[-1]} max_w[{max_w.shape[0]//2}] {max_w[max_w.shape[0]//2]}')

				# Use some simple fits to get an initial guess of all parameters
				# Slice out part of the frequency range when fitting so that we avoid edge effects
				start, stop = (int(max_w.shape[0]*0.1), int(max_w.shape[0]*0.9))
				freq_slice = np.s_[start:stop]

				# fit to a degree 1 polynomial (y = mx + c)
				p1 = np.polynomial.Polynomial([1,1])
				p1_fit_x = numpy_stupid_polynomial_coeffs_fix(np.polynomial.Polynomial.fit(freq_idxs[freq_slice], max_w[freq_slice,0], 1), deg=1) # fit x vs f
				p1_fit_y = numpy_stupid_polynomial_coeffs_fix(np.polynomial.Polynomial.fit(freq_idxs[freq_slice], max_w[freq_slice,1], 1), deg=1) # fit y vs f

				print(f'DEBUGGING: p1_fit_x {p1_fit_x}')
				print(f'DEBUGGING: p1_fit_y {p1_fit_y}')

				# set up initial values of all parameters
				m_xy_0, m_xf_0, m_yf_0, c_xy_0, c_xf_0, c_yf_0 = (m, p1_fit_x[1], p1_fit_y[1], 0, p1_fit_x[0], p1_fit_y[0])

				# choose function to optimise
				opt_func = find_drift_mag

				# set up initial parameters for chosen function
				p0=(m_xf_0, m_yf_0, c_xf_0, c_yf_0)
				print(f'DEBUGGING: p0 {p0}')
			
				# set up bounds for parameters of chosen function
				bounds=(-np.inf, np.inf)

				# set up constants for chosen function
				consts = ()

				try:
					optimize_result = curve_fit(	opt_func, 
													freq_idxs[freq_slice], 
													max_w[freq_slice].ravel(), 
													x0=p0, 
													bounds=bounds, 
													args=consts,
													method='trf', 
													loss='cauchy'
												)
					popt = optimize_result.x
				except RuntimeError:
					# if we haven't found a minimum just continue, use our estimates as the true value
					print('='*79)
					print(f'WARNING: Drift fitting did not converge, using initial guess as result.')
					print('='*79)
					m_xy, m_xf, m_yf, c_xy, c_xf, c_yf = m_xy_0, m_xf_0, m_yf_0, c_xy_0, c_xf_0, c_yf_0
				else:
					# If we have found a minimum then non variable parameters take on either their
					# initial or constant values.
					#print(optimize_result)
					print('INFO: Drift fitting result converged...')
					m_xy, m_xf, m_yf, c_xy, c_xf, c_yf = m_xy_0, m_xf_0, m_yf_0, c_xy_0, c_xf_0, c_yf_0
					# parameters that were varied take on their found values
					m_xf, m_yf, c_xf, c_yf = popt
				finally:
					print(f'INFO:     PARAMETERS: m_xy m_xf m_yf c_xy c_xf c_yf')
					print(f'----: INITIAL VALUES: {m_xy_0} {m_xf_0} {m_yf_0} {c_xy_0} {c_xf_0} {c_yf_0}')
					print(f'----:  FITTED VALUES: {m_xy} {m_xf} {m_yf} {c_xy} {c_xf} {c_yf}')

				print(f'DEBUGGING: corr_w_f.shape {corr_w_f.shape}')
				print(f'DEBUGGING: max_w.shape {max_w.shape}')
				#print(max_w)
				line_2d = lambda x, m, c: m*x + c

				if args['show_plots']:
					# plot the correlation and our fit to the correlation
					nr, nc = 2, 2
					f3 = plt.figure(figsize=[_x/2.54 for _x in (24*nr,24*nc)])
					a3 = f3.subplots(nrows=nr, ncols=nc, gridspec_kw={'hspace':0.2}, squeeze=False)

					a3[0,0].imshow(corr_w_f[400,:,:], origin='lower')
					#aslice = np.s_[start:stop]
					aslice = np.s_[:]

					a3[0,1].scatter(*max_w[aslice,:].T, s=4)
					a3[0,1].plot(max_w[aslice,0], line_2d(max_w[aslice,0], m_xy, c_xy), color='red')

					a3[1,0].scatter(freq_idxs[aslice], max_w[aslice,0], label='x vs freq', s=4)
					a3[1,0].scatter(ref_freq_idx, 0, s=10, facecolor='none', edgecolor='red')
					a3[1,0].plot(freq_idxs[aslice], line_2d(freq_idxs[aslice], m_xf, c_xf), color='red')

					a3[1,1].scatter(freq_idxs[aslice], max_w[aslice,1], label='y vs freq', s=4)
					a3[1,1].scatter(ref_freq_idx, 0, s=10, facecolor='none', edgecolor='red')
					a3[1,1].plot(freq_idxs[aslice], line_2d(freq_idxs[aslice], m_yf, c_yf), color='red')

					plt.show()
					
				# get shifts in x and y directions
				x_shift = line_2d(np.arange(0,hdul[0].data.shape[0], dtype=int), m_xf, c_xf)
				y_shift = line_2d(np.arange(0,hdul[0].data.shape[0], dtype=int), m_yf, c_yf)
			else:
				print('ERROR: Unknown drift_mag_mode "{args["drift_mag_mode"]}", exiting...')
				sys.exit()

			# by now we should have the x and y shifts. We got them by one of our specified methods
			if args['reverse']:
				x_shift *= -1
				y_shift *= -1
			print(f'INFO: x_shift {x_shift}')
			print(f'INFO: y_shift {y_shift}')
			# Apply the shifts
			for k in range(hdul[0].data.shape[0]):
				hdul[0].data[k,:,:] = sp.ndimage.shift(hdul[0].data[k,:,:], (y_shift[0]-y_shift[k],x_shift[0]-x_shift[k]), mode='constant', cval=np.nan, order=1, prefilter=False)

			tc_root, tc_ext = os.path.splitext(tc)
			#outfile = os.path.join(tc_dir,'out_objnod_drift_corrected.fits')
			outfile = '_'.join([tc_root, args['output.suffix']]) + tc_ext
			print(f'INFO: Writing drift-corrected fits file to "{outfile}".')
			hdul.writeto(outfile, overwrite=args['output.overwrite'])
			
		print('-'*79)
	
	return()

def centroid(img):
	ys, xs = np.grid[:img.shape[0], :img.shape[1]]
	x = np.nansum(img*xs)/np.nansum(img)
	y = np.nansum(img*ys)/np.nansum(img)
	return(np.array([x,y]))

def centroid2(img):
	return(np.einsum('ijk,jk->i',np.indices(img.shape),img)/np.einsum('jk->',img))

def gaussian(pos, amp=1, mx=0, my=0, s11=1, s22=1, t=0, amp_is_max=True, restrict_to=None):
	x,y = pos # usually we will have pos=mgrid[:10,:10] and this puts y first for some reason, so remember to do pos=mgrid[:10,:10][::-1]
	a = (np.cos(t)**2)/(2*(s11**2)) + (np.sin(t)**2)/(2*(s22**2))
	b = -np.sin(2*t)/(4*(s11**2)) + np.sin(2*t)/(4*(s22**2))
	c = (np.sin(t)**2)/(2*(s11**2)) + (np.cos(t)**2)/(2*(s22**2))
	g = amp*np.exp(-(a*(x-mx)**2 + 2*b*(x-mx)*(y-my) + c*(y-my)**2))
	if not amp_is_max:
		g/=(2*np.pi*s11*s22)
	if restrict_to is not None:
		g[x>(mx+restrict_to[0])] = 0
		g[x>(mx-restrict_to[0])] = 0
		g[y>(my+restrict_to[1])] = 0
		g[y>(my+restrict_to[1])] = 0
	return(g.ravel())


def curve_fit(func, x, y, **kwargs):
	"""
	My version of scipy.optimize.curve_fit() this one is actually an interface to scipy.optimize.least_squares(), unlike the scipy version which is not quite the same.

	func
		The function to fit to the data "y"
	x
		Independent variables to input to the function "func" (e.g. x values in "y = mx + c")
	y
		Dependent data we are trying to fit to, "func" should return something that has the same shape as y.ravel() (because "least_squares()" operates on 1d arrays only)
	**kwargs
		Arguments to pass through to "least_squares()", see that function's documentation for a full list some important ones are:
		x0=
			The initial guess to the parameters of "func" e.g. "m" and "c" in "y = mx + c"
		args=
			Other arguments to pass to "func" that will not be varied. I use this to pass constants etc.

	"func" must be of form "func(x, *params, *args, **kwargs)" where
		x - independent variables
		*params - all the varying parameters (this is your x0 initially)
		*args - any other arguments (will be passed to "func" via the "args=" keyword argument to "least_squares()"
		**kwargs - any key word arguments (will be passed to "func" via the "kwargs=" argument to "least_squares()"

	RETURNS
		optimize_result
			An instance of "optimize_result" as defined in the "least_squares()" documentation
	"""
	lsq_wrapper = lambda params, x, y, *args, **kwargs: func(x, *params, *args, **kwargs) - y
	args = tuple([x, y] + list(kwargs.pop('args', ())))
	opt_res = sp.optimize.least_squares(lsq_wrapper, args=args, **kwargs)
	return(opt_res)

def numpy_stupid_polynomial_coeffs_fix(p, deg=1):
	"""
	For some reason, numpy doesn't want to give you a way to get all the coefficents of a
	polynomial even if the coeffs are zero. This fixes that oversight
	"""
	pc = p.convert().coef
	if pc.shape[0] < deg+1:
		pc2 = np.array([pc[i] if i<pc.shape[0] else 0 for i in range(deg+1)])
		return(pc2)
	return(pc)

#def find_drift_mag(f, m_xy, m_xf, m_yf, c_xy, c_xf, c_yf):
def find_drift_mag(f, m_xf, m_yf, c_xf, c_yf):
	xy = np.zeros((f.shape[0],2))
	#xy[:,0] = (m_yf*f + c_yf - c_xy)/m_xy
	#xy[:,1] = m_xy*(m_xf*f + c_xf) + c_xy
	xy[:,0] = m_xf*f + c_xf
	xy[:,1] = m_yf*f + c_yf
	return(xy.ravel())

def find_drift_mag_2(f, m_xy, m_xf, c_xf, c_yf):
	m_yf = m_xf*m_xy
	return(find_drift_mag(f, m_xf, m_yf, c_xf, c_yf))

def find_drift_mag_3(f, m_xf, m_xy, f_ref):
	m_yf = m_xf*m_xy
	c_xf = -m_xf*f_ref
	c_yf = -m_yf*f_ref
	return(find_drift_mag(f, m_xf, m_yf, c_xf, c_yf))

def find_drift_mag_4(f, m_xf, m_yf, f_ref=500):
	c_xf = -m_xf*f_ref
	c_yf = -m_yf*f_ref
	return(find_drift_mag(f, m_xf, m_yf, c_xf, c_yf))


def get_field_drift_gradient(drift_ang, wcs, ref_px=(0,0), verbose=False):
	"""
	Gets the direction of spectroscopic drift in terms of the x and y directions of the image field (rather than sky direction)

	ARGUMENTS:
		drift_ang
			Angle from pole (NCP or SCP) to zenith of drift
		wcs
			World coordinate system object (from astropy)
		ref_px
			Reference pixel to use as point to calculate field direction and offset (default=(0,0))
		verbose
			If True will print out extra diagnostic information (default=False)

	RETURNS:
		m
			Gradient of drift direction line (y = mx + c)
		theta
			Angle that the drift direction line makes with the x-axis of the image
		dxdy
			The drift in x and y (pixel) coordinates. The drift magnitude is normalised to 1 arcsecond
		drift_dir_px
			The pixel that would correspond to a drift of 1 arcsecond away from 'ref_px'

	EXAMPLE:
		m, theta, dxdy, drift_dir_px = get_field_drift_gradient(drift_ang[0,0], tc_wcs, ref_px=tc_ra_dec_coord_px)
	"""
	if verbose:
		print('DEBUGGING: in "get_field_drift_gradient()"')
		print(f'\tdrift_ang {drift_ang}')
		print(f'\twcs {wcs}')
		print(f'\tref_px {ref_px}')
	# get bottom left corner pixel
	ref_sky = SkyCoord.from_pixel(*ref_px,wcs)
	drift_dir_sky = ref_sky.directional_offset_by(drift_ang*u.deg, 1*u.arcsec)
	drift_dir_px = drift_dir_sky.to_pixel(wcs)
	# we used (0,0) as our reference pixel so gradient is dy/dx or in this case
	dxdy = np.array((drift_dir_px[0] - ref_px[0], drift_dir_px[1]-ref_px[1]))
	m = dxdy[1]/dxdy[0]
	theta = np.arctan2(drift_dir_px[1], drift_dir_px[0])
	print(m, theta, drift_dir_px)
	return(m, theta, dxdy, drift_dir_px)
	
	

#------------------------------------------------------------------------------
def saturated_wvp_buck(T, unit='kPa'):
	"""
	From https://en.wikipedia.org/wiki/Vapour_pressure_of_water
	Gets saturation vapour pressure of water in kPa
	"""
	units_avail = ('kPa', 'Pa', 'mm Hg', 'torr')
	if unit not in units_avail:
		print('ERROR: Unknown unit asked for "{unit}", function has these units available {units_avail}')
		sys.exit()
	if unit == 'kPa':
		uf = 1
	elif unit == 'Pa':
		uf = 1E3
	elif unit == 'mm Hg':
		uf = 7.50062
	elif unit == 'torr':
		uf = 7.50062
	else:
		return(None)
	
	return(uf*0.61121*np.exp((18.678 - (T/234.5))*(T/(257.14+T))))
"""
These are based on http://articles.adsabs.harvard.edu/pdf/1982PASP...94..715F
referred to as "fillipenko82"

I've re-arranged the equations to give n instead of (n-1)x10^6
"""
def n_s(w):
	"""
	Refractive index as a function of wavelength at Temp =15 deg C, Press=760 mm Hg, water vapour pressure = 0 mm Hg
	w = wavelength (um)
	"""
	return(1 + (64.328 + (29498.1/(146 - (1/w)**2)) + (255.4/(41 - (1/w)**2)))/1E6)

def n_tp(w, T, P):
	"""
	Refractive index as a function of wavelength, temperature, pressure at water vapour pressure = 0 mm Hg
	w = wavelength(um)
	T = temperature (degrees celcius)
	P = pressure (mm Hg)
	"""
	return((n_s(w) - 1)*(P*(1+(1.049-0.0157*T)*P*1E-6))/(720.883*(1 + 0.003661*T)) + 1)

def n_tpf_fillipenko82(w, T, P, f):
	"""
	Refractive index as a function of wavelength, temperature, pressure, and water vapour pressure
	w = wavelength (um)
	T = temperature (degrees celcius)
	P = pressure (mm Hg)
	f = water vapour pressure
	"""
	return(1 + ((n_tp(w, T, P) - 1)*1E6 - ((0.0624 - 0.000680/(w**2))/(1 + 0.003661*T))*f)/1E6)
	
#------------------------------------------------------------------------------
def dR(w, T, P, f, z, w0, refractive_index=n_tpf_fillipenko82):
	"""
	Atmospheric differential refraction as a function of wavelength, temperature, pressure, water vapour pressure, zenith angle, w.r.t a refrerence wavelength in arcsec.
	w = wavelength (um)
	T = temperature (degrees celcius)
	P = pressure (mm Hg)
	f = water vapour pressure (mm Hg)
	z = zenith angle (radians)
	w0 = reference wavelength (um)
	refractive_index = function used to calcuate refractive index of air given w,T,P,f
	"""
	if DEBUG:
		print(f'In "dR()", argument list:')
		print(f'\tw {w[::100]} (um)')
		print(f'\tT {T} (celcius)')
		print(f'\tP {P} (mm Hg)')
		print(f'\tf {f} (mm Hg)')
		print(f'\tz {z} (radians)')
		print(f'\tw0 {w0} (um)')
		print(f'\trefractive_index_function {refractive_index}')
	return(206265*(refractive_index(w,T,P,f) - refractive_index(w0,T,P,f))*np.tan(z))

#------------------------------------------------------------------------------
"""
Alternate version based on http://articles.adsabs.harvard.edu/pdf/1982PASP...94..715F
referrred to as "fillepenko82"

I've not rearranged the equations, instead they are written as is incase I made
a mistake. Only the last function rearranges (n - 1)x10^6 to get n
"""
def dn_e6_s(w):
	return(64.328 + (29498.1/(146 - (1/w)**2)) + (255.4/(41 - (1/w)**2)) )

def dn_e6_tp(w, T, P):
	return((dn_e6_s(w))*(P*(1+(1.049-0.0157*T)*P*1E-6))/(720.883*(1 + 0.003661*T)))

def dn_e6_tpf(w, T, P, f):
	return(dn_e6_tp(w,T,P) - ((0.0624 - 0.000680/(w**2))/(1 + 0.003661*T))*f)
	
def n_tpf_fillipenko82_alt(w, T, P, f):
	return(dn_e6_tpf(w,T,P,f)/1E6 + 1)

#------------------------------------------------------------------------------
"""
These use https://iopscience.iop.org/article/10.1088/0026-1394/2/2/002/pdf
referred to as "edlen65"
Note, torr and mm Hg are the same unit (pretty much).
"""
def dn_e8_s(w):
	s = 1/w
	return(8342.13 + 2406030/(130-s**2) + 15997/(38.9 - s**2))

def dn_e8_tp(w,T,P):
	return(dn_e8_s(w)*(0.00138823*P)/(1+0.003671*T))

def n_tpf_edlen65(w,T,P,f):
	s = 1/w
	n_tp = dn_e8_tp(w,T,P)/1E8 + 1
	return(n_tp - f*(5.722 - 0.0457*s**2)*1E-8)
#------------------------------------------------------------------------------
def n_tpf_mathar07(w,T,P,H):
	"""
	Uses https://iopscience.iop.org/article/10.1088/1464-4258/9/5/008
	valid between 1.3 and 2.5 um
	w - wavelength (um)
	T - Temperature (celcius)
	P - Pressure (mm Hg, torr)
	H - relative humidity (fraction, e.g. 0.1 is 10%)
	"""
	print('IN "n_tpf_mathar07():"')
	if np.any(w>2.5) or np.any(w<1.3):
		print('ERROR: This calculation of refractive index is only valid within the range (1.3 um -> 2.5 um)')
		sys.exit()
	sigma_ref = 1E4/2.25 # cm^-1
	sigma = 1E4/w # wavenumber in cm^-1
	T = T+273.15 # change to Kelvin
	T_ref = 273.15+17.5
	P = 133.322*P # change to Pa
	P_ref = 75000
	H = 100*H #  change to %
	H_ref = 10
	print(f'\tsigma_ref {sigma_ref}')
	print(f'\tsigma {sigma}')
	print(f'\tT {T}')
	print(f'\tT_ref {T_ref}')
	print(f'\tP {P}')
	print(f'\tP_ref {P_ref}')
	print(f'\tH {H}')
	print(f'\tH_ref {H_ref}')

	dt = (1/T - 1/T_ref)
	dh = H - H_ref
	dp = P - P_ref
	ds = sigma - sigma_ref
	d_s = np.array([ds**0, ds**1, ds**2, ds**3, ds**4, ds**5])
	d_thp = np.array([1, dt, dh, dp])
	C = np.array([
		[	[0.2001921E-3, 0.588625E-1, -0.103945E-7, 0.267085E-8],
			[0, -3.01513, 0.497859E-4, 0.779176E-6],
			[0, 0, 0.573256E-12, -0.206567E-15],
			[0, 0, 0, 0.609186E-17]],
		
		
		[	[0.113474E-9, -0.385766E-7, 0.136858E-11, 0.135941E-14],
			[0, 0.406167E-3, -0.661752E-8, 0.396499E-12],
			[0, 0, 0.186367E-16, 0.106141E-20],
			[0, 0, 0, 0.519024E-23]],
		
		
		[	[-0.424595E-14, 0.888019E-10, -0.171039E-14, 0.135295E-18],
			[0, -0.514544E-6, 0.832034E-11, 0.395114E-16],
			[0, 0, -0.228150E-19, -0.149982E-23],
			[0, 0, 0, -0.419477E-27]],
		
		
		[	[0.100957E-16, -0.567650E-13, 0.112908E-17, 0.818218E-23],
			[0, 0.343161E-9, -0.551793E-14, 0.233587E-20],
			[0, 0, 0.150947E-22, 0.984046E-27],
			[0, 0, 0, 0.434120E-30]],
		
		
		[	[-0.293315E-20, 0.166615E-16, -0.329925E-21, -0.222957E-26],
			[0, -0.101189E-12, 0.161899E-17, -0.636441E-24],
			[0, 0, -0.441214E-26, -0.288266E-30],
			[0, 0, 0, -0.122445E-33]],
		
		
		[	[0.307228E-24, -0.174845E-20, 0.344747E-25, 0.249964E-30],
			[0, 0.106749E-16, -0.169901E-21, 0.716868E-28],
			[0, 0, 0.461209E-30, 0.299105E-34],
			[0, 0, 0, 0.134816E-37]]
	])
	print(C.shape)
	print(d_thp.shape)
	print(d_s.shape)
	# this should perform calculation (6) from the paper.
	# n - 1 = sum_j( c_j(T,H,P)*(sigma - sigma_ref)^j
	#
	# or in matrix form
	# n - 1 = sum_j( d_thp C d_thp^T (sigma - sigma_ref)^j
	#
	# of in einstein summation convention form
	# n - 1 = d(thp)_i C_j^i_k d(tph)^k d(s)^j
	n_minus_one = np.einsum('i,jik,k,j...', d_thp, C, d_thp, d_s)
	print(n_minus_one)
	print('-'*79)
	return(1+n_minus_one)
#------------------------------------------------------------------------------

def eta(obs_lat, obj_dec, obj_ha):
	if DEBUG:
		print(f'In "eta()", arguments (all angles in radians):')
		print(f'\tobs_lat {obs_lat}')
		print(f'\tobj_dec {obj_dec}')
		print(f'\tobj_ha {obj_ha}')
		print('\t'+'-'*75)
	ncp_flag = obs_lat > 0
	if not ncp_flag:
		# if we are observing from the southern hemisphere, we just have to swap the sign
		# on the latitude and declination, do the same calculations, and  do 
		# eta = pi - eta if eta >0 and 
		# eta = -pi - eta if eta < 0 after all the calculations are done
		obs_lat = -obs_lat
		obj_dec = -obj_dec
		if DEBUG:
			print(f'\tobs_lat < 0, so not at NCP, taking negatives of obs_lat and obj_dec')
			print(f'\tobs_lat {obs_lat}')
			print(f'\tobj_dec {obj_dec}')
	
	# if we use obj_ha_24 for everything then we don't have to make wierd edge cases
	obj_ha_24 = np.remainder(obj_ha,2*np.pi) # set object hour angle to between 0 and 2*pi, (0 and 24 degree hours)
	if DEBUG:
		print(f'\tRationalising hour angles to range 0->2*pi')
		print(f'\tobj_ha_24 {obj_ha_24}')
	#west_of_meridian_idxs = np.nonzero(obj_ha_24 > np.pi)
	#obj_ha[west_of_meridian_idxs] = -obj_ha[west_of_meridian_idxs]
	
	sin_eta = (np.sin(obj_ha_24)*np.cos(obs_lat))/(np.sqrt(
							1 - (np.sin(obs_lat)*np.sin(obj_dec) 
								+ np.cos(obs_lat)*np.cos(obj_dec)*np.cos(obj_ha_24)
							)**2
						))
	#print(f'obs_lat {obs_lat.T}')
	#print(f'obj_dec {obj_dec.T}')
	#print(f'obj_ha {obj_ha.T}')
	
	eta = np.arcsin(sin_eta) 
	# numpy only returns an angle between -pi/2 and pi/2 (-90 to 90 deg) for arcsin
	# and between 0 and pi (0, 180) for arccos
	# so we have to do some extra logic to get the correct angle of eta
	
	# if obj_dec > obs_lat then there will be some point where the parallatic angle (eta) is 90 degrees
	# if the hour angle is closer to the meridian than this (i.e. obs_ha < z_90), then eta_real = 180 - eta
	
	# length of z when zenith-object line is a tangent to the object's path circle
	# need to use spherical geometry cosine law with eta = pi/2
	# i.e. cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A), with A = pi/2 so cos(pi/2)=0
	# where a, b, c are the lines of the triangle, and A, B, C are the angles opposite the lines a, b, c
	# If the path of the object on the sky is closer to the pole than the zenith
	# (which happens when dec > lat), then there will be two special points where the object-zenith line is
	# a tangent to the circle created by the object's path on the sky
	path_within_zenith_idxs = obs_lat < obj_dec
	z_90 = np.arccos(np.sin(obs_lat)/np.sin(obj_dec)) 
	#z_270 = z_90 #  don't actually need this
	# hour angle of object when zenith-object line is tangent to the object's path circle
	# re-arrange spherical cosine law to get hour angle as the subject
	h_90 = np.arccos((np.cos(z_90)-np.sin(obj_dec)*np.sin(obs_lat))/(np.cos(obj_dec)*np.cos(obs_lat))) 
	h_270 = 2*np.pi - h_90
	
	
	# one more thing to remember is that we can only have an eta>90 (or eta<-90) if
	# the zenith is outside the circular path the object makes on the sky
	# This only happens when the decination of the object is larger than the observer's latitude
	eta_greater_than_90_idxs = np.nonzero(((obj_ha_24 < h_90) & (obj_ha_24 > 0)) & (obs_lat < obj_dec))
	eta_less_than_270_idxs = np.nonzero(((obj_ha_24 > h_270) & (obj_ha_24 < 180)) & (obs_lat < obj_dec)) # less than -90

	eta[eta_greater_than_90_idxs] = np.pi - eta[eta_greater_than_90_idxs]
	#eta[eta_greater_than_180_idxs] = np.pi 
	eta[eta_less_than_270_idxs] = -np.pi - eta[eta_less_than_270_idxs]

	# hour angle of object when rising
	h_r = np.arccos(-np.tan(obj_dec)*np.tan(obs_lat))
	# hour angle of object when setting
	h_s = 2*np.pi - h_r
	print('\tINFO: h_r', h_r)
	mask_for_non_visible = np.zeros_like(eta)
	# if pi/2-dec > lat then the object's path on the sky will be clipped by 
	# the horizon between the rising and setting hour angles
	mask_for_non_visible = (np.pi/2-obj_dec > obs_lat) & ( (h_r < obj_ha_24) & (obj_ha_24 < h_s) )
	
	
	print(eta[:,0])
	if not ncp_flag:
		print('\tINFO: Not at NCP')
		eta[eta>0] = np.pi - eta[eta>0]
		eta[eta<0] = - np.pi - eta[eta<0]
		print(eta[:,0])
	
	
	return(np.ma.array(eta, mask=mask_for_non_visible))



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

	parser.add_argument('target_cubes', type=str, nargs='+', help='fitscubes to operate on, will output a drift-adjusted fits file in the same directory as the target cube called "out_objnod_drift_corrected.fits"')
	parser.add_argument('--show_plots', action='store_true', help='if present, will show intermediate plots')
	drift_mag_mode_choices = ('refractive_index', 'empirical')
	parser.add_argument('--drift_mag_mode', type=str, choices=drift_mag_mode_choices, help=f'Which mode to use to determine drift magnitude. Choices are {drift_mag_mode_choices}.', default='refractive_index')
	parser.add_argument('--reverse', action='store_true', help='If present, will apply the drift correction in the opposite manner so as to reverse any correction that has already been applied')
	parser.add_argument('--output.suffix', type=str, help='The suffix to append to the file name after processing, will be joined to the orginal name using an underscore', default='drift_subtracted')
	parser.add_argument('--output.no_overwrite', action='store_false', dest='output.overwrite', help='If present will not overwrite any output files present')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
