#!/usr/bin/python3
"""
Processes SINFONI datacubes for later analysis

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
import astropy as ap
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.modeling import models, fitting
from astropy.wcs import WCS
import fitscube.process.cfg
import fitscube.header
from astroquery.jplhorizons import Horizons
from astroquery.simbad import Simbad
import matplotlib.pyplot as plt
import scipy.optimize as spo
from astropy import units as u
import textwrap
import copy


class Target():
	def __init__(self, datacube):
		deg2arcsec = 3600.0
		arcsec2ster = (np.pi/(180*3600))**2
		self.datacube = datacube
		self.hdul = fits.open(datacube)
		self.origin_observatory = self.hdul[0].header['ORIGIN'].lower()
		self.object = self.hdul[0].header['ESO OBS TARG NAME'].lower()
		self.observation_date = self.hdul[0].header['DATE-OBS']
		self.jpl_obj = jplh_query_object_at_instant(self.origin_observatory, self.object, self.observation_date)
		self.jpl_eph = self.jpl_obj.ephemerides()
		# airmass between fits file and jpl-horizons is different by quite a lot, why?
		# ANS: need to use ESO TEL AIRM START and ESO TEL AIRM END.
		# ESO OBS AIRM is the MAXIMUM airmass that the observation can cope with, NOT the one USED.
		self.airmass = 0.5*(self.hdul[0].header['ESO TEL AIRM START'] + self.hdul[0].header['ESO TEL AIRM END']) 
		#print(self.jpl_eph['airmass'])
		self.exposure_time = self.hdul[0].header['EXPTIME']
		self.sun_dist = self.jpl_eph['r'][0] # r - heliocentric distance in AU
		self.sub_obs_lat = self.jpl_eph['PDObsLat'][0] # deg, planetodetic sub-observer latitude
		self.obs_dist = self.jpl_eph['delta'][0] # AU, distance from observer to target
		self.obs_vel = self.jpl_eph['delta_rate'][0] # km/s, velocity of target away from observer
		self.ang_width = self.jpl_eph['ang_width'][0] # arcsec, angular width of the disk (uses equatorial radius)
		self.px_scale = self.hdul[0].header['CDELT1']*deg2arcsec
		self.py_scale = self.hdul[0].header['CDELT2']*deg2arcsec
		self.pix_ster = np.abs(self.px_scale*self.py_scale)*arcsec2ster
		self.wavgrid = datacube_wavelength_grid(self.hdul[0])

	def counts_to_spradiance(self, factor):
		# factor should have units W cm-2 um-1 ster-1 counts-1 s
		# self.hdu[0].data has units counts
		# self.exposure_time has units seconds
		f1 = plt.figure(figsize=[_x/2.54 for _x in (24,12)])
		a11 = f1.add_subplot(1,2,1)
		a11.plot(self.wavgrid, self.hdul[0].data[:,30,30])
		a11.set_title("counts")

		self.hdul[0].data *= factor/self.exposure_time

		a12 = f1.add_subplot(1,2,2)
		a12.plot(self.wavgrid, self.hdul[0].data[:,30,30])
		a12.set_title('spectral radiance')
		plt.show()
		return
	
	def H_filter_to_datacube_wavelengths(self):
		hft = fitscube.process.cfg.H_filter_transmittance
		self.h_filter_interp = np.interp(self.wavgrid, hft[:,1], hft[:,2])
		#print(self.wavgrid)
		#print(self.h_filter_interp)
		return

	def apply_tellspec(self, std_odepth_per_airmass, std_polyfit, show_plots=True):
		# have standard star optical depth per airmass
		ut.pINFO("In 'Target.apply_tellspec'")
		print(std_odepth_per_airmass)
		print(std_polyfit)
		print(self.wavgrid)
		print(self.wavgrid.shape)
		self.H_filter_to_datacube_wavelengths()
		odepth = std_odepth_per_airmass*self.airmass
		trans = np.exp(-odepth)
		polyline = std_polyfit[0]*self.wavgrid**2 + std_polyfit[1]*self.wavgrid + std_polyfit[2]# gain
		spec = trans*polyline
		tellmean = np.nansum(spec*self.h_filter_interp)/np.nansum(self.h_filter_interp)
		print(f'tellmean {tellmean}')

		normspec = np.nan_to_num(spec/tellmean, nan=1.0)
		recip_normspec = 1.0/normspec

		before_tellspec_norm = self.hdul[0].data[:,30,30].copy()
		self.hdul[0].data *= recip_normspec[:,None,None] #1.0/normspec[:,None,None]

		if show_plots:		
			f1 = plt.figure(figsize=[_x/2.5 for _x in (48,24)])
			a11 = f1.add_subplot(2,4,1)
			a11.plot(self.wavgrid, before_tellspec_norm)
			a11.set_title("pixel (30,30) before tellspec normalisation")

			a12 = f1.add_subplot(2,4,2)
			a12.plot(self.wavgrid, odepth, label='optical depth')
			a12.plot(self.wavgrid, trans, label='transmittance')
			a12.legend()

			a13 = f1.add_subplot(2,4,3)
			a13.plot(self.wavgrid, normspec)
			a13.set_title("normalised tellspec")

			a14 = f1.add_subplot(2,4,4)
			a14.plot(self.wavgrid, self.hdul[0].data[:,30,30])
			a14.set_title("pixel (30,30) after normalised by tellspec")

			a15 = f1.add_subplot(2,4,5)
			a15.plot(self.wavgrid, spec, label='transmittance*polyline')
			a15.plot(self.wavgrid, polyline, label='polyline')
			a15.legend()

			a16 = f1.add_subplot(2,4,6)
			a16.plot(self.wavgrid, recip_normspec, label='reciprocal of normspec')
			a16.legend()


			plt.show()
	
	def project_lat_lon(self):
		deg2rad = np.pi/180.0
		Re = self.ang_width # equatorial radius in arcseconds
		oblateness = fitscube.process.cfg.planet_data[self.object]['oblateness']
		Rp = Re*(1.0 - oblateness) # polar radius in arcsec
		
		# define an ellipsoid with same oblateness, but major axis = 1
		a = 0.5 # major axis of 1, sma of 0.5
		b = a*(1.0 - oblateness)
		m = np.tan(self.sub_obs_lat)
		if m != 0.0:
			xdiff_sq = 4.0/((1.0/a**2) + (b**2/(m**2*a**4)))
			ydiff_sq = (4.0*b**4)/((m*a)**2 + b**2)
			proj_dist = np.sqrt(xdiff_sq + ydiff_sq)
			Rpp = proj_dist*Re # major axis = 1, so just scale by equatorial diameter
		else:
			Rpp = Rp
		print(f'Equatorial diameter {Re} arcsec')
		print(f'Polar diameter {Rp} arcsec')
		print(f'Projected polar diameter {Rpp} arcsec')

		# auto-find pointing by finding where the greatest signal enclosed by
		# an ellipse with a=Re/2, b=Rpp/2 ?	

		print(self.hdul[0].header['CDELT1'])
		print(self.px_scale)

		a = np.abs(Re/(2*self.px_scale))
		b = np.abs(Rpp/(2*self.py_scale))
		e_idx, ellipse = self.find_max_ellipse_top_hat(a, b, t=0)

		print(f'ellipse a {a} b {b} t {0}')
		print(f'ellipse center indices {e_idx}')

		#### Plot found ellipse	
		f1 = plt.figure(figsize=[_x/2.54 for _x in (36,12)])

		a11 = f1.add_subplot(1,3,1)
		data = np.nanmean(self.hdul[0].data, axis=0)
		cmin = np.nanmin(data)
		cmax = np.nanmax(data)
		img1 = a11.imshow(data, origin='lower', vmin=cmin, vmax=cmax)
		a11.set_title('Median of data')

		print(data.shape)
		print(ellipse.shape)
		ec = np.array([e_idx[1],e_idx[0]])
		aa = np.array([a,0])
		bb = np.array([0,b])
		print(ec, aa, bb)
		print(ec+aa, ec+bb)
		al = np.stack([ec,ec+aa], axis=0)
		bl = np.stack([ec,ec+bb], axis=0)
		print(al)
		print(bl)


		a12 = f1.add_subplot(1,3,2)
		img2 = a12.imshow(ma.array(data,mask=np.logical_not(np.array(ellipse, dtype='bool'))), origin='lower', vmin=cmin, vmax=cmax)
		#img2 = a12.imshow(ellipse, origin='lower', zorder=0)
		a12.scatter(e_idx[1], e_idx[0], edgecolor='red', facecolor='none', zorder=1, label='ellipse center')
		a12.plot(al[:,0], al[:,1], color='red', zorder=2, label='semi-major axis')
		a12.plot(bl[:,0], bl[:,1], color='green', zorder=3, label='semi-minor axis')
		a12.legend()
		a12.set_title('Ellipse fitted to planet disc')

		a13 = f1.add_subplot(1,3,3)
		img3 = a13.imshow(data - ellipse*data, origin='lower', vmin=cmin, vmax=cmax)
		a13.set_title('data median minus fitted ellipse')

		plt.show()
		##### end plotting

		ny, nx = self.hdul[0].data.shape[1:]
		lats = np.ones((ny,nx))*np.nan # fill with junk value
		lons = np.ones((ny,nx))*np.nan # fill with junk value
		zens = np.ones((ny,nx))*np.nan # fill with junk value
		ecy, ecx = e_idx
		print(f'ellipse center x {ecx} y {ecy}')
		for i in range(nx):
			for j in range(ny):
				eoffA = (i-ecx)*np.abs(self.px_scale)
				poffA = (j-ecy)*np.abs(self.py_scale)
				#print(f'i {i} j {j} eoffA {eoffA} poffA {poffA} px_scale {self.px_scale} py_scale {self.px_scale} nx {nx} ny {ny}')
				iflag, xlat, xlon, zen = projpos_ellipse(Re/2, Re/2, self.sub_obs_lat, eoffA, poffA)
				#print(f'Re {Re} Rp {Rp} ecx {ecx} ecy {ecy} iflag {iflag} xlat {xlat} xlon {xlon} zen {zen}')
				if iflag:
					lats[j,i] = xlat
					lons[j,i] = xlon
					zens[j,i] = zen

		yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
		yy = (yy-ecy)*np.abs(self.py_scale)
		xx = (xx-ecx)*np.abs(self.px_scale)

		### plot for debugging
		print(f'lats.shape {lats.shape}')
		print(f'lons.shape {lons.shape}')
		print(f'zens.shape {zens.shape}')

		nx, ny = (4,1)
		f2 = plt.figure(figsize=[_x/2.54 for _x in (12*nx, 12*ny)])

		a21 = f2.add_subplot(ny,nx,1)
		im21 = a21.imshow(data, origin='lower')
		a21.set_title('Median of data')	
	
		a22 = f2.add_subplot(ny,nx,2)
		#im22 = a22.imshow(lats, origin='lower')
		im22 = a22.imshow(ma.array(lats, mask=np.isnan(lats)), origin='lower')
		a22.set_title('Latitude')

		a23 = f2.add_subplot(ny,nx,3)
		#im23 = a23.imshow(lons, origin='lower')
		im23 = a23.imshow(ma.array(lons, mask=np.isnan(lons)), origin='lower')
		a23.set_title('longitude')

		a24 = f2.add_subplot(ny,nx,4)
		#im24 = a24.imshow(zens, origin='lower')
		im24 = a24.imshow(ma.array(zens, mask=np.isnan(zens)), origin='lower')
		a24.set_title('angle of LOS w.r.t zenith')

		plt.show()
		### end plotting

		# store relavent data in class
		self.ellipse_center_x = ecx
		self.ellipse_center_y = ecy
		self.Re = Re
		self.Rp = Rp
		self.Rpp = Rpp
		self.disk = ellipse
		self.lats = lats
		self.lons = lons
		self.zens = zens
		self.xx = xx
		self.yy = yy
		return

	def find_disk_fraction(self, std):
		self.seeing = std.seeing
		shape = self.hdul[0].data.shape[1:]
		x, y = np.mgrid[:shape[0], :shape[1]]
		nz, ny, nx = self.hdul[0].data.shape
		#g_norm = gaussian_2d((x,y), 1, nx/2, ny/2, self.seeing/np.abs(self.px_scale), 
		#						self.seeing/np.abs(self.py_scale), 0, 0).reshape(*shape)
		g_norm = gaussian_2d((x,y), 1, nx/2, ny/2, *std.popt[3:-1], 0.0).reshape(*shape)
		g_norm_sum = np.sum(g_norm)
		self.psf = g_norm/g_norm_sum
		disk_sum = np.sum(self.disk)
		print(f'g_norm_sum {g_norm_sum}')
		print(f'disk_sum {disk_sum}')
		print(f'std.popt {std.popt}')
	
		f1 = plt.figure(figsize = [_x/2.54 for _x in (12,12)])
		a11 = f1.add_subplot(1,1,1)	
		a11.imshow(g_norm, origin='lower')
		a11.set_title('Point spread function')
		plt.show()
		
		df = np.zeros((ny,nx))
		for j in range(ny):
			for i in range(nx):
				g = gaussian_2d((x,y), 1.0/g_norm_sum, j, i, *std.popt[3:-1], 0.0).reshape(*shape)
				df[j,i] = np.sum(g*self.disk)
				#print(f'i {i} j{j} df[{j},{i}] {df[j,i]}')
		self.disk_frac = df
		return

	def find_stat_err(self):
		# get statistical error on data
		nx, ny = (4,1)
		cm_size = 12
		f1 = plt.figure(figsize=[_x/2.54 for _x in (cm_size*nx, cm_size*ny)])

		a11 = f1.add_subplot(ny,nx,1)
		a12 = f1.add_subplot(ny,nx,2)
		a13 = f1.add_subplot(ny,nx,3)


		self.stat_error = np.zeros_like(self.hdul[0].data) # store stat error like this
		self.stat_error_mask = np.zeros_like(self.disk_frac, dtype='float')
		self.stat_error_mask[np.where(self.disk_frac < 0.001)] = 1

		img11 = a11.imshow(self.stat_error_mask, origin='lower')
		a11.set_title('stat error mask')

		mask = np.zeros_like(self.stat_error, dtype='bool')
		mask[:, :, :] = self.stat_error_mask[None, :, :]
		masked_data = ma.array(self.hdul[0].data, mask=np.logical_not(mask))

		img12 = a12.imshow(masked_data[300,:,:], origin='lower')
		a12.set_title('masked data')

		err = np.nanstd(masked_data, axis=(1,2))

		a13.plot(err)
		a13.set_title('err')

		self.stat_error[:, :, :] = err[:,None,None]

		plt.show()
		return


	def assemble_fits(self, std_cal, outfname='out_calibrated.fits', overwrite=False):
		"""
		Assembles a fits file that is congruent to those created by IDL routine 'process_sinf.pro'

		NOTE: IDL routines make an 80x80 image with the planetary disk center and the image center aligned, I haven't done that here, but if some routines fail that may be why

		hdul[0] = data calibrated for spectral radiance using the calibrator
		hdul[1] = error on data (each pixel has a 1:1 correspondence, could change this to be a single wavelength dependet column but IDL routines don't do that (and therefore have loads of redundant data)
		hdul[2] = latitude of pixel
		hdul[3] = longitude of pixel
		hdul[4] = angle LOS makes to zenith of surface
		hdul[5] = arcsec in x-direction from the center of disk
		hdul[6] = arcsec in y-direction from the center of disk

		My extras
		hdul[7] = on-off planetary disk mask
		hdul[8] = pixel fraction that is emission from disk
		hdul[9] = pixels used for statistical data
		"""
		onetruepath = lambda x: os.path.normpath(os.path.abspath(x))

		self.hdul[0].header['PARENT'] = (onetruepath(self.datacube), 'Uncalibrated datacube')
		self.hdul[0].header['CAL_OBJ'] = (std_cal.object, 'The object used to calibrate this observation')
		self.hdul[0].header['CAL_FILE'] = (onetruepath(std_cal.datacube),
											'Datacube used to calibrate this observation')
		self.hdul[0].header['CAL_DATE'] = (std_cal.observation_date, 'Date the calibration object was observed')
		self.hdul[0].header['CAL_AIRM'] = (std_cal.airmass, 'Airmass of calibration observation')
		self.hdul[0].header['CAL_EXTM'] = (std_cal.exposure_time, 'Exposure time of calibration observation')
		self.hdul[0].header['CAL_ORIG'] = (std_cal.origin_observatory, 
											'Location the calibration observation was taken from')
		self.hdul[0].header['CAL_SEE'] = (std_cal.seeing, 
											'(arcsec), seeing found from fitting a function (see "CAL_SEEF")')
		self.hdul[0].header['CAL_SEEF'] = (std_cal.seeing_function, 'Function used to compute seeing')
		self.hdul[0].header['CAL_RA'] = (std_cal.ra, '(deg) Right ascention of calibration object')
		self.hdul[0].header['CAL_DEC'] = (std_cal.dec, '(deg) Declination of calibration target')
		self.hdul[0].header['CAL_RV'] = (std_cal.radial_velocity, '(km/s) Radial veclocity of calibration target')
		self.hdul[0].header['CAL_HFLX'] = (std_cal.h_flux, '(mag) H-band flux of calibration target')
		self.hdul[0].header['CAL_BFLX'] = (std_cal.b_flux, '(mag) B-band flux of calibration target')
		self.hdul[0].header['CAL_VFLX'] = (std_cal.v_flux, '(mag) V-band flux of calibration target')
		
	
		# This approach to choosing just the bits of the main header we need doesn't work	
		#wcs_3d = WCS(self.hdul[0].header)
		#wcs_cel = wcs_3d.sub(['celestial'])
		#wcs_spec = wcs_3d.sub(['spectral'])
		#hdr_3d = wcs_3d.to_header()
		#hdr_cel = wcs_cel.to_header()
		#hdr_spec = wcs_spec.to_header()

		hdr_3d = fitscube.header.get_coord_fields(self.hdul[0].header)
		hdr_cel = hdr_3d
		hdr_spec = hdr_3d

		print(hdr_3d)
		#print(hdr_cel)
		#print(hdr_spec)

		err_hdu = fits.ImageHDU(data=self.stat_error,
								header=hdr_3d,
								name = 'ERROR'
								)

		# may have to change the headers to account for no spectral axis
		lat_hdu = fits.ImageHDU(data=self.lats,
								header=hdr_cel,
								name='LATITUDE'
								)

		lon_hdu = fits.ImageHDU(data=self.lons,
								header=hdr_cel,
								name='LONGITUDE'
								)

		zen_hdu = fits.ImageHDU(data=self.zens,
								header=hdr_cel,
								name='ZENITH'
								)

		xarcsec_hdu = fits.ImageHDU(data=self.xx,
									header=hdr_cel,
									name='X_ARCSEC'
									)
	
		yarcsec_hdu = fits.ImageHDU(data=self.yy,
									header=hdr_cel,
									name='Y_ARCSEC'
									)

		planmask_hdu = fits.ImageHDU(data=self.disk.astype('float'),
									header=hdr_cel,
									name='DISK_MASK'
									)

		diskfrac_hdu = fits.ImageHDU(data=self.disk_frac,
									header=hdr_cel,
									name='DISK_FRAC'
									)

		errmask_hdu = fits.ImageHDU(data=self.stat_error_mask.astype('float'),
									header=hdr_cel,
									name='ERROR_MASK'
									)

		psf_hdu = fits.ImageHDU(data=self.psf, 
								header=hdr_cel, 
								name='PSF'
								)

		err_sp_hdu = fits.ImageHDU(data=self.stat_error[:,0:1,0:1],
									header=hdr_spec,
									name='COUNT_ERROR_VS_SPECTRAL'
									)

		cal_hdu = fits.ImageHDU(data=std_cal.hdul[0].data,
								header = std_cal.hdul[0].header,
								name='CALIBRATOR'
								)
		cal_hdu.header['PARENT'] = (onetruepath(std_cal.datacube),
		  					'Original calibrator datacube')
		cal_hdu.header['AVG_AIRM'] = (std_cal.airmass, 'Average airmass of calibrator')
		cal_hdu.header['SEEING'] = (std_cal.seeing, 
		  					'(arcsec), seeing found from fitting a function (see "SEE_FUNC")')
		cal_hdu.header['SEE_FUNC'] = (std_cal.seeing_function, 'Function used to compute seeing')
		cal_hdu.header['RV'] = (std_cal.radial_velocity, '(km/s) Radial veclocity of calibration target')
		cal_hdu.header['H_FLUX'] = (std_cal.h_flux, '(mag) H-band flux of calibrator')
		cal_hdu.header['B_FLUX'] = (std_cal.b_flux, '(mag) B-band flux of calibrator')
		cal_hdu.header['V_FLUX'] = (std_cal.v_flux, '(mag) V-band flux of calibrator')
		
		list_of_hdus =  [	err_hdu,
							lat_hdu, 
							lon_hdu, 
							zen_hdu, 
							xarcsec_hdu, 
							yarcsec_hdu, 
							planmask_hdu, 
							diskfrac_hdu, 
							errmask_hdu, 
							psf_hdu,
							err_sp_hdu,
							cal_hdu
						]
		for hdu in list_of_hdus:
			self.hdul.append(hdu)

		self.hdul.writeto(outfname, overwrite=overwrite)
		return


	def find_max_ellipse_top_hat(self, a=1, b=1, t=0):
		data = np.nanmean(self.hdul[0].data, axis=(0))
		top_hat_sum = np.zeros_like(data)
		print('data.shape {}'.format(data.shape))
		x, y = np.mgrid[:data.shape[0], :data.shape[1]]
		# simple grid search, make better if I need to
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				eth = ellipse_top_hat_2D((x,y), x[i,j], y[i,j], a, b, t)
				top_hat_sum[i,j] = np.nansum(eth*data)
		max_top_hat_sum_idx = np.unravel_index(np.nanargmax(top_hat_sum), data.shape)

		return(max_top_hat_sum_idx, ellipse_top_hat_2D((x,y), x[max_top_hat_sum_idx], y[max_top_hat_sum_idx],a,b,t))
				

		

	def __repr__(self):
		return(textwrap.dedent(f"""
			<##########################>
			"Target" object at {id(self)}
			datacube {self.datacube}
			origin observatory {self.origin_observatory}
			object {self.object}
			observation date {self.observation_date}
			jpl_obj {self.jpl_obj}
			avg_airmass {self.airmass}
			exposure time {self.exposure_time} s
			distance to sun {self.sun_dist} AU
			sub-observer latitude {self.sub_obs_lat} deg
			distance from observer {self.obs_dist} AU
			velocity from observer {self.obs_vel} km/s
			<-------------------------->
			"""))

class Standard():
	def __init__(self, datacube, tellspec_folder):
		deg2arcsec = 3600.0
		degsq2ster = 360**2/(4*np.pi)
		arcsec2ster = (np.pi/(180*3600))**2
		self.datacube = datacube
		self.tellspec_folder = tellspec_folder
		self.hdul = fits.open(datacube)
		self.origin_observatory = self.hdul[0].header['ORIGIN'].lower()
		self.object = self.hdul[0].header['ESO OBS TARG NAME'].lower()
		self.observation_date = self.hdul[0].header['DATE-OBS']
		# get standard star data from online catalogs
		
		self.airmass = 0.5*(self.hdul[0].header['ESO TEL AIRM START'] + self.hdul[0].header['ESO TEL AIRM END'])
		# airmass from catalog also?
		self.exposure_time = self.hdul[0].header['EXPTIME']
		self.sbd_data = None # simbad data filled outside of constructor because best to use vectorised queries

		# the standard star after it has been adjusted to look like the "standard A0V star"
		self.tellspec_A0V = os.path.normpath(os.path.join(self.tellspec_folder, f'xtellcor_{self.object.upper()}_adj_A0V.dat'))
		# The correction to the spectrum due to telluric lines
		self.tellspec_adj = os.path.normpath(os.path.join(self.tellspec_folder, f'xtellcor_{self.object.upper()}_adj.dat'))
		# The spectrum of the "data" after the telluric correction has been applied 
		self.tellspec = os.path.normpath(os.path.join(self.tellspec_folder, f'xtellcor_{self.object.upper()}_adj_tellspec.dat'))
		self.seeing = None # fill with one of the 'fit_*' methods, should I also have FWHM of a gaussian?
		self.seeing_function = None
		self.px_star_center = None
		self.py_star_center = None
		self.p_star_width = None
		self.px_scale = self.hdul[0].header['CDELT1']*deg2arcsec
		self.py_scale = self.hdul[0].header['CDELT2']*deg2arcsec
		self.pix_ster = np.abs(self.px_scale*self.py_scale)*arcsec2ster
		self.wavgrid = datacube_wavelength_grid(self.hdul[0])
		self.steps = ['raw']
		return

	def get_data_from_simbad_query(self):
		self.h_flux = self.sbd_data['FLUX_H']
		self.b_flux = self.sbd_data['FLUX_B']
		self.v_flux = self.sbd_data['FLUX_V']
		self.radial_velocity = self.sbd_data['RV_VALUE']
		self.ra = hour_angle_str_to_deg(self.sbd_data['RA'])
		self.dec = sexagesimal_str_to_deg(self.sbd_data['DEC'])
		self.proper_motion_ra = self.sbd_data['PMRA']
		self.proper_motion_dec = self.sbd_data['PMDEC'] 

	def set_seeing_pixels(self, seeing_px):
		"""
		Set seeing (which is angluar, in arcsec) from a value in pixels
		"""
		#self.seeing = np.sqrt((seeing_px*self.px_scale)**2+(seeing_px*self.py_scale)**2)
		self.seeing = seeing_px*np.abs(self.px_scale)
		return

	def set_star_details(self, x, y, s):
		self.px_star_center = x
		self.py_star_center = y
		self.p_star_width = s

	def get_boring_region_spectral_background(self):
		# get a pixel number grid for our datacube
		zs, ys,xs = np.mgrid[:self.hdul[0].data.shape[0], :self.hdul[0].data.shape[1], :self.hdul[0].data.shape[2]]
		#print(xs.shape, ys.shape)
		# select the indicies where the distance from the star in pixels is 5 times the seeing size in pixels
		keep_idx = np.nonzero((xs - self.px_star_center)**2 + (ys - self.py_star_center)**2 > (5*self.p_star_width)**2)
		#print(keep_idx)
		#print(keep_idx[0].shape, keep_idx[1].shape, keep_idx[2].shape)
		#plt.scatter(keep_idx[1],keep_idx[2])
		#plt.show()
		# create a true,false mask for the data. In numpy, False=keep data, True=mask data
		# so initially set everything to True, i.e. mask all data out.
		# called 'prism_mask' as the shape of the mask only varies in N-1 dimensions, just like a prism in
		# 3D only varys shape in 2 dimensions
		prism_mask = np.ones_like(self.hdul[0].data, dtype='bool') # masked array False = keep data, True=remove data
		#print(prism_mask)
		#print(prism_mask.shape)
		prism_mask[keep_idx] = False # only keep data that is far from the star, this is our background
		# find the mean of the background for each spectral index
		spectral_background = np.nanmean(np.ma.array(self.hdul[0].data,mask=prism_mask), axis=(1,2))
		#print(spectral_background)	
		#print(spectral_background.shape)

		#plt.plot(spectral_background)
		#plt.show()
		#print(self.hdul[0].data[300,30,30])
		return(spectral_background)

	def subtract_spectral_background(self, spectral_background):
		self.hdul[0].data -= spectral_background[:,None,None]
		self.subtracted_background = spectral_background
		#print(self.hdul[0].data[300,30,30])
		self.steps += ['background subtracted']
		return

	def H_filter_to_datacube_wavelengths(self):
		hft = fitscube.process.cfg.H_filter_transmittance
		self.h_filter_interp = np.interp(self.wavgrid, hft[:,1], hft[:,2])
		print(self.wavgrid)
		print(self.h_filter_interp)
		return

	def get_spfluxangexpfac(self):
		# data is in counts
		# reference is in magnitudes
		# convert magnitudes to W cm-2 um-1
		# have self.spfluxfac in W cm-2 um-1 counts-1
		# want a factor to change from counts --> W cm-2 um-1 ster-1
		
		h_band_mean_counts = np.nansum(self.hdul[0].data*self.h_filter_interp[:,None,None])/(np.nansum(self.h_filter_interp))
		h_band_mean_counts_per_sec = h_band_mean_counts/self.exposure_time # counts sec-1
		h_band_mean_spflux = fitscube.process.cfg.hmag2spflux(self.h_flux) # W cm-2 um-1
		# self.px_ster is pixel size in steradians
		h_band_mean_spflux_per_ster = h_band_mean_spflux/self.pix_ster # W cm-2 um-1 ster-1
		self.spfluxangexpfac = h_band_mean_spflux_per_ster/h_band_mean_counts_per_sec # W cm-2 um-1 ster-1 counts-1 s
		# factor to change from counts per second to Watts per square centimeter per micron per steradian
		return(self.spfluxangexpfac)

	def read_tellspec_data(self):
		self.tellspec_data = np.loadtxt(self.tellspec, comments='#')
		self.tellspec_A0V_data = np.loadtxt(self.tellspec_A0V, comments='#')
		self.tellspec_adj_data = np.loadtxt(self.tellspec_adj, comments='#')

		# xtellcor_general outputs the reciprocal of the corrected solar spectrum, so invert here
		self.recip_tellspec_data = self.tellspec_data.copy()
		self.recip_tellspec_data[:,1] = 1.0/self.recip_tellspec_data[:,1]

		# get a solar transmission spectrum for our wavelength range
		trans_spec_raw = np.loadtxt(os.path.expanduser('~/Documents/reference_data/telluric_features/mauna_kea/sky_transmission/mktrans_zm_50_20.dat'))
		trans_spec = np.zeros(self.tellspec_data.shape[:2])
		trans_spec[:,0] = self.tellspec_data[:,0]
		trans_spec[:,1] = np.interp(trans_spec[:,0], trans_spec_raw[:,0], trans_spec_raw[:,1])
		trans_idxs = np.where(trans_spec[:,1] > 0.990) # get indices where transmittance is nearly 100%
		print('trans idxs')
		print(trans_idxs)
		print('trans_spec[trans_idxs]')
		print(trans_spec[trans_idxs,1][0])
		# assume for transmission close to 1 that the real spectrum is equal to tellspec/transmittance
		real_spec = trans_spec.copy()
		real_spec[:,1] = 1.0/(real_spec[:,1])*self.recip_tellspec_data[:,1]
		real_spec = real_spec[trans_idxs]
		# fit a 2nd order polynomial to the 'real' spectrum
		polyfit = np.polyfit(real_spec[:,0], real_spec[:,1], 2)
		print(polyfit)
		# shift the fitted polynomial so it is just touching the tips of the fitted data
		polyline_small = polyfit[0]*real_spec[:,0]**2 + polyfit[1]*real_spec[:,0] + polyfit[2]
		polymax = np.nanmax(real_spec[:,1]/polyline_small)
		polyline = ((trans_spec[:,0]**2)*polyfit[0] + trans_spec[:,0]*polyfit[1] + polyfit[2])*polymax
		print(polyline)
		flattened_spec = self.recip_tellspec_data.copy()
		flattened_spec[:,1] = flattened_spec[:,1]/polyline # de-trend the tellspec data using the polynomial

		# by now we are at the equivalent point of line 177 in 'sub_scale_airmass_sinf.pro'
		# "flattened_spec" is the equivalent of "trans", "polyline" is the equivalent of "gain",
		# "self.recip_tellspec_data" is the equivalent of "spec"

		odepth = -np.log(flattened_spec[:,1])

		hfilter = flattened_spec.copy()
		hfilter[:,1] = np.interp(hfilter[:,0], fitscube.process.cfg.H_filter_transmittance[:,1], fitscube.process.cfg.H_filter_transmittance[:,2])

		self.odepth_per_airmass = odepth/self.airmass
		self.polyfit = polyfit

		print(self.tellspec_data)
		######### plot for visualisation #############
		plot_x = 4
		plot_y = 2
		plot_cm_size = 12
		f1 = plt.figure(figsize=[_x/2.54 for _x in (plot_x*plot_cm_size, plot_y*plot_cm_size)])

		a11 = f1.add_subplot(plot_y,plot_x,1)
		a11.set_title(os.path.basename(self.tellspec))
		a11.plot(self.tellspec_data[:,0], self.tellspec_data[:,1])

		a12 = f1.add_subplot(plot_y,plot_x,2)
		a12.plot(self.tellspec_A0V_data[:,0], self.tellspec_A0V_data[:,1])
		a12.set_title(os.path.basename(self.tellspec_A0V))

		a13 = f1.add_subplot(plot_y,plot_x,3)
		a13.plot(self.tellspec_adj_data[:,0], self.tellspec_adj_data[:,1])
		a13.set_title(os.path.basename(self.tellspec_adj))

		a14 = f1.add_subplot(plot_y,plot_x,4)
		a14.set_title('reciprocal of tellspec')
		a14_1 = a14.twinx()
		a14_1.plot(trans_spec[trans_idxs,0][0], trans_spec[trans_idxs,1][0], color='tab:orange', lw=1,zorder=1)
		a14_1.set_ylim([0,1])
		a14.plot(self.recip_tellspec_data[:,0], self.recip_tellspec_data[:,1], color='tab:blue', lw=1, zorder=0)
		a14.plot(real_spec[:,0], real_spec[:,1], color='tab:red', lw=1, zorder=2)
		a14.plot(self.recip_tellspec_data[:,0], polyline, lw=1, color='tab:green', zorder=3)

		a15 = f1.add_subplot(plot_y,plot_x,5)
		a15.plot(flattened_spec[:,0], flattened_spec[:,1], lw=1, color='tab:purple')
		a15.set_title('flattened reciprocal of tellspec')
		
		a16 = f1.add_subplot(plot_y,plot_x,6)
		a16.plot(hfilter[:,0], hfilter[:,1], lw=1, color='tab:blue')
		a16.set_title('H-filter')
		plt.show()
		########### end plotting ##################
		return(self.odepth_per_airmass, self.polyfit)
	
	def fit_ricker_wavelet(self):
		self.seeing_function = 'RICKER WAVELET'
		z = self.hdul[0].data[300,:,:]
		y,x = np.mgrid[:z.shape[0],:z.shape[1]]
		
		ixy = np.unravel_index(np.nanargmax(z),z.shape)
		iamp = z[ixy]
		
		def ricker_wavelet2D(pos, a0, x0,y0,s0):
			f_itr = models.RickerWavelet2D(a0,x0,y0,s0)
			return(f_itr(*pos).ravel())
		
		a0,x0,y0,s0 = (iamp, ixy[1], ixy[0], 1)
		popt, pcov = spo.curve_fit(ricker_wavelet2D, (x,y), np.nan_to_num(z, copy=True).ravel(), p0=(a0,x0,y0,s0))
		self.set_seeing_pixels(popt[3]) # set seeing as sigma of mexican hat model	
	
		f_fitted = models.RickerWavelet2D(*popt)
		residual = np.fabs(z-f_fitted(x,y))
		print(f'residual sum {np.nansum(residual)}')
		
		# Plotting to make sure we know what is happening
		afigsize = 16
		f1 = plt.figure(figsize=[_x/2.54 for _x in (3*afigsize,afigsize)])
		
		a11 = f1.add_subplot(1,3,1)
		im11 = a11.imshow(z, origin='lower')
		f1.colorbar(im11,ax=a11)
		a11.set_title('Data')
		
		a12 = f1.add_subplot(1,3,2)
		im12 = a12.imshow(f_fitted(x,y), origin='lower')
		f1.colorbar(im12,ax=a12)
		a12.set_title('Model')
		
		a13 = f1.add_subplot(1,3,3)
		im13 = a13.imshow(residual, origin='lower')
		f1.colorbar(im13,ax=a13)
		a13.set_title('Residual (data-model)')
		
		plt.show()
		# end plotting		
				
		self.popt = popt
		self.pcov = pcov
		return(popt, pcov)
		

	def fit_airy_disk(self):
		import scipy.optimize as spo
		self.seeing_function='AIRY DISK'
		z = self.hdul[0].data[300,:,:]
		y, x = np.mgrid[:z.shape[0], :z.shape[1]] # swap x, y arrays as fits uses z,y,x ordering
		ixy = np.unravel_index(np.nanargmax(z), z.shape)
		iamp = z[ixy]
		irad_to_first_zero = 1

		print(iamp, ixy[1], ixy[0], irad_to_first_zero)
		amp0, x0, y0, r0 = (iamp, ixy[1], ixy[0], irad_to_first_zero)
		print(textwrap.dedent(f"""
			Airy Disk Initial Parameters
				Amplitude {amp0}
				x-pos {x0}
				y-pos {y0}
				radius to first zero {r0}
		"""))
		# np.nan_to_num = treat NANs as zeros
		popt, pcov = spo.curve_fit(airy2D, (x,y), np.nan_to_num(z, copy=True).ravel(), p0=(amp0, x0, y0, r0))
		self.set_seeing_pixels(popt[3]) # define seeing as distance from peak to first zero CHANGE FROM PX TO ARCSEC
		f_fitted = models.AiryDisk2D(*popt)
		print(textwrap.dedent(f"""
			Airy Disk Fitted Parameters
				Amplitude {popt[0]}
				x-pos {popt[1]}
				y-pos {popt[2]}
				radius to first zero {popt[3]}
		"""))
		residual = np.fabs(z - f_fitted(x,y))
		print(f'residual sum {np.nansum(residual)}')

		# Plotting to make sure we know what is happening
		"""
		afigsize = 16
		f1 = plt.figure(figsize=[_x/2.54 for _x in (3*afigsize,afigsize)])

		a11 = f1.add_subplot(1,3,1)
		im11 = a11.imshow(z, origin='lower')
		f1.colorbar(im11,ax=a11)
		a11.set_title('Data')

		a12 = f1.add_subplot(1,3,2)
		im12 = a12.imshow(f_fitted(x,y), origin='lower')
		f1.colorbar(im12,ax=a12)
		a12.set_title('Model')

		a13 = f1.add_subplot(1,3,3)
		im13 = a13.imshow(residual, origin='lower')
		f1.colorbar(im13,ax=a13)
		a13.set_title('Residual (data-model)')

		plt.show()
		"""
		# end plotting		
				
		self.popt = popt
		self.pcov = pcov
		return(popt, pcov)

	def fit_gaussian2D(self):
		
		self.seeing_function = 'GAUSSIAN'
		ixy = np.unravel_index(np.nanargmax(self.hdul[0].data[300,:,:], axis=None), self.hdul[0].data.shape[1:])
		iamp = self.hdul[0].data[300,ixy[0],ixy[1]]
		isx = 1
		isy = 1
		it = 0
		io = 0
		print(textwrap.dedent(f"""
			Gaussian2D Initial Parameters
				amplitude {iamp}
				x-pos {ixy[0]}
				y-pos {ixy[1]}
				sigma_x {isx}
				sigma_y {isy}
				theta {it}
				offset {io}
		"""))
		popt, pcov = fit_gaussian(self.hdul[0].data[300,:,:], iamp, ixy[0], ixy[1], isx, isy, it, io)
		famp, fmx, fmy, fsx, fsy, ft, fo = popt
		self.set_seeing_pixels(np.sqrt(fsx*fsx + fsy*fsy)*3.0) # define seeing as 3 mean standard devaitions away
		p = gaussian_as(self.hdul[0].data.shape[1:], famp, fmx, fmy, fsx, fsy, ft, fo)
		#print(famp, fmx, fmy, fsx, fsy, ft, fo)
		print(textwrap.dedent(f"""
			Gaussian2D Fitted Parameters
				amplitude {famp}
				x-pos {fmx}
				y-pos {fmy}
				sigma_x {fsx}
				sigma_y {fsy}
				theta {ft}
				offset {fo}
		"""))
		residual = np.fabs(self.hdul[0].data[300,:,:] - p)
		print(f'residual sum {np.nansum(residual)}')
		
		# Plotting to make sure we know what is happening
		afigsize = 16
		f1 = plt.figure(figsize=[_x/2.54 for _x in (3*afigsize,afigsize)])

		a11 = f1.add_subplot(1,3,1)
		im11 = a11.imshow(self.hdul[0].data[300, :,:], origin='lower')
		f1.colorbar(im11,ax=a11)
		a11.set_title('Data')

		a12 = f1.add_subplot(1,3,2)
		im12 = a12.imshow(p, origin='lower')
		f1.colorbar(im12,ax=a12)
		a12.set_title('Model')

		a13 = f1.add_subplot(1,3,3)
		im13 = a13.imshow(residual, origin='lower')
		f1.colorbar(im13,ax=a13)
		a13.set_title('Residual (data-model)')

		plt.show()
		# end plotting		
		self.popt = popt
		self.pcov = pcov
		return(popt,pcov)

	def __repr__(self):
		return(textwrap.dedent(f"""
			<##########################>
			"Standard" object at {id(self)}
			steps {self.steps}
			datacube {self.datacube}
			xtellcor A0V file {self.tellspec_A0V}
			xtellcor adj file {self.tellspec_adj}
			xtellcor tellspec file {self.tellspec}
			origin observatory {self.origin_observatory}
			object {self.object}
			observation date {self.observation_date}
			avg_airmass {self.airmass}
			exposure time {self.exposure_time} s
			H band flux {self.h_flux} mag
			B band flux {self.b_flux} mag
			V band flux {self.v_flux} mag
			radial velocity {self.radial_velocity} km/s
			right ascention {self.ra} deg
			declination {self.dec} deg
			proper motion right ascention {self.proper_motion_ra} mas/yr
			proper motion declination {self.proper_motion_dec} mas/yr
			seeing {self.seeing} arcsec
			px_star_center {self.px_star_center}
			py_star_center {self.py_star_center}
			p_star_width {self.p_star_width}
			px_scale {self.px_scale} arcsec
			py_scale {self.py_scale} arcsec
			wavgrid {self.wavgrid} um
			<-------------------------->
			"""))



def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	for tc in args['target_cubes']:
		ut.pINFO('Operating on cube {}'.format(tc))
		targ = Target(tc)
		
		calibs_file = os.path.join(os.path.dirname(tc),args['rel_calibrators_file'])
		tellspec_folder = os.path.join(os.path.dirname(tc), args['rel_tellspec_folder'])
		with open(calibs_file, 'r') as f:
			stds = [Standard(std_fits_path.strip(), tellspec_folder) for std_fits_path in f]
		query_simbad(stds)
		for std in stds:
			std.get_data_from_simbad_query()

		print(targ)
		norm_targs = [copy.deepcopy(targ) for std in stds]
		for std, norm_targ in zip(stds,norm_targs):
			#std.fit_gaussian2D()
			#popt, pcov = std.fit_airy_disk()
			popt, pcov = std.fit_gaussian2D()
			std.set_star_details(popt[1],popt[2],popt[3])
			std.subtract_spectral_background(std.get_boring_region_spectral_background())
			std.H_filter_to_datacube_wavelengths()
			spfluxangexpfac = std.get_spfluxangexpfac()
			norm_targ.counts_to_spradiance(spfluxangexpfac) #  this applies the radiometric calibration in sub_radcal_sinf, I think that the rest of the file that calculates reflectance is just for comparing to reference data, it doesn't seem to be used anywhere else
			optical_depth_per_airmass, polyfit = std.read_tellspec_data()
			norm_targ.apply_tellspec(optical_depth_per_airmass, polyfit)
			print(f'spfluxangexpfac {spfluxangexpfac} W cm-2 um-2 ster-1 counts-1 s')
			#std.fit_ricker_wavelet()
			norm_targ.project_lat_lon()
			print(std)
			norm_targ.find_disk_fraction(std)
			norm_targ.find_stat_err()
			norm_targ.assemble_fits(std, outfname=f'./analysis/obj_{norm_targ.object.upper()}_cal_{std.object.upper()}.fits', overwrite=True)
	return

def airy2D(pos,amp0,x0,y0,r0):
	# create a function that will create an airy disk with desired attributes
	# with the position *pos, and ravel (flatten) it to use it with scipy.optimize.curve_fit
	f_itr = models.AiryDisk2D(amp0, x0, y0, r0)
	return(f_itr(*pos).ravel())

def gaussian_as(shape, amp=1, mx=0, my=0, sx=1, sy=1, t=0, o=0):
	x = np.linspace(0, shape[0]-1, shape[0])
	y = np.linspace(0, shape[1]-1, shape[1])
	x,y = np.meshgrid(x,y)
	return(gaussian_2d((x,y), amp, mx, my, sx, sy, t, o).reshape(*shape))

def fit_gaussian(data, amp=1, mx=0, my=0, sx=1, sy=1, t=0, o=0):
	import scipy.optimize as spo
	import numpy.ma as npm
	x = np.linspace(0, data.shape[0]-1, data.shape[0])
	y = np.linspace(0, data.shape[0]-1, data.shape[0])
	x,y = np.meshgrid(x,y)
	print(x)
	popt, pcov = spo.curve_fit(gaussian_2d, (x,y), np.nan_to_num(data, copy=True).ravel(), p0=(amp, mx, my, sx, sy, t, o))
	return(popt, pcov)

def gaussian_2d(pos,amp, mx, my, s11, s22, t, o):
	"""
	pos - position to evaluate gaussian at (x,y)
	amp - amplitude of gaussian
	mx  - mean in x direction
	my  - mean in y direction
	s11 - standard deviation in x direction
	s22 - standard deviation in y direction
	t   - angle of rotation of the gaussian (theta)
	o   - offset to base of gaussian (offset)
	See https://en.wikipedia.org/wiki/Gaussian_function
	"""
	x, y = pos
	a = (np.cos(t)**2)/(2*(s11**2)) + (np.sin(t)**2)/(2*(s22**2))
	b = -np.sin(2*t)/(4*(s11**2)) + np.sin(2*t)/(4*(s22**2))
	c = (np.sin(t)**2)/(2*(s11**2)) + (np.cos(t)**2)/(2*(s22**2))
	g = amp*np.exp(-(a*(x-mx)**2 + 2*b*(x-mx)*(y-my) + c*(y-my)**2)) + o
	return(g.ravel())
			

def projpos_ellipse(Re, Rp, eps, eoff, poff):
	"""
	Finds the longitude, latitude, and zenith angles of the intercept point of an ellipse and a
	line fired offset from an observers line of sight.

	ARGUMENTS:
		Re
			<float> Equatorial radius of ellipse
		Rp
			<float> Polar radius of ellipse
		eps
			<float> Sub-observer (planetocentric) latitude
		eoff
			<float> equatorial offset of beam (arcsec)
		poff
			<float> polar offset of beam (arcsec)

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
	deg2rad = np.pi/180
	# using line in symmetric form (x-x0)/alpha = (y-y0)/beta = (z-z0)/gamma
	# line goes through the point (x0,y0,z0)
	# line has the direction vector d = (alpha, beta, gamma), d should be a unit vector

	x0 = 0.0 # x-point on the beam-line (zero x-coord means the point the line goes through is in the yz-plane
	y0 = eoff # y-point on the beam-line, equatorial offset of beam (arcsec)
	z0 = poff/np.cos(eps*deg2rad) # z-point on the beam-line

	alpha = np.sin(np.pi/2.0 - eps*deg2rad) # x-dimension of unit direction vector
	beta = 0.0 # y-dimension of unit direction vector, no change in y-direction so y = eoff always, so the line is contained within the xz-plane at y=eoff
	gamma = np.cos(np.pi/2 - eps*deg2rad)# z-dimension of unit direction vector

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
		xlat = 90 - theta/deg2rad
		# planetocentric latiude is the angle between the equatorial plane and a line joining
		# the ICOpoint to the origin

		# the component of orign->ICOpoint in the xy-plane is r*sin(theta), so the angle between the
		# x-axis and the ICOpoint is arccos(x1/r*sin(theta)) (c.f. spherical polars)
		cphi = x1/(r*np.sin(theta))
		cphi = np.clip(cphi, -1, 1)
		phi = np.arccos(cphi)
		if y1 < 0:
			phi*=-1
		xlon = phi/deg2rad

		v1 = np.array([x1/r, y1/r, z1/r]) # unit vector of ICOpoint
		v2 = np.array([alpha, beta, gamma]) # unit vector defining direction of LOS

		zen = np.arccos(np.dot(v1,v2))/deg2rad
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
		a    	real	ellipsoid semi-major axis
		b    	real	ellipsoid semi-minor axis
	      alpha	real	line x-gradient
		beta 	real	line y-gradient
	  gamma	real	line z-gradient
	  x0	real	line x-intercept
	  y0	real	line y-intercept
	  z0	real	line z-intercept
	
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



def ellipse_top_hat_as(shape, x0, y0, a, b, t):
	x, y = np.mgrid[:shape[0], :shape[1]]
	return(ellipse_top_hat_2D((x,y), x0, y0, a, b, t).reshape(shape))
		
def ellipse_top_hat_2D(pos, x0, y0, a, b, t):
	"""
	pos - x, y position to evaluate ellipse at
	x0 - x position of center of ellipse
	y0 - y position of center of ellipse
	a - semi-major axis
	b - semi-minor axis
	t - angle of rotation
	see https://en.wikipedia.org/wiki/Ellipse#General_ellipse
	"""
	x, y = pos
	A = a**2*np.sin(t) + b**2*np.cos(t)
	B = 2*(b**2-a**2)*np.sin(t)*np.cos(t)
	C = a**2*np.cos(t) + b**2*np.sin(t)
	D = -2*A*x0 - B*y0
	E = -B*x0 - 2*C*y0
	F = A*x0**2 + B*x0*y0 + C*y0**2 - a**2*b**2
	
	bg = np.zeros_like(x)
	ellipse_idxs = np.where(A*x**2 + B*x*y + C*y**2 + D*x + E*y < - F)
	bg[ellipse_idxs] = 1.0
	return(bg)



def datacube_wavelength_grid(hdu):
	i = 3
	naxis_i = hdu.header['NAXIS{}'.format(i)]
	# assume spectral axis is 3rd axis
	p = np.linspace(1, naxis_i, num=naxis_i, dtype=int) # do FITS files use 0 or 1 indexing? I think 1.
	#print(p)
	#assume 1d
	ltmi_i = float(hdu.header.get('LTM{}_{}'.format(i,i),1))
	ltvi = float(hdu.header.get('LTV{}'.format(i), 0))
	l = ltmi_i*p + ltvi
	#print(l)

	crpix_i = int(hdu.header['CRPIX{}'.format(i)])
	crval_i = float(hdu.header['CRVAL{}'.format(i)])
	cdelt_i = float(hdu.header['CDELT{}'.format(i)])
	cdi_i = float(hdu.header.get('CD{}_{}'.format(i,i),cdelt_i))
	#print(crpix_i, crval_i, cdelt_i, cdi_i)
	#w = np.zeros([naxis1])
	w = crval_i + cdi_i*(l-crpix_i)
	return(w)

def sign(x):
	if x < 0:
		return(-1)
	return(1)

def sexagesimal_str_to_deg(astr):
	# assume of form XX YY ZZ.ZZZ
	degs, mins, secs = map(float,astr.split())
	print(degs, mins, secs)
	sgn = sign(degs)
	degrees = sgn*(sgn*degs + mins/60.0 + secs/3600.0)
	return(degrees)

def hour_angle_str_to_deg(astr):
	# assome of form XX YY ZZ.ZZZ
	hrs, mins, secs = map(float,astr.split())
	print(hrs, mins, secs)
	sgn = sign(hrs)
	print(sgn)
	degrees = sgn*(sgn*15*hrs + 15/60*mins + 15/3600*secs)
	return(degrees)

def query_simbad(list_of_standards):
	Simbad.add_votable_fields('typed_id')
	Simbad.remove_votable_fields('coordinates', 'main_id')
	Simbad.add_votable_fields(	'coordinates', 'propermotions', 'rv_value', 
								'fluxdata(H)', 'fluxdata(V)','fluxdata(B)')
	sbd_obj = Simbad.query_objects([std.object for std in list_of_standards])
	for i, std in enumerate(list_of_standards):
		std.sbd_data = sbd_obj[i]
	print(sbd_obj)
	print(sbd_obj['PMRA','PMDEC'])
	#sbd_obj.show_in_browser()


def jplh_query_object_at_instant(observer, target, date):
	tgt = fitscube.process.cfg.object_ids[target]
	obs_code = fitscube.process.cfg.observatory_codes[observer]
	t1 = Time(date, format='fits')
	#print(t1.mjd)
	return(Horizons(id=tgt[0], location=obs_code, epochs=[t1.jd], id_type=tgt[1]))

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

	parser.add_argument('target_cubes', type=str, nargs='+', help='data cubes to operate on', default=[])
	parser.add_argument('--rel_calibrators_file', type=str, help='Relative location (w.r.t target_cube) that points to a file that contains a list of fits files of calibration sources (standard stars)', default='./calibrators.txt')
	#parser.add_argument('--optional', help='A description of the optional argument')

	parser.add_argument('--rel_tellspec_folder', type=str, help='Relative location (w.r.t target_cube) that points to a folder that contains telluric spectrum files (created by xtellcor_general) that are named in the format "xtellcor_<STD_STAR_NAME_CAPS>_{A0V,adj,tellspec}.dat', default='./analysis/telluric_correction')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface

	# process arguments
	parsed_args['target_cubes'] = [os.path.abspath(tc) for tc in parsed_args['target_cubes']]
		

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
