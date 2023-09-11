#!/usr/bin/env python3
import utils as ut
import nemesis.common as nc

def spx(data_dict, runname='output', append=False):
	"""Write a *.spx file"""
	nc._ensure_dict_contains(('fwhm', 'latitude', 'longitude', 'ngeom', 
							'nconvs', 'navs', 'fov_averaging_record', 
							'spec_record'), 
							data_dict)
	mode = 'w'
	if append:
		mode = 'a'
	with open(nc._ensure_ext(runname,'.spx'), mode) as f:
		f.write(nc.str_collect([data_dict['fwhm'], data_dict['latitude'], 
							data_dict['longitude'], data_dict['ngeom']], 
							long_format=['{: 0.4e}']*3+['{}']))
		for i in range(data_dict['ngeom']):
			f.write('{}\n'.format(data_dict['nconvs'][i]))
			f.write('{}\n'.format(data_dict['navs'][i]))
			for j in range(data_dict['navs'][i]):
				f.write(nc.str_collect(data_dict['fov_averaging_record'][i][j]))
			for j in range(data_dict['nconvs'][i]):
				f.write(nc.str_collect(data_dict['spec_record'][i][j]))
	return

def forward_model_error(data_dict, filename='forwardnoise.dat'):
	"""
	Write forward model error data to <filename>

	ARGUMENTS:
		data_dict
			A dictionary with the following keys
				fme_n
					<int> Number of (wavelength, error) pairs to write
				fme_we [fme_n,2]
					<float> Array of (wavelength, error) pairs
		filename
			Name of the file to write to

	RETURN:
		None

	EXAMPLE:
		
	"""
	nc._ensure_dict_contains(('fme_n','fme_we'), data_dict)
	with open(filename, 'w') as f:
		f.write('{}\n'.format(data_dict['fme_n']))
		for i in range(data_dict['fme_n']):
			f.write('{} {}\n'.format(data_dict['fme_we'][i,0], data_dict['fme_we'][i,1]))

	return

def set(data_dict, runname='output'):
	"""
	Write a *.set file for a run. Used to define quadrature points for zenith angles
	"""
	nc._ensure_dict_contains(('n_zens', 'quad_points', 'quad_weights', 'n_fourier_components', 'n_azi_angles',
								'sunlight_flag', 'solar_dist', 'lower_boundary_flag', 'ground_albedo', 'surf_temp',
								'base_altitude', 'n_atm_layers', 'layer_type_flag', 'layer_int_flag'), data_dict)

	mode = 'w'
	asterix_line = '*********************************************************'
	with open(nc._ensure_ext(runname, '.set'), mode) as f:
		f.write('{}\n'.format(asterix_line))
		f.write('Number of zenith angles: {}\n'.format(data_dict['n_zens']))
		for p, w in zip(data_dict['quad_points'], data_dict['quad_weights']):
			f.write('{:0.16E}    {:0.16E}\n'.format(p,w))
		f.write('Number of fourier components: {}\n'.format(data_dict['n_fourier_components']))
		f.write('Number of azimuth angles for fourier analysis: {}\n'.format(data_dict['n_azi_angles']))
		f.write('Sunlight on(1) or off(0): {}\n'.format(data_dict['sunlight_flag']))
		f.write('Distance from Sun (AU): {}\n'.format(data_dict['solar_dist']))
		f.write('Lower boundary cond. Thermal(0) or Lambert(1): {}\n'.format(data_dict['lower_boundary_flag']))
		f.write('Ground albedo: {}\n'.format(data_dict['ground_albedo']))
		f.write('Surface temperature: {}\n'.format(data_dict['surf_temp']))
		f.write('{}\n'.format(asterix_line))
		f.write('Alt. at base of bot.layer (not limb): {}\n'.format(data_dict['base_altitude']))
		f.write('Number of atm. layers: {}\n'.format(data_dict['n_atm_layers']))
		f.write('Layer type: {}\n'.format(data_dict['layer_type_flag']))
		f.write('Layer integration: {}\n'.format(data_dict['layer_int_flag']))
		f.write('{}\n'.format(asterix_line))
	return()

def continuous_profile(data_dict, filename='cloud1apr.dat'):
	"""
	Writes a *.dat file for a parameter with varident[3] = 0 (bottom of page 12 in NEMESIS manual)

	ARGUMENTS:
		data_dict
			A dictionary containing the following keys:
			npro
				<int> The number of layers in the profile (should be the same as in <runname>.ref)
			clen
				<float> The correlation length of the profile (in log(pressure))
			p [npro]
				<float> The pressure grid over which the profile is defined (should be the same as the one in <runname>.ref)
			x [npro]
				<float> The value of the profile at each pressure, i.e. the a-priori profile
			err [npro]
				<float> The associated error of the profile at each pressure, i.e. the a-priori error
		filename
			<str> File to write to
	RETURNS:
		None
	"""	
	with open(nc._ensure_ext(filename, '.dat'), 'w') as f:
		f.write(f'{data_dict["npro"]} {data_dict["clen"]}\n')
		for i in range(len(data_dict['p'])):
			f.write(f'{data_dict["p"][i]:>09.6f} {data_dict["x"][i]: 0.3E} {data_dict["err"][i]: 0.3E}\n')
	return
		
def iri(data_dict, filename='imaginary_refractive_index.dat'):
	"""
	Writes a *.dat file that holds information about an aerosol's imaginary refractive index

	ARGUMENTS:
		data_dict
			A dictionary containing the following keys:
			R0
				<float> Radius of aerosol particle
			Rerr
				<float> Error on R0
			V0
				<float> porosity? of aerosol particle
			Verr
				<float> Error on V0
			nwave
				<int> Number of wavelengths that imaginary refractive index is defined at
			clen
				<float> Correlation length
			vref
				<float> reference wavelength
			nreal_vref
				<float> real part of refractive index at reference wavelength?
			v_od_norm
				<float> Wavelength ot normalise everything to?
			wie_array [nwave, 3*float]
				<array, <float,float,float>> Array containing the following columns:
				wavelength, imaginary_refractive_index (iri), error. Set errror to < 1E-6*iri
				to treat as fixed
		filename
			<str> File to write to
	RETURNS:
		None
	"""	
	with open(nc._ensure_ext(filename, '.dat'), 'w') as f:
		f.write(f'{data_dict["R0"]} {data_dict["Rerr"]} !R0, RERR\n')
		f.write(f'{data_dict["V0"]} {data_dict["Verr"]} !V0, VERR\n')
		f.write(f'{data_dict["nwave"]} {data_dict["clen"]} !NWAVE, CLEN\n')
		f.write(f'{data_dict["vref"]} {data_dict["nreal_vref"]} !VREF, NREAL_VREF\n')
		f.write(f'{data_dict["v_od_norm"]} !V_OD_NORM\n')
		for w, i, e in data_dict["wie_array"]:
			f.write(f'{w:05.3f} {i:07.2E} {e:07.2E}\n')
	return

