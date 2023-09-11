#!/usr/bin/env python3
"""
This file contains read-only data to be used with the nemesis modules in it's containing folder
"""
import numpy as np

nir_bands={	'J':{'min':1.1, 'max':1.4, 'unit':'micrometer'},
			'H':{'min':1.5, 'max':1.8, 'unit':'micrometer'},
			'K':{'min':2.0, 'max':2.4, 'unit':'micrometer'},
			'L':{'min':3.0, 'max':4.0, 'unit':'micrometer'},
			'M':{'min':4.6, 'max':5.0, 'unit':'micrometer'},
			'N':{'min':7.5, 'max':14.5, 'unit':'micrometer'},
			'Q':{'min':17, 'max':25, 'unit':'micrometer'},
			'Z':{'min':28, 'max':40, 'unit':'micrometer'}
}
# Here we detail functions used to interpret varident representations (the 3rd of the varident numbers)
# we will then create a dictionary that holds all of the representation functions so we can reference them
# by the third index of varident
# To these funcitons, We will pass the varparams associated with the varident, the section of the state vector,
# the error on that section of the state vector, and the height, pressure, and wavelength grids from <runname>.ref
# and <runname>.xsc.
# We may need to expand this as time goes on, so use KEYWORD arguments.
# We will return a dictionary containing the height OR pressure, the evaluated quantity, the error on that quantity,
# and the x-axis label (the independent variable, e.g. pressure)
# THIS IS REALLY ANNOYING
"""
'varident' contains too much data for it's own good. It has magic numbers that
mean certain specific things (everything +ve that is not a gasid is a magic 
number and has it's own specific interpretation). What I SHOULD do is make a
load of functions that turn the 'varident' and it's associated bit of the 
state vector into a 'continuous' function that I can plot.

E.g. 
	varident[2]==1 represents a quantity as a deep value up to a 'knee 
	pressure' then drops off as a fractional scale height. If I plot those
	three values, then it makes little sense, what I should do is plot the 
	output of a function that computes the quantiy according to the formula

		Q(p) = q_deep if p<p_knee
			 = q_deep*exp(-(p-p_knee)/fsh) otherwise

However, this is going to be REALLY ANNOYING. I would need to write
functions for each of the 29 + magic numbers possible representations, and
return those here, then in the plotting function, run the state vector
components through the function and plot it's output.

I think I will try to make it work ONLY for the profile's I am using RIGHT
NOW. I can add more later, but I'll go crazy if I try to make a general 
solution that covers every case. There are 29 representations, and 5 categories
of quantities, plus 16 special cases. So that's 29*5+16=161 possible
combinations.

NOTE: only include 'paramters' dictionary in output when the representation is a
parameterised quantity. I.e. a specified profile doesn't have parameters, ane therefore
should not have a 'parameters' dictionary in output

See ".../<radtran_repo>/nemesis/subprofretg.f" for the for the fortran implementation
of these profiles.
"""

def var_repr_specified_profile(varparams=None, 
								state_vector=None, 
								state_vector_error=None, 
								height_grid=None, 
								pressure_grid=None,
								wavelength_grid=None,
								runname=None):
	"""
	profile is continuous and is specified in a file, we don't have to put it through a function, just return
	"""
	return({'independent_var':pressure_grid,
			'dependent_var':state_vector,
			'dependent_err':state_vector_error,
			'independent_label':'Pressure',
			'parameters':None})

def var_repr_knee_pressure_exp_decay(varparams=None,
									state_vector=None,
									state_vector_error=None,
									height_grid=None,
									pressure_grid=None,
									wavelength_grid=None,
									runname=None):
	"""
	Profile is represented as a 'deep' value up to a certain 'knee' pressure above which there is
	exponetial decay controlled by the 'fractional scale height'.
	"""

	print(state_vector)
	print(state_vector_error)
	
	kp, kp_err = (varparams[0], varparams[1]) # knee pressure, for some reason this is in 'varparams' not the state vector
	dv, dv_err = (state_vector[0], state_vector_error[0]) # deep value
	fsh, fsh_err = (state_vector[1], state_vector_error[1]) # fractional scale height

	pdict = {'knee_pressure':(kp,kp_err),'deep_value':(dv,dv_err),'fractional_scale_height':(fsh,fsh_err)}

	# dv up to a kp, the exponetial decay with height according to fsh
	def make_profile(kp, dv, fsh):
		profile = np.zeros_like(pressure_grid)
		profile[pressure_grid < kp] = dv
		height_at_kp_idx = np.argmin(np.abs(pressure_grid-kp))
		height_grid_exp_decay = np.exp(-(height_grid - height_grid[height_at_kp_idx])/fsh)
		profile[pressure_grid >= kp] = dv*height_grid_exp_decay[pressure_grid>kp]
		return(profile)

	best_profile = make_profile(kp, dv, fsh)
	min_profile = make_profile(kp-kp_err, dv-dv_err, fsh-fsh_err)
	max_profile = make_profile(kp+kp_err, dv+dv_err, fsh+fsh_err)
	err_profile = max_profile - min_profile
	return({'independent_var':pressure_grid,
			'independent_label':'Pressure',
			'dependent_var': best_profile,
			'dependent_err': err_profile,
			'parameters':pdict})
	


def var_repr_exp_decay_above_base_pressure(varparams=None,
										state_vector=None,
										state_vector_error=None,
										height_grid=None,
										pressure_grid=None,
										wavelength_grid=None,
										runname=None):
	"""
	profile is a cloud represented by a base pressure, opacity, and a fractional scale height
	opacity is zero above the base pressure, with an exponetial drop-off below the base pressure

	numbers are not logs (but opacity and fractional scale height are loged internally in NEMESIS so 
	these parameters can't be negative)
	"""

	print(state_vector)
	print(state_vector_error)

	bp, bp_err = (state_vector[0], state_vector_error[1]) # base pressure
	opa, opa_err = (state_vector[1], state_vector_error[1]) # opacity
	fsh, fsh_err = (state_vector[2], state_vector_error[2]) # fractional scale height
	
	pdict= {'base_pressure':(bp,bp_err), 'opacity':(opa,opa_err), 'fractional_scale_height':(fsh,fsh_err)}
	# zero up to a base height, then exponential decay
	opa_fraction = np.exp((pressure_grid-bp)/fsh)
	opa_fraction[np.where(pressure_grid>bp)] = 0.0
	cloud_opacity = (opa)*opa_fraction
	co_opa_d = opa
	co_bp_d = (opa/fsh)*opa_fraction
	co_fsh_d = opa*(bp-pressure_grid)*opa_fraction
	co_err = np.sqrt( (opa_err*co_opa_d)**2 + (bp_err*co_bp_d)**2 + (fsh_err*co_fsh_d)**2)


	return({'independent_var':pressure_grid,
			'independent_label':'Pressure',
			'dependent_var':cloud_opacity,
			'dependent_err':co_err,
			'parameters':pdict})

def var_repr_exp_decay_above_base_height(varparams=None,
										state_vector=None,
										state_vector_error=None,
										height_grid=None,
										pressure_grid=None,
										wavelength_grid=None,
										runname=None):
	"""
	profile is a cloud represented by a base altitide, opacity, and a fractional scale height
	opacity is zero up to the base altitude, with an exponetial drop-off above the base altitude

	numbers are not logs (but opacity and fractional scale height are loged internally in NEMESIS so 
	these parameters can't be negative)
	"""

	print(state_vector)
	print(state_vector_error)

	ba, ba_err = (state_vector[0], state_vector_error[1]) # base altitide
	opa, opa_err = (state_vector[1], state_vector_error[1]) # opacity
	fsh, fsh_err = (state_vector[2], state_vector_error[2]) # fractional scale height

	pdict = {'base_altitude':(ba, ba_err), 'opacity':(opa,opa_err), 'fractional_scale_height':(fsh,fsh_err)}	
	# zero up to a base height, then exponential decay
	opa_fraction = np.exp(-(height_grid-ba)/fsh)
	opa_fraction[np.where(height_grid<ba)] = 0.0
	cloud_opacity = (opa)*opa_fraction
	co_opa_d = opa
	co_ba_d = (opa/fsh)*opa_fraction
	co_fsh_d = opa*(ba-height_grid)*opa_fraction
	co_err = np.sqrt( (opa_err*co_opa_d)**2 + (ba_err*co_ba_d)**2 + (fsh_err*co_fsh_d)**2)


	return({'independent_var':height_grid,
			'independent_label':'Height',
			'dependent_var':cloud_opacity,
			'dependent_err':co_err,
			'parameters':pdict})


def var_repr_exp_decay_above_base_pressure_adj(varparams=None,
										state_vector=None,
										state_vector_error=None,
										height_grid=None,
										pressure_grid=None,
										wavelength_grid=None,
										runname=None):
	"""
	profile is a cloud represented by a base pressure, opacity, and a fractional scale height
	opacity is zero above the base pressure, with an exponetial drop-off below the base pressure.
	Above the base pressure there is an exponential drop-off with a scale height of 1 km, this is
	to help NEMESIS find a solution when varying the base pressure by a small amount.

	numbers are not logs (but opacity and fractional scale height are loged internally in NEMESIS so 
	these parameters can't be negative)

	Fractional scale height = Cloud scale height / pressure scale height

	If p(z) = P0 exp(-z/Hp)
	and n(z) = n0 exp(-z/Hc)
	fsh = Hc/Hp
	and n(z) = n0 (p(z)/p0)^(1/fsh)

	"""
	print(f'INFO: In "nemesis.cfg.var_repr_exp_decay_above_base_pressure_adj()"')
	print(f'INFO: opacity below the base pressure have been pinned to the smallest opacity above the base pressure, this is to make plotting easier')
	#print(state_vector)
	#print(state_vector_error)
	
	# the parameter order is not necessarily the same in the *.apr file and the *.mre file


	bp, bp_err = (state_vector[2], state_vector_error[2]) # base pressure
	opa, opa_err = (state_vector[0], state_vector_error[0]) # opacity
	fsh, fsh_err = (state_vector[1], state_vector_error[1]) # fractional scale height

	print(f'base pressure {bp}')
	print(f'opacity/abundance {opa}')
	print(f'fractional scale height/pressure {fsh}')

	pdict = {	'base_pressure':(bp, bp_err), 
				'opacity':(opa, opa_err), 
				'fractional_scale_height':(fsh,fsh_err)
			}

	#print('base pressure', bp, bp_err)
	#print('opacity', opa, opa_err)
	#print('fractional scale height', fsh, fsh_err)
	p_offset = pressure_grid - bp
	p_idx = np.nanargmin(np.abs(p_offset))

	fp = (bp - pressure_grid[p_idx-1])/(pressure_grid[p_idx] - pressure_grid[p_idx-1])
	print(fp)

	bh = fp*(height_grid[p_idx] - height_grid[p_idx-1])+height_grid[p_idx-1]

	h_offset = height_grid - bh

	print('p_offset', p_offset)
	print('p_idx', p_idx)
	print('base height', bh)
	print('height offset', h_offset)

	#n = opa*np.exp(-h_offset/fsh)
	#n[h_offset<0] = opa*np.exp(h_offset[h_offset<0]/1)

	n = opa*np.exp(p_offset/fsh)
	n[p_offset>0] = opa*np.exp(-p_offset[p_offset>0]/1)

	print('number density', n)

	cloud_opacity = n

	"""	
	# exponential increase up to base pressure, then quick exponential decay over base pressure
	pressure_offset = pressure_grid - bp
	#print('pressure_offset', pressure_offset)
	
	#opa_fraction = np.exp((pressure_offset)/fsh)


	opa_fraction = opa*(pressure_grid/bp)**(1/fsh)

	#print('opa_fraction', opa_fraction)
	# find index that the base pressure corresponds to
	pressure_idx = np.nanargmin(np.abs(pressure_offset))

	#opa_fraction = (-height_grid/height_grid[pressure_idx])**(1/fsh)

	#print('pressure_idx', pressure_idx)
	# create fast decay below the height the the base pressure corresponds to
	opa_fraction_adj = np.exp((height_grid - height_grid[pressure_idx])/1.0)
	#print('opa_fraction_adj', opa_fraction_adj)
	#print(pressure_offset.shape, opa_fraction_adj.shape, opa_fraction.shape)
	opa_fraction[pressure_offset > 0] = opa_fraction_adj[pressure_offset > 0]
	#print('opa_fraction', opa_fraction)

	# due to the sharp cutoff above the pressure offset, graphs display this wierdly
	# therefore will pin the high-pressure values to the minimum of the low-pressure
	# values to increase readability
	min_low_pressure_opa = np.min(opa_fraction[pressure_offset <= 0])
	min_low_pressure_opa = 1E-6*np.max(opa_fraction) # DEBUGGING to make scale smaller
	opa_fraction = np.where(opa_fraction < min_low_pressure_opa, min_low_pressure_opa, opa_fraction) #pressure_offset > 0] = min_low_pressure_opa

	cloud_opacity = (opa)*opa_fraction
	co_opa_d = opa
	co_bp_d = (opa/fsh)*opa_fraction
	co_fsh_d = opa*(bp-pressure_grid)*opa_fraction
	co_err = np.sqrt( (opa_err*co_opa_d)**2 + (bp_err*co_bp_d)**2 + (fsh_err*co_fsh_d)**2)
	"""

	import nemesis.read
	prf_dat = nemesis.read.prf(runname)
	
	# density in g cm^-3
	# R = 8.3145 (universal gas constant)
	density = prf_dat['molwt']*pressure_grid*100/(8.3145*prf_dat['temp'])

	num_per_gram = 1E-6*cloud_opacity/density

	#DEBUGGING
	co_err = np.zeros_like(num_per_gram)	

	return({'independent_var':pressure_grid,
			'independent_label':'Pressure',
			'dependent_var':num_per_gram,
			'dependent_err':co_err,
			'parameters':pdict})



def var_repr_imaginary_refractive_index(varparams=None,
										state_vector=None,
										state_vector_error=None,
										height_grid=None,
										pressure_grid=None,
										wavelength_grid=None,
										runname=None):
	"""
	profile is delcared in a file, don't have to put it through a function
	but there's some wierdness with the state vector being 2 bigger than the
	wavelength grid, not sure why. For now I am just cutting off the extra values.
	"""
	return({'independent_var':wavelength_grid,
			'independent_label':'Wavelength',
			'dependent_var':state_vector[2:],
			'dependent_err':state_vector_error[2:],
			'parameters':None})

def var_repr_not_implemented(*args, **kwargs):
	raise NotImplementedError('The function for whichever representation code you are using has not been implemented yet. See "nemesis/cfg.py" for examples of implementing functions which turn the representation codes into profiles.')
	return()

# Assemble function handles into a dictionary indexed by the 3rd component of varident
# this must go below all of the 'var_repr*' functions, otherwise they don't exist yet :-P
var_repr_funcs = 	{	0:var_repr_specified_profile,
						1:var_repr_knee_pressure_exp_decay,
						8:var_repr_exp_decay_above_base_pressure,
						9:var_repr_exp_decay_above_base_height,
						32:var_repr_exp_decay_above_base_pressure_adj,
						444:var_repr_imaginary_refractive_index
			}
	















