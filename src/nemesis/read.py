#!/usr/bin/env python3
import sys, os
import utils as ut
import numpy as np
import nemesis.common as nc
import nemesis.exceptions

# create a dictionary to cache data from files
# maybe put a 'cache_inputs' flag in nemesis.cgf?
already_opened_dict = {}

def spx(runname, use_cache=False):
	"""
	Reads the spectrum *.spx file for a NEMESIS run. See manual Nemesis_B2.pdf for more details.

	ARGUMENTS
		runname
			<str> Name of the NEMESIS run
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.

	RETURNS
		Dictionary containing the following keys:
			fwhm
				<float> The spectral bin size, zero if using channel integrated k-tables, 
				-ve if providing a *.fil file, +ve otherwise.
			latitude
				<float> The planetocentric latitude at the center of the field of view (FOV)
			longitude
				<float> The planetocentric longitude at the center of the FOV
			ngeom
				<int> The number of different observation geometries in the file
			nconvs[ngeom]
				<int> The number of (wavenumber|wavelength, radiance, error) entries in each
				spectrum, one for each ngeom geometries.
			navs[ngeom]
				<int> The number of different spectral calculations needed to construct the
				field of view averaged spectrum. One for each ngeom geometries
			fov_averaging_record[ngeom,[navs,6]]
				[[[<float>]]] The field of view information for each of the ngeom observing
				geometries. The i^th geometry has navs[i] number of different spectral 
				calculations needed to construct the field of view averaged spectrum. And
				each of the calculations has six (6) floating point variables that describe
				it. These variables are (in order):
				flat
					<float> The integration point latitude
				flon
					<float> The integration point longitude
				sol_ang
					<float> The angle of the sun w.r.t. the observers line of sight to the
					point on the planet being observed
				emiss_ang
					<float> ?? UNSURE ABOUT THIS ?? The angle of emission (of what?) w.r.t
					the observers line of sight to the point on the planet being observed
					I think this is the zenith angle
				azi_ang
					<float> ?? UNSURE ABOUT THIS ?? The angle of emission (of what) w.r.t
					the ?planetary north? in the plane perpendicular to the observers line
					of sight
				wgeom
					<float> Weighting that is needed to recreate the field of view.
			spec_record[ngeom,[nconvs,3]]
				[[[<float>]]] The spectrum information for each of the ngeom observing
				geometries. The i^th geometry has nconvs[i] (vconv, y, err)
				entries that describe the detected spectrum.
				vconv
					Measured wavelength or wavenmber
				y
					Measured radiance
					in units: W cm-2 sr-1 um-1
				err
					Error on measurement
	"""
	import numpy as np
	
	saved_data = already_opened_dict.get('spx',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)
	# otherwise, we haven't opened this file yet so grab the data

	spec_record = []
	fov_averaging_record = []
	if not runname.endswith('.spx'):
		runname += '.spx'
	with open(runname, 'r') as f:
		fwhm, latitude, longitude, ngeom = [cast(word) for word, cast in zip(f.readline().strip().split(), (float, float, float, int))]
		nconvs = np.zeros(ngeom, dtype='int')
		navs = np.zeros(ngeom, dtype='int')
		for i in range(ngeom):
			nconvs[i] = f.readline().strip()
			navs[i] = f.readline().strip()

			llseaw_s = np.zeros((navs[i],6))
			for j in range(navs[i]):
				llseaw_s[j] = nc._distribute(llseaw_s[j], f.readline().strip().split())
			vye_s = np.zeros((nconvs[i],3))
			for j in range(nconvs[i]):
				vye_s[j] = nc._distribute(vye_s[j], f.readline().strip().split())
			fov_averaging_record.append(llseaw_s)
			spec_record.append(vye_s)
	saved_data = nc._varname2dict(('fwhm', 'latitude', 'longitude', 'ngeom', 'nconvs', 'navs', 'fov_averaging_record', 'spec_record'),locals())
	already_opened_dict['spx'] = saved_data
	return(saved_data)

def prf(runname, use_cache=False):
	"""
	Reads the <runname>.prf file for a given nemesis run. These files are 
	calculated in between interations based upon the <runname>.ref files
	and the profiles given in <runname>.apr. You can think of the <runname>.ref 
	files as defaults that are only used if you haven't set a profile
	for some parameter. You cannot use this to read in a <runname>.ref file as 
	they have a slightly different format to the <runname>.prf files.

	ARGUMENTS
		runname
			<str> Name of the nemesis run
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then 
			return the stored data any other time the file is requested to be 
			read. If FALSE, will always read the file from scratch.


	RETURNS:
		Dictionary containing the following keys:
			nplanet
				<int> Planet number we are looking at
			xlat
				<int> Latitude we are looking down at
			npro
				<int> Number of levels in the *.prf file
			ngas
				<int> Number of gasses in the *.prf file
			mean_molwt
				<float> Mean molecular weight of atmosphere (kg/mol)
			height[npro]
				[<float>] Height of each level (km)
			press[npro]
				[<float>] Pressure of each level (bar, or 100 kPa)
			temp[npro]
				[<float>] Temperature of each level (K)
			gas_vmrs[ngas,npro]
				[[<float>]] Gas volume mixing ratios at each level for each gas
			gasID[ngas]
				[<int>] Gas ID number
			isoID[ngas]
				[<int>] Gas isotopologue ID number
			molwt [npro]
				[<float>] Molecular weight of each level (g/mol)
	"""
	saved_data = already_opened_dict.get('prf',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)
	
	with open(nc._ensure_ext(runname,'.prf'), 'r') as f:
		iform = int(nc._read_line_from(f))
		if (iform==0):
			nplanet, xlat, npro, ngas, mean_molwt = map(nc._str2type, nc._read_line_from(f).split()[:5])
		else:
			nplanet, xlat, npro, ngas = map(nc._str2type, nc._read_line_from(f).split()[:4])
			mean_molwt = -1
		gasID = np.zeros([ngas], dtype='int')
		isoID = np.zeros([ngas], dtype='int')
		for i in range(ngas):
			gasID[i], isoID[i] = nc._read_line_from(f).split()[:2]
		nc._read_line_from(f)
		height = np.zeros([npro])
		press = np.zeros([npro])
		temp = np.zeros([npro])
		gas_vmrs = np.zeros([ngas, npro])
		for i in range(npro):
			gas_vmr_tbl = nc._read_line_from(f).split()
			height[i] = gas_vmr_tbl[0]
			press[i] = gas_vmr_tbl[1]
			press[i] *= 1.01325 # convert to bar (100 kPa, or 1E5 Pa, or 1E5 kg m^-1 s^-2)
			temp[i] = gas_vmr_tbl[2]
			gas_vmrs[:,i] = gas_vmr_tbl[3:]

	# need the gas id mapping to get molecular weights for 
	gas_id_mapping = gasinforef_raddata()
	molwt = np.zeros([npro])
	gas_molwt_arr = np.zeros([ngas])
	for _i in range(ngas):
		if isoID[_i] == 0:
			# then we have all of the isotopes present
			for _k, _v in gas_id_mapping[gasID[_i]]['isotopologue'].items():
				gas_molwt_arr[_i] += _v[0]*_v[1] # relative abundance * molecular mass (g/mol)
		else:
			gas_molwt_arr[_i] = gas_id_mapping[gasID[_i]]['isotopologue'][isoID[_i]][1]

	for i in range(npro):
		molwt[i] = np.sum(gas_molwt_arr*gas_vmrs[:,i])

	if iform == 1:
		# all volume mixing ratios add up to one, so we can calculate molwt directly
		mean_molwt = np.mean(molwt)
	elif iform == 0:
		# vmrs do not necessarily add up to one, but we can use mean_molwt to
		# guess a correction factor
		molwt *= (mean_molwt/np.mean(molwt))
	elif iform ==2:
		# vmrs do not necessarily add up to one, but we have no extra information
		# so the molecular weight only represents those contained in the file
		mean_molwt = np.mean(molwt)
	else:
		ut.pERROR(f'Do not have prescription when "iform={iform}" for <runname>.prf file. Exiting...')
		raise ValueError

	saved_data = nc._varname2dict(('nplanet', 'xlat', 'npro', 'ngas', 'mean_molwt', 'height', 'press', 'temp', 'gas_vmrs', 'gasID', 'isoID', 'molwt'),locals())
	already_opened_dict['prf'] = saved_data
	return(saved_data)

def apr(runname, npro, use_cache=False):
	"""
	Read in the <runname>.apr file for a nemesis run

	ARGUMENTS:
		runname
			<str> Name of the nemesis run to read the *.apr file of
		npro
			<int> Number of vertical levels in a profile
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.

	RETURNS:
		A dictionary containing the following keys:
			nvar
				<int> Number of variables
			varident[nvar,3]
				[[<int>,<int>,<int>]] Identity of each variable
			varparam[nvar,nparam]
				[[<float>]] Extra parameters for each variable
			nx
				<int> Length of the a-priori (and also state) vector
			xa[nx]
				[<float>] A-priori vector (elements beyond nx are zero)
			erra[nx]
				[<float>] A-priori erros (elements beyond nx are zero)
	EXAMPLE:
		{nvar, varident, varparam, nx, xa, erra} = nemesis.read.apr(runname)
	"""
	import numpy as numpy

	saved_data = already_opened_dict('apr',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	mx = 400
	mvar = 12
	np=1
	mparam = 405
	xa = numpy.zeros([mx]) # a-priori vector
	erra = numpy.zeros([mx]) # a-priori error
	varident = numpy.zeros([mvar,3], dtype='int')
	#varparam = numpy.zeros([mvar,mparam])
	varparam = [None]*mvar
	istart = 0

	# quick function to read a single value + error from 'num' lines
	def read_val_err(num, fhdl):
		for i in range(num):
			val, err = map(float, nc._read_line_from(f).split[:2])
			xa[istart+i] = val
			erra[istart+i] = err
		return

	with open(nc._ensure_ext(runname,'.apr'), 'r') as f:
		nc._read_line_from(f) #ignore header line
		nvar = int(nc._read_line_from(f).split()[0])
		for ivar in range(nvar):
			varparam[ivar] = numpy.zeros([mparam])
			varident[ivar] = nc._read_line_from(f).split()[:3]
			itype = varident[ivar][2]
			nvp = 0 # number of entries in varparam for variable 'ivar'

			print(F'ITYPE = {itype}')
			if itype==0:
				np = npro
				apfile = nc._read_line_from(f).split()[0]
				with open(apfile, 'r') as g:
					npro_1, clen = map(nc._str2type, nc._read_line_from(g).split()[:2]) #clen = correlation length
					if npro_1 != npro:
						print('ERROR: Error reading variable {} data-file {}'.format(ivar, apfile))
						print('-----: npro_1 {} does not match npro {}'.format(npro_1, npro))
						sys.exit()
					for j in range(np):
						press, x, err = map(float, nc._read_line_from(g).split())
						xa[istart+j] = x
						erra[istart+j] = err

			if itype in [1,6]:
				np=2
				varparam[ivar][0] = float(nc._read_line_from(f).split()[0]) #knee pressure	
				deep_vmr, deep_vmr_err = map(float, nc._read_line_from(f).split()[:2]) #deep volume mixing ratio
				fsh, fsh_err = map(float, nc._read_line_from(f).split()[:2]) # fractional scale height
				xa[istart] = deep_vmr
				erra[istart] = deep_vmr_err
				xa[istart+1] = fsh
				erra[istart+1] = fsh_err
				nvp=1

			if itype in [2,3]:
				np = 1
				fac, fac_err = map(float, nc._read_line_from(f).split()[:2])
				xa[istart] = fac
				erra[istart] = fac_err

			if itype in [4,24]:
				np=3
				read_val_err(np, f)
				#for j in range(np):
				#	val, err = map(float, nc._read_line_from(f).split()[:2])
				#	xa[istart+j] = val
				#	erra[istart+j] = err

			if itype == 23:
				np=4
				read_val_err(np,f)

			if itype == 25:
				apfile = nc._read_line_from(f).split()[1]
				with open(apfile, 'r') as g:
					np, clen, layer = map(nc._str2type, nc._read_line_from(g).split()[:3])
					varparam[ivar][0] = np
					for j in range(np):
						press, x, err = map(float, nc._read_line_from(g).split())
						xa[istart+j] = x
						erra[istart+j] = err
						varparam[ivar][istart+j+1] = press
				nvp = istart+np+1

			if itype in [8,9]:
				np=3
				read_val_err(np,f)

			if itype ==10:
				np=4
				read_val_err(np,f)
				nc._read_line_from(f) #read empty line?

			if itype==11:
				np=2
				read_val_err(np,f)
				nc._read_line_from(f) #read empty line?

			if itype==21:
				np=2
				read_val_err(np,f)
				nc._read_line_from(f) #read empty line?

			if itype in [12,13,14,15]:
				np=3
				read_val_err(np,f)

			if itype==19:
				np=4
				read_val_err(np,f)
			
			if itype==555:
				np=1
				read_val_err(np,f)
	
			if itype==888:
				np = int(nc._read_line_from(f).split()[0])
				varparam[ivar][0]=np
				for j in range(np):
					wav, x, err = map(float, nc._read_line_from(f))
					xa[istart+j] = x
					erra[istart+j] = err
				nvp = 1

			if itype==887:
				np = int(nc._read_line_from(f).split()[0])
				varparam[ivar][0]=np
				for j in range(np):
					wav, x, err = map(float, nc._read_line_from(f))
					xa[istart+j] = x
					erra[istart+j] = err
				nvp = 1

			if itype==222:
				np = 8
				read_val_err(np,f)
				varparam[ivar][0:4] = nc._read_line_from(f).split()[:5]
				nvp = 4

			if itype==225:
				np = 11
				read_val_err(np,f)
				varparam[ivar][0:4] = nc._read_line_from(f).split()[:5]
				nvp = 4

			if itype in [223,224]:
				np=9
				read_val_err(np,f)
				varparam[ivar][0:4] = nc._read_line_from(f).split()[:5]
				nvp = 4

			if itype ==227:
				np=7
				read_val_err(np,f)
				varparam[ivar][0] = float(nc._read_line_from(f).split()[0])
				nvp = 1

			if itype==444:
				cloudfile=nc._read_line_from(f).split()[0]
				with open(cloudfile, 'r') as g:
					np = 2
					read_val_err(np, g)
					nlam, x = map(nc._str2type, nc._read_line_from(f).split()[:2])
					varparam[ivar][0:2] = [nlam, x]
					varparam[ivar][2:4] = nc._read_line_from(f).split()[:2]
					varparam[ivar][4] = nc._read_line_from(f).split()[0]
					walb=numpy.zeros([nlam])
					for j in range(nlam):
						a,b,c = map(float, nc._read_line_from(f).split()[:3])
						xa[istart+np+j] = b
						erra[istart+np+j] = c
						walb[j] = a # for the 4** types, should 'walb' be added to varparam?
					np = np+nlam
				nvp = 5

			if itype ==445:
				cloudfile = nc._read_line_from(f).split()[0]
				with open(cloudfile, 'r') as g:
					np = 3
					read_val_err(np,g)
					nlam, x = map(nc._str2type, nc._read_line_from(g).split()[:2])
					varparam[ivar][0:2] = [nlam, x]
					a,b,c = map(float, nc._read_line_from(g).split()[:2])
					varparam[ivar][2:4] = [a,b]
					varparam[ivar][5] = c
					varparam[ivar][4] = nc._read_line_from(g).split()[0]
					walb = numpy.zeros([nlam])
					for j in range(nlam):
						w,a,b,c,d = nc._read_line_from(g).split()[:4]
						xa[istart+np+j] = a
						erra[istart+np+j] = b
						xa[istart+np+nlam+j] = c
						erra[istart+np+nlam+j] = d
						walb[j] = w
					np = np+(2*nlam)
				nvp = 6

			if itype==446:
				cloudfile=nc._read_line_from(f).split()[0]
				with open(cloudfile, 'r') as g:
					np = 3
					read_val_err(np, g)
					nlam, x = map(nc._str2type, nc._read_line_from(f).split()[:2])
					varparam[ivar][0:2] = [nlam, x]
					varparam[ivar][2:4] = nc._read_line_from(f).split()[:2]
					varparam[ivar][4] = nc._read_line_from(f).split()[0]
					walb=numpy.zeros([nlam])
					for j in range(nlam):
						a,b,c = map(float, nc._read_line_from(f).split()[:3])
						xa[istart+np+j] = b
						erra[istart+np+j] = c
						walb[j] = a
					np = np+nlam
				nvp = 5

			if itype==666:
				np=1
				read_val_err(np,f)
			
			if itype in [999, 777]:
				np=1
				read_val_err(np,f)
			varparam[ivar] = varparam[ivar][:nvp]
			istart += np
	nx = istart
	
	varident = varident[:nvar]
	varparam = varparam[:nvar]
	xa = xa[:nx]
	erra = erra[:nx]
	saved_data = nc._varname2dict(('nvar', 'varident', 'varparam', 'nx', 'xa', 'erra'), locals())
	already_opened_dict['apr'] = saved_data
	return(saved_data)

def itr(runname, use_cache=False):
	"""
	Reads <runname>.itr file from a nemesis run. Adapted from "plotiternewX.pro" IDL routine
	
	ARGUMENTS:
		runname
			<str> Name of the NEMESIS run to read <runname>.mre file for
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary containing the following keys:
			nx
				<int> Number of entries in the state (and a-priori) vector
			ny
				<int> Number of entries in the measured (and fitted) spectrum
			niter
				<int> Number of iterations in the file (this may be incorrect, check)
			chisq_arr[niter]
				[<float>] Value of the chi-squared cost function for each iteration
			phi_arr[niter]
				[<float>] Value of the phi cost function for each iteration
			xn_arr[niter, nx]
				[[<float>]] Value of the state vector at each iteration
			xa_arr[niter,nx]
				[[<float>]] Value of the a-priori vector at each iteration (should all be identical)
			y_arr[niter,ny]
				[[<float>]] Measured value of the spectrum
			se_arr[niter, ny]
				[[<float>]] Variances of the measured value of the spectrum (i.e. diagonal elements of covariance matrix) (I think)
			yn_arr[niter, ny]
				[[<float>]] Fitted value of the spectrum (nth iteration?)
			yn1_arr[niter,ny]
				[[<float>]] Fitted value of the spectrum (n-1th iteration?). Unsure how this is different to yn_arr
			kk_arr[niter,nx,ny]
				[[[<float>]]] The Jacobian (otherwise known as "sensitivity matrix" or "functional derivative") for each iteration.

	EXAMPLE:
		{nx, ny, niter, chisq_arr, phi_arr, xn_arr, xa_arr, y_arr, se_arr, yn_arr, yn1_arr, kk_arr} = nemesis.read.itr(runname)
	"""
	import numpy as np

	saved_data = already_opened_dict.get('itr',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	phi = 1.
	chisq = 1.

	xn_list = [] # list of retrieved state vectors (one for each iteration)
	xa_list = [] # list of a-priori state vectors (one for each iteration, but they should be identical)
	
	with open(runname+'.itr', 'r') as f:
		try:
			mx = 400
			nx = 1
			ny = 1
			niter = 1
			nc._read_line_from(f) #skip empty line
			i = 0
			chisq_arr = np.zeros((niter,))
			phi_arr = np.zeros((niter,))
			xn_arr = np.zeros((niter, nx))
			xa_arr = np.zeros((niter, nx))
			y_arr = np.zeros((niter, ny))
			se_arr = np.zeros((niter, ny))
			yn_arr = np.zeros((niter, ny))
			yn1_arr = np.zeros((niter, ny))
			kk_arr = np.zeros((niter, nx, ny))
			while i<niter:
				# for some reason, some files have less iterations in them than the 'niter' variable indicates
				# we must set the value of 'niter' correctly and slice all of the arrays to the right size
				if (f.readline()==''): #if <file>.readline() returns an empty string, we are at EOF, otherwise this should be an empty line
					niter = i
					chisq_arr = chisq_arr[:niter]
					phi_arr = phi_arr[:niter]
					xn_arr = xn_arr[:niter]
					xa_arr = xa_arr[:niter]
					y_arr = y_arr[:niter]
					se_arr = se_arr[:niter]
					yn_arr = yn_arr[:niter]
					yn1_arr = yn1_arr[:niter]
					kk_arr = kk_arr[:niter]
					break
				nx, ny, niter = map(int, nc._read_line_from(f).split()) # niter may not be the number of iterations
				if (i==0):
					chisq_arr = np.zeros((niter,))
					phi_arr = np.zeros((niter,))
					xn_arr = np.zeros((niter, nx))
					xa_arr = np.zeros((niter, nx))
					y_arr = np.zeros((niter, ny))
					se_arr = np.zeros((niter, ny))
					yn_arr = np.zeros((niter, ny))
					yn1_arr = np.zeros((niter, ny))
					kk_arr = np.zeros((niter, nx, ny))
				chisq, phi = map(float, nc._read_line_from(f).split())
				chisq_arr[i] = chisq
				phi_arr[i] = phi
				xn_arr[i] = nc._read_line_from(f).split()
				xa_arr[i] = nc._read_line_from(f).split()
				y_arr[i] = nc._read_line_from(f).split()
				se_arr[i] = nc._read_line_from(f).split()
				yn1_arr[i] = nc._read_line_from(f).split()
				yn_arr[i] = nc._read_line_from(f).split()
				for j in range(nx):
					kk_arr[i][j] = nc._read_line_from(f).split()
				i+=1
		except:
			print(sys.exc_info())
			raise nemesis.exceptions.NemesisReadError("Something went wrong when reading file {}. It did not conform to expected structure or some other problem".format(nc._ensure_ext(runname,'.itr')))
			
	saved_data = nc._varname2dict(('nx', 'ny', 'niter', 'chisq_arr', 'phi_arr', 'xn_arr', 'xa_arr', 'y_arr', 'se_arr', 'yn_arr', 'yn1_arr', 'kk_arr'), locals())
	already_opened_dict['itr'] = saved_data
	return(saved_data)

def mre(runname, comment_str='#', use_cache=False):
	"""
	Reads the *.mre file associated with a NEMESIS run name

	ARGUMENTS:
		runname
			<str> Name of the nemesis run
		comment_str (optional) (default='#')
			<str> Ignore lines that start with this string as comments
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.
	
	RETURNS:
		A dictionary containing the following keys:
			nspec
				<int> Total number of retrevals performed
			ispec
				<int> Unknown exactly what this does
			ngeom
				<int> Number of geometries in the file
			ny
				<int> Number of points in the measurement vector
			nx
				<int> Number of points in the state vector
			ny2
				<int> Number of points in the measurement vector (seems to mirror ny, check with more complex run)
			lat
				<float> Latitude (deg)
			lon
				<float> Longitude (deg)
			waveln[ngeom, ny/ngeom]
				<float,array> Wavelength (or wavenumber) of each point in the measurement vector
			radiance_meas[ngeom, ny/ngeom]
				<float,array> Measured (from observation) spectrum for each of the geometries
				in units: uW cm-2 sr-1 um-1
			radiance_err[ngeom,ny/ngeom]
				<float,array> Absolute error on the measured spectrum
				in units: uW cm-2 sr-1 um-1
			radiance_perc_err[ngeom,ny/ngeom]
				<float,array> Percentage error on the measured spectrum
				in units: uW cm-2 sr-1 um-1
			radiance_retr[ngeom,ny/ngeom]
				<float,array> Retrieved spectrum from NEMESIS
				in units: uW cm-2 sr-1 um-1
			radiance_perc_diff[ngeom,ny/ngeom]
				<float,array> Percentage difference between measurement and retrieved spectrum
			nvar
				<int> Number of variables in the state vector
			nxvar[nvar]
				<int,array> Number of values for each variable in the state vector
			varident[nvar,3]
				<float,array> Retrieved variable ID as defined by NEMESIS manual
			varparam[nvar,5]
				<float,array> Extra parameters about how to read the retrieved variable
			aprprof[nvar,nx]
				<<float,array>,array> A-priori profile for each variable in the state vector
			aprerr[nvar,nx]
				<<float,array>,array> A-priori errors for each varaiable in the state vector
			retprof[nvar,nx]
				<<float,array>,array> Retrieved profile for each variable in the state vector
			reterr[nvar,nx]
				<<float,array>,array> Retrieved errors for each variable in the state vector

	EXAMPLE:
		{nspec,ispec,ngeom, ny, nx, ny2, lat, lon, waveln, radiance_meas, radiance_err, 
		radiance_perc_err, radiance_retr, radiance_perc_diff, nvar, nxvar, varident, 
		varparam, aprprof, aprerr, retprof, reterr} = nemesis.read.mre(runname)
		
	"""
	saved_data = already_opened_dict.get('mre',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	#ut.pDEBUG('runname {}'.format(runname))
	cs = comment_str
	#get npro from *.ref file
	npro = ref(runname[:-4] if runname.endswith('.mre') else runname, use_cache=use_cache)['npro']
	ln = 0 #line number
	waveln = []
	radiance_measured = []
	radiance_err = []
	radiance_percent_err = []
	radiance_retrieved = []
	radiance_percent_diff = []
	with open(nc._ensure_ext(runname,'.mre'), 'r') as f:
		try:
			nspec = int(nc._read_line_from(f, cs).split()[0])
			ispec, ngeom, ny, nx, ny2 = map(int, nc._read_line_from(f, cs).split()[:5])
			ny_per_geom = int(ny/ngeom)
			lat, lon = map(float, nc._read_line_from(f, cs).split()[:2])
			# skip two lines
			nc._read_line_from(f,cs)
			nc._read_line_from(f,cs)
			#make arrays for reading spectra
			arr_ordering = (ngeom, ny_per_geom)
			waveln = np.zeros(arr_ordering)
			radiance_meas = np.zeros(arr_ordering)	
			radiance_err = np.zeros(arr_ordering)
			radiance_perc_err = np.zeros(arr_ordering)
			radiance_retr = np.zeros(arr_ordering)
			radiance_perc_diff = np.zeros(arr_ordering)
			
			for i in range(ngeom):
				for j in range(ny_per_geom):
					k, w, rm, re, rpe, rr, rpd = map(float, nc._read_line_from(f, cs).split())
					waveln [i,j] = w
					radiance_meas[i,j] = rm
					radiance_err[i,j] = re
					radiance_perc_err[i,j] = rpe
					radiance_retr[i,j] = rr
					radiance_perc_diff[i,j] = rpd

			nc._read_line_from(f,cs) #skip empty line
			
			#make arrays for reading state vector
			nvar = int(nc._read_line_from(f,cs).split()[-1]) #number of variable profiles
			#print(f'INFO: nvar {nvar}')
			nxvar = np.zeros([nvar], dtype='int')
			
			# This uses extra storage than is needed because numpy only understands rectangular arrays
			# the performance hit should be minimal unless massive retrievals are being done.
			# See nemesis.plot.mre for an example of how to read-out this formatted data.
			# If you need to make this memory-efficient, you can use a list of numpy arrays.
			arr_ordering = (nvar, nx)
			aprprof = np.zeros(arr_ordering)
			aprerr = np.zeros(arr_ordering)
			retprof = np.zeros(arr_ordering)
			reterr = np.zeros(arr_ordering)
			varident = np.zeros((nvar,3)) # There are always 3 values for varident
			varparam = np.zeros((nvar,5)) # I see nothing in the manual that ensures we never have more than 5 values for varparam
			for i in range(nvar):
				which_var = int(nc._read_line_from(f,cs).split()[-1]) # which variable (out of nvar variables) are we giving information for?
				varident[i,:] = list(map(int, nc._read_line_from(f,cs).split()))
				varparam[i,:] = list(map(float, nc._read_line_from(f,cs).split()))
				nxvar[i] = nc._get_number_of_state_vector_entries_for_profile(npro, varident[i], varparam[i])
				nc._read_line_from(f,cs) #skip line
				for j in range(nxvar[i]): #may have to change later
					apr_x, apr_err, ret_x, ret_err = map(float, nc._read_line_from(f,cs).split()[2:])
					aprprof[i,j] = apr_x
					aprerr[i,j] = apr_err
					retprof[i,j] = ret_x
					reterr[i,j] = ret_err
		except:
			print(sys.exc_info())
			raise nemesis.exceptions.NemesisReadError("Something went wrong when reading file {}. It did not conform to expected structure or some other problem".format(nc._ensure_ext(runname,'.mre')))
			
		
	saved_data = nc._varname2dict((	'nspec','ispec','ngeom','ny', 'nx', 'ny2', 'lat', 'lon', 'waveln', 
							'radiance_meas', 'radiance_err', 'radiance_perc_err', 'radiance_retr', 
							'radiance_perc_diff', 'nvar', 'nxvar', 'varident', 'varparam', 'aprprof', 
							'aprerr', 'retprof', 'reterr'), locals())
	already_opened_dict['mre'] = saved_data
	return(saved_data)

def gasinforef_raddata(gir_file="~/Documents/repos/radtrancode/raddata/gasinforef.dat", use_cache=False):
	"""
	Reads the gasinforef.dat file from the radtrancode/raddata directory in the NEMESIS repository

	ARGUMENTS:
		gir_file (optional) (default="~/Documents/repos/radtrancode/raddata/gasinforef.dat")
			The gasinforef.dat file to read	
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		gas_id_mapping
			{	gID:{	'name':gName, 
						'isotopologue':{	isoID:	[	relAbubdance,
														molwt,
														unknown,
														[	coeff1,
															coeff2,
															coeff3,
															coeff4
														]
													]
										}
					}
			}
			Where the variables in the dictonary are:
				gID
					<int> An integer serving as a gas identity number
				gName
					<str> A string of the gas' name
				isoID
					<int> An integer serving as a identity number for the
					gas isotope
				relAbundance
					<float> The relative abundance of the gas isotope
				molwt
					<float> The molecular weight of the gas isotope (g cm^-3)
				unknown
					<int> An integer of unknown purpose, possible a HITRAN 
					isotopologue ID number.
				coeff1, coeff2, coeff3, coeff4
					<float>,<float>,<float>,<float> List of coefficients of a 
					4th order fit to the 70->300 K section of the total 
					partition function (vibrational+rotational)
			
	EXAMPLE:
		gas_id_mapping = nemesis.read.gasinforef_raddata()
	"""
	saved_data = already_opened_dict.get('gasinforef_raddata',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	gas_id_mapping = {}
	with open(os.path.expanduser(gir_file), 'r') as f:
		#skip empty line just after comments
		nc._read_line_from(f)
		#each gas record starts with a line of hyphens (-)
		while nc._read_line_from(f).startswith('-'):
			gid = int(nc._read_line_from(f).split()[0]) #first line is gas id
			gn = nc._read_line_from(f) #second line is gas name
			niso = int(nc._read_line_from(f).split()[0]) #third line is number of isotopologues
			gas_id_mapping[gid]={'name':gn, 'isotopologue':{}}
			for i in range(0,niso): #next lines are isotopologue information, 2 lines per isotopologue
				aline = nc._read_line_from(f) #this first line is tricky
				#print(aline)
				iid, ra, mn, x = aline.split(None, 3)
				# x is always of format "(102) ..." or "( 12) ..." or "(29) ...", so we can't just split by spaces but we want the characters inside the brackets
				x = x[1:x.find(')')]
				#iid, ra, mn, x = read_line_from(f).split()[:4] #need to remove brackets from 'x' later
				pfc1, pfc2, pfc3, pfc4 = map(float, nc._read_line_from(f).split())
				gas_id_mapping[gid]['isotopologue'][int(iid)] = [float(ra), float(mn), int(x), [pfc1, pfc2, pfc3, pfc4]]

	saved_data = gas_id_mapping
	already_opened_dict['gasinforef_raddata'] = saved_data
	return(saved_data)

def ref(runname, comment_str='#', use_cache=False):
	"""
	Reads the *.ref file associated with a NEMESIS run name

	ARGUMENTS:
		runname
			<str> The run name of the *.ref file to look for.
		comment_str (optional) (default='#')
			<str> If present at the beginning of a line, this
			string determines that the whole line is a comment.
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then 
			return the stored data any other time the file is requested to be 
			read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary with the following keys:
			amform
				<int> If amform==1 then it is assumed that all volume mixing 
				ratios sum to 1
			unknown_var_1
				<int> I do not know what this variable is for
			nplanet
				<int> Planet ID in order of distance to sun (Mercury=1, 
				Venus=2, ...)
			xlat
				<float> Planetocentric latitude
			npro
				<int> Number of points in the profile
			ngas
				<int> Number of gasses whose volume mixing ratios are included 
				in the file
			mean_molwt
				<float> Molecular weight of atmosphere (kg/mol)
			gasID[ngas]
				<int,array> HITRAN ID's of the gas that needs to be incuded 
				(see "radtrans" manual)
			isoID[ngas]
				<int,array> ID numbers of the isotopologue to include (0 for 
				all)
			height[npro]
				<float,array> Height profile in km
			press[npro]
				<float,array> Presure profile in bar
			temp[npro]
				<float,array> Temperature profile in K
			vmr[npro,ngas]
				<<float,array>,array> volume mixing ratio of the different gasses
			molwt [npro]
				<float,array> Molecular weight of profile in (g/mol)
	
	EXAMPLE:
		{amform, unknown_var_1, nplanet, xlat, npro, ngas, mean_molwt, gasID, isoID, height, press, temp, vmr} = nemesis.read.ref(runname)

	"""
	import numpy as np
	cs = comment_str

	# Check to see if we've already got the data from this file
	saved_data = already_opened_dict.get('ref', None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)
	# If we don't already have it, grab it and save it

	refname = nc._ensure_ext(runname, '.ref')

	with open(refname, 'r') as f:
		line_num = 0 # zero index to be consistent with python
		after_gas = -1 # placeholder variable
		amform = int(nc._read_line_from(f,cs))
		unknown_var_1 = nc._read_line_from(f,cs)
		# for some reason, some files exclude 'mean_molwt' from the file
		nxnnm = list(map(nc._str2type, nc._read_line_from(f,cs).split()))
		if amform == 0:
			nplanet, xlat, npro, ngas, mean_molwt = nxnnm
		elif amform in (1,2):
			nplanet, xlat, npro, ngas = nxnnm
			mean_molwt = -1 # calculate mean_molwt later
			#ut.pWARN('File {} has no entry for "mean_molwt", setting to -1.'.format(refname))
		else:
			ut.pERROR('File {} does not conform to normal standards for a *.ref file'.format(refname))
		gasID = np.zeros([ngas], dtype='int')
		isoID = np.zeros([ngas], dtype='int')
		for i in range(ngas):
			gid, iid = map(int, nc._read_line_from(f,cs).split())
			gasID[i] = gid
			isoID[i] = iid

		profile_hdr = nc._read_line_from(f,cs) #ignore this line
		height = np.zeros([npro])
		press = np.zeros([npro])
		temp = np.zeros([npro])
		vmr = np.zeros([npro, ngas])
		for i in range(npro):
			lf = list(map(float, nc._read_line_from(f,cs).split()))
			height[i] = lf[0]
			press[i] = lf[1]
			press[i] *= 1.01325 #convert from atm to bar
			temp[i] = lf[2]
			vmr[i] = lf[3:]

	vmr = np.transpose(vmr)

	# need the gas id mapping to get molecular weights for 
	gas_id_mapping = gasinforef_raddata()
	molwt = np.zeros([npro])
	gas_molwt_arr = np.zeros([ngas])
	for _i in range(ngas):
		if isoID[_i] == 0:
			# then we have all of the isotopes present
			for _k, _v in gas_id_mapping[gasID[_i]]['isotopologue'].items():
				gas_molwt_arr[_i] += _v[0]*_v[1] # relative abundance * molecular mass (g/mol)
		else:
			gas_molwt_arr[_i] = gas_id_mapping[gasID[_i]]['isotopologue'][isoID[_i]][1]

	for i in range(npro):
		molwt[i] = np.sum(gas_molwt_arr*vmr[:,i])

	if amform == 1:
		# all volume mixing ratios add up to one, so we can calculate molwt directly
		mean_molwt = np.mean(molwt)
	elif amform == 0:
		# vmrs do not necessarily add up to one, but we can use mean_molwt to
		# guess a correction factor
		molwt *= (mean_molwt/np.mean(molwt))
	elif amform ==2:
		# vmrs do not necessarily add up to one, but we have no extra information
		# so the molecular weight only represents those contained in the file
		mean_molwt = np.mean(molwt)
	else:
		ut.pERROR(f'Do not have prescription when "iform={iform}" for <runname>.prf file. Exiting...')
		raise ValueError


	saved_data = nc._varname2dict(('amform', 'unknown_var_1', 'nplanet', 'xlat', 'npro', 'ngas', 'mean_molwt', 'gasID', 'isoID', 'height', 'press', 'temp', 'vmr', 'molwt'),locals())
	# remember the data so if we open this file again we don't have to bother the filesystem
	already_opened_dict['ref'] = saved_data
	return(saved_data)

def cov(runname, use_cache=False):
	"""
	Reads the <runname>.cov file associated with a nemesis run

	ARGUMENTS:
		runname
			name of the nemesis run, can optionally end in ".cov" 
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary that contains the following keys:
			npro
				<int> The number of points in the profile
			nvar
				<int> Number of variable profiles
			varident [nvar,3]
				<int, array> Identity code of the i^th variable
			varparams [nvar, 5]
				<float,array> Parameters of the i^th variable
			nx
				<int> Number of points in the state vector
			ny
				<int> Number of points in the measurement vector
			sa [nx,nx]
				<float,array> The a-priori covariance matrix
			st [nx,nx]
				<float,array> The total covariance matrix
			sm [nx,nx]
				<float,array> The measurement covariance matrix
			sn [nx,nx]
				<float,array> The smoothing covariance matrix
			aa [nx,nx]
				<float,array> Averaging kernel
			dd [nx,ny]
				<float,array> Contribution function (What is this?)
			kk [ny,nx]
				<float,array> The Jacobian matrix (or 'kernel')
			kt [nx,ny]
				<float,array> The transpose of kk
			se1 [ny]
				<float,array> Measurement errors, includes forward modelling errors
	EXAMPLE:
		An example of calling the function
	"""
	saved_data = already_opened_dict.get('cov', None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	refname = nc._ensure_ext(runname, '.cov')
	
	with open(refname, 'r') as f:
		npro, nvar = nc.read_line_as(f, int)
		varident = np.zeros((nvar,3), dtype=int)
		varparams = np.zeros((nvar,5), dtype=float)
		for i in range(0,nvar):
			varident[i] = nc.read_line_as(f, int)
			varparams[i] = nc.read_line_as(f, float)

		nx, ny = nc.read_line_as(f, int)
		sa = np.zeros((nx,nx),dtype=float)
		sm = np.zeros((nx,nx),dtype=float)
		sn = np.zeros((nx,nx),dtype=float)
		st = np.zeros((nx,nx),dtype=float)
		
		for i in range(0,nx):
			sa[i] = nc.read_line_as(f, float)
			sm[i] = nc.read_line_as(f, float)
			sn[i] = nc.read_line_as(f, float)
			st[i] = nc.read_line_as(f, float)
		
		aa = np.zeros((nx,nx),dtype=float)
		for i in range(0,nx):
			aa[i] = nc.read_line_as(f, float)
		
		dd = np.zeros((nx,ny), dtype=float)
		for i in range(0,ny):
			dd[:,i] = nc.read_line_as(f, float)
		
		kk = np.zeros((ny,nx), dtype=float)
		for i in range(0, nx):
			kk[:,i] = nc.read_line_as(f, float)
		
		kt = np.transpose(kk)

		se1 = np.zeros((ny,), dtype=float)
		se1[:] = nc.read_line_as(f, float)

	# TESTING
	print('npro {} nvar {}'.format(npro, nvar))
	for i in range(0,nvar):
		print('variable {}'.format(i))
		print('\tvarident {}'.format(varident[i]))
		print('\tvarparams {}'.format(varparams[i]))

	print('nx {} ny {}'.format(nx, ny))
	print('sa.shape {} st.shape {} sm.shape {} sn.shape {}'.format(
			sa.shape, st.shape, sm.shape, sn.shape))
	print('aa.shape {} dd.shape {} kk.shape {} kt.shape {} se1.shape {}'.format(
			aa.shape, dd.shape, kk.shape, kt.shape, se1.shape))
	# END TESTING

	saved_data = nc._varname2dict(('npro','nvar','varident','varparams','nx','ny','sa','st','sm','sn','aa', 'dd', 'kk', 'kt', 'se1'), locals())
	already_opened_dict['cov'] = saved_data
	return(saved_data)

def forward_model_error(filename, use_cache=False):
	"""
	Reads the forward model error file specified in the <runname>.inp file

	ARGUMENTS:
		filename
			<str> The filename of the forward model error file
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary that contains the following keys
			fme_n
				<int> Number of forward model error (wavelength, error) points defined, will be interpolated.

			fme_we [fme_n,2]
				<float,float> (Wavelength, Error) pairs that the forward model error is defined over

	EXAMPLE:
		An example of calling the function
	"""
	# We rely on this changing in some of our code so cannot cache the data
	saved_data = already_opened_dict.get('forward_model_error', None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)


	with open(filename, 'r') as f:
		fme_n = nc.read_line_as(f, int)[0]
		fme_we = np.zeros((fme_n,2), dtype=float)
		for i in range(0,fme_n):
			fme_we[i,:] = nc.read_line_as(f, (float,float))
	saved_data = nc._varname2dict(('fme_n','fme_we'),locals())
	already_opened_dict['forward_model_error']=saved_data
	return(saved_data)

def aerosol(filename, comment_str='#', use_cache=False):
	"""
	Reads in an 'aerosol.ref' or 'aerosol.prf' file that contains information about the aerosol species vertical distribution profile. 
	'areosol.ref' contains the 'reference' values, i.e. the values NEMESIS will use unless they are overwritten using a profile in <runname>.apr
	'areosol.prf' is written by NEMESIS to contain the profiles actually used, this involves taking profiles described in <runname>.apr,
	if a species does not have a profile in <runname>.apr, then that species' profile from 'aerosol.ref' is used.

	ARGUMENTS:
		filename
			<str> The name of the file to read.
		comment_str (optional)
			<str> Lines that start with this string will be treated as comments
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary with the following keys:
			npro
				<int> Number of points in the profile (vertical levels)
			ncont
				<int> Number of aerosol types defined in the file
			height[npro]
				<float,array> Height in km (?) above a reference position
			density[npro,ncont]
				<<float, array>,array> Concentration of aerosol at specific height (unknown unit, possibly g/cm^{-3})

	EXAMPLE:
		{npro, ncont, height, density} = aerosol(filename)
	"""
	saved_data = already_opened_dict.get('aerosol',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	cs = comment_str
	with open(filename, 'r') as f:
		npro, ncont = map(int, nc._read_line_from(f, cs).split())
		height = np.zeros((npro))
		density = np.zeros((npro,ncont))
		for i in range(npro):
			lf = list(map(float, nc._read_line_from(f,cs).split()))
			height[i] = lf[1]
			for j in range(ncont):
				density[i,j] = lf[1+j]

	saved_data = nc._varname2dict(('npro','ncont','height','density'),locals())
	already_opened_dict['areosol'] = saved_data
	return(saved_data)

def xsc(runname, use_cache=False):
	"""
	Reads in the <runname>.xsc file for a nemesis run

	ARGUMENTS:
		runname
			<str> The name of the nemesis run
		use_cache (optional) (default=False)
			<bool> If TRUE, will open a file once and store the data, then return the stored data any other time the file is 
			requested to be read. If FALSE, will always read the file from scratch.


	RETURNS:
		A dictionary containing the following keys:
			ncont
				<int> Number of aerosol species
			nwavs
				<int> Number of wavelenght/wavenumber points in the file
			wavs[nwavs]
				<float,array> Wavelength/wavenumber grid used
			xsection[ncont,nwavs]
				<<float,array>,array> cross-section at wavelength/wavenumber for each areosol species (was unknown_1)
			ssa[ncont,nwavs]
				<<float,array>,array> single scattering albedo at wavelength/wavenumber for each aerosol species (was unknown_2)

	EXAMPLE:
		{ncont, nwavs, wavs, unknown_1, unknown_2} = xsc(runname)
	"""
	saved_data = already_opened_dict.get('xsc',None)
	if use_cache and type(saved_data)!=type(None):
		return(saved_data)

	with open(nc._ensure_ext(runname, '.xsc'), 'r') as f:
		ncont = int(nc._read_line_from(f))
		w_xs_l = []
		ssa_l = []
		# we don't know the length of the file, but we do know that we have alternating lines
		i = 0
		for aline in f:
			if i%2==0:
				w_xs_l.append(list(map(float, aline.strip().split())))
			else:
				ssa_l.append(list(map(float, aline.strip().split())))
			i+=1
		nwavs = len(w_xs_l)
		wavs = np.array([x[0] for x in w_xs_l])
		xsection = np.array([x[1:] for x in w_xs_l])
		ssa = np.array(ssa_l)
	saved_data = nc._varname2dict(('ncont','nwavs','wavs','xsection','ssa'),locals())
	already_opened_dict['xsc'] = saved_data
	return(saved_data)

def set(runname):
	"""
	Reads a <runname>.set file.

	ARGUMENTS:
		filename
			<str> The file to read, can just be a runname without the '.set' extension

	RETURNS: 
		A dictionary containing the following keys:
			n_zens
				<int> The number of zenith angles
			quad_points [n_zens]
				<float, array> The quadrature points in cos(zenith_angle) space (should use gauss-lobatto quadrature)
			quad_weights [n_zens]
				<float, array> The weight for each quadrature point
			n_fourier_components
				<int> This is set automatically by the code so it gets overwritten
			n_azi_angles
				<int> The number of azimuthal angles used by the integration scheme, has been set at 100 for years
			sunlight_flag
				<int> Sunlight on (1) or off (0)
			solar_dist
				<float> Distance from sun to object beign modelled
			lower_boundary_flag
				<int> How should we treat the lower boundary, Thermal (0) or Lambert (1)
			ground_albedo
				<float> Albedo of the ground (how much light it absorbs/refects)
			surf_temp
				<float> Surface tempertaure of the 'ground', usually set to the same as the bottom layer.
			base_altitude
				<float> The altitude at the base of the bottom layer, set this deep enough so that the optical dept
				is very large
			n_atm_layers
				<int> Number of atmospheric layers to use, for scattering calculations, 39 is the largest number
				supported
			layer_type_flag
				<int> How the layers are split up, see RadTrans manual
			layer_int_flag
				<int> How the integration for each layer is performed, see the RadTrans manual

	EXAMPLE:
		set_data = nemesis.read.set('./neptune')
	"""	

	with open(nc._ensure_ext(runname, '.set'), 'r') as f:
		f.readline() # skip line of asterixes
		n_zens = int(f.readline().split(':')[-1].strip())
		quad_pw = np.zeros((2,n_zens))
		for i in range(n_zens):
			quad_pw[:,i] = nc.read_line_as(f, (float,float))
		n_fourier_components = int(f.readline().split(':')[-1].strip())
		n_azi_angles = int(f.readline().split(':')[-1].strip())
		sunlight_flag = int(f.readline().split(':')[-1].strip())
		solar_dist = float(f.readline().split(':')[-1].strip())
		lower_boundary_flag = int(f.readline().split(':')[-1].strip())
		ground_albedo = float(f.readline().split(':')[-1].strip())
		surf_temp = float(f.readline().split(':')[-1].strip())
		f.readline() #  ignore line of asterixes
		base_altitude = float(f.readline().split(':')[-1].strip())
		n_atm_layers = int(f.readline().split(':')[-1].strip())
		layer_type_flag = int(f.readline().split(':')[-1].strip())
		layer_int_flag = int(f.readline().split(':')[-1].strip())

	quad_points = quad_pw[0,:]
	quad_weights = quad_pw[1,:]

	set_data = nc._varname2dict(('n_zens', 'quad_points', 'quad_weights', 'n_fourier_components', 'n_azi_angles', 
									'sunlight_flag', 'solar_dist', 'lower_boundary_flag', 'ground_albedo', 
									'surf_temp', 'base_altitude', 'n_atm_layers', 'layer_type_flag', 
									'layer_int_flag'), locals())
	return(set_data)

def inp(runname):
	"""
	reads in a <runname>.inp file

	ARGUMENTS:
		runname
			<str> The name of the nemesis run

	RETURNS:
		A dictionary containing the following keys:
			ispace
				How the wavelengths are described (wavenumber cm^-1) or wavelength (um)
			iscat
				Is multiple scattering required (1) or is thermal calculation ok (0)
			ilbl
				Layer by layer flag. 0 = correlated k calc, 1 = line by line from scratch, 2 = line by line using precalculated tables
			woff
				wavelength offset (used for calibration error) to be added to synthetic spectra
			ename
				name of file containing forward modelling error
			niter
				number of iterations of NEMESIS to do
			philimit
				percentage conversion limit of cost function
			nspec
				total number of retrievals to perform
			ioff
				which retrieval to start with (1 for start at the beginning of <runname>.spx)
			lin
				controls how any previous retrievals are factored into this one (0=do not read in previous retrievals, default)
			iform
				unit of calculated spectrum (0 is defalt)
			percbool
				if false then error in ename is a systematic offset, otherwise error in ename is percentage error (false if default)
	"""	
	with open(nc._ensure_ext(runname, '.inp'), 'r') as f:
		ispace, iscat, ilbl = nc.read_line_as(f, (int,int,int), line_comment_char='!')
		woff = nc.read_line_as(f, (float,), line_comment_char='!')
		ename = nc.read_line_as(f, (str,), line_comment_char='!')
		niter = nc.read_line_as(f, (int,), line_comment_char='!')
		philimit = nc.read_line_as(f, (float,), line_comment_char='!')
		nspec, ioff = nc.read_line_as(f, (int,int), line_comment_char='!')
		lin = nc.read_line_as(f, (int,), line_comment_char='!')
		try:
			iform = int(f.readline().split('!')[0].strip())
		except:
			iform = 0
		try:
			percbool = int(f.readline().split('!')[0].strip())
		except:
			percbool = False
	saved_data = nc._varname2dict(('ispace','iscat','ilbl','woff','ename','niter','philimit','nspec','ioff','lin','iform','percbool'),locals())
	return(saved_data)

def iri(filename):
	"""
	Reads a *.dat file that holds information about an areosol's imaginary refractive index

	ARGUMENTS
		filename
			<str> The file to read

	RETURNS
		A dictonary contaning the following keys:
			R0
				<float> Radius of aerosol particle
			Rerr
				<float> Error on R0
			V0
				<float> porosity? of aerosol particle
			Verr
				<float> Error on V0
			nwave
				<int> Number of wavelenghts in file
			clen
				<float> Correlation length
			vref
				<float> reference wavelength
			nreal_vref
				<float> real part of refractive index at reference wavelength?
			v_od_norm
				<float> wavelength to normalise everything to?
			wie_array [nwave,3*float]
				<array, <float,float,float>> Array containing the following columns: 
				wavelength, imaginary refractive index (iri), error. Set error to < 1E-6*iri to
				treat as fixed.  
	"""
	
	with open(nc._ensure_ext(filename, '.dat'), 'r') as f:
		R0, Rerr = nc.read_line_as(f, (float, float), line_comment_char='!')
		V0, Verr = nc.read_line_as(f, (float, float), line_comment_char='!')
		nwave, clen = nc.read_line_as(f, (int, float), line_comment_char='!')
		vref, nreal_vref = nc.read_line_as(f, (float, float), line_comment_char='!')
		(v_od_norm,) = nc.read_line_as(f, (float,), line_comment_char='!')
		wie_array = np.full((nwave, 3), fill_value=np.nan)
		for i in range(nwave):
			wie_array[i,:] = nc.read_line_as(f, (float,float,float), line_comment_char='!')

	saved_data = nc._varname2dict(		('R0', 'Rerr', 'V0', 'Verr', 'nwave', 'clen', 
										'vref', 'nreal_vref', 'v_od_norm', 'wie_array'), 
									locals()
									)
	return(saved_data)	
