#!/usr/bin/env python3
"""
A collection of common functions used by many nemesis routines
"""

import sys
import utils as ut
import nemesis.cfg

class RegionContainer():
	def __init__(self, filename, description='GENERIC REGION'):
		self.filename = filename
		self.description = description
		self.data = {}
		self.read_region_file()
		return

	def read_region_file(self):
		import keyval as kv
		with open(self.filename, 'r') as f:
			region_kvutc = kv.readKV(f)
		self.data = {}
		for k, v, u, t, c in zip(*region_kvutc):
			if t=='int':
				v = int(v)
			elif t=='float':
				v = float(v)
			else:
				v = v
			self.data[k] = {'value':v, 'unit':u, 'type':t, 'comment':c}

	def __getitem__(self, idx):
		return(self.data[idx])

	def __repr__(self):
		astr = '<RegionContainer>\n'
		astr += '\tdescription: {}\n'.format(self.description)
		astr += '\tsuffix: {}\n'.format(self.suffix)
		astr += '\tfitscube: {}\n'.format(self.fitscube)
		astr += '\tfilename: {}\n'.format(self.filename)
		astr += '\tdata:\n'
		for k in sorted(self.data.keys()):
			astr += '\t\t{}\n'.format(k)
			for s in ('value', 'unit', 'type', 'comment'):
				astr += '\t\t\t{}: {}\n'.format(s, self.data[k][s])
		astr += '<--------------->\n'
		return(astr)

	def get_filename(self):
		return(self.filename)

	def get_description(self):
		return(self.description)
	
	def set_description(self, desc):
		self.description = desc

	def get_region_type(self):
		"""
		Returns a string that defines the type of region contained
		Type is determined by which keys are present in the self.data member.
		"""
		circle_keys = ['cxp','cyp','rp','cra','cdec','rang']
		rect_keys = ['cxp','cyp','wp','hp','ap','cra','cdec','wang','hang','aang']

		if all([_x in self.data.keys() for _x in circle_keys]):
			return('circle')
		elif all([_x in self.data.keys() for _x in rect_keys]):
			return('rect')
		else:
			ut.pWARN('Unrecognised region type for region:')
			self.__repr__()
			return('undefined_region_type')
		


def _distribute(arr, collection):
	"""Assigns an array from the elements of a list-like object"""
	#print(arr, collection)
	for i, c in enumerate(collection):
		arr[i] = c
	return(arr)

def _varname2dict(varnames, local_dict):
	vardict = {}
	for varname in varnames:
		vardict[varname] = local_dict[varname]
	return(vardict)

def _read_line_from(fhdl, comment_str='#', skip_comments=True, strip_whitespace=True, debug=False, skip_empty=False):
	al = fhdl.readline()
	if(debug):
		ut.pDEBUG(al)
	if(skip_comments):
		while (al.startswith(comment_str)):
			al = fhdl.readline()
	if(skip_empty):
		while (len(al)==0):
			al = fhdl.readline()
	if(strip_whitespace):
		return(al.strip())
	return(al)

def _str2type(astr):
	"""
	Attempt to convert a string to a basic python type, will attempt bool, int, float in that order.
	If no conversion can be found, will leave as a string.

	ARGUMENTS:
		astr
			<str> String to convert

	RETURNS:
		converted_astr
			<bool|int|float|str> "astr" converted to the simplest type that will hold it.
	"""
	if astr.strip() in ('true', 'TRUE', 'True'):
		return(True)
	elif astr.strip() in ('false', 'FALSE', 'False'):
		return(False)
	else:
		try:
			return(int(astr))
		except ValueError:
			pass

		try:
			return(float(astr))
		except ValueError:
			pass

	return(astr)

def _ensure_ext(fname, ext):
	if not fname.endswith(ext):
		return(fname+ext)
	return(fname)

def _str_collect(collection, short_fmt='{: 0.4e}', sep='\t', pref='', suff='\n', long_format=None):
	"""Creates a formatted string from an arbitrary-sized list-like object"""
	n = len(collection)
	if long_format==None:
		astr = pref+sep.join([short_fmt]*n).format(*collection)+suff
	else:
		astr = pref+sep.join(long_format).format(*collection)+suff
	return(astr)

def _get_number_of_state_vector_entries_for_profile(npro, varident, varparam):
	"""
	Gets the number of state vector entries for a given parameter, identified as in section 3.1 of NEMESIS manual

	ARGUMENTS:
		npro 
			<int> The number of points in an atmospheric profile (e.g. a temperature profile).
		varident[3]
			<int,array> An array describing the identity of the profile
		varparam[]
			<float,array> An array with extra parameters for describing the retrieved variable

	RETURNS:
		xvar
			<int> The number of points associated with the variable
	"""
	# varident has 3 parts. The first number describes the type of profile, 
	# the second number describes the the sub-type of profile (for example the particle ID for clouds, or ISOGAS number for volume mixing ratios, 
	# the third number is a parameterisation code that describes how the profile is represented (see NEMESIS manual pg.12[17PDF] for a list)
	pcode = varident[2] #3rd entry of varident
	if(pcode==0):
		state_vec_entries = npro	
	elif pcode == 1:
		state_vec_entries = 2
	elif pcode == 2:
		state_vec_entries   = 1
	elif pcode == 3:
		state_vec_entries  = 1
	elif pcode == 4:
		state_vec_entries  = 3       
	elif pcode == 5:
		state_vec_entries  = 1
	elif pcode == 6:
		state_vec_entries  = 2
	elif pcode == 7:
		state_vec_entries = 2
	elif pcode == 8:
		state_vec_entries  = 3
	elif pcode == 9:
		state_vec_entries  = 3
	elif pcode == 10:
		state_vec_entries  = 4
	elif pcode == 11:
		state_vec_entries  = 2
	elif pcode == 12:
		state_vec_entries  = 3
	elif pcode == 13:
		state_vec_entries  = 3
	elif pcode == 14:
		state_vec_entries  = 3
	elif pcode == 15:
		state_vec_entries  = 3
	elif pcode == 16:
		state_vec_entries = 4
	elif pcode == 17:
		state_vec_entries = 2
	elif pcode == 18:
		state_vec_entries = 2
	elif pcode == 19:
		state_vec_entries  = 4
	elif pcode == 20:
		state_vec_entries = 2
	elif pcode == 21:
		state_vec_entries  = 2
	elif pcode == 22:
		state_vec_entries = 5
	elif pcode == 23:
		state_vec_entries = 4
	elif pcode == 24:
		state_vec_entries = 3
	elif pcode == 25:
		state_vec_entries  = int(varparam[0])
	elif pcode == 26:
		state_vec_entries = 4
	elif pcode == 27:
		state_vec_entries  = 3
	elif pcode == 28:
		state_vec_entries  = 1
	elif pcode == 32:
		state_vec_entries = 3
	elif pcode == 228:
		state_vec_entries  = 7
	elif pcode == 229:
		state_vec_entries  = 7
	elif pcode == 444:
		state_vec_entries  = 1 + 1 + int(varparam[0])
	elif pcode == 666:
		state_vec_entries  = 1
	else:
		sys.exit('ERROR: Parameterisation code {} not included in nemesis.common._get_number_of_state_vector_entries_for_profile'.format(pcode)) 
	return(state_vec_entries)

def _ensure_dict_contains(keys, adict):
	contains_list = [k in adict for k in keys]
	if not all(contains_list):
		contained = [k for k,c in zip(keys,contains_list) if c]
		not_contained = [k for k,c in zip(keys, contains_list) if (not c)]
		ut.pERROR('Expected keys {} not found in passed dictionary'.format(not_contained))
		raise IndexError

def read_line_as(fhdl, types, line_comment_char=None):
	al = fhdl.readline()
	if line_comment_char is not None:
		al = al.split('!')[0]
	words = [w.strip() for w in al.strip().split()] # ensure we have a list of non-whitespace characters
	try:
		if len(words) == len(types):
			vals = [t(w) for w,t in zip(words, types)]
		else:
			ut.pERROR('read_line_as() must have the same number of types as words in the line, or a single type to cast everything as')
			sys.exit(1)
	except TypeError:
		vals = [types(w) for w in words]

	return(vals)	

def which_band(wavlen, bands=nemesis.cfg.nir_bands):
	# assume wavlen is indexable first
	try:
		bandl=[]
		for w in wavlen:
			bandl.append(which_band(w))
	except TypeError:
		# assume wavlen is a number
		b = None
		for ab in bands.keys():
			if bands[ab]['min'] <= wavlen <= bands[ab]['max']:
				b = ab
		return(b)
	return(bandl)

def str_collect(collection, short_fmt='{: 0.4e}', sep='\t', pref='', suff='\n', long_format=None):
	"""Creates a formatted string from an arbitrary-sized list-like object"""
	n = len(collection)
	if long_format==None:
		astr = pref+sep.join([short_fmt]*n).format(*collection)+suff
	else:
		astr = pref+sep.join(long_format).format(*collection)+suff
	return(astr)
