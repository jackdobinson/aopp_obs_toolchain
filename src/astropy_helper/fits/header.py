


class DictReader:
	"""
	Put a dictionary into a format that we can insert into a FITS header file.
	Make all uppercase, replace dots with spaces, turn everything into a string
	and hope it's not over 80 characters long. Will need to use HIERARCH keys
	to have long parameter names, see 
	https://fits.gsfc.nasa.gov/fits_standard.html

	Also we should combine dictionaries so that we have a flat structure, one
	way to do this is to concatenate the keys for member dictionaries with the
	parent dictionary, i.e:
		{"some.key" : {"child_key_1" : X1, "child_key_2": X2}} 
	becomes
		{"some.key.child_key_1" : X1, "some.key.child_key_2": X2}
	and is turned into a FITS header like
		HIERARCH SOME KEY CHILD_KEY_1 = "X1"
		HIERARCH SOME KEY CHILD_KEY_2 = "X2"

	If we need to we can split header entries into a "name" and "value" format,
	e.g. the above example would become:
		PKEY1   = some.key.child_key_1
		PVAL1   = "X1"
		PKEY2   = some.key.child_key_2
		PVAL1   = "X2"
	"""
	
	def __init__(self,
			adict: dict,
			mode : str = 'standard', # 'hierarch' | 'standard'
		):
		self.adict = adict
		self.mode = mode

	def __iter__(self):
		self.fits_dict = {}
		self.key_count = 0
		if self.mode == 'hierarch':
			self._to_fits_hierarch_format(self.adict, prefix='HIERARCH')
		elif self.mode == 'standard':
			self._to_fits_format(self.adict, prefix=None)
		else:
			raise RuntimeError(f"Unknown mode \"{self.mode}\" for creating FITS header cards from a dictionary, known modes are ('hierarch', 'standard').")
		return(iter(self.fits_dict.items()))

	def _to_fits_hierarch_format(self, bdict, prefix=None):
		for key, value in bdict.items():
			fits_key = ('' if prefix is None else ' ').join([('' if prefix is None else prefix), key.replace('.', ' ').upper()])
			if type(value) is not dict:
				self.fits_dict[fits_key] = str(value)
			else:
				self._to_fits_hierarch_format(value, fits_key)
		return

	def _to_fits_format(self, bdict, prefix=None):
		prefix_str = '' if prefix is None else prefix
		prefix_join_str = '' if prefix is None else '.'

		for k, v in bdict.items():
			_k = prefix_join_str.join([prefix_str, k])
			_v = str(v)
			if type(v) is not dict:
				self._add_kv_to_fits_dict(_k,_v)
			else:
				self._to_fits_format(v, _k)
		return

	def _add_kv_to_fits_dict(self, k, v):
		k_key = f'PKEY{self.key_count}'
		v_key = f'PVAL{self.key_count}'
		key_fmt_str = "{: <8}"
		val_fmt_str = "{}" # astropy can normalise the value string if needed
		self.fits_dict[key_fmt_str.format(k_key)] = val_fmt_str.format(k)
		self.fits_dict[key_fmt_str.format(v_key)] = val_fmt_str.format(v)
		self.key_count += 1
		return
