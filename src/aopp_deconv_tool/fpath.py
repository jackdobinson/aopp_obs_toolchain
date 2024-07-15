from pathlib import PurePath, Path
import re

class FPath(PurePath):
	"""
	A field-aware abstract path (PurePath). Instantiate via FPath('path/{with}_some_{field}'). String used in instantiation will be passed to str.format(...) when used.
	
	Defaults are set via keyword arguments on construction.
	
	Use the `FPath.with_fields(**kwargs)` method to create a concrete path (Path) with **kwargs passed to str.format(...). These **kwargs overwrite defaults.
	
	"""
	
	field_re = re.compile(r'{{1}(.*?)(:*?)?}{1}', flags=re.S) # Regular expression that finds field names in passed path strings
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args)
		self._fields = []
		self._defaults = {}
		self._parts = []
		for a in args:
			self._parts.extend(PurePath(a).parts if (type(a) in (str, bytes)) else a.parts)
		self._parts = tuple(self._parts)
		
		for p in self.parts:
			if type(p) is str:
				for match in FPath.field_re.finditer(p):
					self._fields.append(match.group(1))
		self._fields = tuple(set(self._fields))
		
		# update only the fields present so we don't carry around unused data
		for field in self.fields:
			if field in kwargs:
				self._defaults[field] = kwargs[field]
	
	@property
	def fields(self):
		return self._fields
	
	@property
	def defaults(self):
		return self._defaults
	
	@defaults.setter
	def defaults(self, **kwargs):
		self._defaults = dict((field, kwargs[field]) for field in self.fields)
	
	def with_fields(self, **kwargs):
		temp_dict = {}
		temp_dict.update(self._defaults)
		temp_dict.update(kwargs)
		try:
			return Path(*(p.format(**temp_dict) for p in self._parts))
		except KeyError as e:
			e.add_note(f'All fields of FPath must have a default or set a value using `FPath.with_fields(**kwargs)`. Fields with no value for this instance: {', '.join((f'"{f}"' for f in self.fields if f not in temp_dict))}')
			raise e
	
	def __fspath__(self):
		return self.with_fields().__fspath__()