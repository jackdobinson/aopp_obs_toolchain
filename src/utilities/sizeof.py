#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import typing
try:
	from reprlib import repr
except ImportError:
	pass

def info(
		obj_dict, 
		handlers={}, 
		ignore_double_underscore=True, 
		ignore_modules=True, 
		ignore_builtins=True,
		sort_by : typing.Literal['name', 'type', 'size'] = 'name',
		sort_ascending = True,
	):
	import types
	import inspect
	hdr = ('name', 'type', 'size')
	hdr_units = ('(str)', '(str)', '(bytes)')
	if ignore_double_underscore:
		filtered_objs = filter(lambda k: not(k.startswith('__') and k.endswith('__')), obj_dict)
		obj_dict = dict((k,obj_dict[k]) for k in filtered_objs)
	if ignore_modules:
		filtered_objs = filter(lambda k: not(type(obj_dict[k]) is types.ModuleType), obj_dict)
		obj_dict = dict((k,obj_dict[k]) for k in filtered_objs)
	if ignore_builtins:
		filtered_objs = filter(lambda k: not(type(obj_dict[k]) in (types.BuiltinFunctionType, types.BuiltinMethodType)), obj_dict)
		obj_dict = dict((k,obj_dict[k]) for k in filtered_objs)

	data = [None]*len(obj_dict)
	for i, (obj_name, obj) in enumerate(obj_dict.items()):
		data[i] = (obj_name, type(obj).__name__ if not inspect.isclass(obj) else 'class' , sizeof(obj, handlers=handlers, verbose=False))
	
	clens =tuple(max(map(len,map(str,col))) for col in zip(*data))
	cfmts = tuple('{:<'+str(clen)+'}' for clen in clens)

	hdr_lines = ' | '.join((cfmts[i].format(str(h)) for i, h in enumerate(hdr)))
	tbl_line_len = len(hdr_lines)
	hdr_lines += '\n'+' | '.join((cfmts[i].format(str(h)) for i, h in enumerate(hdr_units)))
	data_lines = [None]*len(data)

	sort_idx = hdr.index(sort_by)
	data.sort(key=lambda x: x[sort_idx], reverse=not sort_ascending)
	for i, d in enumerate(data):
		data_lines[i]=' | '.join((cfmts[i].format(str(x)) for i, x in enumerate(d)))

	tbl_str = '\n'.join((hdr_lines, '-'*tbl_line_len, *data_lines))
	print(tbl_str)



def sizeof(o, handlers={}, verbose=False):
	"""
	Returns the approximate memory footprint an object and all of its contents.

	Automatically finds the contents of the following builtin containers and
	their subclasses:  tuple, list, deque, dict, set and frozenset.
	To search other containers, add handlers to iterate over their contents:

		handlers = {SomeContainerClass: iter,
					OtherContainerClass: OtherContainerClass.get_elements}
	
	Should play nice with custom classes, but if there's an error, add
	custom handlers.

	"""
	dict_handler = lambda d: chain.from_iterable(d.items())
	all_handlers = {tuple: iter,
					list: iter,
					deque: iter,
					dict: dict_handler,
					set: iter,
					frozenset: iter,
				   }
	all_handlers.update(handlers)	  # user handlers take precedence
	seen = set()					  # track which object id's have already been seen
	default_size = getsizeof(0)		  # estimate sizeof object without __sizeof__

	def _sizeof(o):
		if id(o) in seen:		# do not double count the same object
			return 0
		seen.add(id(o))
		s = getsizeof(o, default_size)

		if verbose:
			print(s, type(o), repr(o), file=stderr)

		for typ, handler in all_handlers.items():
			if isinstance(o, typ):
				s += sum(map(_sizeof, handler(o)))
				break
		else:
			if not hasattr(o.__class__, '__slots__'):
				if hasattr(o, '__dict__'):
					 # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `type(None)`) have neither
					s+=_sizeof(o.__dict__)
			else:
				# else, `o` has no attributes at all, so sys.getsizeof() actually returned the correct value
				s+=sum(_sizeof(getattr(o, x)) for x in o.__class__.__slots__ if hasattr(o, x))

		return s

	return _sizeof(o)


##### Example call #####

if __name__ == '__main__':
	_lgr.setLevel('DEBUG')

	d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
	print(sizeof(d, verbose=True))

