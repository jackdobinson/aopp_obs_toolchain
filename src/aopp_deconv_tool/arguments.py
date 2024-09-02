"""
Classes and functions for working with command-line arguments alongside the `argparse` module.
"""

import sys, os
import dataclasses as dc
import re
import typing
from typing import Any, Type
import argparse

import aopp_deconv_tool
import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')

re_empty_line = re.compile(r'^\s*$\s*', flags=re.MULTILINE)

class DataclassArgFormatter (argparse.RawTextHelpFormatter):#, argparse.MetavarTypeHelpFormatter):
	"""
	Formatter for command-line arguments found from a dataclass.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

class TypeConverterFactory:
	"""
	When instantiated with a type, returns a callable that converts a passed string to that type.
	"""
	def __init__(self, type : Type):
		"""
		Register which type the converter will convert strings to.
		"""
		self.type = type
		self.meta_type = typing.get_origin(type)
		self.types = typing.get_args(type)
	
	def __call__(self, astring : str):
		"""
		Convert `astring` to the type specified on object creation.
		"""
		if self.meta_type is None:
			return self.type(astring)
		else:
			for atype in self.types:
				return atype(astring)
		raise RuntimeError(f"Cannot convert '{astring}' to any of the supported types for {self.type}")

def parse_args_of_dataclass(
		dataclass     : Type, 
		argv          : list[str] | tuple[str], 
		show_help     : bool                   = False, 
		prog          : str                    = None, 
		description   : str                    = None, 
		arg_prefix    : str                    = '', 
		metadata_keys : list[str]              = None,
	) -> dict[str,Any]:
	"""
	Use the `argparse` package to read the fields of a DataClass. Fields that are "init" fields will be
	used as command-line arguments. A "description" entry will be looked for in a field's `metadata` dict,
	and used if it is found. Type conversion is handled by converting command-line argument string to the
	field's type as specified in the `field.type` attribute. The type conversion uses `TypeConverterFactory`.
	"""
	
	parser = argparse.ArgumentParser(
		prog=prog,
		description = re_empty_line.split(dataclass.__doc__,2)[1] if description is None else description, 
		formatter_class=DataclassArgFormatter,
		add_help=False
	)
	
	def on_parser_error(err_str):
		print(err_str)
		parser.print_help()
		sys.exit(1)
	
	parser.error = on_parser_error
	
	max_string_length = 50
	try:
		max_string_length = os.get_terminal_size().columns - 30
	except Exception:
		pass
	
	# always want a 'defaut' value, but will get it from the field, but add it here so
	# padding is correct
	if metadata_keys is None:
		metadata_keys = {'default'}
		for field in dc.fields(dataclass):
			metadata_keys.update(set(field.metadata.keys()))
		metadata_keys.discard('description') # discard this as the description does not get printed with it's key.
	else:
		metadata_keys.update({'default'})
	
	max_metadata_key_size = max(len(k) for k in metadata_keys)
	metadata_keys.discard('default') # We don't actually have a "default" entry, so discard it
	
	# Get correct format string so colons line up when metadata is printed
	metadata_fmt = '{:<'+str(max_metadata_key_size)+'} : {}'
	
	negative_bool_flag_args = []
	for field in dc.fields(dataclass):
		if field.init != True: # only include parameters that are passed to init
			continue
			
		field_default = field.default if field.default != dc.MISSING else (field.default_factory() if field.default_factory != dc.MISSING else None)
		
		field_arg_string = f'--{arg_prefix}{field.name}'
		field_help_string = aopp_deconv_tool.text.wrap(
			'\n'.join((
				field.metadata.get('description', 'DESCRIPTION NOT FOUND'),
				metadata_fmt.format('default', str(field_default)),
				*(metadata_fmt.format(k,field.metadata.get(k, f'{k.upper()} NOT FOUND')) for k in metadata_keys)
			)),
			max_string_length,
			combine_strings_of_same_indent_level = False
		)
		
		field_type_string = str(field.type)
		if field_type_string.startswith("<class '") and field_type_string.endswith("'>"):
			field_type_string = field_type_string[8:-2]
		elif field_type_string.startswith("typing.Optional[") and field_type_string.endswith(']'):
			field_type_string = 'None | ' + field_type_string[16:-1]
		else:
			field_type_string = 'UNKNOWN TYPE'
		
		
		if field.type is bool:
			arg_string = field.name
			if field_default is True:
				arg_string = 'no_'+field.name
				negative_bool_flag_args.append(arg_string)
				
			parser.add_argument(
				'--'+arg_string,
				#type=field.type, 
				action = 'store_true' if field_default is False else 'store_false',
				help= field_help_string,
				#metavar=str(field.type)[8:-2]
			)
		else:
			_lgr.debug(f'{field.name=} {field.type=}')
			parser.add_argument(
				'--'+field.name, 
				type=TypeConverterFactory(field.type), 
				default= field_default,
				help= field_help_string,
				metavar=field_type_string
			)
	
	
	args = vars(parser.parse_args(argv))
	
	if show_help:
		parser.print_help()
		sys.exit()
	
	# Remove the "no_" prefix from negative boolean flags
	for k in negative_bool_flag_args:
		args[k[3:]] = args[k]
		del args[k]
	
	return args


def construct_arglist_from_locals(
		dict_of_locals : dict[str, Any],
		n_positional_args : int = 0
	):
	"""
	Given a key-value dictionary, construct something we can pass to an argparse parser.
	Entries with `None` as their value will NOT be constructed into the list.
	"""
	dict_of_locals.update(dict_of_locals.get('kwargs',{}))
	if 'kwargs' in dict_of_locals:
		del dict_of_locals['kwargs']
	
	arglist = []
	
	for i,(k,v) in enumerate(dict_of_locals.items()):
		# Positional arguments get stuffed in without names
		if i < n_positional_args:
			arglist.append(str(v))
			continue
		# `None` means that we didn't pass anything, so don't send to the argument parser.
		if v is not None:
			# if the key ends in "_FLAG" we don't send a value, and remove the suffix
			if k.endswith('_FLAG'):
				arglist.append(f'--{k[:-5]}')
			else:
				arglist.append(f'--{k}')
				if type(v) in (list, tuple):
					arglist.extend([str(x) for x in v])
				else:
					arglist.append(str(v))
	
	_lgr.debug(f'{arglist=}')
	return arglist

