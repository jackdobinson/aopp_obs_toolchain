
import sys, os
import dataclasses as dc
import re
from typing import Any, Type
import argparse

import aopp_deconv_tool

re_empty_line = re.compile(r'^\s*$\s*', flags=re.MULTILINE)

class DataclassArgFormatter (argparse.RawTextHelpFormatter):#, argparse.MetavarTypeHelpFormatter):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


def parse_args_of_dataclass(
		dataclass     : Type, 
		argv          : list[str] | tuple[str], 
		show_help     : bool                   = False, 
		prog          : str                    = None, 
		description   : str                    = None, 
		arg_prefix    : str                    = '', 
		metadata_keys : list[str]              = None,
	) -> dict[str,Any]:
	
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
	
	max_string_length = os.get_terminal_size().columns - 30
	
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
		
		parser.add_argument(
			'--'+field.name, 
			type=field.type, 
			default= field_default,
			help= field_help_string,
			metavar=str(field.type)[8:-2]
		)
	
	
	args = parser.parse_args(argv)
	
	if show_help:
		parser.print_help()
		sys.exit()
	
	return vars(args)