#!/usr/bin/env python3
"""
Utilities for dealing with arguments
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import os
import argparse
import functools
import dataclasses as dc
import typing
import inspect
import utilities as ut
import utilities.text

prefix_sep_str = '.'

# classes

class DummyClass:
	__slots__=tuple()
	@staticmethod
	def __getattr__(value):
		return(None)
	@staticmethod
	def __setattr__(name, value):
		return(None)

class HetroType:
	def __init__(self, *args):
		self._component_types = args
		self._call_counter = 0
		names = [x.__name__ for x in self._component_types]
		self.__name__ = f'({", ".join(names)})'

	def __call__(self, *args, **kwargs):
		ret = self._component_types[self._call_counter](*args, **kwargs)
		self._call_counter = (self._call_counter + 1) % len(self._component_types)
		return(ret)

class UnionType:
	def __init__(self, *args):
		self._component_types = args
		self.__name__ = '|'.join([x.__name__ for x in self._component_types])

	def __call__(self, arg):
		for atype in self._component_types:
			try:
				return(atype(arg))
			except (ValueError, TypeError) as error:
				#_lgr.DEBUG(error)
				pass
		raise ValueError(f"UnionType could not interpret argument '{arg}' as one of {self._component_types}")

	def __eq__(self, val):
		for ct in self._component_types:
			if (val==ct):
				return(True)
		return(False)
	
	def __hash__(self):
		return(self._component_types.__hash__())

class OptionalType(UnionType):
	def __init__(sels, *args):
		super().__init__(NoneTypeShadow, *args)

class NoneTypeShadowType:
	__name__ = "None"
	@staticmethod
	def __call__(value=None):
		if value in (None, 'None', 'NONE', 'none'):
			return(None)
		raise ValueError(f"NoneTypeShadow cannot convert {value} to None")
NoneTypeShadow = NoneTypeShadowType()


class ActionTF(argparse.Action):
	"""
	Creates an action that you can pass to "parser.add_argument(..., action=ActionTF, ...)" that gives the user a
	True or False switch in the form of "--show" and "--no_show".
	
	# DESCRIPTION #
	More generally the argument can take the form "--<prefix>{<true_tag>,<false_tag>}<argument>".
	Where <prefix> is an unchanging prefix to the argument, <true_tag> is present when the argument is true,
	<false_tag> is present when the argument is false, and <argument> is the rest of the argument name.
	E.g. "--plots.do_save" and "--plots.no_save", or even "--plots.save" (if <true_tag>="")
	
	Access the argument using the name you set in the "add_argument()" function.
	
	# EXAMPLE #
	```
	# an argument added with
	parser.add_argument('--plots.show', action=ActionTF, prefix='plots.', default=False, help='show (or not) plots')

	# the parsed as
	parsed_args = vars(parser.parse_args(argv))
	
	# is accessed via the 'plots.show' member
	print(parsed_args['plots.show'])
	```
	"""
	def __init__(self, option_strings, dest, default=None, required=False, help=None,
					   prefix='', true_tag='', false_tag='no_'):
		self.prefix = prefix
		self.arg_prefix = '--'+self.prefix
		self.true_tag = true_tag
		self.false_tag = false_tag
		
		if default not in (True,False,None):
			raise ValueError('True/False action requires a True, False, or None default')
		if len(option_strings)!=1:
			raise ValueError('True/False action expects only a single argument string')
			
		opt = option_strings[0]
		if not opt.startswith(self.arg_prefix):
			raise ValueError(f'True/False action expects format {prefix}<argument> option_string is {opt!r} should start with {self.arg_prefix!r}')
			
		
		# get the base option name without prefix
		opt = opt[len(self.arg_prefix):]
		opts = [self.arg_prefix + self.true_tag + opt, self.arg_prefix + self.false_tag + opt]
		super(ActionTF, self).__init__(	opts, 
										dest, 
										nargs=0, 
										const=None, 
										default=default, 
										required=required, 
										help=help,
										type=UnionType(bool,type(None)))
		return

	def __call__(self, parser, namespace, values, option_strings=None):
		if option_strings.startswith(self.arg_prefix + self.false_tag):
			setattr(namespace, self.dest, False)
		elif option_strings.startswith(self.arg_prefix + self.true_tag):
			setattr(namespace, self.dest, True)
		else:
			raise Exception(f'True/False action has unknown option strings "{option_strings}", this should never happen.')
		return

	def _get_kwargs(self):
		names = [
			'option_strings',
			'dest',
			'default',
			'help',
			'prefix',
			'true_tag',
			'false_tag'
		]
		return([(name, getattr(self, name)) for name in names])


class ActionTf(ActionTF):
	"""
	As "ActionTF()" except assumes the default to be True
	"""
	def __init__(self, option_strings, dest, required=False, help=None,
					   prefix='', true_tag='', false_tag='no_'):
			super(ActionTf, self).__init__(	option_strings, 
											dest, 
											default=True, 
											required=required, 
											help=help,
											prefix=prefix,
											true_tag=true_tag,
											false_tag=false_tag)
			return

	def _get_kwargs(self):
		names = [
			'option_strings',
			'dest',
			'help',
			'prefix',
			'true_tag',
			'false_tag'
		]
		return([(name, getattr(self, name)) for name in names])


class ActiontF(ActionTF):
	"""
	As "ActionTF()" except assumes the default to be False
	"""
	def __init__(self, option_strings, dest, required=False, help=None,
				   prefix='', true_tag='', false_tag='no_'):
		super(ActiontF, self).__init__(	option_strings, 
										dest, 
										default=False, 
										required=required, 
										help=help,
										prefix=prefix,
										true_tag=true_tag,
										false_tag=false_tag)
		return

	def _get_kwargs(self):
		names = [
			'option_strings',
			'dest',
			'help',
			'prefix',
			'true_tag',
			'false_tag'
		]
		return([(name, getattr(self, name)) for name in names])



class RawDefaultTypeFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
	pass
class RawDefaultFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
	pass
class TextDefaultTypeFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
	pass
class TextDefaultFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
	pass
class RawTextDefaultTypeFormatter(argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter, 
		argparse.MetavarTypeHelpFormatter
	):
	pass

class ParaFormatter(RawTextDefaultTypeFormatter):
	text_parser = ut.text.TextParser()
	
	def __init__(self, *args, **kwargs):
		_width = kwargs.pop('width', None)
		super().__init__(*args, **kwargs)
		if _width is not None:
			self._width=_width

	def _prep_text(self, text):
		import textwrap
		import re
		text = textwrap.dedent(text)
		prefix_newline_re = re.compile(r'\A\n*')
		text = prefix_newline_re.sub('', text)
		text = prefix_newline_re.sub('', text[::-1])[::-1]
		return(text)

	def _fill_text(self, text, width, indent):
		#return(str(self.text_parser(self._prep_text(text)).wrap(width, pad='')).replace('\t', '    '))
		return(str(self.text_parser(self._prep_text(text)).wrap(width, pad='')))

	def _split_lines(self, text, width):
		#print(f'{text=!r}')
		#print(f'{width=}')
		#for k, v in vars(self).items():
		#	print(f'{k} = {v}')
		return(self._fill_text(text,width,'').split('\n'))

class wrapper_helper:
	_slots_ = ('width','_wrapper')

	def __init__(self, width):
		self.width =width
		self._wrapper = ut.text.Splitter(width=width)
		return

	def __call__(self, astr, width=None):
		return('\n'.join(self._wrapper.wrap(astr, width=width)))


class FileArgParser(argparse.ArgumentParser):
	def convert_arg_line_to_args(self, arg_line):
		#print(arg_line.split('#',1))
		split_str = ' '
		#if '=' in arg_line:
		#	split_str = '='
		argval = arg_line.split('#',1)[0].strip()
		if (argval==''):
			args = []
		else:
			args = argval.split(split_str)
		for x in args: 
			#print(x)
			yield(x)
		#return(argval.split(split_str))

class RawArgParser(FileArgParser):
	def __init__(self, **kwargs):
		#self.wrapper = ut.str.Wrapper(preserve_existing_linebreaks=False, block_pad=False, remove_whitespace=True, suffix='\n->', width=kwargs.pop('width',79))
		self.wrapper = wrapper_helper(80)
		kwargs.update(
			dict(
				description = kwargs.get('description', ''),
				formatter_class = kwargs.get('formatter_class', RawTextDefaultTypeFormatter),
				epilog = kwargs.get('epilog', 'END OF USAGE')
			)
		)
		super().__init__(**kwargs)

class DocStrArgParser(FileArgParser):
	def __init__(self, **kwargs):
		#wrapper = lambda x: ut.str.block_indent_raw(ut.str.rationalise_newline_for_wrap(x), width=79)
		width = kwargs.pop('width',79)
		#self.wrapper = ut.str.Wrapper(preserve_existing_linebreaks=False, block_pad=False, remove_whitespace=True, width=width)
		#self.wrapper = ut.text.Splitter(width=width)
		#self._wrapper = ut.text.Splitter(width=width)
		#self.wrapper = lambda astr, width=None: self._wrapper.wrap(astr, width=width)
		self.wrapper = wrapper_helper(width=width)
		_description = kwargs.get('description', None) # has a default value of None, so have to filter it

		kwargs.update(
			dict(
				description = self.wrapper('' if _description is None else _description, width=width-8),
				#formatter_class = kwargs.get('formatter_class', RawDefaultTypeFormatter),
				formatter_class = kwargs.get('formatter_class', ParaFormatter),
				epilog = self.wrapper(kwargs.get('epilog', 'END OF USAGE'))
			)
		)
		super().__init__(**kwargs)

argparse.RawArgParser = RawArgParser
argparse.DocStrArgParser = DocStrArgParser

def with_prefix(arg_dict, prefix, strip_prefix=True, min_depth=1, max_depth=None):
	"""
	All arguments that have the prefix "prefix", down to some maximum depth (default all)
	"""
	chosen_dict = {}
	prefix_components = prefix.split(prefix_sep_str)
	for k, v in arg_dict.items():
		ks = k.split(prefix_sep_str)
		k_depth = len(ks) - len(prefix_components)
		k_in_depth = (min_depth <= k_depth) and ((k_depth <= max_depth) if max_depth is not None else True)
		if k.startswith(prefix) and k_in_depth:
			k = k if not strip_prefix else prefix_sep_str.join(ks[len(prefix_components):])
			chosen_dict[k] = v
	return(chosen_dict)
	#return(dict((k[len(prefix):] if strip_prefix else k, v) for k,v in arg_dict.items() if k[:len(prefix)] == prefix))

def of_prefix(arg_dict, prefix, strip_prefix=True):
	"""
	All arguments that only have the prefix "prefix", does not include "prefix.prefix2.argname" arguments.
	"""
	return(with_prefix(arg_dict, prefix, strip_prefix=strip_prefix, max_depth=1))

def is_like(obj, thing):
	get_origin = lambda x: x.__origin__ if hasattr(x, '__origin__') else x
	get_abstractmethods = lambda x: x.__abstractmethods__ if hasattr(x, '__abstractmethods__') else None

	obj_abc = get_origin(obj)
	if get_abstractmethods(obj_abc) is not None:
		obj_methods = frozenset.union(*tuple(x.__abstractmethods__ for x in obj_abc.__mro__[:-1]))
	else:
		obj_methods = tuple(x[0] for x in inspect.getmembers(obj_abc))

	thing_abc = get_origin(thing)
	if get_abstractmethods(thing_abc) is not None:
		thing_methods = frozenset.union(*tuple(x.__abstractmethods__ for x in thing_abc.__mro__[:-1]))
	else:
		thing_methods = tuple(x[0] for x in inspect.getmembers(thing_abc))
	#_lgr.DEBUG('\n'.join(obj_methods)+'\n')
	#_lgr.DEBUG('\n'.join(tuple(f'{j in obj_methods} {j}' for j in thing_methods))+'\n')
	return(all(z in obj_methods for z in thing_methods))

def get_action_from_field(field):
	if field.type is bool:
		if field.default == True:
			return(ActionTf)
		elif field.default == False:
			return(ActiontF)
		else:
			return(ActionTF)
	if field.type is str:
		return('store')
	if is_like(field.type, typing.MutableSequence):
		return('extend')
	return('store')

def get_nargs_from_field(field):
	if field.type is str:
		return(1)
	if is_like(field.type, typing.MutableSequence):
		return('+')
	if is_like(field.type, typing.Collection):
		if field.default is None or field.default == dc._MISSING_TYPE:
			# if the default is none, or not present, we just have to take what we get
			return('+')
		else:
			# otherwise we assume that the field default is an example of a valid entry
			return(len(field.default))

	# if we don't know, just assume the default of the action is best
	return(None)

def get_const_from_field(field):
	_lgr.DEBUG('Generally do not use "store_const" action, so not implemented for now.')
	return(None)

def get_default_from_field(field):
	_lgr.DEBUG(f'In get_default_from_field( {field.name!r} )"')
	_lgr.DEBUG(f'{field.default=} {field.default_factory=}')
	_lgr.DEBUG(f'{isinstance(field.default,dc._MISSING_TYPE)=}')
	_lgr.DEBUG(f'{isinstance(field.default_factory,dc._MISSING_TYPE)=}')
	if not isinstance(field.default, dc._MISSING_TYPE):
		return(field.default)
	if not isinstance(field.default_factory, dc._MISSING_TYPE):
		return(field.default_factory())
	return(None)


def get_type_of_annotation(type_ann, default_type):
	if type_ann is type(None):
		return(NoneTypeShadow)
	#print(type_ann)
	if type_ann in (list, tuple):
		# we can't make an informed choice, so it's the broadest type
		return(type(object))
	if hasattr(type_ann, '__args__'):
		if (str(type_ann).startswith(str(typing.Optional)) | str(type_ann).startswith(str(typing.Union))):
			return(UnionType(*(get_type_of_annotation(x, default_type) for x in type_ann.__args__)))
			
		return(HetroType(*(get_type_of_annotation(x, default_type) for x in type_ann.__args__)))
	if hasattr(type_ann, '__name__'):
		return(type_ann)

	if default_type is type(None):
		return(NoneTypeShadow)
	return(default_type)


def get_type_from_field(field):
	#if not hasattr(field.type, '__name__') or field.type in (tuple, list):
	ret_type = get_type_of_annotation(field.type, type(get_default_from_field(field)))
	#else:
	#	ret_type = field.type
	_lgr.DEBUG('In get_type_from_field()')
	_lgr.DEBUG(f'{ret_type=}')
	if field.type is not typing.Any:
		return(ret_type)
	return(str)

def get_choices_from_field(field):
	#_lgr.DEBUG('In "get_choices_from_field()"')
	#_lgr.DEBUG(f"{field.type=}")
	#_lgr.DEBUG(f"{dir(field.type)=}")
	# only really possible when type is typing.Literal[arg1,...]
	# or typing.Optional[typing.Literal[arg1,...]]
	ft = field.type
	if is_like(ft, typing.Optional) and hasattr(ft, '__args__') and is_like(ft.__args__[0], typing.Literal):
		ft = field.type.__args__[0]
	if is_like(ft, typing.Literal) and hasattr(ft, '__args__'):
		return(ft.__args__)
	# otherwise we have no information, so don't have defined choices
	return(None)

def get_required_from_field(field):
	# only fields missing defaults should be required, but I shouldn't do that anyway
	if (field.default == dc._MISSING_TYPE) and (field.default_factory ==  dc._MISSING_TYPE):
		return(True)
	return(False)

def get_help_from_field(field):
	# if the help has "description" or "help" as a part of it's metadata, use that
	for k in ('description', 'help'):
		if k in field.metadata:
			return(field.metadata[k])
	return(None)

def get_metavar_from_field(field):
	# if present in metadata, try and find it
	return(field.metadata.get('metavar',None))

def get_dest_from_field(field):
	# if present in metadata, try and find it
	return(field.metadata.get('dest',None))

def get_action_expected_kwargs(action):
	kwargs = action._get_kwargs(DummyClass())
	_lgr.DEBUG(f'{kwargs=}')
	return(tuple(x[0] for x in kwargs))

def get_field_descriptions_from_comments(adataclass):
	import inspect
	_lgr.DEBUG('In "get_field_descriptions_from_comments()"')
	field_comments = {}

	# should have sub-classes shadow their parents
	for aclass in adataclass.__mro__:
		if not dc.is_dataclass(aclass):
			continue

		src = inspect.getsource(aclass)

		field_names = tuple(x.name for x in dc.fields(adataclass))
		field_found_comments = dict((name, name in field_comments) for name in field_names)

		current_field=-1
		in_block_comment = False
		at_method_def = False
		block_comment_start = -1
		block_comment_end = -1
		last_had_line_continue = False
		for i, aline in enumerate(src.split(os.linesep)):
			# If we're in a block comment, then ignore it
			if ("'''" in aline) or ('"""' in aline):
				if not in_block_comment:
					block_comment_start = i
					block_comment_end = float('inf')
				else:
					block_comment_end = i
			if i >= block_comment_start and i <= block_comment_end:
				in_block_comment = True
			else:
				in_block_comment = False

			if in_block_comment: continue

			aline_s = aline.strip()
			aline_ss = aline_s.split()

			# only interested in dataclass variables, so ignore everything after the first "def"
			if aline_s.startswith('def'):
				# "def" is only used outside of a comment at the start of a line
				# to define a method or a function.
				at_method_def = True

			if at_method_def: break # no more dataclass variables

			# if this line has a comment in it, get the comment information
			is_comment = aline_s.startswith('#')
			has_comment = '#' in aline
			comment = None
			if has_comment:
				comment = aline_s.split('#',1)[1]

			# work out if there is a line-continuation character
			has_line_continue = False
			line_continue_pos = aline.find('\\')
			if line_continue_pos != -1:
				comment_pos = aline.find('#')
				if comment_pos ==-1 or (comment_pos > line_continue_pos):
					has_line_continue = True

			# work out if this line defines a field
			is_field = False
			for j, fn in enumerate(field_names):
				tests = (
					aline_s.startswith(fn+':'),
					aline_ss[0] == fn and aline_ss[1].startswith(':') if len(aline_ss) > 1 else False,
				)
				if any(tests):
					is_field=True
					break

			# update which field we are finding comments about
			if is_field:
				current_field = j
			if not (is_field or is_comment or last_had_line_continue) and current_field >= 0:
				current_field = -2

			if current_field >= 0 and has_comment and (not field_found_comments[field_names[current_field]]):
				field_comments[field_names[current_field]] = field_comments.get(field_names[current_field], '') + comment + '\n' # always a new line

			#print(f"{i:4} {in_block_comment:1} {is_field:1} {j:2} {current_field:2} {aline}")	
			last_had_line_continue = has_line_continue
			last_was_comment = is_comment
	return(field_comments)


def add_args_from_dataclass(parser, adataclass, prefix=None, wrapper=None, as_group=False):
	"""
	Add arguments from the fields of a dataclass to a specified parser.
	"""
	if wrapper is None and hasattr(parser, 'wrapper'):
		wrapper = parser.wrapper
	elif wrapper is None:
		wrapper = lambda x,*args,**kwargs: x
	
	
	if not dc.is_dataclass(adataclass):
		raise TypeError(f'Class {adataclass} is not a dataclass')

	parser_actions = tuple(parser._registries['action'].items())
	if prefix is None:
		if adataclass.__module__ == '__main__':
			prefix = '' + adataclass.__qualname__
		else:
			prefix = '' + prefix_sep_str.join((adataclass.__module__, adataclass.__qualname__))

	# add "prefix_sep_str" to the end of the prefix if we need it for joining with field name.
	if prefix != '' and prefix[-len(prefix_sep_str):]!=prefix_sep_str:
		join_prefix = prefix + prefix_sep_str
	else:
		join_prefix = prefix
	

	# Change parser to group parser if we want to make a group
	if as_group:
		#parser = parser.add_argument_group(prefix, wrapper(adataclass.__doc__))
		parser = parser.add_argument_group(prefix, adataclass.__doc__)

	field_comment_descs = get_field_descriptions_from_comments(adataclass)

	fields = dc.fields(adataclass)
	for i, field in enumerate(fields):
		_lgr.DEBUG(f'{field=}')
		argstr = join_prefix + field.name
		argstr = parser.prefix_chars*(2 if len(argstr) > 1 else 1) + argstr
		action = get_action_from_field(field)
		nargs = get_nargs_from_field(field)
		argdefault = get_default_from_field(field)
		argtype = get_type_from_field(field)
		choices = get_choices_from_field(field)
		required = get_required_from_field(field)
		arghelp = get_help_from_field(field)
		arghelp = arghelp if arghelp != None else field_comment_descs.get(field.name,' ')
		metavar = get_metavar_from_field(field)
		dest = get_dest_from_field(field)

		# wrap arghelp if we have a wrapper passed or in the parser
		#if wrapper is not None:
		#	arghelp = wrapper(arghelp, width=wrapper.width-24)
		#_lgr.DEBUG(f'{wrapper=}')
		#_lgr.DEBUG('arghelp=')
		#_lgr.DEBUG(repr(arghelp))

		add_argument_kwargs = dict(nargs=nargs, default=argdefault, 
			type=argtype, choices=choices, required=required, help=arghelp, 
			metavar=metavar, dest=dest
		)
		if action in (ActionTF, ActionTf, ActiontF):
			add_argument_kwargs['prefix'] = join_prefix

		# don't include "option_strings" as this is positonal in "add_argument()"
		action_expected_kwargs = get_action_expected_kwargs(parser._registries['action'].get(action, action))[1:]
		add_argument_kwargs = dict((k,v) for k,v in add_argument_kwargs.items() if k in action_expected_kwargs)

		_lgr.DEBUG(f'{add_argument_kwargs=}\n{type(add_argument_kwargs)=}')
		_lgr.DEBUG('CALL: parser.add_argument({}, action={}, {})'.format(repr(argstr), action, ', '.join(f'{k}={v}' for k,v in add_argument_kwargs.items())))
		parser.add_argument(argstr, action=action, **add_argument_kwargs)


	return(parser)

@dc.dataclass
class ParamToFieldProxy:
	name : str
	type : typing.Any = typing.Any
	default : typing.Any = dc._MISSING_TYPE()
	default_factory : typing.Callable = dc._MISSING_TYPE()
	init : bool = True
	repr : bool = True
	hash : bool = True
	compare : bool = True
	metadata : dict = dc.field(default_factory=dict)

def add_args_from_callable(parser, acallable, /,
		prefix=None, 
		wrapper=None, 
		as_group=False, 
		kind_filter = lambda p: (
			(p.default!=inspect.Parameter.empty) 
			and (
				(p.kind == inspect.Parameter.KEYWORD_ONLY) 
				or (p.kind==inspect.Parameter.POSITIONAL_OR_KEYWORD)
			)
		),
	) -> argparse.ArgumentParser :
	"""
	Add arguments from the arguments of a callable to a specified parser.
	"""
	if wrapper is None and hasattr(parser, 'wrapper'):
		wrapper = parser.wrapper
	elif wrapper is None:
		wrapper = lambda x,*args,**kwargs: x
	
	if not callable(acallable):
		raise TypeError(f'Object {acallable} is not callable')

	parser_actions = tuple(parser._registries['action'].items())
	if prefix is None:
		if adataclass.__module__ == '__main__':
			prefix = '' + adataclass.__qualname__
		else:
			prefix = '' + prefix_sep_str.join((adataclass.__module__, adataclass.__qualname__))

	# add "prefix_sep_str" to the end of the prefix if we need it for joining with field name.
	if prefix != '' and prefix[-len(prefix_sep_str):]!=prefix_sep_str:
		join_prefix = prefix + prefix_sep_str
	else:
		join_prefix = prefix
	

	# Change parser to group parser if we want to make a group
	if as_group:
		#parser = parser.add_argument_group(prefix, wrapper(acallable.__doc__.split('# ARGUMENTS #')[0]))
		parser = parser.add_argument_group(prefix, acallable.__doc__.split('# ARGUMENTS #')[0])

	#param_comment_descs = get_param_descriptions_from_comments(acallable)

	params = inspect.signature(acallable).parameters.values()
	params = tuple(x for x in inspect.signature(acallable).parameters.values() if kind_filter(x))
	for i, param in enumerate(params):
		_lgr.DEBUG(f'{param=}')
		#if not kind_filter(param): continue
		field = ParamToFieldProxy(
			name=param.name, 
			default=param.default if param.default != inspect.Parameter.empty else dc._MISSING_TYPE(), 
			type=param.annotation if param.annotation != inspect.Parameter.empty else (type(param.default) if param.default != inspect.Parameter.empty else typing.Any),
		)
		argstr = join_prefix + field.name
		argstr = parser.prefix_chars*(2 if len(argstr) > 1 else 1) + argstr
		action = get_action_from_field(field)
		nargs = get_nargs_from_field(field)
		argdefault = get_default_from_field(field)
		argtype = get_type_from_field(field)
		choices = get_choices_from_field(field)
		required = get_required_from_field(field)
		arghelp = get_help_from_field(field)
		#arghelp = arghelp if arghelp != None else field_comment_descs.get(field.name,' ')
		metavar = get_metavar_from_field(field)
		dest = get_dest_from_field(field)

		# wrap arghelp if we have a wrapper passed or in the parser
		#if (wrapper is not None) and (arghelp is not None):
		#	arghelp = wrapper(arghelp, width=wrapper.width-24)
		if arghelp is None:
			arghelp = '->'
		#_lgr.DEBUG(f'{wrapper=}')
		#_lgr.DEBUG('arghelp=')
		#_lgr.DEBUG(repr(arghelp))

		add_argument_kwargs = dict(nargs=nargs, default=argdefault, 
			type=argtype, choices=choices, required=required, help=arghelp, 
			metavar=metavar, dest=dest
		)
		if action in (ActionTF, ActionTf, ActiontF):
			add_argument_kwargs['prefix'] = join_prefix

		# don't include "option_strings" as this is positonal in "add_argument()"
		action_expected_kwargs = get_action_expected_kwargs(parser._registries['action'].get(action, action))[1:]
		add_argument_kwargs = dict((k,v) for k,v in add_argument_kwargs.items() if k in action_expected_kwargs)

		_lgr.DEBUG(f'{add_argument_kwargs=}\n{type(add_argument_kwargs)=}')
		_lgr.DEBUG('CALL: parser.add_argument({}, action={}, {})'.format(repr(argstr), action, ', '.join(f'{k}={v}' for k,v in add_argument_kwargs.items())))
		parser.add_argument(argstr, action=action, **add_argument_kwargs)


	return(parser)
	
def add_args_classes_from_module_set_as_choices(
		parser,
		module, 
		set_prefix=None, 
		set_help=None,
		set_default=None,
		as_group=True, 
		class_filter=lambda cls_name_obj_tpl: dc.is_dataclass(cls_name_obj_tpl[1])
	) -> dict :
	import inspect
	name_to_obj_map = {} 
	set_prefix = module.__name__ if set_prefix is None else set_prefix
	chosen_classes = [cls_obj for cls_name, cls_obj in 
		filter(
			class_filter, 
			inspect.getmembers(module, inspect.isclass)
		)
	]
	cls_names = tuple(cls_obj.__name__ for cls_obj in chosen_classes)
	parser.add_argument(f'--{set_prefix}', type=str, choices=cls_names, help=set_help, default=set_default)
	for cls, cls_name in zip(chosen_classes, cls_names):
		name_to_obj_map[cls_name] = cls
		add_args_from_dataclass(parser, cls, prefix=prefix_sep_str.join([set_prefix,cls_name]), as_group=as_group)
	return({set_prefix : name_to_obj_map})

def add_args_callable_set_as_choices(
		parser,
		callable_list,
		set_prefix = "callable_set",
		set_help = None,
		set_default = None,
		as_group=True,
	) -> dict :
	name_to_callable_map = {}
	callable_names = tuple(c.__name__ for c in callable_list)
	set_default = callable_names[0] if set_default is None else set_default
	parser.add_argument(f'--{set_prefix}', type=str, choices=callable_names, default=set_default, help=set_help)
	
	for acallable, acallable_name in zip(callable_list, callable_names):
		name_to_callable_map[acallable_name] = acallable
		add_args_from_callable(
			parser, 
			acallable, 
			prefix=prefix_sep_str.join([set_prefix, acallable_name]), 
			as_group=as_group
		)
	return({set_prefix : name_to_callable_map})

def output_current_parser_args_to_file(
		parser,
		args,
		fname, 
		action_filter_func=lambda action: (
			not (isinstance(action, argparse._HelpAction)
				or ('output.argfile' == action.dest)
			)
		),
	) -> None :
	#_lgr.setLevel("DEBUG") # DEBUGGING

	def get_justified_help_string(action, out_str, width=120):
		if (action.help is not None) and (action.help.strip() not in ('','->')):
			_lgr.DEBUG(f'{action.help=}')
			_lgr.DEBUG(f'{parser.wrapper(action.help,width=width-len(out_str))=}')
			help_str = ('\n'+' '*len(out_str)+'# ').join(
				parser.wrapper(' '.join(action.help.split('\n')), width=width-len(out_str)).split('\n')
			)
			return('# '+help_str)
		return('')
	def write_arg_and_help(f, out_str, action, write_help=True):
		help_str = get_justified_help_string(action, "", width=80)
		help_str = '#' + '='*79 + ('\n' if len(help_str)>0 else '') + help_str if write_help else ''
		f.write(f'{help_str}\n\n{out_str}\n')

	def write_val_sequence(f, ostr, val, action, write_help=True):
		if type(val) == type(None):
			#out_str = ' '.join([ostr, str(val)])
			out_str = '#'+' '.join([ostr, "<VALUE_PLACEHOLDER>"])
			#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
			write_arg_and_help(f, out_str, action, write_help)
			return
			
		if (type(val) != action.type) and (type(val) not in (int,float,bool,str)):
			# args were combined into a list or tuple
			# strings starting with "-" not handled well by argparse, so move them to their own line
			vstrl = []
			first_written=False
			for i, x in enumerate(val):
				if (x==float('-inf')) or ((type(x) is str) and (x[0]=='-')):
					if not first_written:
						out_str = ' '.join([ostr, ' '.join(vstrl)])
						#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
						write_arg_and_help(f, out_str, action, write_help)
						first_written = True
					else:
						f.write(f'{ostr} {" ".join(vstrl)}\n')
					f.write(f'{ostr}={x}\n')
					vstrl = []
				else:
					vstrl.append(str(x))
			if not first_written:
				out_str = ' '.join([ostr, ' '.join(vstrl)])
				#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
				write_arg_and_help(f, out_str, action, write_help)
				first_written = True
			else:
				f.write(f'{str} {" ".join(vstrl)}\n')
			return

		argjoin = ' '
		if val==float('-inf') or ((type(val) == str) and (val[0]=='-')):
			argjoin='='
		out_str = argjoin.join([ostr, str(val)])
		#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
		write_arg_and_help(f, out_str, action, write_help)
		return

	

	with open(fname, 'w') as f:
		f.write('#'*80+'\n')
		for aline in parser.format_help().split('\n'):
			f.write(f'## {aline}\n')
		f.write('#'*80+'\n')

		for action in filter(action_filter_func, parser._actions):
			f.write('\n')
			_lgr.DEBUG(f'{action=}')
			is_option = len(action.option_strings) > 0
			dest = action.dest
			_lgr.DEBUG(f'{dest=}')
			option_prefix = ('' if not is_option else ('-' if len(dest)==1 else '--'))
			val = args[dest]
			_lgr.DEBUG(f'{val=} {type(val)=} {action.type=}')
			ostr = option_prefix + dest

			if not is_option:
				# just put positional arguments at the front
				out_str = f'# {ostr}\n{val}\n'
				write_arg_and_help(f, out_str, action)
				continue

			if isinstance(action, argparse._CountAction):
				if args[dest]>0:
					out_str = f'{option_prefix}{dest*val}'
					#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
					write_arg_and_help(f, out_str, action)
				continue

			if isinstance(action, ut.args.ActionTF):
				out_str = ostr
				#f.write(f'{out_str}{get_justified_help_string(action, out_str)}\n')
				write_arg_and_help(f, out_str, action)
				continue

			if isinstance(action, argparse._StoreConstAction):
				out_str = ostr
				write_arg_and_help(f, out_str, action)
				continue

			if isinstance(action, argparse._AppendAction):
				if (val is None) or (len(val) == 0): # write a placeholder value for zero length lists
					write_val_sequence(f, ostr, None, action)
				else:
					for i, item in enumerate(val):
						write_val_sequence(f, ostr, item, action, write_help=i==0)
				continue
				
			write_val_sequence(f, ostr, val, action, write_help=True)


		return
	


				

			

		

if __name__=='__main__':
	# Testing #
	import sys
	if '' in sys.path: sys.path.remove('')
	import fitscube.deconvolve.algorithms
	parser = argparse.RawArgParser()
	add_args_from_dataclass(parser, fitscube.deconvolve.algorithms.LucyRichardson, prefix='algorithm.lucy-richardson', as_group=True)
	#print(parser.format_help())
	
