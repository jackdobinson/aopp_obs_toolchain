#!/usr/bin/env python3
"""
Setting up logging so it's not quite as horrific as it usually is

# USAGE #
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'WARNING')

import other_module
# set logging level for "other_module" when running as script
if __name__=='__main__': logging.setLevel('other_module', 'INFO')
# or you can use the module object itself
if __name__=='__main__': logging.setLevel(other_module, 'INFO')

_lgr.INFO('Some message to log')
"""
import logging
import os, string, shutil, sys
import importlib, site

import utilities.text
import utilities.str


# GLOBALS
default_logging_level = logging.NOTSET

class IndentLogFormatter(logging.Formatter):
	def __init__(self, 
			            fmt :  str ='{message}', 
			        datefmt :  str = None, 
			          style :  str = '{', 
			       validate : bool = True, 
			         indent :  str = None, 
			 hanging_indent : bool = True, 
			auto_indent_str :  str = '-',
			     indent_end :  str = ':',
			  indent_suffix :  str = ' ',
			          width :  int = 0,
		) -> None:
		super().__init__(fmt, datefmt, style, validate)
		# there is only one instance, so don't write to attributes outside of __init__
		self.default_time_format='%Y-%m-%dT%H:%M:%S'
		self.default_msec_format='%s.%03d'
		self.indent=indent
		self.hanging_indent=hanging_indent
		self.auto_indent_str=auto_indent_str
		self.indent_end = indent_end if indent_end is not None else ''
		self.indent_suffix=indent_suffix
		self.width = width

		"""
		# used with an alternate "wrapMessageAt" method
		self.wrapper = utilities.str.Wrapper(
			width = width, 
			split_on_strings = tuple(), 
			split_after_strings = tuple(x for x in '}])'+string.whitespace),
			split_before_strings = tuple(),
			split_hard_dashes = tuple(x for x in string.ascii_letters),
			block_pad=False
		)
		"""

		self.msg_parser = utilities.text.TextParser()
	
		return
	
	def format(self, record):
		full_message = super().format(record)
		message = record.getMessage() # don't change this one, we need it to replace data in full message
		message_copy = message[:] # alter this one as much as I like

		# get and/or construct the indent
		if type(self.indent) is not str:
			if self.indent is None:
				hdr = full_message[:full_message.index(message)]
				indent_size = max([len(_h.rstrip()) for _h in hdr.split('\n')])

			elif type(self.indent) is int:
				indent_size = self.indent

			n = indent_size - len(self.indent_end)
			nf = n // len(self.auto_indent_str)
			np = n % len(self.auto_indent_str)
			indent=self.auto_indent_str*(nf)+self.auto_indent_str[:np]+self.indent_end+self.indent_suffix
		else:
			indent = self.indent

		# if a non-zero width is given, chop up the message into sub-messages such that
		# an indent + sub_message is always less than or equal to self.width
		if self.width > 0:
			message_copy = self.wrapMessageAt(message, self.width-len(indent))

		# apply the indent
		message_copy = message_copy.replace(os.linesep,'\n{indent}'.format(indent=indent))
		if not self.hanging_indent:
			message_copy = '{indent}'.format(indent=indent)+message_copy
		return(full_message.replace(message, message_copy))
	
	def formatTime(self, record, datefmt=None):
		return(super().formatTime(record,datefmt))

	"""
	def wrapMessageAt(self, message, width, remove_whitespace=True):
		# uses utilities.str for the wrapping
		return(self.wrapper(message, width, remove_whitespace=remove_whitespace))
	"""

	def wrapMessageAt(self, message, width, remove_whitespace=True):
		return(str(self.msg_parser(message).wrap(width)))

def change_default_formatter(formatter):
	root_logger = logging.getLogger('root')
	handlers = root_logger.handlers
	# remove previous handlers/formatters
	for hdlr in handlers:
		root_logger.removeHandler(hdlr)
	hdlr = logging.StreamHandler()
	hdlr.setFormatter(default_formatter)
	root_logger.addHandler(hdlr)
	return	
	
msg_fmts = dict(
	full_fmt = '{asctime:s} -> line {lineno:d} in "{filename:s}" in {funcName:s}()\n{levelname:-<8s}: {message:s}',
	short_fmt = '{levelname: <4.4s} {filename}-{lineno:d} {funcName:s}(): {message:s}',
	two_line_fmt = '{levelname: <4.4s} {filename}-{lineno:d} {funcName:s}():\n{message:s}',
	vshort_fmt = '{levelname} {module}-{lineno}: {message}',
)

#default_formatter = IndentLogFormatter(fmt=full_fmt, indent='--------: ')
#default_formatter = IndentLogFormatter(fmt=short_fmt, indent=None)
#default_formatter = IndentLogFormatter(fmt=two_line_fmt, indent='----: ', hanging_indent=False)
#default_formatter = IndentLogFormatter(fmt=two_line_fmt, indent=4, hanging_indent=False)
#default_formatter = IndentLogFormatter(fmt=vshort_fmt, indent=None, hanging_indent=True)
default_formatter = IndentLogFormatter(fmt=msg_fmts['vshort_fmt'], indent=None, hanging_indent=True, width=shutil.get_terminal_size().columns)


# set the default formatter
change_default_formatter(default_formatter)





# get loggers for other modules, and set their level. Also return the 'logging' module
# call this in other modules as
# >>> import utilities.logging_setup
# >>> logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'level=WARNING')
def getLoggers(name, level=logging.NOTSET):
	if type(name) is str:
		named_logger = logging.getLogger(name)
		named_logger.setLevel(level)
		# set up some aliases, easier to see in code
		named_logger.DEBUG = named_logger.debug
		named_logger.INFO = named_logger.info
		named_logger.WARN = named_logger.warning
		named_logger.ERROR = named_logger.error
		named_logger.CRIT = named_logger.critical
		return(logging, named_logger)
	# in this case, assume we have a sequence of names, not just a single name
	named_loggers = []
	for aname in name:
		_logging, named_logger = getLoggers(aname, level)
		named_loggers.append(named_logger)
	return(logging, named_loggers)


#-----------------------------------------------------------------------------
# helper functions attached to "logging" module
import types

def setLevel(name, level=default_logging_level):
	lgr = logging.getLogger(name.__name__ if type(name) is types.ModuleType else name)
	lgr.setLevel(level)
	return

def setLevelExcept(names, level=default_logging_level, only_on_python_path=True, exclude_builtins=True):
	import sys, os
	if type(names) not in (tuple, list):
		names = [names]
	names = list(map(lambda n: n.__name__ if type(n) is types.ModuleType else n, names))
	get_module_name = lambda m: m.__name__
	get_module_path = lambda m:	(m.__spec__.origin 
		if (
			hasattr(m, "__spec__") 
			and (m.__spec__ is not None) 
			and (m.__spec__.origin is not None)
		) else (
			m.__path__[0]
		) if (
			hasattr(m, "__path__")
		) else (
			''
		)
	)
	get_module_is_builtin = lambda m: m.__spec__.origin == 'built-in' if m.__spec__ is not None else False
	
	filtered_modules = tuple(filter(lambda m: type(m) is types.ModuleType, sys.modules.values()))
	#print(filtered_modules)
	filtered_modules = tuple(filter(lambda m: not(get_module_name(m) in names), filtered_modules))
	#print(filtered_modules)
	if exclude_builtins:
		filtered_modules = tuple(filter(lambda m: not get_module_is_builtin(m), filtered_modules))
	#print(filtered_modules)
	if only_on_python_path:
		filtered_modules = tuple(filter(lambda m: (get_module_path(m).startswith(os.environ["PYTHONPATH"])), filtered_modules))
	#print(filtered_modules)

	logger_names_to_set = [module.__spec__.name if module.__spec__ is not None else module.__name__ for module in filtered_modules]
	for logger_name in logger_names_to_set:
		lgr = logging.getLogger(logger_name)
		lgr.setLevel(level)

	return

	

logging.setLevel = setLevel
logging.change_default_formatter = change_default_formatter
logging.setLevelExcept = setLevelExcept

# set up root logger
root_logger = logging.getLogger('root')
root_logger.setLevel(level=logging.WARNING)

_lgr = getLoggers(__name__)[1] # create a logger for the current module
_lgr.setLevel(level=logging.INFO) # set the logging level for this module, overwrites the global logging level for this module only


def add_arguments(parser, prefix=__name__, suppress_help=False):
	"""
	Adds arguments that control logging to a parser
	"""
	import argparse

	def add_argument(*args, **kwargs):
		if suppress_help:
			kwargs['help'] = argparse.SUPPRESS
		parser.add_argument(*args, **kwargs)
		return

	add_argument(f'--{prefix}.disable', action='store_true', help='If present will disable logging')
	add_argument(f'--{prefix}.set_level', type=str, action='append', help='The level of logging to set the associated modules to', default=[])
	add_argument(f'--{prefix}.modules', type=str, nargs='+', action='append', help='The modules to have their level of logging set by the associated "--logging.set_level" argument, these should be module names. A special value of "*" will set all user defined modules to the specified logging level.', default=[])
	add_argument(f'--{prefix}.to_file', type=str, help='File to log to, pass a filename ending in a ".number" (e.g. "script.log.4") to enable log rotation.', default=None)

	return(parser)

def handle_arguments(args):
	if args['disable']:
		logging.disable()
	levels = args['set_level']
	module_lists = args['modules']

	if args['to_file'] is not None:
		try:
			n = int(args['to_file'].rsplit('.',1)[1])
		except:
			n = None
		logging_handler = logging.FileHandler(args['to_file'], mode='w') if n is None else logging.handlers.RotatingFileHandler(args['to_file'].rsplit('.',1)[0], mode='w', backupCount=n, maxBytes=2^24) 
		logging.getLogger().addHandler(logging_handler)

	for modules, level in zip(module_lists, levels):
		for module in modules:
			if module == '*':
				logger_names = [name for name in logging.root.manager.loggerDict]
				loggers = [logging.getLogger(name) for name in logger_names]
				specs = [importlib.util.find_spec(name) for name in logger_names]
				is_user_module_flags = []
				non_user_prefixes = [sys.prefix, site.USER_SITE] + site.PREFIXES
				for spec in specs:
					if (spec.origin is not None) and (spec.origin != "built-in"):
						is_user_module_flags = not any([os.path.abspath(spec.orign).startswith(os.path.abspath(x)) for x in non_user_prefixes ])
				for logger, is_user_module_flag in zip(loggers, is_user_module_flags):
					if is_user_module_flag:
						logger.setLevel(level)
			else:
				logging.getLogger(module).setLevel(level)
	



if __name__=='__main__':
	# test logging here

	_lgr.WARN("Astropy does not understand the FITS conventions of long-string keywords (i.e. the CONTINUE keyword) and the ESO-developed HIERARCH keyword. Astropy has therefore been set to warn users when a FITS file does not conform to it's standards, but NOT to fix any errors it detects. If any warnings astropy is giving you are of the form \"Card 'CONTINUE' is not FITS standard...\", then those warnings can be safely ignored. See [the FITS standard](https://fits.gsfc.nasa.gov/fits_standard.html) for more information.")

	_lgr.CRIT(
		'\n'.join((
		'idx',
		'	Index of a list/array etc.',
		'len',
		'	Length in number of elements',
		'clen',
		'	Length of a string in number of printed characters',
	))
	)


