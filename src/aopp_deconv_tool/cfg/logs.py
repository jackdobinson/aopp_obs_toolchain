"""
Logging setup for the package
"""
import sys
import types
import logging

pkg_name = __name__.split('.',1)[0]
pkg_lgr = logging.getLogger(pkg_name)
pkg_lgr.propagate = False

pkg_lgr.setLevel(logging.DEBUG)

pkg_stream_hdlr = logging.StreamHandler()
pkg_stream_hdlr.setLevel(logging.DEBUG)

pkg_stream_hdlr_formatter = logging.Formatter(
	fmt="%(asctime)s %(filename)s:%(lineno)d \"%(funcName)s\" %(levelname)s: %(message)s",
	datefmt="%Y-%m-%dT%H:%M:%S %z",
)
pkg_stream_hdlr.setFormatter(pkg_stream_hdlr_formatter)


pkg_lgr.addHandler(pkg_stream_hdlr)



excepthook_lgr = logging.getLogger(f'{pkg_name}.excepthook')
def log_excepthook(type, value, traceback):
	"""
	Override for `sys.excepthook`.
	"""
	
	# Go through the traceback and find out
	# where the exception was raised initially.
	tb = traceback
	while tb.tb_next is not None:
		tb = tb.tb_next
	
	# Use the original source of the exception as the source of the log.
	co = tb.tb_frame.f_code
	
	rcrd_fac = lambda msg: logging.LogRecord(excepthook_lgr.name, logging.ERROR, co.co_filename, tb.tb_lineno, msg, {}, None,func=co.co_qualname)
	
	rcrd = rcrd_fac(value)
	excepthook_lgr.handle(rcrd)
	#root_lgr.error(value, exc_info=(type, value, traceback), stack_info=True, stacklevel=5)
	if hasattr(value, '__notes__'):
		for note in value.__notes__:
			rcrd = rcrd_fac(f'NOTE: {note}')
			excepthook_lgr.handle(rcrd)
	sys.__excepthook__(type, value, traceback)

# Override except hook so we can log all uncaught exceptions
sys.excepthook = log_excepthook


def get_logger_at_level(name : str | types.ModuleType, level : str|int = logging.NOTSET) -> logging.Logger:
	""":
	Return the `name`d logger that reports `level` logs
	"""
	if type(name) is types.ModuleType:
		name = name.__name__
		
	if name.split('.',1)[0] != pkg_name:
		raise RuntimeError(f'Trying to get logger "{name}" but logger is not a child of "{pkg_name}"')
		
	_lgr = logging.getLogger(name)
	_lgr.setLevel(level)
	return _lgr

def set_logger_at_level(name : str | types.ModuleType, level : str|int = logging.NOTSET) -> None:
	"""
	Set the logger with `name` to report `level` logs
	"""
	if type(name) is types.ModuleType:
		name = name.__name__
	
	if name.split('.',1)[0] != pkg_name:
		raise RuntimeError(f'Trying to set level of logger "{name}" but logger is not a child of "{pkg_name}"')
	
	get_logger_at_level(name, level)
	return None
