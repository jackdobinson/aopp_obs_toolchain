"""
Logging setup for the package
"""
import sys
import types
import logging

logging.basicConfig(
	format="%(asctime)s %(filename)s:%(lineno)d \"%(funcName)s\" %(levelname)s: %(message)s",
	datefmt="%Y-%m-%dT%H:%M:%S %z",
	force=True
)

excepthook_lgr = logging.getLogger('excepthook')
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
	
	_lgr = logging.getLogger(name)
	_lgr.setLevel(level)
	return _lgr

def set_logger_at_level(name : str | types.ModuleType, level : str|int = logging.NOTSET) -> None:
	"""
	Set the logger with `name` to report `level` logs
	"""
	if type(name) is types.ModuleType:
		name = name.__name__
	get_logger_at_level(name, level)
	return None
