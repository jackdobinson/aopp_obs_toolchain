from contextlib import contextmanager
from typing import Iterator, Any
from types import TracebackType

class Next:
	"""
	Context manager wrapper around the next(iter) function, to be used in
	when separating blocks of code by indent level would be useful
	"""
	slots=('el',)
	def __init__(self, iterator : Iterator[Any]):
		self.el = next(iterator)
	def __enter__(self):
		return(self.el)
	def __exit__(self, etype : type, evalue : Exception, traceback : TracebackType):
		return(False)