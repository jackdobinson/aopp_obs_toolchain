from typing import Any
from types import TracebackType

class Alias:
	"""
	Context manager wrapper around a variable to temporarily change it's name
	"""
	slots=('ptr',)
	def __init__(self, var : Any):
		self.ptr = var
	def __enter__(self):
		return(self.ptr.val)
	def __exit__(self, etype : type, evalue : Exception, traceback : TracebackType):
		return(False)