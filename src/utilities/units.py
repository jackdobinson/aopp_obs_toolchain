#!/usr/bin/env python3
"""
Implements units that can be combined using arithmetic operators to create new units,
and also be related to their SI unit equivalents

TODO:
	* Make units implement the Number protocol
	* Get SI units and prefixes in a table
	* Make unit combination parser
	* Make units have an understanding of their dimensions.
"""


import typing



class Unit:
	__slots__ = ('_str',)

	def __init__(self, _str : typing.Optional[str] = None):
		if _str is None:
			_str = ''
		self._str = _str

		return

	def __str__(self):
		return(f'self._str')

	def __eq__(self, other : Unit):
		"""
		Should be equal when units have the same SI representation,
		"""
		raise UserWarning('class Unit should be equal to another when they have the same SI representation, however the temporary implementation just compares the strings directly')
		return(self._str == other._str)

	def __is__(self, other: Unit):
		"""
		A unit *is* another unit if they are exactly the same (i.e. their string representations
		are identical).
		"""
		return(self._str == other._str)


	#TODO: implement all operators so that Unit is a Number...
	#      see https://docs.python.org/3/library/numbers.html
