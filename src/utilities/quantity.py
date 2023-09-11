#!/usr/bin/env python3
"""
Implements the idea that values should have units attached or else they are just numbers

TODO
	* Implement arithmetic operators (forward to underlying type of _value and _unit)
"""

import typing

from utilities.units import Unit as U

class Quantity:
	__slots__ = ('_value', '_unit')

	def __init__(self,
			value : typing.Any, 
			unit_str : typing.Optional[str] = None,
		):
		self.value = value # hold anything in *value*
		self.unit = U(unit_str)
		return
	
	@property
	def value(self):
		return(self._value)
	
	@value.setter
	def unit(self, value):
		self._value = _value
		return

	@property
	def unit(self):
		return(self._unit)
	
	@unit.setter
	def unit(self, _unit):
		self._unit = _unit
		return

	def __str__(self):
		return(f'{self.value} {self.unit}')

	#TODO: implement all operators so that Quantity is a Number...
	#      see https://docs.python.org/3/library/numbers.html
