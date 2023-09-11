#!/usr/bin/env python3
"""
Implements the idea that Measurements are Quantities (have a value and a unit), and that
they should also have Uncertainties (which are also Quantities, maybe a separate class?)

TODO:
	* Implement arithmetic operators that forward to the underlying class operators.

	* Try to get numpy working with everything, see 
	  <https://stackoverflow.com/questions/43493256/python-how-to-implement-a-custom-class-compatible-with-numpy-functions>
"""
import typing

from utilities.quantity import Quantity as Q

class Measurement:
	__slots__ = ('value', 'uncertainty')

	def __init__(self,
			value : Q,
			uncertainty : Q = Q(0),
		):
		self.value = value
		self.uncertainty = uncertainty
		if self.uncertainty.unit is None:
			self.uncertainty.unit = self.value.unit
		return
	
	@classmethod
	def from_veu(cls, value, error, unit):
		return(cls(Quantity(value, unit), Quantity(error, unit)))

	def __str__(self):
		if self.uncertainty.unit == self.value.unit:
			return(f'{self.value.value} +/- {self.uncertainty.value} {self.value.unit}')
		return(f'{str(self.value)} +/- {(self.uncertainty)}')

	
	#TODO: implement all operators so that Measurement is a Number...
	#      see https://docs.python.org/3/library/numbers.html
