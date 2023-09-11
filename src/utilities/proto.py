#!/usr/bin/env python3

from typing import Protocol, runtime_checkable, Union
from abc import abstractmethod
import typing

@runtime_checkable
class Addable(Protocol):
	@abstractmethod
	def __add__(self, other):
		raise NotImplementedError
	def __radd__(self, other):
		return(self + other)




if __name__=='__main__':
	pass