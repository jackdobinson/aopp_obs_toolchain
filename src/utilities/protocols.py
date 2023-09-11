#!/usr/bin/env python3
from typing import Protocol, runtime_checkable, Union
import typing
import numbers
from abc import abstractmethod
import types
	
def hasprotocol(obj, proto):
	if (type(proto) is typing._UnionGenericAlias) and \
		all([isinstance(_x, typing.Protocol) for _x in typing.get_args(proto)]):
		return(any([isinstance(obj,_x) for _x in typing.get_args(proto)]))
	return(isinstance(obj, proto))

def protocol_desc(proto):
	ignore_keys = (	'__slots__', '_is_protocol','_is_runtime_protocol',
					'__init_subclass__','__parameters__','__abstractmethods__', 
					'_abc_impl_', '__dict__', '__weakref__', '__subclasshook__', 
					'__module__','__doc__', '_abc_impl', '__init__')
	
	slot_attrs = proto.__dict__.get('__slots__',tuple([]))
	
	dyn_attr_dict = dict([(k,v) for k,v in proto.__dict__.items() 
						   if k not in ignore_keys and not (hasprotocol(v,Callable) 
															  or type(v) is classmethod)])
	meth_dict = dict([(k,v) for k,v in proto.__dict__.items() 
					   if k not in ignore_keys and (hasprotocol(v,Callable) 
													 or type(v) is classmethod)])
	astrl = []
	astrl.append(f'PROTOCOL "{proto.__name__}"')
	astrl.append('\tattributes:')
	for k, v in dyn_attr_dict.items():
		desc = f'= {v} (default)' if type(v) not in (property, types.MemberDescriptorType) else ''
		desc += ' SLOT' if k in slot_attrs else ''
		desc += ' PROPERTY' if type(v) is property else ''
		astrl.append(f'\t\t{k} :{desc}')
	astrl.append(f'\tmethods:')
	for k, v in meth_dict.items():
		astrl.append('\t\t'
						+ construct_call_signature(v)
						+(' <ABSTRACT>' if k in proto.__dict__.get('__abstractmethods__',[]) else '')
						)
	return('\n'.join(astrl))
			
def construct_call_signature(function):
	v = function
	pd = v.__defaults__
	kd = v.__kwdefaults__
	ac = v.__code__.co_argcount
	ap = v.__code__.co_posonlyargcount
	ak = v.__code__.co_kwonlyargcount
	va = v.__code__.co_varnames
	fl = v.__code__.co_flags
	star_pos = bool(fl & 0x04)
	star_kw = bool(fl & 0x08)
	isgen = bool(fl & 0x20)
	desc = []
	desc += list(va[:ap])+['/'] if ap >0 else []
	desc += [k if (i - ac +(len(pd)+1 if pd is not None else 0)) <0 
					else f'{k}={repr(pd[i - ac +(len(pd)+1 if pd is not None else 0)])}' 
						for i,k in enumerate(va[ap:ac])]  \
					+ (['*args'] if star_pos else [])
	desc += [f'{k}={repr(kd[k])}' for k in va[ac:ac+ak]] + (['**kwargs'] if star_kw else [])
	desc = v.__name__+'('+', '.join(desc)+')' + (' <GENERATOR>' if isgen else '')
	return(desc)

@runtime_checkable
class Addable(Protocol):
	__slots__ = ('a')
	@abstractmethod
	def __add__(self, other):
		return(self + other)
	def __radd__(self, other):
		return(self+other)
	def __pos__(self, other):
		return(self+other)

@runtime_checkable	
class Multiplyable(Protocol):
	@property
	def b(self):
		raise NotImplementedError
	def __mul__(self, other):
		return(self*other)
	def __rmul__(self, other):
		return(self*other)

@runtime_checkable
class Subable(Protocol):
	def __sub__(self, other):
		return(self-other)
	def __rsub__(self, other):
		return(-self+other)
	def __neg__(self, other):
		return(-self)

@runtime_checkable
class Divideable(Protocol):
	def __div__(self, other):
		return(self/other)
	def __rdiv__(self, other):
		return(other/self)

@runtime_checkable
class Powable(Multiplyable, Protocol):
	def __pow__(self, other):
		return(self**other)

@runtime_checkable
class Cmpable(Subable, Protocol):
	__slots__=()
	def __cmp__(self,other):
		return(self - other)

@runtime_checkable
class Comparable(Protocol):
	@abstractmethod
	def __lt__(self, other):
		raise NotImplementedError
	def __le__(self, other):
		return(self < other or self == other)
	@abstractmethod
	def __eq__(self, other):
		raise NotImplementedError
	def __ge__(self, other):
		return(self > other or self == other)
	@abstractmethod
	def __gt__(self, other):
		raise NotImplementedError

@runtime_checkable
class Callable(Protocol):
	@abstractmethod
	def __call__(self):
		raise NotImplementedError


Orderable = Union[Comparable, Cmpable] #if we have one of these, the class can be ordered

if __name__=='__main__':
	for astr in map(protocol_desc, (Addable, Subable, Multiplyable, Divideable, Powable, Cmpable, Comparable, Callable)):
		print(astr)
		print()
