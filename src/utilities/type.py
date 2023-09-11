#!/usr/bin/env python3
from typing import * #type: ignore
import types

T1 = TypeVar('T1') # an input type of a function
R = TypeVar('R') # the return type of a function

def coerce_to_type(
		var : T1,
		atype : Type[R], # usually the name of a type also constructs that type, e.g. int('7') or str(1.289) but not always
		cast_func : Optional[Callable[[T1], R]] = None,
		fail_msg : Optional[str] = None
		) -> R:
	"""
	Takes a variable "var" and attempts to transform it to the type "atype".
	"""
	if type(var) is not atype:
		try:
			if (cast_func is None) and isinstance(atype, Union[Callable]): #type: ignore # mypy is complaining about this even though it's fine
				cast_func = atype
			else:
				raise ValueError(f'"atype" is not callable as a casting function. I.e. `{atype}({var})` is not valid code. Pass a function to "cast_func" that accepts one argument "var" and returns the desired type "atype".')
			var = cast(R, cast_func(var)) # type: ignore # mypy fails here too
		except TypeError:
			if fail_msg is None:
				fail_msg  = f'Could not coerce variable with representation {repr(var)} of type {type(var)} into type {atype}'
			raise TypeError(fail_msg)
	return(cast(R,var)) # at this point, 'var' must be of type R

def has_attrs(obj : Any, *args : Sequence[Any]) -> tuple[bool]:
	return(tuple(hasattr(obj, x) for x in args))

def is_iter(obj : Any) -> bool:
	# just having obj.__iter__ means that you can get an iterable *to* the
	# object, not that is *is* an iterator. It needs obj.__next__ also to *be* an iterator
	return(has_attrs(obj, '__iter__', '__next__'))

def is_gen_func(obj : Any) -> bool:
	# functions that create generators have bit 0x20 set in their code flags.
	return((type(obj) is types.FunctionType) and (0x20 & obj.__code__.flags))





