"""
Contains routines for casting to and from types. Generally useful for simple data marshalling to and from text
"""

from typing import Iterable, Any

class CastException(Exception):
	"""
	If casts fail, throw this exception
	"""
	pass


def to(x : Any, atype : type) -> type:
	"""
	Casts an object to a type
	
	Arguments:
		x : any
			Object to cast
		atype : type
			type to cast to
	
	Returns:
		x casted to `atype`, throws CastException on failure.
	"""
	try:
		return atype(x)
	except:
		raise CastException(f"Cannot cast type {type(x)} = '{x}' to type {atype}")

def to_any(x : Any, types : Iterable[type]) -> Any:
	"""
	Casts and object to any of the given types.
	
	Arguments:
		x : Any
			Object to cast
		types : Iterable[type]
			Iterable (e.g. list) of the types to attempt to cast to. Once one
			cast succeeds, the others are not attempted.
	
	Returns:
		`x` casted to the first type that succeeds. Throws CastException on failure.
	"""
	for atype in types: 
		try:
			return(to(x, atype))
		except CastException:
			continue
	raise CastException(f"Cannot cast type {type(x)} = '{x}' to any of the types {tuple(t for t in types)}")
