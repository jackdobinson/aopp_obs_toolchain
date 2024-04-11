 
from contextlib import contextmanager

class NotPresent:
	pass

@contextmanager
def attributes(obj, **kwargs):
	"""
	Overwrites the attributes on an object while the context is active. Replaces them with original values on exit.
	"""
	old_attrs = {}
	for k,v in kwargs.items():
		
		if hasattr(obj, k):
			old_attrs[k] = getattr(obj, k)
			setattr(obj, k, v)
		else:
			old_attrs[k] = NotPresent
			if hasattr(obj, "__dict__"):
				setattr(obj, k, v)
			else:
				raise AttributeError(f'Attribute "{k}" not present on object of type "{obj.__class__.__qualname__}", and new attributes cannot be created on this object')
	
	yield obj
	
	for k,v in old_attrs.items():
		if v is NotPresent:
			delattr(obj,k)
		else:
			setattr(obj, k, v)
	
