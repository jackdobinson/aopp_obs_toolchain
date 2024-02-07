"""
Contains decorators for use in tests
"""

from typing import Any, Callable
from functools import wraps, update_wrapper, partial
import itertools as it
import inspect


def decorator(f : callable):
	@wraps(f)
	def as_decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			# the decorator has been called as `@name`
			#print("decorator has been called as `@name`")
			return f(args[0])
		else:
			# the decorator has been called as `@name(...)`
			#print("decorator has been called as `@name(...)`")
			return lambda func: f(func, *args, **kwargs)
	return as_decorator

class DecoratorClass:
	"""
	Class that all classes that are decorators should inherit from
	"""
	_debug_print_decorator_info : bool = True
	
	def __new__(cls, *args, **kwargs):
		"""
		Decorating classes are used in one of two ways
		
		1)
			```
			@DecoratorClass
			def func(...):
			```
		2)
			```
			@DecoratorClass(...)
			def func(...):
			```
		
		In case (1), this expands to `DecoratorClass(func)`, case (2) expands to
		`DecoratorClass(...)(func)`.
		
		For (1) we therefore need to detect if the class is instantiated with
		a function as its only argument, forward the remaining arguments to
		`__init__...)` and then return a callable that wraps `func`. This
		callable is the result of the `self.__decorator__(func)` method.
		
		For (2) we forward all arguments to `__init__(...)`, and return a
		callable, `self.__decorator__ in this case, that accepts `func` as its 
		only argument and wraps it.
		"""
		self = super().__new__(cls)
		
		func = None
		if (len(args) == 1) and (len(kwargs) == 0) and callable(args[0]):
			func = args[0]
			args = args[1:]
		
		self.__init__(*args, **kwargs)
		if func is not None:
			if self._debug_print_decorator_info: print('Called as @DecoratorClass, expands to `DecoratorClass(func)`')
			return self.__call__(func)
		else:
			if self._debug_print_decorator_info: print('Called as @DecoratorClass(...), expands to `DecoratorClass(...)(func)`')
			return self.__call__

	
	def __init__(self, *args, **kwargs):
		"""
		`args` and `kwargs` are forwarded only if they are not a function. 
		Therefore, can use a "normal" __init__ that only accepts inputs to the
		class. Do not have to worry about handling a callable differently from
		arguments.
		"""
		raise NotImplementedError
		#self._args = args
		#self._kwargs = kwargs
	
	def __call__(self, func):
		"""
		Accepts a callable `func`, stores it in the `self._wrapped_callable`
		attribute, and returns `__wrapper__` that forwards any argument to the
		`self.__call__` method.
		
		Example:
		
		>>> @wraps(func)
		>>> def __wrapper__(*args, **kwargs):
		>>> 	if self._debug_print_decorator_info: print('__wrapper__ wraps the __call__ method of DecoratorClass')
		>>> 	return func(*args, **kwargs)
		>>> return __wrapper__
		"""
		raise NotImplementedError

class TestSkippedException(Exception):
	pass
	

# TODO: Make this work by adding an attribute to the function instead of
# changing it's name. Then alter 'run.py' to skip when the attribute is found,
# also alter other wrappers to preserve the attributes of the function so that
# the skip is propagated correctly.
@decorator
def skip(func, 
		do_skip : bool = True, 
		predicate : Callable[[],bool] = lambda : True, 
		message : str | None = None
	):
	"""
	Marks test for skipping
	"""
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		if do_skip and predicate():
			raise TestSkippedException(message)
		func(*args, **kwargs)
	return __wrapper__

@decorator
def mark(func, mark, payload = None):
	"""
	Mark test for debugging, testing only runs this test
	"""
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		mark_actions.action_map[mark](payload)
		func(*args, **kwargs)
			
	return __wrapper__


@decorator
def debug(func):
	"""
	Mark test for debugging, testing only runs this test
	"""
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		func(*args, **kwargs)
	
	setattr(__wrapper__, 'scientest_attributes', getattr(__wrapper__, 'scientest_attributes', {}))
	__wrapper__.scientest_attributes['debug'] = True
	
	return __wrapper__


def get_base_wrapped_callable(func):
	"""
	Gets the function that is wrapped by following the chain of "__wrapped__"
	attributes.
	"""
	while hasattr(func, '__wrapped__'):
		func = func.__wrapped__
	return func

class DummyWrapper:
	"""
	Very simply wraps a function, does nothing else.
	"""
	def __init__(self, func):
		self.__wrapped__ = func
	def __call__(self, *args, **kwargs):
		pass

@decorator
def pass_args(func, *args, **kwargs):
	"""
	When given a function `func` and some arguments `*args` and `**kwargs`, 
	creates a new function in the module of the old function that, when called,
	returns the result of `func(*args, **kwargs)`.
	
	Used for feeding argument sets to tests.
	"""
	#print(f'in pass_args {args=} {kwargs=}')
	# Can chain these, so get the base function
	f_base = get_base_wrapped_callable(func)
	
	# Find out how many we have already saved
	if hasattr(f_base, '_scientest_n_variants'):
		f_base._scientest_n_variants+=1
	else:
		f_base._scientest_n_variants=0
	
	
	module = inspect.getmodule(f_base)
	
	# get a new name for the generated function, create the new function and
	# ensure it gets the correct name
	new_name = f'{f_base.__name__}_{f_base._scientest_n_variants}'
	f_new = lambda : f_base(*args, **kwargs)
	update_wrapper(f_new, f_base)
	f_new.__name__ = new_name
	
	# Add the new function to the module under the new name
	setattr(module, new_name, f_new) 
		
	return DummyWrapper(func)

@decorator
def pass_arg_sets(func, arg_sets):
	"""
	Do the same thing as `pass_args`, but assume each element of `arg_sets` is a
	set of arguments (*args,**kwargs) to pass.
	"""
	func = get_base_wrapped_callable(func)
	#print(f'in pass_arg_sets() {func} {arg_sets=}')
	for args in arg_sets:
		pass_args(*args)(func) # call decorators like this due to "@decorator" decorator
	
	return DummyWrapper(func)


@decorator
def pass_arg_combinations(func, *args, **kwargs):
	"""
	Do the same thing as `pass_args`, but assume each argument is a sequence of
	values that should be iterated over.
	"""
	func = get_base_wrapped_callable(func)
	#print(f'in pass_arg_sets() {func} {args=} {kwargs=}')
	for kwarg_set in (dict(zip(kwargs.keys(),values)) for values in it.product(*kwargs.values())):
		for arg_set in it.product(*args):
			#print(f'{arg_set=} {kwarg_set=}')
			pass_args(*arg_set, **kwarg_set)(func) # call decorators like this due to "@decorator" decorator
	
	return DummyWrapper(func)

@decorator
def argument(func, arg_name : int | str, arg_value : Any):
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		new_args = []
		# argument should be positional
		for i in range(max(len(args), arg_name+1 if type(arg_name) is int else 0)):
			if arg_name == i:
				new_args.append(arg_value)
			elif i < len(args):
				new_args.append(args[i])
			else:
				new_args.append(None)
		if type(arg_name) is str:
			# argument is keyword
			kwargs.update({arg_name : arg_value})
		return func(*new_args, **kwargs)
	return __wrapper__


@decorator
def argument_set(func, arg_name : int | str, arg_value_set : tuple[Any]):
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		for arg_value in arg_value_set:
			new_args = []
			# argument should be positional
			for i in range(max(len(args), arg_name+1 if type(arg_name) is int else 0)):
				if arg_name == i:
					new_args.append(arg_value)
				elif i < len(args):
					new_args.append(args[i])
				else:
					new_args.append(None)
			if type(arg_name) is str:
				# argument is keyword
				kwargs.update({arg_name : arg_value})
			func(*new_args, **kwargs)
	return __wrapper__
