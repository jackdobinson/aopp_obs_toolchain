from typing import Any
from functools import wraps, update_wrapper

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
			#return self.__decorator__(func)
			return self._wrap_callable(func)
		else:
			if self._debug_print_decorator_info: print('Called as @DecoratorClass(...), expands to `DecoratorClass(...)(func)`')
			#return self.__decorator__
			return self._wrap_callable
	
	def _wrap_callable(self, acallable):
		if self._debug_print_decorator_info: print(f'Wrapping {acallable=}')
		self._wrapped_callable = acallable
		update_wrapper(self, self._wrapped_callable)
		if self._debug_print_decorator_info: print(f'{self=} {inspect.isfunction(self)=} {callable(self)=} {inspect.isclass(self)=} {inspect.ismethod(self)=}')
		return self
	
	def __wrapper__(self, *args, **kwargs):
		if self._debug_print_decorator_info: print('__wrapper__ wraps the __call__ method of DecoratorClass')
		return self(*args, **kwargs)
			
	def __decorator__(self, func):
		"""
		Accepts a callable `func`, stores it in the `self._wrapped_callable`
		attribute, and returns `__wrapper__` that forwards any argument to the
		`self.__call__` method.
		"""
		self._wrapped_callable = func
		update_wrapper(self.__wrapper__, func)
		#@wraps(func)
		#def __wrapper__(*args, **kwargs):
		#	if self._debug_print_decorator_info: print('__wrapper__ wraps the __call__ method of DecoratorClass')
		#	return self(*args, **kwargs)
		return self.__wrapper__
	
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
	
	def __call__(self, *args, **kwargs):
		"""
		Is called instead of the wrapped function `func`. The wrapped function
		is stored in the `self._wrapped_callable` attribute.
		"""
		raise NotImplementedError


class TestSkippedException(Exception):
	pass
	
	
@decorator
def skip(func, do_skip : bool = True):
	@wraps(func)
	def __wrapper__(*args, **kwargs):
		if do_skip:
			raise TestSkippedException
		func(*args, **kwargs)
	return __wrapper__


class pass_args(DecoratorClass):
	def __init__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs
	
	def __call__(self, *args, **kwargs):
		print(f'in self.__call__(...) {args=} {kwargs=}')
		


	
		
		
		


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
