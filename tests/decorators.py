from functools import wraps

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
	
