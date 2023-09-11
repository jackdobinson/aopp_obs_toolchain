#!/usr/bin/env python3
"""
Useful decorators
"""
import functools


def code_block(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		return(func(*args, **kwargs))
	wrapper()
	return(wrapper)


def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
		

	
if __name__=='__main__':
	from classes import Pointer
	
	def test_function():
		v = 8
		a = [1,2,3]
		a_iter = iter(a)
		
		@code_block
		def BLOCK(b = a):
			nonlocal v # binds to the enclosing scope variable when the function is created
			print(f'{id(b)=}')
			print(f'{id(v)=}')
			print(f'{b=}')
			print(f'{v=}')
			
			v = 1
			print(f'{b=}')
			print(f'{v=}')
			
		print(a)
		print(v)
		v = 6
		print(v)
		print()
		def tf2():
			v = 100.0
			a = 7
			print(v, id(v))
			BLOCK() # still uses previous "v" variable here
			print(v)
			
		tf2()
		print(v)
	test_function()
