"""
Definitions of generic types so I can type hint more easily
"""

from typing import TypeVar

T = TypeVar('T')

class NumVar(TypeVar, _root=True):
	"""
	Defines a number that can be compared to other numbers.
	
	E.g.
	
	N = NumVar('N')
	M = NumVar('M')
	
	def some_function(a : N, b : M<N):
		...
		
	"""
	def __lt__(self, other : T): pass
	def __gt__(self, other : T): pass
	def __eq__(self, other : T): pass
	def __le__(self, other : T): pass
	def __ge__(self, other : T): pass


class ShapeVar(TypeVar, _root=True):
	"""
	Defines a tuple of length N with integer elements, i.e. ShapeVar[2] is 
	equivalent to tuple[int,int].
	
	E.g.
	
	S = ShapeVar('S')
	
	(2,3,2) is an S[3]
	(1,2,3) is an S[3]
	(100,99,154) is an S[3]
	(1,2,3,4,5) is an S[5]
	(101,201) is an S[2]
	
	Q = ShapeVar('Q')
	
	Q[N] is not neccesarily the same as S[N]
	i.e. Q[2] could be (640,400), whereas S[2] could be (1280,1024)
	
	[1,*S[N]] means that the shape "S[N]" is expanded like a tuple, so
	if S[2] is (640,400), then [320,*S[N]] is (320,640,400)
	
	"""
	def __getitem__(self, item : T): 
		return tuple[T,...]
	def __iter__(self):
		pass
