#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import numpy as np

class Interval:
	__slots__ = ('span', 'endpoints_closed')
	def __init__(self, start=-np.inf, stop=np.inf, endpoints='[]'):
		self.span = np.array((start,stop))
		self.endpoints_closed = np.zeros((2,), dtype=bool)
		for i, char in enumerate(endpoints):
			if (i==0 and char == '[') or (i==1 and char==']'):
				self.endpoints_closed[i] = True
			elif (i==0 and char == '(') or (i==1 and char==')'):
				self.endpoints_closed[i] = False
			else:
				raise ValueError('Unknown interval endpoint description "{endpoints}", should be two of "[]()" to denote closed, or open endpoints')
		return

	@classmethod
	def from_attributes(cls, span, endpoints_closed):
		interval = cls()
		interval.span = span
		interval.endpoints_closed = endpoints_closed.astype(bool)
		return(interval)

	@classmethod
	def copy(cls, instance):
		other = cls()
		other.span = np.array(instance.span)
		other.endpoints_closed = np.array(instance.span)
		return(other)

	def frac_value(self, a):
		"""
		Gets the value that is the fraction "a" between the endpoints of the interval,
		treats a=0 and a=1 by correctly by giving the closest representable floating point
		number to the relevant endpoint that is within the interval.
		"""
		if type(a) is np.ndarray:
			# this does work, but "print()" rounds arrays, so
			# use "print(f[0])" and "print(f[1])" to see the
			# adjusted values
			f = a*self.span[1]+(1-a)*self.span[0]
			if not self.endpoints_closed[0]:
				f[a==0] = np.nextafter(self.span[0],self.span[1])
			if not self.endpoints_closed[1]:
				f[a==1] = np.nextafter(self.span[1],self.span[0])
		else:
			f = a*self.span[1]+(1-a)*self.span[0]
			if a == 0 and not self.endpoints_closed[0]:
				f = np.nextafter(f, self.span[1])
			if a == 1 and not self.endpoints_closed[1]:
				f = np.nextafter(f, self.span[0])
		return(f)

	def is_degenrate():
		return((self.span[0] == self.span[1]) and np.all(self.endpoints_closed))

	def is_empty():
		return(
			(
				(self.span[0] == self.span[1]) 
				and (
					~np.all(self.endpoints_closed) 
					or (self.endpoints_closed[0]!=self.endpoints_closed[1])
				)
			) 
			or (self.span[1] < self.span[0])
		)
	
	def __repr__(self):
		return('Interval'
			+ ('[' if self.endpoints_closed[0] else '(')
			+ f'{self.span[0]}, {self.span[1]}'
			+ (']' if self.endpoints_closed[1] else ')')
		)

	def __add__(self, a):
		other = self.copy(self)
		other.span += a
		return(other)

	def __radd__(self, a):
		return(self.__add__(a))
	
	def __sub__(self, a):
		other = self.copy(self)
		other.span -= a
		return(other)

	def __rsub__(self, a):
		return(-self + a)

	def __mul__(self, a):
		other = self.copy(self)
		other.span *= a
		return(other)

	def __rmul__(self, a):
		return(self*a)
	
	def __truediv__(self, a):
		other = self.copy(self)
		other.span = other.span / a
		return(other)

	def __floordiv__(self, a):
		other = self.copy(self)
		other.span //= a
		return(other)

	def __neg__(self):
		other = self.copy(self)
		other.span *= -1
		return(other)
		
	def __invert__(self):
		other1 = Interval.from_attributes(np.array([-np.inf,self.span[0]]), np.array([True,~self.endpoints_closed[0]]))
		other2 = Interval.from_attributes(np.array([self.span[1], np.inf]), np.array([~self.endpoints_closed[1], True]))
		return(TypeSet(other1, other2, member_type=Interval))



	def __contains__(self, item):
		is_in = True
		if self.endpoints_closed[0]:
			is_in = is_in and (self.span[0] <= item)
		else:
			is_in = is_in and(self.span[0] < item)

		if self.endpoints_closed[1]:
			is_in = is_in and (item <= self.span[1])
		else:
			is_in = is_in and (item < self.span[1])
		return(is_in)

	def range(self):
		return(self.span[1] - self.span[0])

	def sample_every(self, a):
		r = self.range()
		n = r//a if self.endpoints_closed[1] else (self.range()//a)-1
		n_dash = n+1
		start_dash = 0
		if not self.endpoints_closed[0]:
			start_dash = a
			n_dash -= 1
		
		end_dash = n*a
		if not self.endpoints_closed[1]:
			if end_dash == r:
				end_dash = (n-1)*a
		return(np.linspace(start_dash, end_dash, n_dash)+self.span[0])
	
	def sample_n(self, n):
		r = self.range()
		a = self.span[0] + (r/(n+1) if not self.endpoints_closed[0] else 0)
		b = self.span[1] - (r/(n+1) if not self.endpoints_closed[1] else 0)
		return(np.linspace(a, b, n))
	
	def get_bin_edges(self, n=None, width=None):
		if (n is None) and (width is None):
			raise ValueError(f'One of "n" and "width" must not be None')
		if n is None:
			n = (self.range()//width)
		if width is None:
			width = self.range()/n
		n_edges = n+1
		return(self.frac_value(np.linspace(0,1,n_edges)))

	def is_disjoint(self, other):
		return(not any([x in other for x in self.frac_value(np.array([0,1]))] + [x in self for x in other.frac_value(np.array([0,1]))]))


	def lowest_endpoint(self, other):
		if self.span[0] < other.span[0]:
			return(self.span[0], self.endpoints_closed[0])
		elif other.span[0] < self.span[0]:
			return(other.span[0], other.endpoints_closed[0])
		return(self.span[0], any((self.endpoints_closed[0],other.endpoints_closed[0])))

	def highest_endpoint(self, other):
		if self.span[1] > other.span[1]:
			return(self.span[1], self.endpoints_closed[1])
		elif other.span[1] > self.span[1]:
			return(other.span[1], other.endpoints_closed[1])
		return(self.span[1], any((self.endpoints_closed[1],other.endpoints_closed[1])))


	def union(self, other):
		if self.is_disjoint(other):
			return(TypeSet(self, other, member_type=Interval))
		else:
			a, af = self.lowest_endpoint(other)
			b, bf = self.highest_endpoint(other)
			return(Interval.from_attributes(np.array([a,b]), np.array([af,bf])))
	
class NoMember:
	__slots__=tuple()

class NoMethod:
	__slots__=tuple()

class ObjectSet:
	__slots__ = ('_members',)
	def __init__(self, *args):
		self._members = set(args)
		return

	def __getattr__(self, name):
		_lgr.DEBUG(f'In {self.__class__.__name__}.__getattr__({name})')
		member_list = []
		method_list = []
		for item in self._members:
			if hasattr(item, name):
				attr = getattr(item,name)
				if not callable(attr):
					member_list.append(attr)
					method_list.append(NoMethod)
				else:
					method_list.append(attr)
					member_list.append(NoMember)
			else:
				member_list.append(NoMember)
				method_list.append(NoMethod)

		output_list = []
		return(tuple(member_list), lambda *a, **k: tuple(_m(*a,**k) if _m is not NoMethod else NoMethod for _m in method_list))

	def __repr__(self):
		return(f'{self.__class__.__name__}'+'{' + ', '.join([str(item) for item in self._members]) + '}')

	def __contains__(self, item):
		_lgr.DEBUG(f'In {self.__class__.__name__}.__contains__({item})')
		a = tuple(item in _m for _m in self._members)
		return(any(a))

	def __iter__(self):
		for _m in self._members:
			yield(_m)
	
class TypeSet(ObjectSet):
	__slots__ = ('_type',)
	def __init__(self, *args, member_type=None):
		if member_type is None:
			self._type = type(args[0])
		else:
			self._type = member_type
		for arg in args:
			if not issubclass(type(arg), self._type):
				raise TypeError(f'TypeSet can only be constructed with homogenous data, expected {self._type}, encountered {type(arg)}')
		super().__init__(*args)
	
	def __getattr__(self, name):
		_lgr.DEBUG(f'In {self.__class__.__name__}.__getattr__({name})')
		members, methods = super().__getattr__(name)
		# all of the same type, so should only be members or methods
		if any([x is not NoMember for x in members]):
			return(members)
		return(methods)

	def __repr__(self):
		return(f'{self._type.__name__}Set' + '{' + ', '.join([str(item) for item in self._members]) + '}')
		

if __name__=='__main__':
	_lgr.setLevel('DEBUG')

	_lgr.DEBUG('DEBUG TESTING...')

	i1 = Interval()
	print(f'{i1=}')


	i2 = Interval(2,16)
	print(f'{i2=}')
	print(f'{i2.frac_value(0) = }')
	print(f'{i2.frac_value(1) = }')
	print(f'{i2.frac_value(0.5) = }')
	print(f'{i2.sample_n(3) = }')
	print(f'{i2.sample_every(2) = }')
	print(f'{i2.get_bin_edges(n=5) = }')
	print(f'{i2.get_bin_edges(width=2) = }')


	i3 = Interval(2,16,'()')
	print(f'{i3=}')
	print(f'{i3.frac_value(0) = }')
	print(f'{i3.frac_value(1) = }')
	print(f'{i3.frac_value(0.5) = }')	
	print(f'{i3.sample_n(3) = }')
	print(f'{i3.sample_every(2) = }')
	print(f'{i3.get_bin_edges(n=5) = }')
	print(f'{i3.get_bin_edges(width=2) = }')



	i4 = Interval(2,16,'[)')
	print(f'{i4=}')
	print(f'{i4.frac_value(0) = }')
	print(f'{i4.frac_value(1) = }')
	print(f'{i4.frac_value(0.5) = }')
	print(f'{i4.sample_n(3) = }')
	print(f'{i4.sample_every(2) = }')
	print(f'{i4.get_bin_edges(n=5) = }')
	print(f'{i4.get_bin_edges(width=2) = }')


	i5 = Interval(2,16,'(]')
	print(f'{i5=}')
	print(f'{i5.frac_value(0) = }')
	print(f'{i5.frac_value(1) = }')
	print(f'{i5.frac_value(0.5) = }')
	print(f'{i5.sample_n(3) = }')
	print(f'{i5.sample_every(2) = }')
	print(f'{i5.get_bin_edges(n=5) = }')
	print(f'{i5.get_bin_edges(width=2) = }')

	i6 = Interval(5,10)
	i7 = Interval(11,15,'(]')
	print(f'{i6 = } {i7 = }')

	iset1 = i6.union(i7)
	print(f'{iset1 = }')
