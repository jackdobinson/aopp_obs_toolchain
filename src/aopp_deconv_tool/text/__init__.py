"""
Text wrapping, filling, formatting, indenting etc.
"""

import dataclasses as dc
from collections import namedtuple
from typing import Iterable
import textwrap

import aopp_deconv_tool.cast as cast

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


newline='\n'
tab='\t'

def indent(x : str, n : int, s : str = tab):
	"""
	Indent all lines in `x` by `n` copys of string `s`
	"""
	return textwrap.indent(x, n*s, lambda _: True)

def to_tuple(x : str, t : type[type] | Iterable[type]):
	"""
	Converts a text representation of a tuple to a tuple of the specified type.
	
	Arguments:
		x : str
			string to convert to a tuple
		t : type[type] | Iterable[type]
			Type of tuple to convert to. If `t` is not an iterable, then will
			convert to a homogeneous tuple, otherwise will convert each element
			of `x` to the corresponding element of `t`
	"""
	assert type(x) == str
	
	# if x ends with a comma, remove it
	if x.endswith(','):
		x = x[:-1]
	
	if isinstance(t, type):
		# assume tuple homogeneous
		try:
			return tuple(cast.to(y,t) for y in x.split(','))
		except Exception as e:
			e.add_note(f"string '{x}' could not be cast to a homogeneous {t} tuple")
			raise
	
	assert len(t) == x.count(',')+1, "Must have same number of tuple entries as types"
	try:
		return tuple(cast.to(y,u) for y,u in zip(x.split(','), t))
	except Exception as e:
		e.add_note(f"string '{x}' could not be cast to a tuple of types {t}")
		raise
	
@dc.dataclass
class BracketState:
	"""
	Class for holding the depth of nested brackets when parsing text.
	"""
	r : int = 0 # round
	c : int = 0 # curly
	s : int = 0 # square
	
	def update(self, c : str):
		"""
		Given a character `c` will update the nested bracket depth counters
		"""
		if c == '[': self.s+=1; return
		if c == ']': self.s-=1; return
		if c == '{': self.c+=1; return
		if c == '}': self.c-=1; return
		if c == '(': self.r+=1; return
		if c == ')': self.r-=1; return
	
	def total(self) -> int:
		"""
		Returns the sum of the bracket counters.
		"""
		return self.r + self.c + self.s


def split_around_brackets(x : str, sep : str = ','):
	"""
	Split on a separator `sep`, do not split on any separator inside any types
	of bracket.
	"""
	bstate = BracketState(0,0,0)
	stack = []
	idx = 0
	for i, c in enumerate(x):
		_lgr.debug(i,c)
		if c == sep and bstate.total()==0: 
			stack.append(x[idx:i])
			idx = i+1
			continue
		bstate.update(c)
	stack.append(x[idx:])
	return(stack)
