#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'WARNING')


import os
import dataclasses as dc
import functools
import enum
import string
import utilities
import array
import re

# TODO:
# * Treat TABs correctly

class TempAttrs:
	def __init__(self, obj, **kwargs):
		self.obj = obj # store reference to object
		self.stored_obj_attrs = {}
		self.temp_obj_attrs = {}
		for k in kwargs.keys():
			self.stored_obj_attrs[k] = getattr(self.obj, k)
			self.temp_obj_attrs = kwargs
		return

	def __enter__(self):
		for k, v in self.temp_obj_attrs.items():
			# set new attribute value
			setattr(self.obj, k, v)
		return(self.obj)
	
	def __exit__(self, ex_type, ex_value, traceback):
		for k, v in self.stored_obj_attrs.items():
			setattr(self.obj, k, v)
		return(False)

# lengths of strings in character lengths
# strings not in this map are assumed to have
# a character-length of 1
str_charlen_map = {
	"\t" : 4,
	"\r" : 0,
	"\n" : 0,
	"\v" : 0,
	"\f" : 0,
	"\u001b": 0,
	# already handled "\u001b" escape character, so escape strings have a length of 1 to account for separate handling of "\u001b"
	re.compile(r"\u001b\[\d+[A-Za-z]") : 1, # escape strings like "\u001b[41m" which is the escape sequence to turn text background red
}

newline_strings = ("\r\n", "\n\r", "\n", "\r")
whitespace_strings = tuple(x for x in string.whitespace)


def getCharLen(astr, str_charlen_map=str_charlen_map):
	#_lgr.DEBUG('In "getCharLen()"')
	l = len(astr)
	#_lgr.DEBUG(f'{astr=!r} {l=}')
	for k, v in str_charlen_map.items():
		#_lgr.DEBUG(f"{k=!r} {v=} {l=}")
		if type(k) == re.Pattern:
			matches = k.findall(astr)
			n = len(matches)
			x = sum((len(match) for match in matches)) # sum of the lengths of all matches
			# all matches are assumed to take up "x" characters initially, therefore we must subtract x and add v to l
			l += n*v - x
			#_lgr.DEBUG(f"{matches=} {x=} {n*v=} {l=}")
		elif k =='\t':
			n = astr.count(k)
			#_lgr.DEBUG(f"{n=}")
			if n == 0: continue
			y = astr.find(k)
			vv = 0
			while y!=-1:
				#_lgr.DEBUG(f"{y=} {vv=} {astr[:y]!r}")
				nl_pos = astr.rfind('\n', 0, y)
				nl_pos = 0 if nl_pos == -1 else nl_pos
				ll = getCharLen(astr[nl_pos:y])
				ns = ll%v
				ns = v if ns==0 else ns
				astr = astr[:y]+' '*ns+astr[y+len(k):]
				vv += ns
				#_lgr.DEBUG(f"{nl_pos=} {ll=} {ns=} {astr=!r}")
				y = astr.find(k, y+len(k))
			l += vv - n*len(k)
		else:
			n = astr.count(k)
			l += n*(v - len(k)) # they are already assumed to take up len(k) characters
		#_lgr.DEBUG(f'{n=}')
	return(l)

def getCharLenNew(astr, str_charlen_map=str_charlen_map):
	n = 0
	for i in range(len(astr)):
		nch = 0
		for k, v in str_charlen_map.items():
			if k =='\t' and astr[i]==k:
				nch = v - (n % v) # go to next tabstop
			elif type(k) == re.Pattern:
				if k.match(astr) is not None:
					nch = v
			elif astr[i:i+len(k)] == k:
				nch = v
		n += nch if nch != 0 else 1
	return(n)
	


def getCharLenIdx(astr, length, str_charlen_map=str_charlen_map):
	"""Returns the last index of astr which has a textlength of < length"""
	#_lgr.DEBUG('In "getCharLenIdx()"')
	tmp=-1
	idx_high = len(astr)
	idx_low = idx_high//2
	while idx_low != idx_high:
		#_lgr.DEBUG(f"{idx_low=} {idx_high=}")
		cl_low = getCharLen(astr[:idx_low])

		if cl_low < length:
			idx_low = (idx_low + idx_high + 1)//2
			tmp = idx_low
		else:
			if tmp!=-1:
				idx_low = 2*tmp - idx_high -1
				idx_high = tmp
			else:
				idx_high = idx_low
				idx_low = idx_high//2
	#_lgr.DEBUG(f"{idx_low=} {idx_high=}")
	#raise NotImplementedError('DEBUGGING')
	return(idx_low)


@dc.dataclass
class Splitter:
	split_types = enum.IntEnum(
		"SPLIT_TYPE", (
			"ON",
			"AFTER",
			"BEFORE",
			"HARD"
		)
	)
	str_modes = enum.IntEnum(
		"STR_MODE", (
			"RESPECT",
			"IGNORE",
		)
	)
	newline_strings : tuple = newline_strings
	split_strategy : dict = dc.field(default_factory=lambda : {
			Splitter.split_types.ON : ("\r\n", "\n\r", "\n", "\r"),
			Splitter.split_types.AFTER : tuple(x for x in "}])"+string.whitespace if x not in ("\n","\r","\t")),
			Splitter.split_types.BEFORE : ("\t",),
		}
	)
	width : int = utilities.cfg.wrap_width
	whitespace_mode = str_modes.RESPECT
	newline_mode = str_modes.RESPECT

	
	def splitPoint(self, astr):
		_lgr.DEBUG('In "splitPoint()"')
		n = len(astr)
		stop_idx = getCharLenIdx(astr, self.width)
		_lgr.DEBUG(f'{astr[:stop_idx]=}')
		
		split_idx = None
		split_str = None
		split_type = None

		for i in range(stop_idx):
			if self.newline_mode == self.str_modes.RESPECT:
				for s in self.newline_strings:
					if astr[i:i+len(s)] == s:
						_lgr.DEBUG(f'{i=} {s=} {astr[i:i+len(s)]=!r}')
						return(i, s, self.split_types.ON)


		for i in range(stop_idx,0,-1):
			for split_type, split_strings in self.split_strategy.items():
				for split_str in split_strings:
					if astr[i-len(split_str):i] == split_str:
						_lgr.DEBUG(f'{i=} {split_str=} {split_type=} {astr[i-len(split_str):i]=!r}')
						if split_type in (self.split_types.ON, self.split_types.BEFORE):
							i -= len(split_str)
							if split_type == self.split_types.BEFORE and i ==0 and getCharLen(split_str) > self.width:
								raise RuntimeError(f"Splitting Error, encountered a string we should split 'BEFORE', but splitting string is larger than splitter.width. {split_str=!r} {self.width=}")
						return(i, split_str, split_type)

		return(stop_idx, "", self.split_types.HARD)

			
	def __call__(self, astr, **kwargs):
		"""
		Want to return the start and stop indicies of each line
		"""
		_lgr.DEBUG('In "__call__()"')
		with TempAttrs(self, **kwargs) as self:

			start_idxs = array.array('i')
			end_idxs = array.array('i')
			split_strings = []
			split_types = []

			split_idx = 0
			n = len(astr)
			while split_idx < n:
				start = split_idx

				split_idx, split_str, split_type = self.splitPoint(astr[start:])
				_lgr.DEBUG(f"{split_idx=} {split_str=} {split_type=}")
				split_idx += start
				
				end = split_idx

				if split_type == self.split_types.ON:
					split_idx += len(split_str)

				if self.whitespace_mode == self.str_modes.IGNORE:
					while (split_idx < n) and (astr[split_idx] in string.whitespace):
						split_idx += 1

				_lgr.DEBUG(f"{start=} {end=} {split_str=} {split_type=}")
				start_idxs.append(start)
				end_idxs.append(end)
				split_strings.append(split_str)
				split_types.append(split_type)

		return(start_idxs, end_idxs, split_strings, split_types)

	def __iter__(self, astr, **kwargs):
		returns = self.__call__(astr,**kwargs)
		start_idxs = returns[0], end_idxs=returns[1]
		for k, (i, j) in enumerate(zip(start_idxs,end_idxs)):
			yield(astr[i,j],*returns[2:])


@dc.dataclass
class Wrapper(Splitter):
	pad_str : str = ''

	def __call__(self, astr, **kwargs):
		with TempAttrs(self, **kwargs) as self:
			starts, ends, ss, st = super().__call__(astr)
			return_lines = []
			for s, e in zip(starts,ends): # start and end string indices
				n = getCharLen(astr[s:e]) # n is character length of substring 
				_lgr.DEBUG(f'{n=} {self.width=} {astr[s:e]=}')
				padding = ''
				if self.pad_str != '' and n < self.width:
					padding = self.pad_str*((self.width-n)//len(self.pad_str)) + self.pad_str[:(self.width-n)%len(self.pad_str)]
				return_lines.append(astr[s:e]+padding)
		return(return_lines)

@dc.dataclass
class Reader:
	newline_strings : tuple = newline_strings
	whitespace_strings : tuple = whitespace_strings

	@staticmethod
	def mrk_is_wordbreak(interval, astr, break_str='-'):
		if (astr[interval.i-len(break_str):interval.i]==break_str) and (astr[interval.j+1] in string.ascii_letters):
			return(True)
		return(False)

	@staticmethod
	def mrk_is_splitword(interval, astr):
		if astr[interval.i-1:interval.i] in string.ascii_letters+'.,' and  (astr[interval.j:interval.j+1] in string.ascii_letters):
			return(True)
		return(False)

	def mrk_is_beside_whitespace(self,interval, astr):
		if (astr[interval.i-1:interval.i] in self.whitespace_strings) or (astr[interval.j:interval.j+1] in self.whitespace_strings):
			return(True)
		return(False)

	def mkr_is_besideL_strs(self, interval, astr, strings):
		for ss in strings:
			if astr[interval.i-len(ss):interval.i] == ss:
				return(True)
		return(False)
	
	def mkr_is_besideR_strs(self, interval, astr, strings):
		for ss in strings:
			if astr[interval.j:interval_j+len(ss)] == ss:
				return(True)
		return(False)

	def mkr_is_not_besideL_strs(self, interval, astr, strings):
		for ss in strings:
			if astr[interval.i-len(ss):interval.i] == ss:
				return(False)
		return(True)
	
	def mkr_is_not_besideR_strs(self, interval, astr, strings):
		for ss in strings:
			if astr[interval.j:interval_j+len(ss)] == ss:
				return(False)
		return(True)

	def mkr_is_beside_strs(self, interval, astr, strings):
		return(self.mkr_is_besideL_strs(interval, astr, strings) or self.mkr_is_besideR_strs(interval, astr, strings))

	def mkr_is_not_beside_strs(self, interval, astr, strings):
		return(self.mkr_is_not_besideL_strs(interval, astr, strings) and self.mkr_is_not_besideR_strs(interval, astr, strings))


	def action_delete_linebreak_default(self, mrk, s):
		if mrk.is_empty():
			s.delete(mrk)
		if self.mrk_is_wordbreak(mrk, s.data, '-'): 
			mrk.i-=1
			s.delete(mrk)
		if self.mrk_is_beside_whitespace(mrk,s.data): 
			s.delete(mrk)
		if self.mrk_is_splitword(mrk,s.data):
			s.delete(mrk)
			s.insert(mrk.i,' ')
			
	
	def action_highlight_in_red(self, mrk, s):
		s.insert(mrk.i,'\u001b[41m')
		s.insert(mrk.j,'\u001b[0m')


	def __call__(self, astr, **kwargs):
		# this should deal with reading text in and getting it into some
		# sort of representable format
		
		kwargs['delete'] = kwargs.get('delete',(
			lambda astr: self.mark_consecutive_str(astr, '\n', nmark=lambda nf, pch, nch: 1 if (nf==1) and (nch in string.ascii_letters) else 0),
			lambda mrk,s: self.action_delete_linebreak_default(mrk,s)
		))
		
		"""	
		# example of using highlighting on new line characters
		kwargs['highlight'] = (
			lambda astr: self.mark_consecutive_str(astr, '\n', nmark=lambda nf, pch, nch: 1 if (nf==1) and (nch in string.ascii_letters) else 0),
			lambda mrk, s: self.action_highlight_in_red(mrk,s)
		)
		"""

		marks = {}
		actions = {}
		for k, v in kwargs.items():
			if type(v) in (tuple, list):
				marks[k] = v[0](astr)
				actions[k] = v[1]
			else:
				marks[k] = v(astr)

		astr = self.handle_marks(astr, marks, **actions)

		#print('\n######################################')
		#print(repr(astr))
		#print(astr)
		return(astr)

	def handle_marks(self, astr,  marks, **kwargs):
		astr = MarkedString(astr)
		for v in marks.values():
			for m in v:
				astr.marks.append(m)
		for k, v in marks.items():
			mark_operation = kwargs.get(k, lambda mrk, astr: None) # no op, return the marked region unchanged
			for mark in reversed(v):
				mark_operation(mark, astr)
		return(astr.data)
		
		
	def mark_consecutive_str(self, astr, ss, nmark = lambda nfound, pch, nch: 1):
		#print('In "Reader.mark_consecutive_str()"')
		#print(repr(astr))
		#print(repr(ss))
		#print(repr(nmark))
		#print()
		marks = []
		slen = len(astr)
		l = len(ss)
		x, i, j = 0, 0, 0
		while (i>=0) and (j>=0):
			i,j = find_consecutive(astr[x:], ss)
			# number found
			nf = (j-i)//l
			nd = nmark(nf, astr[x+i-1] if x+i-1 < slen else '', astr[x+j] if x+j < slen else '')
			# number of characters to not mark = w
			if nf < nd:
				w = nf*l
			elif nf > nd:
				w = (nf-nd)*l
			else:
				w = 0
			#print(i,j,nf,nd)
			if x+i+w == x+j:
				# we have an empty mark, ignore it
				pass
			else:
				amark = Interval(x+i+w, x+j)
				updated_marks=False
				for i, mark in enumerate(marks):
					if amark in mark:
						mark.update_from(amark.union(mark))
						updated_marks=True
						break
				if not updated_marks:
					marks.append(amark)
			x = x+j
		return(marks)

@dc.dataclass
class MarkedString:
	data : str = ''
	marks : list = dc.field(default_factory=list)
	
	def insert(self, idx, astr):
		#print('INSERT', idx, repr(astr))
		self.data = self.data[:idx]+astr+self.data[idx:]
		for mark in self.marks:
			if mark.i > idx: mark += len(astr)
			elif mark.j >= idx: mark.j += len(astr)
	
	def delete(self, aslice):
		#print('DELETE', aslice)
		if type(aslice) is slice:
			aslice.indices(len(self))
			x, y = aslice.start, aslice.stop
		elif type(aslice) is Interval:
			x, y = aslice.i, aslice.j
		else:
			x, y = aslice
		self.data = self.data[:x]+self.data[y:]
		for mark in self.marks:
			if mark.i > x and mark.i < y: mark.i = x
			elif mark.i >= y: mark.i -= (y-x)
			if mark.j > x and mark.j < y: mark.j = x
			elif mark.j >=y: mark.j-=(y-x)
	
	def __getattr__(self, name):
		if name in dir(self.data):
			return(getattr(self.data, name))
	
	def __getitem__(self, key):
		self.data = self.data[key]
		return(self)
	
class Interval:
	num_comp_funcs = {
		'[' : lambda i,k: i<=k,
		'(' : lambda i,k: i<k,
		']' : lambda j,k: k<=j,
		')' : lambda j,k: k<j,
	}
	_i = None
	_j = None

	@classmethod
	def from_string(cls, astr, casts=(int, float)):
		co_type = astr[0]+astr[-1]
		i_str, j_str = (x.strip() for x in astr[1:-1].split(','))
		i, j= None,None
		for cast in casts:
			try:
				i = cast(i_str)
				j = cast(j_str)
				break # break out as soon as we have a valid conversion
			except (ValueError,TypeError) as ERROR:
				pass
		if (i is None) or (j is None):
			raise ValueError(f'Could not cast one of i={i_str!r}, j={j_str!r} to any of the types {casts}')
		return(cls(i,j,co_type))

	@classmethod
	def copy(cls, other):
		return(cls(other.i, other.j, other.type))

	def __init__(self, i, j, type="[]"):
		#print(i, j, type)
		self.type=type
		self.set_bounds(i,j)

	def update_from(self, other):
		self.i = other.i
		self.j=other.j
		self.type=other.type
		return


	def intersection(self, other):
		if self in other:
			return(Interval.copy(self))
		if other in self:
			return(Interval.copy(other))
		if self.i in other and self.j not in other:
			return(Interval(self.i,other.j,self.type[0]+other.type[1]))
		if self.i not in other and self.j in other:
			return(Interval(other.i,self.j,other.type[0]+self.type[1]))
		if self.i not in other and self.j not in other:
			return(Interval(0,0,'()')) # empty interval

	def union(self, other):
		if (not self.intersection(other).is_empty()) or (self.beside(other)):
			return(Interval(
					min(self.i,other.i), 
					max(self.j,other.j), 
					('[' if self.type[0]=='[' or other.type[0]=='[' else '(')+(']' if self.type[1]==']' or other.type[1]==']' else ')')
				)
			)
		return(Interval(0,0,'()'))# empty interval

	def beside(self, other):
		if other.j==self.i and ((other.type[1]+self.type[0]) in ('](',')[')):
			return(True)
		if self.j==other.i and ((self.type[1]+other.type[0]) in ('](',')[')):
			return(True)
		return(False)

	def is_empty(self):
		if self.i==self.j and self.type=='()':
			return(True)
		return(False)

	def __contains__(self, k):
		if type(k) in (int, float):
			return(self.num_comp_funcs[self.type[0]](self.i,k) and self.num_comp_funcs[self.type[1]](self.j,k))
		elif type(k) is type(self):
			return(self.__contains__(k.i) and self.__contains__(k.j))
		else:
			raise NotImplementedError(f"operator 'in' not implemented for type {type(k)} and {type(self)}")

	def __len__(self):
		return(self.j-self.i)

	def __lt__(self, other):
		if (self.type[1]==']') and (other.type[0]=='['):
			return(self.i < other.i and self.j < other.j)
		return(self.i <= other.i and self.j <= other.j)

	def __gt__(self, other):
		if self.type[0]=='[' and other.type[1]==']':
			return(self.i > other.i and self.j > other.j)
		return(self.i >= other.i and self.j >= other.j)

	def __eq__(self, other):
		if self.is_empty() and other.is_empty():
			return(True)
		return(self.type==other.type and self.i==other.i and self.j==other.j)

	def __add__(self, value):
		new = self.__class__.copy(self)
		new.set_bounds(new.i+value, new.j+value)
		return(new)
	
	def __iadd__(self, value):
		self.set_bounds(self.i+value, self.j+value)
		return
	
	def __sub__(self, value):
		return(self + -value)

	def __isub__(self, value):
		self += -value

	def set_bounds(self, i,j):
		if i > j:
			raise ValueError(f"Interval(i,j) must be passed a lower and upper bound in that order, currently have {i=} {j=}")
		self._i = i
		self._j = j
		return
		
	def get_bounds(self):
		return(self.i,self.j)

	@property
	def type(self):
		return(self._type)

	@type.setter
	def type(self, value):
		if value not in ("[]","[)","(]","()"):
			raise ValueError(f'Interval(i,j,type), type must be one of "[]","[)","(]","()" denoting closed, half-open (upper), half-open (lower), and open intervals. Currently have {type=!r}')
		else:
			self._type=value
		return

	@property
	def i(self):
		return(self._i)

	@i.setter
	def i(self,value):
		self.set_bounds(value, self.j)

	@property
	def j(self):
		return(self._j)

	@j.setter
	def j(self,value):
		self.set_bounds(self.i, value)
	
	def __repr__(self):
		return(f"{self.__class__.__qualname__}(i={self.i}, j={self.j}, type={self.type})")


class Render:
	_repr_flags_defaults = {
		'read' : True,
		'wrap' : True,
	}

	def __init__(self, reader=Reader(), wrapper=Wrapper(), **kwargs):
		self.repr_flags = Render._repr_flags_defaults
		self.repr_flags.update(kwargs)
		self.reader = reader
		self.wrapper= wrapper
		self.width = self.wrapper.width
		return

	def __call__(self, data_string, **kwargs):
		with TempAttrs(self, **kwargs) as self:
			return(self._get_rendered_string(data_string))
		
	@property
	def width(self):
		return(self.wrapper.width)
	@width.setter
	def width(self, value):
		self.wrapper.width = value
		return

	@property
	def pad_str(self):
		return(self.wrapper.pad_str)
	@pad_str.setter
	def pad_str(self, value):
		self.wrapper.pad_str = value
		return

	def _get_rendered_string(self, data_string):
		#print('='*80)
		rendered_string = data_string[:]
		if self.repr_flags['read']:
			rendered_string = self.reader(rendered_string)
		if self.repr_flags['wrap']:
			rendered_string = self.wrapper(rendered_string)	
		return(self.expand_tabs('\n'.join(rendered_string)))

	def expand_tabs(self, aline):
		# assume aline[0] is the start of a new line
		x=0
		y = aline.find('\t', x)
		while y != -1:
			nl_pos = aline.rfind('\n',0,y)
			if nl_pos == -1: nl_pos = 0
			n = getCharLen(aline[nl_pos:y])
			ns = n % str_charlen_map['\t'] # how far are we from a tab stop?
			ns = str_charlen_map['\t'] if ns == 0 else ns # go to next tab stop if neccessary
			aline = aline[:y] + ' '*ns + aline[y+1:]
			y = aline.find('\t',y+1)

		return(aline)
			

	def __str__(self):
		return(f"{self.__class__}("+', '.join([f"{str(k)}={str(v)}" for k,v in self.__dict__.items() if k[0]!='_'])+")")
		#if self.repr_flags['wrap']:
		#	return('\n'.join(self._wrapper(self._repr_string)))
		#return(self._repr_string)
	
	def __repr__(self):
		return(f"{self.__class__}("+', '.join([f"{str(k)}={str(v)}" for k,v in self.__dict__.items() if k[0]!='_'])+")")
		#if self.repr_flags['wrap']:
		#	return('\n'.join(self._wrapper(self._repr_string)))
		#return(repr(self._repr_string))



@dc.dataclass
class Symbol:
	name : str
	pattern : str # string representation of a regex pattern
	mutator : callable = lambda name, match_str: name # use this to make tokens that can vary depending on the qualities of the match
													# e.g. an INDENT token that holds the character-length of the indent

@dc.dataclass
class Token:
	name : str
	start : int
	end : int

@dc.dataclass
class SymbolSet:
	symbols : tuple
	flags : tuple(re.RegexFlag) = (re.MULTILINE,re.DOTALL)

	def __post_init__(self):
		symbol_group_patterns = []
		for symbol in self.symbols:
			symbol_group_patterns.append( f"(?P<{symbol.name}>{symbol.pattern})" )
	
		self.symbol_group_regex = re.compile('|'.join(symbol_group_patterns), flags=functools.reduce(lambda a,b: a|b, self.flags))
		return

	def tokenise(self, astr):
		tokens = []
		i = 0
		re_match = self.symbol_group_regex.search(astr, i)
		while re_match is not None:
			print(i, re_match)
			for k, v in re_match.groupdict().items():
				if v is not None:
					matching_symbol = [s for s in self.symbols if s.name==k][0]
					tokens.append(Token(matching_symbol.mutator(k, v), re_match.start(k), re_match.end(k)))
			i = re_match.end()
			re_match = self.symbol_group_regex.search(astr, i)
		return(tokens) # tokens should be in order start->end
			

@dc.dataclass
class Parser:

	def action_para_sep(self, astr, token):
		return(astr[:token.start])

	def action_contiguous_indent(self, astr, tokens):
		pass
		
	def __call__(self, astr, tokens):
		token_NONE = Token('NONE',-1,-1) # a 'does not exists' token
		token_at = lambda i: tokens[i] if i < len(tokens) else token_NONE
		mode_count = {}
		blocks = []
		EOL_STR='\n'
		PARA_STR='\n\n'
		n_list_idt = 0
		n_list_item_idt = -1
		n_list_mode = 'key'
		ostr = ''
		prev_indent = 0
		cur_indent = 0
		cur_mode = ''
		j=0
		for i, token in enumerate(tokens):
			print(j, token)
			ostr = astr[j:token.start]
			if token.name == 'PARA_SEP':
				ostr += PARA_STR
			
			if token.name == 'EOL':
				ostr +=  ' ' if astr[token.start-1] not in string.whitespace or astr[token.end] not in string.whitespace else ''
				pass
		
			if token.name.startswith('CHAR_INDENT'):
				cur_indent += int(token.name.rsplit('-')[1])
				if cur_indent != prev_indent:
					if cur_mode in ('N_LIST_START_DEF') and cur_indent > n_list_idt:
						if n_list_item_idt == -1:
							n_list_item_idt = cur_indent
						if cur_indent == n_list_item_idt:
							ostr += EOL_STR+astr[token.start:token.end]
							n_list_mode = 'key'
						if cur_indent > n_list_item_idt:
							if n_list_mode =='key': # first line of value, add newline and indent
								ostr += EOL_STR+astr[token.start:token.end]
							n_list_mode = 'value'
						if cur_indent < n_list_item_idt:
							cur_mode = ''
							n_list_item_idt = -1
						else:
							cur_indent = n_list_idt

					else:
						cur_mode = '' # reset mode
						n_list_item_idt = -1


			if token.name in ('UO_LIST_DEF','O_LIST_DEF'):
				cur_indent += token.end-token.start
				ostr += astr[token.start:token.end]
				cur_mode = token.name
				mode_count[cur_mode] = mode_count.get(cur_mode,0)+1
			
			if token.name in ('N_LIST_START_DEF',):
				ostr += astr[token.start:token.end]
				cur_mode = token.name
				mode_count[cur_mode] = mode_count.get(cur_mode,0)+1
				n_list_idt = cur_indent
			
			indent_mode = (cur_indent,cur_mode, mode_count.get(cur_mode,0))
			if len(blocks) > 0 and blocks[-1][0] == indent_mode:
				blocks[-1][1] += ostr
			else:
				blocks.append([indent_mode,ostr])	
			
			# at the start of a new paragraph or line, update indent
			if token.name == 'EOL' or token.name=='PARA_SEP':
				prev_indent = cur_indent
				cur_indent = 0
	
			j = token.end

		for indent_mode, ostr in blocks:
			print(indent_mode)
			print(ostr)
		return(blocks)
		

@dc.dataclass
class Reader2:
	"""
	Try to read things like markdown style text.
	"""
	symbols : SymbolSet = SymbolSet(
		(
			Symbol('PARA_SEP', r'\n\n'),
			#Symbol('PARA', r'.+?\n\n'),
			Symbol('EOL', r'\n{1}'),
			Symbol('UO_LIST_DEF', r"^[ \t]*[*o-] "),
			Symbol('O_LIST_DEF', r"^[ \t]*\d+\) "),
			Symbol('N_LIST_START_DEF', r"\S+:$"),
			#Symbol('N_LIST_DEF', r"(?:\S+:$)(?P<NL_IDT1>^[ \t]+).+?$((?P=NL_IDT1)(?P<NL_IDT2>[ \t]+).+?$)*"),
			#Symbol('ZERO_INDENT', r"^[ \t]{0}"),
			Symbol('CHAR_INDENT', r"^[ \t]+", lambda name, match_str: f'{name}-{getCharLenNew(match_str)}'),
			#Symbol('SPACE_INDENT', r"^ +", lambda name, match_str: f'{name}-{len(match_str)}'),
			#Symbol('TAB_INDENT', r"^\t+", lambda name, match_str: f'{name}-{len(match_str)}'),
		)
	)
	"""
		re.compile(r"(?P<BLOCK_SEP>\n\n)", flags=re.MULTILINE), # two new lines
		re.compile(r"(?P<UO_LIST>^[ \t]*[*o-] )", flags=re.MULTILINE), # bullet points "* ", "- ", "o " at the start of new lines with optional indent
		re.compile(r"(?P<O_LIST>^[ \t]*\d+\) )", flags=re.MULTILINE), # numbered lists "1) ", "2) ", etc, at the start of new lines with optional indent
		re.compile(r"(?P<INDENT>^[ \t]+)", flags=re.MULTILINE), # whitespace indent
		re.compile(r"(?P<BLOCK_SEP>\n\n)|(?P<UO_LIST>^[ \t]*[*o-] )|(?P<O_LIST>^[ \t]*\d+\) )|(?P<INDENT>^[ \t]+)", flags=re.MULTILINE),
	)
	"""

	def __call__(self, astr):

		print(self.symbols.symbol_group_regex)
		tokens = self.symbols.tokenise(astr)
		print(tokens)
		self.print_tokens_per_line(astr, tokens)
		self.print_tokens_inline(astr, tokens)

		parser = Parser()
		parser(astr, tokens)
		

	def print_tokens_inline(self, astr, tokens):
		i, j, k = 0, 0, 0
		sout = ''
		while i < len(astr):
			if k < len(tokens):
				sout += f'{astr[i:tokens[k].start]}[{tokens[k].name}({astr[tokens[k].start:tokens[k].end]})]'
				i = tokens[k].end
				k += 1
			else:
				sout += astr[i:]
				i = len(astr)
		print(sout)
		return	

	def print_tokens_per_line(self, astr, tokens):
		maxlen = 2*max(len(symbol.name) for symbol in self.symbols.symbols)
		fmtstr = '{:<'+f'{maxlen+1}'+'} |{}'
		i, j, k = 0, 0, 0
		sout = ''
		for aline in astr.splitlines(keepends=True):
			j = i + len(aline)
			print(i, j, k, tokens[k] if k < len(tokens) else None)

			name = ''
			next_token_flag = True
			while k < len(tokens) and next_token_flag:
				next_token_flag = False
				if tokens[k].start >= i and tokens[k].start < j: # if current token starts on this line
					name += '['+tokens[k].name
				if tokens[k].start < i and tokens[k].end >= j: # if current token starts before and ends after or on this line
					name += ' '*len(tokens[k].name)+'+'
				if tokens[k].end >= i and tokens[k].end < j: # if the current token ends on this line
					name += ']'
					k += 1
					next_token_flag = True

			sout += fmtstr.format(name, aline)

			i = j

		print(sout)
		return	

def find_consecutive(string, astr):
	l = len(astr)
	i = string.find(astr)
	idx0 = i
	ii = i-l
	while (i != -1) and (i-l == ii):
		ii = i
		i = string[ii+l:].find(astr)+ii+l
	return(idx0,ii+l)

def find_all(string, astr):
	l = len(astr)
	i = string.find(astr)
	j=0
	while i!=-1:
		j += i
		yield(j)
		j += l
		i = string[j:].find(astr)

def find_all_consecutive(string,astr):
	s, e = find_consecutive(string, astr)
	ss, ee = 0, 0
	while (s>0) and (e>0):
		yield(s+ee,e+ee)
		ee += e
		s,e = find_consecutive(string[ee:], astr)

	


msg = """\
Creates NEMESIS compatible *.spx files from a set of target fits cubes and NEMESIS run templates.

The `if __name__=='__main__':` statement allows execution of code if the script is called directly.
eveything else not in that block will be executed when a script is imported. 
Import statements that the rest of the code relies upon should not be in the if statement, python
is quite clever and will only import a given package once, but will give it multiple names if it
has been imported under different names.

Standard library documentation can be found at https://docs.python.org/3/library/

Packages used in this program are:
	sys
		system routines,
		useful for interacting with low-level system
		stuff
	os
	astropy

TODO:
* Write detailed tests for text.Render()
* Slowly turn this into a markdown parser :-P however that is probably something I should use a library for...
1) I would also like to be able to wrap within numbered blocks
   even when they extend over multiple lines. Maybe I can do something
   with a 'common indent'. I.e. the first word in a line matches a 
   markdown tag, I can treat all subsequent lines that have the same
   character-size indent as a single block of text.
"""


if __name__=='__main__':
	logging.setLevel(__name__, 'WARNING')
	print(utilities.cfg.wrap_width*'=')
	print(repr(msg))

	print(20*'#' + ' text.Splitter() output ' + '#'*20)
	splitter = Splitter()
	msg_splits = splitter(msg)
	for s,e in zip(msg_splits[0],msg_splits[1]):
		print(msg[s:e])

	print(20*'#' + ' text.Reader() output ' + '#'*20)
	reader = Reader()
	print(reader(msg))

	print(20*'#' + ' text.Render() output ' + '#'*20)
	text_renderer = Render()
	msg_render = text_renderer(msg, width=50, pad_str='')
	print(msg_render)

	print(20*'#' + ' text.Reader2() output ' + '#'*20)
	reader2 = Reader2()
	reader2(msg)


