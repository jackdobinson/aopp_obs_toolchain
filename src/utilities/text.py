#!/usr/bin/env python3

# As this file is used in "utilities.logging_setup" I can't use routines from that file in this one...
#import utilities.logging_setup
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')


import sys, os
import dataclasses as dc
import functools
import enum
import string
import array
import re
import typing
import itertools as it
import numpy as np

import utilities as ut
import utilities.cfg


"""
idx
	Index of a list/array etc.
len
	Length in number of elements
clen
	Length of a string in number of printed characters

"""

# lengths of strings in character lengths
# strings not in this map are assumed to have
# a character-length of 1
str_charlen_map = {
	None : 1, # if a character/string is not in this dictionary, this is it's length
	"\t" : lambda cur_line_clen, tab_size=ut.cfg.tab_size: tab_size-cur_line_clen%tab_size if cur_line_clen%tab_size!=0 else tab_size,
	"\r" : 0,
	"\n" : 0,
	"\v" : 0,
	"\f" : 0,
	"\u001b": { # we have more than one possibility for strings that start with this character
		None: 0, # If we don't match anything else in this dictionary, this is it's length
		re.compile(r"\u001b\[\d+[A-Za-z]") : 0, # escape strings like "\u001b[41m" which is the escape sequence to turn text background red
	},
}

newline_strings = ("\r\n", "\n\r", "\n", "\r")
whitespace_strings = tuple(x for x in string.whitespace)


def get_next_char_len(astr, cur_line_clen, sc_map = str_charlen_map, default_char=None):
	default = sc_map.pop(None,1)
	default_len = 1
	for k, v in sc_map.items():
		match  = k.match(astr) if type(k) == re.Pattern else astr.startswith(k)
		if match:
			next_char = match.group() if type(k) == re.Pattern else k
			if type(v) is dict:
				return(get_next_char_len(astr, cur_line_clen, sc_map = v, default_char=next_char))
			elif callable(v):
				return(v(cur_line_clen), next_char)
			elif type(v) is int:
				return(v, next_char)
			else:
				raise ValueError(f'{type(v)=}, must be one of (dict, callable, int)')
	return(default, astr[0] if default_char is None else default_char)

def get_str_char_len(astr, sc_map = str_charlen_map):
	i = 0
	clen = 0
	while i < len(astr):
		#print(f'{i=} {clen=}')
		char_clen, char = get_next_char_len(astr[i:], clen, sc_map = sc_map)
		clen += char_clen
		i+= len(char)
	#print(f'{i=} {clen=}')
	return(clen)
	
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
		

def regex_replace_all(astr, regex, rstr, group=0):
	if type(regex) is str:
		regex = re.compile(regex)
	match = regex.search(astr)
	while match is not None:
		replacement_str = rstr if not callable(rstr) else rstr(match)
		astr = regex_match_replace(astr, match.span(group), replacement_str)
		match = regex.search(astr, pos=match.start(group)+len(replacement_str))
	return(astr)

def regex_match_replace(astr, match_start_stop, rstr):
	return(astr[:match_start_stop[0]] + rstr + astr[match_start_stop[1]:])


def term_colour_code(n):
	if n >= 0 and n <= 255:
		return(u"\u001b[38;5;"+str(n)+'m')
		
def term_reset_all_code():
	return(u"\u001b[0m")

def term_reset_bgc_code():
	return('\x1b[49m')

def term_reset_fgc_code():
	return('\x1b[39m')


def term_bgcolour_code(n):
	if n >= 0 and n <= 255:
		return(u"\u001b[48;5;"+str(n)+'m')


@dc.dataclass
class Splitter:
	str_charlen_map : typing.Mapping = dc.field(default_factory = lambda :str_charlen_map)
	newline_strings : typing.Sequence[str] = newline_strings
	whitespace_strings : typing.Sequence[str] = whitespace_strings
	can_lbrk_strings : typing.Sequence[str] = (')',']','}')
	width : int = ut.cfg.wrap_width
	special_width : np.ndarray = np.zeros((0,2), dtype=int)

	def get_split_idxs(self, astr):
		#_lgr.DEBUG('Splitter::__call__()')
		split_idx_stack = [0,len(astr)]
		line_clen_stack = []
		cur_line_clen = 0
		last_potential_split_idx = None
		to_last_potential_split_idx_clen = 0

		i = 0
		while i < len(astr):
			width = self.special_width[np.argwhere(self.special_width[:,0]==i)][0,0,1] if (i in self.special_width[:,0]) else self.width

			next_char_clen, next_char = get_next_char_len(astr[i:], cur_line_clen, sc_map = self.str_charlen_map)
			#_lgr.DEBUG(f'{i=} {next_char_clen=} {next_char=!r} {cur_line_clen=}\n{width=} {to_last_potential_split_idx_clen=}')

			if cur_line_clen + next_char_clen > width:
				#_lgr.DEBUG('SPLIT due to width exceeded')
				if last_potential_split_idx is None: 
					last_potential_split_idx = i

				line_clen_stack.append(cur_line_clen - to_last_potential_split_idx_clen)
				split_idx_stack.insert(-1, last_potential_split_idx)
				

				cur_line_clen = to_last_potential_split_idx_clen
				last_potential_split_idx = None
				from_last_potential_split_idx_clen = 0
				
				#print(astr[slice(*split_idx_stack[-2:])],'\n')

			if ((next_char in self.whitespace_strings) or (next_char in self.can_lbrk_strings)) and (next_char not in self.newline_strings):
				#_lgr.DEBUG('update last potential split index')
				last_potential_split_idx = i + len(next_char)
				to_last_potential_split_idx_clen = -next_char_clen

			if next_char in self.newline_strings:
				#_lgr.DEBUG('SPLIT on existing newline')
				_split_idx = i +len(next_char)
				if _split_idx != len(astr):
					line_clen_stack.append(cur_line_clen)
					split_idx_stack.insert(-1, i+len(next_char))

					last_potential_split_idx = None
					to_last_potential_split_idx_clen = 0
					cur_line_clen = -next_char_clen

			to_last_potential_split_idx_clen += 0 if last_potential_split_idx is None else next_char_clen
			cur_line_clen += next_char_clen

			i += len(next_char)

		line_clen_stack.append(cur_line_clen)
		return(split_idx_stack, line_clen_stack)
	
	def split(self, astr, width=None, special_width=None):
		if width is not None: self.width = width
		if special_width is not None: self.special_width = special_width
		split_idx_stack, line_clen_stack = self.get_split_idxs(astr)
		#_lgr.DEBUG(f'{split_idx_stack=}')
		for i, j in zip(split_idx_stack[:-1], split_idx_stack[1:]):
			yield(astr[i:j])
	
	def wrap(self, astr, width=None, pad='', special_width=None):
		#print(f'{astr=}\n{width=}\n{pad=}\n{special_width=}')
		if width is not None: self.width=width
		if special_width is not None: self.special_width = special_width
		split_idx_stack, line_clen_stack = self.get_split_idxs(astr)
		#_lgr.DEBUG(f'{split_idx_stack=}')
		#_lgr.DEBUG(f'{line_clen_stack=}')
		#_lgr.DEBUG(f'{len(split_idx_stack)} {len(line_clen_stack)}')
		#_lgr.DEBUG(f'{self.special_width=}')
		for k, (i, j) in enumerate(zip(split_idx_stack[:-1], split_idx_stack[1:])):
			_width = self.special_width[np.argwhere(self.special_width[:,0]==i)][0,0,1] if (i in self.special_width[:,0]) else self.width
			#_lgr.DEBUG(f'{self.width=} {_width=}')
			#_lgr.DEBUG(f'{astr[i:j]=!r}')
			yield_str = (astr[i:j-1] if ((i!=j) and (astr[i:j][-1] == '\n')) else astr[i:j]) + (_width-line_clen_stack[k])*pad
			#_lgr.DEBUG(f'{yield_str=!r}')

			yield(yield_str)




@dc.dataclass
class IndentedBlock:
	char_indent : int = -1
	idx_indent : int = -1
	special_indent : list[tuple[int, str]] = dc.field(default_factory=list)
	data : list[str] = dc.field(default_factory=list)

	def __str__(self, wrapper=None, tab_size=4):
		astrl = None

		if wrapper != None:
			astrl = list(wrapper(''.join(self.data)))

		astrl = self.data.splitlines(True) if astrl is None else astrl

		j = 0
		for i in range(len(astrl)):
			if j < len(self.special_indent) and self.special_indent[j][0]==i:
				indent_str = self.special_indent[j][1]
				j += 1
			else:
				indent_str = ((self.char_indent//tab_size)*'\t' + (self.char_indent%tab_size)*' ') if (tab_size != 0) else (self.char_indent*' ')
			astrl[i] = indent_str + astrl[i]
			
		
		return(''.join(astrl))

@dc.dataclass
class RegexLexer:
	default_regex_dict = {
		###########################################################################
		# ## EXAMPLE OF INPUT ##
		#
		# Modified regex patterns, use "[KEY_NAME]" to match a previous regex,
		# be sure to make "blank named group" to match the bit of the pattern that
		# corresponds to the key we operate on.
		#
		# I.e. if we only want a subset of the regex to be identified with a token
		# wrap it in "(?P<>...)" and will programatically substutute for the name
		# later.
		#
		# E.g. 'SECOND_WORD' : r'\w+\s+(?P<>\w+)'
		# will be expanded to r'\w+\s+(?P<SECOND_WORD>\w+)', i.e. we don't have to 
		# write the name twice.
		###########################################################################


		# TERMINAL PATTERNS
		'LINE_START' 		: r'(?P<>(?m:^))',
		'LINE_END' 			: r'(?P<>(?m:$))',
		'INLINE_WS' 		: r'(?P<>[ \t])',
		'TEXT_CHARS'		: r'(?P<>[a-zA-Z,.])',
		'CLOSE_BRACKETS'	: r'(?P<>[])}])',
		'OPEN_BRACKETS'		: r'(?P<>[\[({])',
		'TEXT_WORD' 		: r'(?P<>[a-zA-Z]+)',
		'DOUBLE_QUOTED_STR' : r'(?P<>".+?")',
		'SINGLE_QUOTED_STR' : r"(?P<>'.+?')",
		'BACK_QUOTED_STR' 	: r'(?P<>`.+?`)',
		'DIGITS' 			: r'(?P<>\d+)',
		'SIGN'				: r'(?P<>[+-])',
		'NEW_LINE_CHAR'		: r'(?P<>\n)',
		'UO_LIST_START_RAW' : r'(?P<>[-o*] )',

		# NON-TERMINAL PATTERNS
		'SEPARATOR' 		: r'(?P<>([INLINE_WS])|([,]))',
		# have to use lookahead and lookbehind patterns as otherwise
		# the pattern will consume the separator and not count adjacent matches
		# as separate matches (as they technically 'share' the separator.
		'L_SEP' 			: r'(?P<>(?<=[LINE_START])|(?<=[SEPARATOR]))',
		'R_SEP' 			: r'(?P<>(?=[SEPARATOR])|(?=[LINE_END]))',


		'SINGLE_LBRK'		: r'(?P<>((?<![NEW_LINE_CHAR])(?<!\A))[NEW_LINE_CHAR]((?![NEW_LINE_CHAR])(?!\Z)))',
		'EXPANDABLE_LBRK'	: r'(?<=[TEXT_CHARS]|[CLOSE_BRACKETS])(?P<>[SINGLE_LBRK])(?=[TEXT_CHARS])',
		'EMPTY_LINE' 		: r'[LINE_START](?P<>\s*?\n)',
		'INDENT' 			: r'[LINE_START](?P<>[INLINE_WS]*)',
		'UO_LIST_START' 	: r'[INDENT](?P<>[UO_LIST_START_RAW])',
		'NUM_LIST_START_RAW': r'(?P<>[DIGITS](\.[DIGITS])*[])] )',
		'NUM_LIST_START' 	: r'[INDENT](?P<>[NUM_LIST_START_RAW])',
		'INTEGER_RAW' 		: r'(?P<>[SIGN]?[DIGITS])',
		'INTEGER' 			: r'[L_SEP](?P<>[INTEGER_RAW])[R_SEP]',
		'FLOAT_RAW' 		: r'(?P<>[INTEGER_RAW]((\.\d+([eE][INTEGER_RAW])?)|(\.)|([eE][INTEGER_RAW])))',
		'FLOAT' 			: r'[L_SEP](?P<>[FLOAT_RAW])[R_SEP]',
		'NUMBER_RAW' 		: r'(?P<>[FLOAT_RAW]|[INTEGER_RAW])',
		'NUMBER' 			: r'[L_SEP](?P<>[NUMBER_RAW])[R_SEP]',
		'COMPLEX_RAW' 		: r'(?P<>([NUMBER_RAW])[INLINE_WS]*[+-][INLINE_WS]*([NUMBER_RAW])[ij])',
		'COMPLEX' 			: r'[L_SEP](?P<>[COMPLEX_RAW])[R_SEP]',
	}
		
	
	# May have to make this into an ordered dict
	regex_dict : dict = dc.field(default_factory = lambda : {**RegexLexer.default_regex_dict})
	
	def __post_init__(self):
		or_str = '('+')|('.join(tuple(self.regex_dict.keys()))+r')' # matches any "regex_dict" key
		cr_str = r'\[(?P<TOKEN_NAME>'+or_str+')\]' # grabs the "regex_dict" key name from a modified regex
		nr_str = r'\(\?P<>.*\)' # finds "blank named group", will be modified so the "regex_dict" key is the group name
		mr_str = r'\(\?P<(?P<GROUP_NAME>\w+)>'
		#_lgr.DEBUG(f'{or_str=}')
		#_lgr.DEBUG(f'{cr_str=}')
		#_lgr.DEBUG(f'{nr_str=}')
		combiner_regex = re.compile(cr_str)
		namer_regex = re.compile(nr_str)
		mutator_regex = re.compile(mr_str)

		#i = 0
		for k, v in self.regex_dict.items():
			#_lgr.INFO(f'{k}\n\t{v}')
			"""
			for match in list(combiner_regex.finditer(v))[::-1]:
				_lgr.DEBUG(match)
				# do I want to surround the substituted regex in a non-capturing group?
				matched_regex_str = self.regex_dict[match["TOKEN_NAME"]]
				#_lgr.DEBUG(matched_regex_str)
				v = regex_match_replace(v, match.span(), matched_regex_str)
				#i+=1
			"""
			v = regex_replace_all(v, combiner_regex, lambda match: self.regex_dict[match["TOKEN_NAME"]])
			
			"""
			for match in namer_regex.finditer(v):
				_lgr.DEBUG(match)
				v = regex_match_replace(v, match.span(), f'(?P<{k}>{match[0][5:-1]})')
			"""
			v = regex_replace_all(v, namer_regex, lambda match: f"(?P<{k}>{match[0][5:]}")

			self.regex_dict[k] = v

		# ensure that each named group has a unique name
		# alter repeated group names to avoid collisions
		for k, v in self.regex_dict.items():
			#_lgr.INFO(f'{k}\n\t{v}')
			idx_tracker = {}
			match = mutator_regex.search(v, pos=0)
			while match is not None:
				match_str = match['GROUP_NAME']
				n = len(match_str)
				if match_str != k:
					match_replacement_str = f'{match_str}_{idx_tracker.get(match_str, 0)}'
					n = len(match_replacement_str)
					v = regex_match_replace(v, match.span('GROUP_NAME'), match_replacement_str)
					idx_tracker[match_str] = idx_tracker.get(match_str, 0) + 1
				match = mutator_regex.search(v, pos=match.start()+n)
			self.regex_dict[k] = v


		for k, v in self.regex_dict.items():
			#_lgr.INFO(f'{k}\n\t{v}')
			#print(f'{k}\n\t{v}')
			self.regex_dict[k] = re.compile(v)
		return


	def __call__(self, astr, tokens=None, colour_mstr=None):
		tokens = self.regex_dict.keys() if tokens==None else tokens

		#_lgr.INFO('\n'+'-'*10)
		#for k, v in self.regex_dict.items():
		for k in tokens:
			v = self.regex_dict[k]
			#_lgr.INFO(f"{k}\n\t'{v.pattern!s}'")

		#_lgr.INFO('\n'+'-'*10)
		matched_intervals = {}
		#for k, v in self.regex_dict.items():
		for k in tokens:
			v = self.regex_dict[k]
			#_lgr.INFO(f'{k}')
			matched_intervals[k] = []
			for match in v.finditer(astr):
				#_lgr.INFO(f'{match}')
				#_lgr.INFO(f'\t{match.span(k)} {repr(match[k])}')
				matched_intervals[k].append(Interval.from_regex_match(match, k))

		# now that intervals are all matched, we can colour the matches
		# for debugging and understanding.

		return(MarkedString(astr, matched_intervals))



@dc.dataclass
class TextBlock:
	text : str
	special_indent_lines : np.ndarray = dc.field(default_factory=np.zeros((0,2),dtype=int))
	cindent : int = 0 # number of characters worth of indent
	expand_tabs = True
	tab_size = ut.cfg.tab_size
	wrapper = Splitter()

	def __post_init__(self):
		self._raw_text = self.text[:]
		self._raw_width = max(len(x) for x in self._raw_text.split('\n'))
		self._raw_height = self._raw_text.count('\n')
		self.width = self._raw_width
		self.height = self._raw_height


	@classmethod
	def get_indent_str(cls, cindent):
		istr = '\t'*(cindent//cls.tab_size)+' '*(cindent%cls.tab_size)
		return(istr if not cls.expand_tabs else istr.replace('\t',' '*cls.tab_size))

	def get_indent_strs(self, n):
		if self.special_indent_lines.shape[0]== 0:
			for i in range(0,n):
				yield(self.get_indent_str(self.cindent))

		else:
			for i in range(0,n):
				special_indent = self.special_indent_lines[np.argwhere(self.special_indent_lines[:,0]==i)]
				cindent = special_indent[0,0,1] if special_indent.shape[0] > 0 else self.cindent
				yield(self.get_indent_str(cindent))
			
		

	def unwrap(self):
		self.text = self._raw_text[:]
		self.width = self._raw_width
		self.height = self._raw_height

	def wrap(self, width, pad=''):
		#lbrk_str = '\n'+self.get_indent_str()
		#self.text = lbrk_str.join(tuple(self.wrapper.wrap(self.raw_text, width=width-self.cindent, pad=pad)))
		special_line_widths = width - np.array(self.special_indent_lines)
		special_line_widths[:,0] = self.special_indent_lines[:,0]
		ws = tuple(self.wrapper.wrap(self._raw_text, width=width-self.cindent, pad=pad, special_width=special_line_widths))
		self.text = '\n'.join([x+y for y,x in zip(ws, self.get_indent_strs(len(ws)))])
		self.width = width
		self.height = self.text.count('\n')

	def pad_text(self, pad=' '):
		text = '\n'.join((aline + pad*(self.width-len(aline)) for aline in self.text))


	def __str__(self):
		#return(self.prefix+self.text+self.suffix)
		return(self.text)

@dc.dataclass
class Text:
	text_blocks : list[TextBlock]
	expand_tabs : bool = True
	tab_expansion : str = ' '*ut.cfg.tab_size

	def wrap(self, width, pad=''):
		for tb in self.text_blocks:
			tb.expand_tabs = self.expand_tabs
			tb.tab_size = len(self.tab_expansion)
			tb.wrap(width, pad)
		return(self)

	def unwrap(self):
		for tb in self.text_blocks:
			tb.unwrap()
		return(self)

	def to_column_pages(self, page_height, col_width, col_n, col_sep=' | ', page_sep='-', pad=' ', lbrk='\n', frame_strs=['-','-','| ',' |'], frame=True):
		get_page_sep = lambda chars, extra_width=0: ('\n'+chars*(col_width*col_n+(col_n-1)*len(col_sep)+extra_width)+'\n')
		page_sep =  get_page_sep(page_sep)
		
		# wrap to column width
		self.wrap(col_width, pad=pad)
		#print('#'*30)
		#print(self)

		# split into "page_height" chunks
		lineset = str(self).split('\n')
		n_total_cols = int(np.ceil(len(lineset)//page_height))
		total_cols = [lineset[i*page_height:j*page_height] for (i,j) in zip(range(0,n_total_cols),range(1,n_total_cols+1))]

		#print(total_cols)

		# combine chunks into columns
		pages = []
		_i = 0
		_j = 0
		while _j < n_total_cols:
			pages.append([])
			for _k in range(col_n):
				if _j >= n_total_cols: 
					pages[_i].append([])
				else:
					#print(f'{_i=} {_j=} {len(pages)=} {len(total_cols)=}')
					pages[_i].append(total_cols[_j])
				_j+=1
			_i +=1
		#print(pages)
		

		astr = []
		for cols in pages:
			#print(f'{cols=}')
			max_lines = max(len(lines) for lines in cols)
			page_lines = []
			for _i in range(page_height):
				aline = col_sep.join([(lines[_i] if _i < len(lines) else ' '*col_width) for lines in cols])
				#print(f'{aline=}')
				page_lines.append(aline)
			#print(page_lines)
			astr.append(lbrk.join(page_lines))

		# have to change tabs to spaces here otherwise they don't
		# have the correct size as a column boundary != tab boundary 
		astr = page_sep.join(astr).replace('\t',tab_expansion)
		
		if frame:
			astr = self.enframe(astr, frame_strs)

		return(astr)
				
	@classmethod
	def enframe(cls, astr, frame_strs=['-','-','| ',' |']):
		get_page_sep = lambda chars, extra_width=0: ('\n'+chars*(len(astr.split('\n',1)[0])+extra_width)+'\n')
		frame_strs[0] = get_page_sep(frame_strs[0], len(frame_strs[2]+frame_strs[3]))
		frame_strs[1] = get_page_sep(frame_strs[1], len(frame_strs[2]+frame_strs[3]))
		return(frame_strs[0] + '\n'.join((frame_strs[2]+x+frame_strs[3] for x in astr.split('\n'))) + frame_strs[1])



	def __str__(self):
		astr = '\n'.join((str(tb) for tb in self.text_blocks))
		if self.expand_tabs:
			astr.replace('\t', self.tab_expansion)
		return(astr)
		

@dc.dataclass
class Parser:
	lex_dict : dict = dc.field(default_factory = lambda: {**RegexLexer.default_regex_dict})
	lexer_class : callable = RegexLexer
	
	def __post_init__(self) -> None:
		"""
		Sets up the "self.lex_dict" (to have any extra regexes not included in the
		default one, use "self.lex_dict.update({'EXTRA': 'KEY-VALUE_PAIRS'})").

		Use super().__post_init__() to then create the self.lexer from the
		self.lex_dict that you want.
		"""
		self.lexer = self.lexer_class(self.lex_dict)

	def __call__(self, astr : str) -> typing.Any:
		"""
		Parses the inputted string and emits whatever you want it to.
		"""
		raise NotImplementedError


@dc.dataclass
class Token:
	name : str
	match : str
	span : tuple[int, int]

@dc.dataclass
class ParserContext:
	tokens : list[Token] 
	stack : list = dc.field(default_factory=list)
	
	def __post_init__(self):
		self.current_token_idx = -1
		self.current_token = None
		return
	
	def __iter__(self):
		for i, token in self.tokens:
			self.current_token = token
			self.current_token_idx = i
			yield(self)


@dc.dataclass
class TextParser(Parser):

	def __post_init__(self):
		self.lex_dict.update({
			'INDENT_CONTRIBUTING_TOKENS' 	: r'(?P<>[UO_LIST_START_RAW]|[NUM_LIST_START_RAW])',
			'TEXT_BLOCK_BREAKING_TOKENS'	: r'(?P<>[UO_LIST_START_RAW]|[NUM_LIST_START_RAW])',
			'TEXT_BLOCK_BREAKING_LINE'		: r'(?P<>[LINE_START])[INLINE_WS]*[TEXT_BLOCK_BREAKING_TOKENS]',
			'ICT'							: r'[INDENT](?P<>[INDENT_CONTRIBUTING_TOKENS])',
			'INDENT_PLUS_ICT'				: r'(?P<>[INDENT][INDENT_CONTRIBUTING_TOKENS]?)'
		})
		super().__post_init__()
		#print(self.lex_dict['INDENT'])
		#print(self.lex_dict['INDENT_PLUS_ICT'])

		return

	def __call__(self, astr):
		#mstr = self.lexer(astr, ['TEXT_BLOCK_BREAKING_LINE', 'INDENT', 'ICT'])

		#print('\n'.join([f'{x[1].i:<8} {x[0]:<40} {mstr.data[x[1].i:x[1].j]!r}' for x in mstr.get_marks_in_order()]))

		mstr = self.lexer(astr, ['INDENT_PLUS_ICT'])
	
		#print(mstr.marks['INDENT_PLUS_ICT'])
		#print(mstr['INDENT_PLUS_ICT'])
		cindent_list = [get_str_char_len(s) for s in mstr['INDENT_PLUS_ICT']]
		#print(cindent_list)

		contiguous_text_blocks = []
		text_prefixes = ['']
		text_cindents = [0]
		text_nindents = [[0]]
		text_suffixes = []
		
		start = 0
		last_cindent = 0
		for m in mstr.marks['INDENT_PLUS_ICT']:
			s = mstr.data[m.i:m.j]
			#print(repr(s))
			this_cindent = get_str_char_len(s)
			#print(this_cindent)
			if last_cindent != this_cindent or (len(s)!=0 and not s.isspace()):
				stop = m.i
				text_suffixes.append('')
				contiguous_text_blocks.append((start,stop))
				start = m.i
				text_prefixes.append(s)
				text_cindents.append(this_cindent)
				text_nindents.append([])
			text_nindents[-1].append(len(s))
			last_cindent = this_cindent


		stop = len(mstr.data)
		text_suffixes.append('')
		contiguous_text_blocks.append((start,stop))


		#print(contiguous_text_blocks)
		#for i, j in contiguous_text_blocks:
			#print(mstr.data[i:j], end='')
			#print('--------')

		text_blocks = []
		non_ws_regex = re.compile(r'\S')
		for n in range(len(contiguous_text_blocks)):
			i, j  = contiguous_text_blocks[n]
			prefix = text_prefixes[n]
			cindent = text_cindents[n]
			suffix = text_suffixes[n]
			nindents = text_nindents[n]
	

			text = mstr.data[i:j]
			#mtext = self.lexer(text, ['SINGLE_LBRK'])
			#for mark, nindent in zip(mtext.marks['SINGLE_LBRK'], nindents):
			#	mtext.delete((mark.i+1, mark.j+nindent))

			#print(text)
			#mtext = self.lexer(text, ['INDENT_PLUS_ICT'])
			#mtext.delete('INDENT_PLUS_ICT')
			mtext = self.lexer(text, ['INDENT'])
			mtext.delete('INDENT')
			#print(repr(mtext.data))
			mtext = self.lexer(mtext.data, ['EXPANDABLE_LBRK'])
			mtext.replace('EXPANDABLE_LBRK', ' ')
			#print(repr(mtext.data))
			mtext = self.lexer(mtext.data, ['SINGLE_LBRK'])
			mtext.delete('SINGLE_LBRK')
			#print(repr(mtext.data))

			sil = np.zeros((0,2),dtype=int)
			match = non_ws_regex.search(prefix)
			if match:
				sil = np.array([[0,get_str_char_len(prefix[:match.span()[0]])]], dtype=int)
			text_blocks.append(TextBlock(mtext.data, special_indent_lines=sil, cindent=cindent))


		return(Text(text_blocks))
		


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

	@classmethod
	def from_regex_match(cls, match, group=0):
		return(cls(match.start(group), match.end(group), type='[]'))

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


@dc.dataclass
class MarkedString:
	data : str = ''
	marks : dict[str, list[Interval]] = dc.field(default_factory=dict)
	
	def insert(self, idx, astr):
		#print('INSERT', idx, repr(astr))
		self.data = self.data[:idx]+astr+self.data[idx:]
		for name, mark_list in self.marks.items():
			for mark in mark_list:
				if mark.i > idx: mark += len(astr)
				elif mark.j >= idx: mark.j += len(astr)

	def insert_left(self, idx, astr):
		self.data = self.data[:idx]+astr+self.data[idx:]
		for name, mark_list in self.marks.items():
			for mark in mark_list:
				if mark.i >= idx: mark += len(astr)
				elif mark.j > idx: mark.j += len(astr)


	def delete(self, ivar):
		#print('DELETE', aslice)
		if type(ivar) is str:
			for mark in self.marks[ivar]:
				self.delete(mark)
			return

		x, y = self._input_as_xy(ivar)
		self.data = self.data[:x]+self.data[y:]
		for name, mark_list in self.marks.items():
			for mark in mark_list:
				if mark.i > x and mark.i < y: mark.i = x
				elif mark.i >= y: mark.i -= (y-x)
				if mark.j > x and mark.j < y: mark.j = x
				elif mark.j >=y: mark.j-=(y-x)

	def replace(self, ivar, astr):
		if type(ivar) is str:
			for mark in self.marks[ivar]:
				self.replace(mark, astr)
			return

		x, y = self._input_as_xy(ivar)
		self.delete(ivar)
		self.insert(x,astr)
		return

	def _input_as_xy(self, var):
		if type(var) is slice:
			var.indices(len(self.data))
			x, y = var.start, var.stop
		elif type(var) is Interval:
			x,y = var.i, var.j
		elif type(var) is re.Match:
			x,y = var.span()
		else:
			x, y = var
		return(x,y)


	def encase(self, ivar, str1, str2):
		if type(ivar) is str:
			#print(ivar)
			for mark in self.marks[ivar]:
				self.encase(mark, str1, str2)
				#print(mark, self.data[mark.i:mark.j].replace('\x1b','\ESC'), self.data[mark.i-10:mark.j+10].replace('\x1b','\ESC'))
				#print(mark, repr(self.data[mark.i:mark.j]))#, self.data[mark.i-10:mark.j+10].replace('\x1b','\ESC'))
			return

		x,y = self._input_as_xy(ivar)
		self.insert(x,str1)
		y += len(str1)
		self.insert_left(y,str2)

	def __getitem__(self, key):
		return([self.data[x.i:x.j] for x in self.marks[key]])
	
	def __str__(self):
		return(self.data)
	
	
	def colour_matches(self, match_sets, cc_func = term_bgcolour_code, xx=term_reset_bgc_code()):
		match_sets_colours = []
		i = 1
		for k in match_sets:
			cc = cc_func(i)
			match_sets_colours.append(cc+k+xx)
			self.encase(k, cc, xx)
			i += 1

		self.pair_up_bgcolour_matches()
		for x in match_sets_colours: print(x)
		
	
	def pair_up_bgcolour_matches(self):
	
		term_esc_bgc_regex=re.compile('\x1b'+r'\[48.*?m')
		term_esc_bgc_reset_regex = re.compile('\x1b'+r'\[49m')

		stack = ['\x1b[0m']

		last_bgc_match_end = 0

		#print('finding next r_match')
		r_match = term_esc_bgc_reset_regex.search(self.data, pos=0)
		
		while r_match is not None:
			last2_bgc_match=None
			last1_bgc_match = None

			#print('finding next bgc match')
			bgc_match = term_esc_bgc_regex.search(self.data, pos=last_bgc_match_end)
			while bgc_match is not None and bgc_match.start() < r_match.start():
				last2_bgc_match = last1_bgc_match
				if last2_bgc_match is not None: stack.append(last2_bgc_match[0])
				last1_bgc_match = bgc_match
				last_bgc_match_end = last1_bgc_match.end()
				#print('finding next bgc match')
				bgc_match = term_esc_bgc_regex.search(self.data, pos=last_bgc_match_end)
			
			#print(r_match)
			#print(bgc_match)
			#print(stack)

			replacement_str = stack.pop() if len(stack) > 1 else stack[0]
			self.replace(r_match, replacement_str)

			#print('finding next r_match')
			r_match = term_esc_bgc_reset_regex.search(self.data, pos=r_match.end())
		
		self.term_reset_attributes_at_eol()

	def term_reset_attributes_at_eol(self, term_reset_cc='\x1b[m'):
		"""
		If you don't tell the terminal to block-colour to the end of line,
		you can get spurious background colours if a colour code is active
		when the terminal wraps a line and then a linebreak happens.
		"""
		eol_regex = re.compile(r'\n', flags=re.MULTILINE)
		eol_reset_command = term_reset_cc+'\x1b[K'
		eol_match = eol_regex.search(self.data)
		while eol_match is not None:
			#print('EOL MATCH: ', eol_match)
			self.insert(eol_match.start(), eol_reset_command)
			eol_match = eol_regex.search(self.data, pos=eol_match.end()+len(eol_reset_command))
		return
	
	def get_marks_in_order(self):
		ml = []
		for k, v in self.marks.items():
			ml += [(k, x) for x in v]
		return(sorted(ml, key=lambda x: x[1].i))




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
	- Also I can try to nest bullets like this
	- and this
1) I would also like to be able to wrap within numbered blocks
   even when they extend over multiple lines. Maybe I can do something
   with a 'common indent'. I.e. the first word in a line matches a 
   markdown tag, I can treat all subsequent lines that have the same
   character-size indent as a single block of text.
2) And have mutiple nested numbered lists also
	2.1) like this list here.
	2.2) and this second entry as well. It helps if I make this line really long so I can see the wrapping at work.

I should also try to get numbers recognised e.g. 1, 2, 3, 1.2, 1.4, 1E-99, -7, 1.989e+009 etc.

99 7+9i 1E-09 + 33.99j
"""


def centered(astr, width=ut.cfg.wrap_width, ends=(' ',' '), pad_char='='):
	fmt = '{:' + pad_char + '^' + f'{width}' + '}'
	return(fmt.format(f'{ends[0]}{astr}{ends[1]}'))

if __name__=='__main__':
	#_lgr.setLevel('INFO')

	print(centered('Original Message'))
	print(repr(msg),end='\n\n')

	print(centered('splitter() output'))
	print(''.join(Splitter().wrap(msg, pad=' ')))

	print(centered('RegexCombiner() output'))
	lexer = RegexLexer()
	mstr = lexer(msg)
	mstr.replace('EXPANDABLE_LBRK', 'X')
	mstr.colour_matches(['INDENT','UO_LIST_START', 'NUM_LIST_START', 'INTEGER', 'FLOAT', 'COMPLEX'])

	mstr.data = mstr.data.replace('\t', '    ') # for some reason tabs do not get coloured in with terminal command codes. Maybe "\t" skips the cursor alonginstead of printing characters?


	#print(repr(mstr.data))
	print(mstr)
	print(repr(msg))


	parser = TextParser()
	mtxt = parser(msg)
	mtxt.wrap(30, pad='')
	print(mtxt)

	print(mtxt.to_column_pages(20, 30, 2))
