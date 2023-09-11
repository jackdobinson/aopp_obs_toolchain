#!/usr/bin/env python3
"""
Utilities for string operations and information

"""
import os
import dataclasses as dc
import utilities.cfg
import string
import enum

@dc.dataclass
class Wrapper:
	split_on_strings : tuple = tuple()
	split_after_strings : tuple = tuple(x for x in '}])'+string.whitespace)
	split_before_strings : tuple = tuple()
	split_hard_dashes : tuple = tuple(x for x in string.ascii_letters)
	width : int = utilities.cfg.wrap_width
	block_pad : bool = False, # pad each returned line to `width`
	remove_whitespace : bool = True
	preserve_existing_linebreaks : bool = True
	linebreak : str = os.linesep
	prefix : str = ''
	suffix : str = ''
	split_modes = enum.IntEnum(
		'SPLIT_MODES', (
			"AFTER", 		# split after a string
			"BEFORE", 		# split before a string
			"ON",			# split on a string (string turns into new line)
			"HARD_CONT",	# split at `self.wrap` characters, but we are in the middle of a word, so use word contunation string `self.str_word_cont`
			"HARD_FILL", 	# split at `self.wrap` characters, but whole word is too short to continue, so move word to next line and fill space with `self.char_fill` string
			"HARD_BR"		# split at `self.wrap` characters, nothing special so just break the line.
		)
	)
	str_word_cont = '-'
	char_fill = ' '

	def __post_init__(self):
		self.split_strategy = (
			(self.split_on_strings, self.split_modes.ON),
			(self.split_after_strings, self.split_modes.AFTER),
			(self.split_before_strings, self.split_modes.BEFORE)
		)
		return

	"""
	The "split*()" methods always return a tuple of the form (split_idx, split_str, split_type).
		split_idx
			The index where the split should take place, -1 indicates no split found.
		split_str
			The substring the split is being performed on
		split_type
			The type of split
	"""
	def splitAtStr(self, astr, start=None, stop=None, split_strings=tuple(), split_mode=split_modes.AFTER):
		if len(split_strings) > 0:
			return(max((astr.rfind(x, start, stop), x, split_mode) for x in split_strings))
		else:
			return((-1, '', split_mode))


	def splitHard(self, astr, start=None, stop=None):
		if len(astr) == 0:
			return(stop, '', self.split_modes.HARD_CONT)
		stop = len(astr)-1 if ((stop is None) or (stop >= len(astr))) else stop
		start = 0 if start is None else start
		#print(f'{start:4} {stop:4} {len(astr):4} {astr!r}')
		#print(repr(astr[stop]), repr(astr[stop-1]), astr[stop] in self.split_hard_dashes, astr[stop-1] in self.split_hard_dashes)
		if (astr[stop] in self.split_hard_dashes) and (astr[stop-1] in self.split_hard_dashes):
			hard_dash_extent = tuple(x in self.split_hard_dashes for x in astr[stop-len(self.str_word_cont)-1:stop-1])
			try:
				false_offset = hard_dash_extent[::-1].index(False) + 1
				stop -= false_offset
				return(stop, astr[stop], self.split_modes.HARD_FILL)
			except ValueError: # `False` not present in tuple
				stop -= len(self.str_word_cont)
				return(stop, astr[stop], self.split_modes.HARD_CONT)
				

		"""
		if ((astr[stop] in self.split_hard_dashes) and (astr[stop-1] in self.split_hard_dashes) and (astr[stop-2] in self.split_hard_dashes)):
			stop -= len(self.str_word_cont)
			return((stop, astr[stop], self.split_modes.HARD_CONT))
		if ((astr[stop] in self.split_hard_dashes) and (astr[stop-1] in self.split_hard_dashes) and (astr[stop-2] not in self.split_hard_dashes)):
			stop -= len(self.str_fill)
			return((stop, astr[stop], self.split_modes.HARD_FILL))
		"""
		return((stop, astr[stop], self.split_modes.HARD_BR))


	# TODO: 
	# * Deal with TABS correctly
	def splitPoint(self, astr, start=None, stop=None):
		#sp = max((sf(astr, start, stop) for sf in (self.splitOn, self.splitAfter, self.splitBefore)))
		sp = max((self.splitAtStr(astr, start, stop, split_strings, split_mode) for split_strings, split_mode in self.split_strategy))
		return(sp if sp[0] != -1 else self.splitHard(astr,start,stop))

	
	def __call__(self, astr, width=None, preserve_existing_linebreaks=None, remove_whitespace=None, linebreak=None):
		if width == 0: raise ValueError('width to Wrapper() cannot be zero')
		width = self.width if width is None else width # can overwrite instance value by passing parameter
		preserve_existing_linebreaks = self.preserve_existing_linebreaks if preserve_existing_linebreaks is None else preserve_existing_linebreaks # can overwrite instance value by passing parameter
		remove_whitespace = self.remove_whitespace if remove_whitespace is None else remove_whitespace # can overwrite instance value by passing parameter
		linebreak = self.linebreak if linebreak is None else linebreak # can overwrite instance value by passing parameter

		if preserve_existing_linebreaks:
			lines = astr.split(linebreak)
		else:
			lines = [astr]
		wrapped_lines = []
		for aline in lines:
			start = 0
			end = len(aline)
			#DEBUG_I=0
			while (end-start) > width :
				#print(f'DEBUG: {start=} {end=} {width=}')
				split_idx, split_str, split_mode = self.splitPoint(astr, start, start+width)
				
				if split_mode == self.split_modes.AFTER:
					split_idx += len(split_str)

				split_line = aline[start:split_idx]

				if split_mode == self.split_modes.ON:
					# need to skip over the rest of the string, it has been converted to a linbreak
					split_idx += len(split_str)
				elif split_mode == self.split_modes.HARD_CONT:
					split_line += self.str_word_cont
				elif split_mode == self.split_modes.HARD_FILL:
					split_line += self.char_fill
				
				if remove_whitespace:
					split_line = split_line.strip()
					while (split_idx < len(aline)) and (aline[split_idx] in string.whitespace):
						split_idx += 1

				if self.block_pad:
					split_line += self.char_fill * (width - len(split_line))

				wrapped_lines.append(split_line)
				#print(f'{start:4} {split_idx:4} {len(split_line):4} {split_mode._name_:12} {repr(split_str):4} {split_line!r}')

				#if DEBUG_I > 4: raise RuntimeError
				start = split_idx
				#DEBUG_I+=1

			final_line = aline[start:]
			if len(final_line) > 0:
				if remove_whitespace:
					final_line = final_line.strip()
				if self.block_pad:
					final_line += self.char_fill * (width - len(final_line))
				wrapped_lines.append(final_line)
				#print(f'{start:4} {end:4} {len(aline[start:]):4} {"final_line"!r:12} {repr(""):4} {aline[start:]!r}')

		return(self.prefix + ('\n'.join(wrapped_lines)) + self.suffix)

block_wrapper = Wrapper(block_pad=True, remove_whitespace=True)

def as_block(astr, width=utilities.cfg.wrap_width):
	#print('DEBUG: In "as_block()"')
	return(block_wrapper(astr, width=width))

def block_below(str_list, width = None, sep=' '):
	#print('DEBUG: In "block_below()"')
	#print(f'DEBUG: {width=}')
	#print(str_list)
	#print(str_list[0])
	if width is None:
		nl_pos = str_list[0].find(os.linesep)
		width = utilities.cfg.wrap_width if nl_pos==-1 else nl_pos+1
		#print(f'INSIDE IF: {width=}')
	#print(f'DEBUG: {width=}')
	if sep in('', '\n'):
		block_sep = '\n'
	elif len(sep)>=width:
		block_sep = f'\n{sep[:width]}\n'
	else:
		block_sep = '\n'+sep*(width//len(sep))+sep[:width%len(sep)]+'\n'
	return(block_sep.join((block_wrapper(astr, width=width) for astr in str_list)))

def block_right(str_list, width=None, sep=' | '):
	#print('DEBUG: In "block_right()"')
	if width is None:
		nl_pos = tuple(astr.find(os.linesep) for astr in str_list)
		width = tuple(utilities.cfg.wrap_width if x==-1 else x+1 for x in nl_pos)
	elif type(width) is int:
		width = tuple([width]*len(str_list))
	#print(f'DEBUG: {width=}')
	str_blocks = tuple(block_wrapper(astr, w).split(os.linesep) for (astr, w) in zip(str_list, width))
	max_lines = max(len(block) for block in str_blocks)
	combined_block_lines = []
	for i in range(max_lines):
		lines = tuple(block[i] if i < len(block) else width[j]*' ' for j, block in enumerate(str_blocks))
		combined_block_lines.append(sep.join(lines))
	return('\n'.join(combined_block_lines))

def block_frame(block_str, hframe=u'\u2550', lvframe=u'\u2551 ', rvframe=None, cframe=u'\u2554\u2557\u255A\u255D'):
	if rvframe is None:
		rvframe = lvframe[::-1]
	block_strs = block_str.split(os.linesep)
	width = len(block_strs[0])
	nwidth = width + len(lvframe) +len(rvframe)
	cwidth = nwidth -2*(len(cframe)//4)
	height = len(block_strs)
	nheight = height + 2*len(hframe)
	block_lines = []
	for i,hf in enumerate(hframe):
		if i==0:
			block_lines.append(cframe[0]+hf*cwidth+cframe[1])
		else:
			block_lines.append(lvframe+hf*width+rvframe)
	for i in range(height):
		block_lines.append(lvframe+block_strs[i]+rvframe)
	for i, hf in enumerate(hframe[::-1]):
		if i!=len(hframe)-1:
			block_lines.append(lvframe+hf*width+rvframe)
		else:
			block_lines.append(cframe[2]+hf*cwidth+cframe[3])
	return('\n'.join(block_lines))
	

def find_all(astr, substring):
	start = 0
	start = astr.find(substring, start+1)
	yield(start-1)

def wrap_in_tag(astr, starttag, endtag=False, indent=True, numhashes=3, 
				indentstr=utilities.cfg.indent_str):
	f"""
	Wraps a string in a starting and ending tag
	
	ARGUMENTS
		astr
			the string to wrap

		starttag
			the tag to put at the start of the string

		endtag=False
			the tag to put at the end of the string, if false will
			place a string of dashes the same length as 'starttag'

		indent=True
			should we indent the new lines in the string? if an integer,
			by how many levels?

		indentstr={utilities.cfg.indent_str}
			string to use for indentation
		
		numhashes=3
			Wraps the start and end tags in 'numhashes' '#' characters? 

	RETURNS
		a string wrapped in a starting and ending tag, and possibly indented

	EXAMPLE
		>>> wrap_in_tag('this string\nwill be\nwrapped', 'WRAPPER')
		--- ### WRAPPER ###
		--- 	this string
		--- 	will be
		--- 	wrapped
		---	### ------ ###

	"""
	start_tag_line = ('#'*numhashes+' '+starttag+' '+numhashes*'#').strip()
	end_tag_line =  ('#'*numhashes+' '+endtag+' '+numhashes*'#').strip() if endtag else  ('#'*numhashes+' '+'-'*len(starttag)+' '+numhashes*'#').strip()
	if type(indent)==bool:
		if indent:
			indent = 1
		else:
			indent=0
	lines = []
	lines.append(start_tag_line)
	lines += [indent*indentstr+s for s in astr.rstrip().split('\n')]
	lines.append(end_tag_line)
	return('\n'.join(lines))

def detect_nl_type(astr):
	nl_candidates = ('\r', '\n', '\r\n')
	candidate_counts = []
	#print(repr(astr))
	for nlc in nl_candidates:
		#print(repr(nlc))
		#print(nlc)
		#print(astr.count(nlc))
		candidate_counts.append(astr.count(nlc))
	# assume most common candidate is the correct one
	maxc = max(candidate_counts)
	#print(maxc)
	most = [(n,c) for n,c in zip(nl_candidates, candidate_counts) if c==maxc]
	#print(most)
	best = sorted(most, key=lambda x: len(x[0]), reverse=True)
	return(best[0][0], len(best[0][0]))

def block_indent(astr, 
				 from_indent_str=utilities.cfg.indent_str, 
				 to_indent_str=utilities.cfg.indent_str, 
				 tab_size=utilities.cfg.tab_size, 
				 width=utilities.cfg.wrap_width):
	"""Wraps a block of text while also changing the indent"""
	bi_string = ''
	for aline in astr.split('\n'):
		nindent = count_start(aline, from_indent_str)
		#print(nindent)
		wrapped_string_list = wrap(aline.strip(), width-nindent*charlen(to_indent_str, tab_size=tab_size))
		#print(wrapped_string_list)
		bi_string += '\n'.join([nindent*to_indent_str+ws for ws in wrapped_string_list]) + '\n'
	return(bi_string[:-1])

def block_indent_raw(astr, 
					 from_indent_str=(' ', utilities.cfg.indent_str), 
					 tab_size=utilities.cfg.tab_size, 
					 width=utilities.cfg.wrap_width):
	"""Wraps a block of text without changing the indent"""
	bi_string = ''
	for aline in astr.split('\n'):
		indent_str = count_start_list(aline, from_indent_str)
		wrapped_string_list = wrap(aline.strip(), width-charlen(indent_str, tab_size=tab_size))
		#print(wrapped_string_list)
		bi_string += '\n'.join([indent_str+ws for ws in wrapped_string_list]) + '\n'
	return(bi_string[:-1])

def wrap(astr, width=utilities.cfg.wrap_width):
	"""Wrap a string to width, preferentially split on spaces, preserve newlines"""
	#print('###')
	split_str = []
	last_space = 0
	last_split = 0
	for i, ch in enumerate(astr):
		#print(i, ch)
		if ch==' ':
			last_space = i
		if ch=='\n':
			split_str.append(astr[last_split:i])
			last_split = i
		if i - last_split >= width:
			if i-last_space < width:
				split_str.append(astr[last_split:last_space])
				last_split = last_space
			else:
				split_str.append(astr[last_split:i])
				last_split = i
	split_str.append(astr[last_split:])
	#print('---')
	return([ss.strip() for ss in split_str]) 

def charlen(astring, tab_size=utilities.cfg.tab_size):
	"""Finds the number of characters a string has the same length as"""
	slen = len(astring)
	ntabs = astring.count('\t')
	charlen = slen - ntabs + ntabs*tab_size
	return(charlen)

def count_start_list(astr, slist):
	slist_len = [len(s) for s in slist]
	init_str_lst = []
	i = 0
	while True:
		does_start_with = [astr[i:].startswith(s) for s in slist]
		if any(does_start_with):
			idx = does_start_with.index(True)
			i+=slist_len[idx]
			init_str_lst.append(slist[idx])
		else:
			break
	return(''.join(init_str_lst))
	

def count_start(astr, start_str):
	"""Counts the number of times 'start_str' appears at the start of astr"""
	if start_str == '':
		return(0)
	#print('## s')
	#print(repr(astr), repr(start_str))
	slen = len(start_str)
	i = 0
	#print(repr(astr[:(i+1)*slen]))
	while(astr[:(i+1)*slen] == (i+1)*start_str):
		i+=1
	#print(i)
	#print('## e')
	return(i)

def rationalise_newline_for_wrap(astr):
	"""Remove new lines that have text directly after them, do not remove them if they have whitespace directly after them"""
	bstr = ''
	if len(astr) == 1:
		return(astr)
	a = 0
	for i, ch in enumerate(astr):
		if ch == '\n':
			if i == 0:
				continue
			#if i == 0:
			#	bstr += astr[a:i+1]
			#	sys.stdout.write(astr[a:i+1]+'##')
			#	a = i+1
			if i+1==len(astr):
				bstr += astr[a:i]
				#sys.stdout.write(astr[a:i]+'##')
			else:
				if (astr[i-1]!='\n') and (not astr[i+1].isspace()):
					bstr+=' '+astr[a:i]
					#sys.stdout.write(astr[a:i]+'##')
					a = i+1
	return(bstr)

def line_wrap(astr, width=utilities.cfg.wrap_width, newline_mode='single_to_space'):
	if not (newline_mode in ('single_to_space', 'collapse_all', 'collapse_last', 'ignore')):
		print('ERROR: Unknown option to argument "newline_mode":"{}"'.format(newline_mode))
	
	unl_cha, unl_len = detect_nl_type(astr)
	#print(repr(unl_cha), unl_len)
	if newline_mode != 'ignore':
		lines = astr.splitlines(keepends=True)
		# only care about '\r', '\n', '\r\n'
		unl_tags = ('\n', '\r', '\r\n')
		lines =[]
		j=0
		for i,s in enumerate(astr):
			if astr[i:i+unl_len]==unl_cha:
				if (i<(len(astr)-unl_len)) and (astr[i+unl_len:i+2*unl_len]==unl_cha):
					continue
				lines.append(astr[j:i+unl_len])
				j=i+unl_len
		lines.append(astr[j:])
	
		for i, al in enumerate(lines):
			#unl_len = 2 if al.endswith('\r\n') else 1 if al.endswith('\n') or al.endswith('\r') else None
			#unl_cha = '\r\n' if al.endswith('\r\n') else '\r' if al.endswith('\r') else '\n' if al.endswith('\n') else None
			if unl_len == None and unl_cha==None:
				continue
			elif unl_len==None or unl_cha==None:
				print('ERROR: Unversal new line detection problem, new line tag detected as {} but is of length {}'.format(unl_cha, unl_len))
			if newline_mode=='single_to_space':
				#print(repr(al))
				if al.endswith(2*unl_cha):
					#print('TWO')
					a = lines[i]
				elif al.endswith(unl_cha):
					#print('ONE')
					a = lines[i][:-unl_len]
				else:
					#print('OTHER')
					a = lines[i]
				lines[i] = a
			elif newline_mode=='collapse_all':
				lines[i] = lines[i].rstrip(unl_cha)+' ' if al.endswith(unl_cha) else lines[i]
			elif newline_mode=='remove_last':
				if al.endswith(unl_cha):
					a = lines[i].rstrip(unl_cha,1)
				else:
					a = lines[i]
				lines[i] = a
		astr = ''.join(lines)

	#print(repr(astr))

	tstr = ''
	j = 0
	k = 0
	while i < len(astr):
		#print(i,j,k,repr(astr[i]), len(astr))
		if astr[i:i+unl_len]==unl_cha:
			tstr += astr[j:i+unl_len]
			j = i +unl_len
			i += unl_len
		if (astr[i] in (' ', '-')):
			k = i
		if i-j > width:
			if (k-j) > width:
				#print('\t{}'.format(astr[j:i]))
				tstr += astr[j:i]+unl_cha
				j = i
			else:
				#print('\t{}'.format(astr[j:k]))
				tstr += astr[j:k]+unl_cha
				j = k
		i+=1
	tstr += astr[j:]
	#print(tstr)
	return(tstr)
	
def wrap_msg(astr, msg, sep=': ', pad_char='-', linewrap=79, colour=[]):
	"""
	Formats a string to be an error, warning, or information message.

	TODO
	* fix to not use textwrap, it doesn't handle wrapping multiple 
	  paragraphs well.
	"""
	tcc_start = ''
	tcc_end = ''
	if len(colour)!=0:
		tcc_start+='\33['
		tcc_end += '\33[0m'
		if 'txt_norm' in colour:
			tcc_start += '0'
		if 'txt_bold' in colour:
			tcc_start += '1'

		if 'red' in colour:
			tcc_start += ';31'#'\33[91m'
		if 'yellow' in colour:
			tcc_start += ';33'#'\33[93m'
		if 'green' in colour:
			tcc_start += ';32'#'\33[92m'
		if 'blue' in colour:
			tcc_start += ';34'#'\33[94m'

		if 'bg_red' in colour:
			tcc_start += ';41'
		if 'bg_green' in colour:
			tcc_start += ';42'

		tcc_start += 'm'

	import textwrap as tw
	lines = tw.wrap(astr, width=linewrap)
	msgs = [tcc_start+msg+tcc_end]+[tcc_start+pad_char*len(msg)+tcc_end]*(len(lines)-1)
	lines = [m+sep+s for m,s in zip(msgs,lines)]
	return('\n'.join(lines))

def wrap_str(
		astring, 
		width=utilities.cfg.wrap_width, 
		remove_leading_whitespace=True, 
		preserve_existing_linebreaks=True, 
		split_strings=tuple([x for x in string.whitespace])
	) -> str:
	# assume any previous linebreaks should be preserved
	if preserve_existing_linebreaks:
		lines = astring.split(os.linesep)
	else:
		lines = (astring,)

	mutated_lines = []
	rfind_splitpoint = lambda astr, start=None, end=None: max((astr.rfind(x,start,end),x) for x in split_strings)
	
	for aline in lines:
		start = 0
		while len(aline[start:]) > width:
			# try to find a split point before hard split at 'width'
			split_idx, split_str = rfind_splitpoint(aline, start, end=start+width)
			if split_idx == -1:
				# have not found a valid splitting point, just do a hard split.
				split_idx = start+width
			split_line = aline[start:split_idx]
			start = split_idx

			if remove_whitespace:
				split_line = split_line.strip()
				while (start < len(aline)) and (aline[start] in string.whitespace):
					start += 1

			mutated_lines.append(split_line)
		# add final message section
		if len(aline[start:]) > 0:
			mutated_lines.append(aline[start:] if not remove_whitespace else aline[start:].strip())
	return('\n'.join(mutated_lines))

def count(i):
	"""Turns an integer (1,2,3,...) into a 'count' string ('1st', '2nd', '3rd', '4th', ...)"""
	return('{}{}'.format(i, count_suffix(i)))

def latex_count(i):
	"""same as 'count()' but formats uses latex superscripts"""
	return('{}'.format(i)+'^{' + '{}'.format(count_suffix(i))+'}')

def count_suffix(i):
	""" gets the suffix of a 'count' string"""
	# special cases for eleventh twelvth thirteenth
	if i%100 == 11 or i%100==12 or i%100==13:
		return('th')
	
	#get units column of integer
	if i > 0:
		unit = i%10
	else:
		unit = 10 - i%10
		
	if unit == 1:
		return('st')
	elif unit==2:
		return('nd')
	elif unit==3:
		return('rd')
	else:
		return('th')


def strip_suffix(astr, suffix):
	"""Removes 'suffix' from a string"""
	if astr.endswith(suffix):
		return(astr[:-len(suffix)])
	return(astr)

def strip_prefix(astr, prefix):
	"""Removes 'prefix' from a string"""
	if astr.startswith(prefix):
		return(astr[len(prefix):])
	return(astr)


def infer_type(x):
	'''
	Type casts a string, filters from most restrictive to least restrictive
	'''
	if x in ['None', 'none', 'NONE']: return(None)
	if x in ['True', 'true', 'TRUE']: return(True)
	if x in ['False', 'false', 'FALSE']: return(False)
	
	try:
		return(int(x)) # Automatically promotes to long if needed
	except ValueError:
		pass
	
	try:
		return(float(x))
	except ValueError:
		pass
	
	return(x) # fall back on string


def as_type(astr):
	"""Returns the type a string is trying to specify"""
	if astr in ('str', 'STR', 'string', 'String', 'STRING'):
		return(type(str))
	if astr in ('bool', 'Bool', 'BOOL', 'boolean', 'Boolean', 'BOOLEAN'):
		return(type(bool))
	if astr in ('int', 'Int', 'INT', 'integer', 'Integer', 'INTEGER'):
		return(type(int))
	if astr in ('float', 'Float', 'FLOAT', 'double', 'Double', 'DOUBLE'):
		return(type(float))
	if astr in ('imag', 'imaginary', 'Imag', 'Imaginary', 'IMAG', 'IMAGINARY', 'complex', 'Complex', 'COMPLEX'):
		return(type(complex))
	if astr in ('none', 'NONE', 'None', 'NULL', 'null', 'Null'):
		return(type(None))
	sys.exit('ERROR: No mapping from from string "{}" to a data type, exiting...'.format(astr))
	return

def create_rand():
	"""
	Create a random string for testing
	"""
	import numpy as np
	import itertools as it
	n_lines = np.random.randint(1, 5)
	n_words = np.random.randint(1, 20, n_lines)
	n_chars = [np.random.randint(1, 10, _x) for _x in n_words]

	astr = '\n'.join([' '.join([''.join(np.random.choice([_a for _a in utilities.cfg.alphabet], _x)) for _x in _y]) for _y in n_chars])
	return(astr)
	


_testing_text = """
Creates NEMESIS compatible *.spx files from a set of target fits cubes and NEMESIS run templates.

The `if __name__=='__main__':` statement allows execution of code if the script is called directly.
eveything else not in that block will be executed when a script is imported. 
Import statements that the rest of the code relies upon should not be in the if statement, python
is quite clever and will only import a given package once, but will give it multiple names if it
has been imported under different names.

Standard library documentation can be found at https://docs.python.org/3/library/

Packages used in this program are:
	sys
	os 
"""

if __name__=='__main__':
	import sys, os
	
	# DEBUGGING
	#import matplotlib.pyplot as plt
	#
	#plt.plot([1,2,3,4,5,6], [1,45,2,6,8,65])
	#plt.show()
	
	print('HELLO WORLD')
	
	astr = create_rand()
	
	print(utilities.cfg.wrap_width*'=')
	print('create_rand()')
	print(astr)
	
	print(utilities.cfg.wrap_width*'=')
	print('wrap_in_tag()')
	print(wrap_in_tag(astr, 'TAG TO WRAP'))
	
	print(utilities.cfg.wrap_width*'=')
	print('block_indent(wrap_in_tag())')
	print(block_indent(wrap_in_tag(astr, 'TAG TO WRAP')))
	
	print(utilities.cfg.wrap_width*'=')
	print('wrap_in_tag(block_indent())')
	print(wrap_in_tag(block_indent(astr), 'TAG TO WRAP'))
	
	print(utilities.cfg.wrap_width*'=')
	print('block_indent()')
	print(block_indent(astr))
	
	print(utilities.cfg.wrap_width*'=')
	print('block_indent_raw()')
	print(block_indent_raw(astr))
	
	print(utilities.cfg.wrap_width*'=')
	print('wrap()')
	print(wrap(astr))
	
	print(utilities.cfg.wrap_width*'=')
	print('charlen()')
	print(charlen(astr))
	
	print(utilities.cfg.wrap_width*'=')
	print('rationalise_newline_for_wrap()')
	print(rationalise_newline_for_wrap(astr))

	print(utilities.cfg.wrap_width*'=')
	print('line_wrap()')
	print(line_wrap(astr))
	
	print(utilities.cfg.wrap_width*'=')
	print('count()')
	print([count(_i) for _i in range(0,25)])
	
	print(utilities.cfg.wrap_width*'=')
	print('strip_suffix()')
	print(strip_suffix('alchemist', 'ist'), strip_suffix('volleyball', 'volley'))
	
	print(utilities.cfg.wrap_width*'=')
	print('strip_prefix()')
	print(strip_prefix('alchemist', 'ist'), strip_prefix('volleyball', 'volley'))
	
	print(utilities.cfg.wrap_width*'=')
	print('infer_type()')
	print([infer_type(_x) for _x in 'none false True 1 536 22.56 2E55 one two three'.split()])
	
	print(utilities.cfg.wrap_width*'=')
	print('as_type()')
	print([as_type(_x) for _x in 'int float complex str BOOL NONE'.split()])

	print(utilities.cfg.wrap_width*'=')
	wrapper = Wrapper()
	print(wrapper(_testing_text))


	print(utilities.cfg.wrap_width*'=')
	ablock = as_block(_testing_text)
	print(ablock)
	print('---')
	print(block_below((_testing_text, _testing_text)))
	print('---')
	col2 = block_right((_testing_text, _testing_text), width=30)
	print(col2)
	print('---')
	col1col2 = block_below((_testing_text,col2), width=len(col2.split(os.sep)[0]), sep='-')
	print(col1col2)
	print(block_frame(col1col2))








