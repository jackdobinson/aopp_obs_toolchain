"""
Routines for outputting to terminal in a kinda pretty way
"""

import textwrap
import shutil



terminal_wrapper = textwrap.TextWrapper(
	expand_tabs = True,
	tabsize = 8,
	replace_whitespace= False,
	drop_whitespace= False,
	fix_sentence_endings= False,
	break_long_words= True,
	break_on_hyphens=True,
	initial_indent='',
	subsequent_indent=''
	
)


def get_indent(text):
	return text[:-len(text.lstrip())]

def terminal_wrap(text, indent_level, indent_str='\t', d=0):
	indent = indent_level*indent_str
	terminal_wrapper.width = shutil.get_terminal_size().columns - terminal_wrapper.tabsize*indent_level + d
	#terminal_wrapper.width = 60 - terminal_wrapper.tabsize*indent_level
	
	wrapped_text_lines = []
	for line in text.splitlines(False):
		wls = terminal_wrapper.wrap(line)
		if len(wls) > 1:
			windent = get_indent(wls[0])
			wls[1:] = [windent+x for x in wls[1:]]
		wrapped_text_lines += wls
	return(indent + ('\n'+indent).join(wrapped_text_lines))

def terminal_fill(text, d=0):
	assert len(text) > 0, "must have some amount of text to repeat"
	c = shutil.get_terminal_size().columns + d
	n = c // len(text)
	r = c % len(text)
	return(text*n + text[:r])

def terminal_center(text, fill, d=0):
	assert len(text) > 0
	assert len(fill) > 0
	c = shutil.get_terminal_size().columns + d
	x = (c - len(text))
	h = x//2
	b = x%2
	n = h // len(fill)
	r = h % len(fill)
	return(fill*n + fill[:r] + text + fill[:b] + fill*n + fill[:r])

def terminal_left(text, fill, d=0):
	assert len(text) > 0
	assert len(fill) > 0
	c = shutil.get_terminal_size().columns + d
	x = (c - len(text))
	n = x // len(fill)
	r = x % len(fill)
	return(text + fill*n + fill[:r])
	
def terminal_right(text, fill, d=0):
	assert len(text) > 0
	assert len(fill) > 0
	c = shutil.get_terminal_size().columns + d
	x = (c - len(text))
	n = x // len(fill)
	r = x % len(fill)
	return(fill*n + fill[:r] + text)


