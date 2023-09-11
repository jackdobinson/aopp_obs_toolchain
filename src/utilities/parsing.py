#!/usr/bin/env python3

import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')



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



if __name__=='__main__':
	
