 
import sys, os
import importlib
from pathlib import Path
import inspect
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import textwrap
import shutil


terminal_wrapper = textwrap.TextWrapper(
	expand_tabs = True,
	tabsize = 4,
	replace_whitespace= False,
	drop_whitespace= False,
	fix_sentence_endings= False,
	break_long_words= True,
	break_on_hyphens=True,
	initial_indent='',
	subsequent_indent=''
	
)
def terminal_wrap(text, indent_level, indent_str='\t'):
	indent = indent_level*indent_str
	terminal_wrapper.width = shutil.get_terminal_size().columns - terminal_wrapper.tabsize*indent_level
	#terminal_wrapper.width = 60 - terminal_wrapper.tabsize*indent_level
	
	wrapped_text_lines = []
	for line in text.splitlines(False):
		wrapped_text_lines += terminal_wrapper.wrap(line)
	return(indent + ('\n'+indent).join(wrapped_text_lines))

def terminal_fill(text):
	assert len(text) > 0, "must have some amount of text to repeat"
	c = shutil.get_terminal_size().columns
	n = c // len(text)
	r = c % len(text)
	return(text*n + text[:r])

def terminal_center(text, fill):
	assert len(text) > 0
	assert len(fill) > 0
	c = shutil.get_terminal_size().columns
	x = (c - len(text))
	h = x//2
	b = x%2
	n = h // len(fill)
	r = h % len(fill)
	return(fill*n + fill[:r] + text + fill[:b] + fill*n + fill[:r])

def main(continue_on_fail=True):
	test_dir = Path(__file__).parent
	sys.path = [str(test_dir), *sys.path]
	print(sys.path)
	print(f'{test_dir=}')


	test_results = {}

	for prefix, dirs, files in os.walk(test_dir, topdown=True):
		dirs[:] = list(filter(lambda adir: not adir.startswith('__'), dirs))
		files[:] = list(filter(lambda afile: (not afile.startswith('__')) and afile.endswith('.py') and ('test' in afile), files))
		
		for item in files:
			module_file = Path(prefix) / item
			
			module_name = str(module_file.relative_to(test_dir)).replace(os.sep, '.')
			if not module_name.startswith('.'): module_name = '.'+module_name
			if module_name.endswith('.py'): module_name = module_name[:-3]
			module = importlib.import_module(module_name, test_dir.name)
			
			print(f'Imported {module}')
			
			for test_name, test_function in inspect.getmembers(module, lambda m: inspect.isfunction(m) and ('test' in m.__name__)):
				test_identifier = f"{test_dir.name}{module_name}::{test_name}"
				test_results[test_identifier] = {}
				
				test_results[test_identifier].update({'stdout': StringIO()})
				
				print(terminal_center(f' Running test "{test_identifier}" ', '='))
				try:
					with redirect_stdout(test_results[test_identifier]['stdout']), redirect_stderr(test_results[test_identifier]['stdout']):
						test_function()
				except BaseException as e:
					print('Failed')
					test_results[test_identifier].update({'success': False, 'message':'\n'.join(traceback.format_exception(e))})
					if not continue_on_fail: raise e
				else:
					test_results[test_identifier].update({'success': True, 'message':''})
					print('Passed')
				finally:
					print()
					
	print(terminal_fill('='))
	for test_identifier, test_result in test_results.items():
		print(("PASSED" if test_result['success'] else "FAILED" )
			+ ': ' 
			+ f'{test_identifier} '
			+ '-'*(shutil.get_terminal_size().columns-len(test_identifier)-9) 
		)
		print(terminal_wrap(test_result['stdout'].getvalue(), 1))
		print(terminal_wrap(test_result["message"], 1))



if __name__=='__main__':
	main()
