 
import sys, os
import importlib
from pathlib import Path
import inspect
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
import textwrap
import shutil
import subprocess as sp

from decorators import TestSkippedException

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

def dict_diff(a, b):
	ak = set(a.keys())
	bk = set(b.keys())
	uk = ak & bk
	ak = ak - uk # keys only in a
	bk = bk - uk # keys only in b
	
	ck = set() # keys changed from a to b
	for k in uk:
		if a[k] != b[k]:
			ck |= {k}
	
	return(ak, bk, ck)


def discover_tests(
		test_dir : str, # directory to start searching for tests in

		 # If this is true, directory will be included in test search
		directory_search_predicate = \
			lambda adir: \
				not adir.startswith('__'),		
		
		# if this is true, modulefile will be included in test search
		modulefile_search_predicate = \
			lambda mf: \
				(not mf.startswith('__')) \
					and mf.endswith('.py') \
					and ('test' in mf), 
		
		# if this is true, the member of the module is a test and will be run
		member_search_predicate = \
			lambda m: \
				inspect.isfunction(m) \
					and ('test' in m.__name__),
	):

	test_discovery_data = {}

	# Perform test discovery
	print(terminal_center(f' Starting test discovery from folder "{test_dir}" ', '='))
	
	for prefix, dirs, files in os.walk(test_dir, topdown=True):
		print(f'{dirs=}')
		print(f'{files=}')
		dirs[:] = list(filter(directory_search_predicate, dirs))
		files[:] = list(filter(modulefile_search_predicate, files))
		
		for item in files:
			package_file = Path(prefix).relative_to(test_dir)
			#module_file = (Path(prefix) / item).relative_to(test_dir)
			module_file = Path(item)
			print(f'Searching "{package_file}/{module_file}", discovering tests:')
			
			
			package_name = str(package_file)
			#if not package_name.startswith('.'):
			#	package_name = '.'+package_name
			#module_name = str(module_file).replace(os.sep, '.')
			module_name = str(module_file)
			
			if module_name.endswith('.py'): 
				module_name = module_name[:-3]
			
			#print(f'{Path(prefix)=} {test_dir=}')
			#print(f'{package_name=} {module_name=}')
			
			if Path(prefix) != test_dir:
				module = importlib.import_module('.'.join((package_name,module_name)))
				full_module_name = '.'.join((package_name, module_name))
			else:
				module = importlib.import_module(module_name)
				full_module_name = module_name
			
			
			test_discovery_data[full_module_name] = {}
			
			for test_name, test_member_callable in inspect.getmembers(module, member_search_predicate):
				print(f'    "{test_name}"')
				tid = f"{full_module_name}::{test_name}"
				
				test_discovery_data[full_module_name][tid] = {'callable':test_member_callable,'name':test_name}

	return test_discovery_data

def run_tests(test_discovery_data, continue_on_fail=True, live_output=False):
	test_results = {}
	for full_module_name, test_data in test_discovery_data.items():
		for tid, td in test_data.items():
			if not live_output:
				print(terminal_left(f'{tid} ', '-', -8), end='')
			else:
				print(terminal_left(f'{tid} ', '-'))
			
			test_results[tid] = {}

			# Enable output capture
			test_results[tid]['stdout'] = StringIO() if not live_output else sys.stdout
			
			
			# Run the test
			try:
				with redirect_stdout(test_results[tid]['stdout']), redirect_stderr(test_results[tid]['stdout']):
					
					# Remember program state before the test
					pre_globals = dict(globals())
					pre_locals = dict(locals())
					for item in ('pre_globals', 'pre_locals', 'new_globals', 'new_locals'):
						if item in pre_locals:
							del pre_locals[item]
					if 'item' in pre_locals:
						del pre_locals['item']
					
					td['callable']()
					
					# Save program state directly after test
					new_globals = dict(globals())
					new_locals = dict(locals())
					for item in ('pre_globals', 'pre_locals', 'new_globals', 'new_locals'):
						if item in new_locals:
							del new_locals[item]
					if 'item' in new_locals:
						del new_locals['item']
					
					# Compare to see if program state has changed
					if pre_globals != new_globals:
						print('WARINING: Test altered global variables see below for details ("-" = removed, "+" = added):')
						pk, nk, ck = dict_diff(pre_globals, new_globals)
						for k in pk:
							print(f"    - '{k}' : {pre_globals[k]}")
						for k in nk:
							print(f"    + '{k}' : {pre_globals[k]}")
						for k in ck:
							print(f"    - '{k}' : {pre_globals[k]}")
							print(f"    + '{k}' : {new_globals[k]}")
							
					if pre_locals  != new_locals:
						print('WARNING: Test altered local variables see below for details ("-" = removed, "+" = added):')
						pk, nk, ck = dict_diff(pre_locals, new_locals)
						for k in pk:
							print(f"    - '{k}' : {pre_locals[k]}")
						for k in nk:
							print(f"    + '{k}' : {new_locals[k]}")
						for k in ck:
							print(f"    - '{k}' : {pre_locals[k]}")
							print(f"    + '{k}' : {new_locals[k]}")
						
					
			except TestSkippedException as e:
				print(' Skipped')
				test_results[tid]['success'] = True
				test_results[tid]['exception'] = e
			except BaseException as e:
				print(' Failed')
				test_results[tid]['success'] = False
				test_results[tid]['exception'] = e
				if not continue_on_fail:
					if not live_output: print(terminal_wrap(td['stdout'].getvalue(),1))
					raise e
			else:
				test_results[tid]['success'] = True
				test_results[tid]['exception'] = None
				print(' Passed')
	return(test_results)



def main(
		test_dir = None, # Top of directory tree to search for tests
		continue_on_fail=False, # Should we continue executing tests if one fails?
		only_report_failures = True, # If true, will only show result details when failures happen.
		live_output=True, # If true, will output to terminal as the tests are run.

		 # If this is true, directory will be included in test search
		directory_search_predicate = \
			lambda adir: \
				not adir.startswith('__'),		
		
		# if this is true, modulefile will be included in test search
		modulefile_search_predicate = \
			lambda mf: \
				(not mf.startswith('__')) \
					and mf.endswith('.py') \
					and ('test' in mf), 
		
		# if this is true, the member of the module is a test and will be run
		member_search_predicate = \
			lambda m: \
				inspect.isfunction(m) \
					and ('test' in m.__name__),

	):
	if test_dir is None: test_dir = Path(__file__).parent 
	#print(f'{test_dir=}')
	
	# Add test_dir to END of sys.path
	sys.path += [str(test_dir)]
	#print(f'{sys.path=}')


	test_discovery_data = discover_tests(test_dir, directory_search_predicate, modulefile_search_predicate, member_search_predicate)

	print()
	print(terminal_center(' Discovery Summary ', '='))
	for full_module_name, test_data in test_discovery_data.items():
		print(f'    module "{full_module_name}" contains tests:')
		for tid, td in test_data.items():
			print(f'        {td["name"]}')
	
	
	print()
	print(terminal_center(' Running Tests ', '='))
	test_results = run_tests(test_discovery_data, continue_on_fail, live_output)


	print()
	if (not only_report_failures) or any(not td['success'] for td in test_results.values()):
		print(terminal_center(' Test Result Details ', '='))
		for tid, td in test_results.items():
			
			if (not only_report_failures) or not td['success']:
				
				msg = None
				if td['success']:
					if td['exception'] is None:
						msg = 'PASSED'
					elif type(td['exception']) is TestSkippedException:
						msg = 'SKIPPED'
					else:
						msg = 'UNKNOWN STATE'
				else:
					msg = 'FAILED'
				
				print(msg
					+ ': ' 
					+ f'{tid} '
					+ '-'*(shutil.get_terminal_size().columns-len(tid)-10) 
				)
				if not live_output:
					print(terminal_wrap(td['stdout'].getvalue(), 1))
				for aline in traceback.format_exception(td["exception"]):
					print(terminal_wrap(aline, 1))



if __name__=='__main__':
	main()
