"""
Runs discovers and runs tests in parent folder. Used as an alternative to pytest
that is more configurable and I know how it works.

## Magic Variables ##

scientest.test_output_dir
	A directory to store test output in. A new output directory is created for each run
	based off of the time the run is started. They all live in the "test_output" folder
	that is created in the current working directory by default.

TODO:
	* Enable test discovery to be influenced by decorators, will need to make
	the list of discovered tests a module scope variable. But the extra
	configuration flexibility will be worth it.
	* Turn this into it's own package, it's useful enough that it would be
	useful, possible name "scientest"
	* Move all of the text handling code into it's own module.
"""

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
import datetime as dt
from typing import Callable
import logging
import random

import scientest.cfg.logs
import scientest.cfg.settings

_lgr = scientest.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


from scientest.term_output import terminal_right, terminal_left, terminal_center, terminal_wrap #,terminal_fill, 
import scientest.discover


from scientest.decorators import TestSkippedException
from scientest.output_directory import TestsOutputDirectory

import scientest.intercepts.matplotlib


def dict_diff(a : dict, b : dict) -> tuple[set,set,set]:
	'''
	Accepts two dictionaries: `a` and `b`
	
	Returns:
	
	tuple(ak, bk, ck)
	
	ak : set
		Keys *only* in `a`
		
	bk : set
		Keys *only* in `b`
	
	ck : set
		Keys *changed* from `a` to `b`
	'''
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




@contextmanager
def redirect_logging(stream, old_handler : logging.Handler = None):
	if old_handler is None:
		old_handler = logging.getLogger().handlers[0]
	logger = logging.getLogger()
	logger.removeHandler(old_handler)
	
	new_handler : logging.Handler = logging.StreamHandler(stream)
	new_handler.setFormatter(old_handler.formatter)
	logger.addHandler(new_handler)
	
	yield
	
	logger.removeHandler(new_handler)
	logger.addHandler(old_handler)


def run_tests(test_discovery_data, continue_on_fail=True, live_output=False):
	"""
	Run discovered tests
	"""
	
	# Hacky way to get scientest module on path for now
	#sys.path.append(Path(__file__).parent.parent)
	
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
			
			
			module_full_name, test_callable_name = tid.split('::')
			scientest.test_output_dir = TestsOutputDirectory(
				scientest.cfg.settings.root_test_output_dir / module_full_name.replace('.',os.sep),
				dir_fmt="{test_callable_name}",
				format_parameter_providers = {"test_callable_name":lambda : test_callable_name}
			)
			
			# Run the test
			try:
				with (	redirect_stdout(test_results[tid]['stdout']), 
						redirect_stderr(test_results[tid]['stdout']), 
						redirect_logging(test_results[tid]['stdout'])
					):
					
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
				if not live_output:
					del test_results[tid]['stdout'] # delete cached output on skip
			except BaseException as e:
				print(' Failed')
				test_results[tid]['success'] = False
				test_results[tid]['exception'] = e
				if not continue_on_fail:
					if not live_output: print(terminal_wrap(test_results[tid]['stdout'].getvalue(),1))
					raise e
			else:
				test_results[tid]['success'] = True
				test_results[tid]['exception'] = None
				print(' Passed')
				if not live_output:
					del test_results[tid]['stdout'] # delete cached output on pass
	return(test_results)



def main(
		test_dir = None, # Top of directory tree to search for tests, if None use parent directory of this file
		continue_on_fail=True, # Should we continue executing tests if one fails?
		only_report_failures = True, # If true, will only show result details when failures happen.
		live_output=False, # If true, will output to terminal as the tests are run.

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


	test_discovery_data = scientest.discover.discover_tests(test_dir, directory_search_predicate, modulefile_search_predicate, member_search_predicate)

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
	main(sys.argv[1])
