#!/usr/bin/env python3

import sys
import os
import ast
import importlib
import re
import shutil


def parse_args(argv):
	import argparse
	parser = argparse.ArgumentParser(description=__doc__)

	parser.add_argument('files', type=str, nargs='+', help='files to find modules for')

	parser.add_argument('--rebase.from', type=str, help='If present, will rebase modules so that ones with this path are at the top of the tree.', default=None)
	parser.add_argument('--rebase.to', type=str, help='When rebasing, will copy files to this location for the rebasing process.', default='./build/scripts')
	parser.add_argument('--rebase.include_non_user', action='store_true', help='Should we include non-user defined scripts and modules in the rebase process?', default=False)

	args = vars(parser.parse_args(argv))

	return(args)

def get_modules_in_file(afile):
	print(f'INFO: Reading file "{afile}"')
	modules = set()

	def visit_Import(node):
		for name in node.names:
			#modules.add(name.name.split(".")[0])
			modules.add(name.name)


	def visit_ImportFrom(node):
		if node.module is not None and node.level ==0:
			#modules.add(node.module.split(".")[0])
			modules.add(node.module)


	node_iter = ast.NodeVisitor()
	node_iter.visit_Import = visit_Import
	node_iter.visit_ImportFrom = visit_ImportFrom

	with open(afile) as f:
		node_iter.visit(ast.parse(f.read()))
	return(modules)


def get_rebase_paths(specs, module_paths, rebase_from, rebase_to):
	"""
	TODO: This could definitely be tidied up, I wrote it as a stream of conciousness.
	"""
	module_paths = [os.path.abspath(module_path) for module_path in module_paths]
	rebase_from = os.path.abspath(rebase_from)
	rebase_to = os.path.abspath(rebase_to)
	
	top_modules = []
	lower_modules = []
	for spec, m_path in zip(specs, module_paths):
		if m_path.startswith(rebase_from):
			top_modules.append((spec, m_path[len(rebase_from)+1:]))
		else:
			lower_modules.append((spec, m_path))
	
	lm_paths = [x[1] for x in lower_modules]
	print(lm_paths)
	lm_groups = ['']*len(lm_paths)

	abs_sys_path = [os.path.abspath(x) for x in sys.path]
	for i in range(len(lm_paths)):
		for prefix_path in abs_sys_path:
			if lm_paths[i].startswith(prefix_path) and (len(prefix_path) > len(lm_groups[i])):
				lm_groups[i] = prefix_path
	
	path_groups = dict((x,[]) for x in set(lm_groups))
	

	for i in range(len(lm_paths)):
		lm_paths[i] = lm_paths[i][len(lm_groups[i])+1:] # need a +1 here for some reason
		path_groups[lm_groups[i]].append([i,lm_paths[i]])

	print(lm_paths)

	for k, v in path_groups.items():
		print([x[1] for x in v])
		common_path = os.path.commonpath([x[1] for x in v])
		for i in range(len(v)):
			v[i][1] = v[i][1][len(common_path):]
	
	print(path_groups)

	n = 1
	temp_keys = tuple(path_groups.keys())
	for i, k1 in enumerate(temp_keys):
		for j, k2 in enumerate(temp_keys):
			if j<=i: continue
			common_path = os.path_commonpath([x[1] for x in path_groups[k1]]+[x[1] for x in path_groups[k2]])
			if len(common_path)>0:
				path_groups[k2] = [(i,f'temp_{n}'+os.sep+p) for i, p in path_groups[k2]]
				n += 1
	
	for k, v in path_groups.items():
		for x in v:
			lm_paths[x[0]] = x[1]

	common_path = os.path.commonpath(lm_paths+[x[1] for x in top_modules])
	if len(common_path) > 0:
		lm_paths = ['modules'+os.sep+x for x in lm_path]

	m_dests = []
	top_specs = [x[0] for x in top_modules]
	lower_specs = [x[0] for x in lower_modules]
	for spec in specs:
		if spec in top_specs:
			m_dests.append(top_modules[top_specs.index(spec)][1])
		if spec in lower_specs:
			m_dests.append(lm_paths[lower_specs.index(spec)])


	for i, m_path in enumerate(module_paths):
		print(m_path)
		if m_path.endswith('__init__.py'):
			if not any([os.path.dirname(x)==os.path.dirname(m_path) if i!=j else False for j, x in enumerate(module_paths)]):
				m_dests[i] = None
	
	print(m_dests)
	m_names = [x.replace(os.sep,'.').replace('.__init__.py','').replace('.py','').replace('__init__','') if x is not None else None for x in m_dests]
	m_dests = [os.path.join(rebase_to,x) if x is not None else None for x in m_dests]
	return(m_names, m_dests)



def find_import_statements(afile):
	print(f'####################################################### {afile}')
	module_names = []
	module_as_names = []
	import_statement_pos = [] # start line, start column, end line, end column
	import_statement_type = [] # "import", "from"

	def visit_Import(node):
		print([name.name for name in node.names])
		for name in node.names:
			#modules.add(name.name.split(".")[0])
			import_statement_pos.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
			import_statement_type.append('import')
			module_names.append(name.name)
			module_as_names.append(name.asname)


	def visit_ImportFrom(node):
		if node.module is not None and node.level ==0:
			print(node.module)
			import_statement_type.append('from')
			import_statement_pos.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
			#modules.add(node.module.split(".")[0])
			module_names.append(node.module)
			module_as_names.append(None)


	node_iter = ast.NodeVisitor()
	node_iter.visit_Import = visit_Import
	node_iter.visit_ImportFrom = visit_ImportFrom

	with open(afile) as f:
		node_iter.visit(ast.parse(f.read()))
	return(module_names, import_statement_pos, import_statement_type)



################ MAIN ROUTINE ####################################
def main():
	args = parse_args(sys.argv[1:])
	all_modules = set()
	rebase_modules = set()

	file_list = args['files'][:] # copy

	for afile in file_list:
		modules = get_modules_in_file(afile)
		for module in modules:
			spec = importlib.util.find_spec(module)
			is_on_python_path = os.path.abspath(spec.origin).startswith(os.path.abspath(os.environ['PYTHONPATH'])) if ((spec is not None) and (type(spec.origin) is str)) else False
			if is_on_python_path or args['rebase.include_non_user']:
				if spec.origin not in file_list: file_list.append(spec.origin) # search imported files as well as passed ones, but only once each
				if args['rebase.from'] is not None: 
					rebase_modules.add(module)
			all_modules.add(module)
		

	print('All modules found:')
	for x in sorted(all_modules):
		print(f'\t{x}')

	print('Modules to rebase:')
	for x in sorted(rebase_modules):
		print(f'\t{x}')

	m_specs, m_paths = [], []
	if args['rebase.from'] is not None:
		for m in rebase_modules:
			spec = importlib.util.find_spec(m)
			m_path = spec.origin
			m_paths.append(m_path)
			m_specs.append(spec)
			#m_dest = os.path.join(args['rebase.to'], *m.split('.')) 
			#	+ (os.path.sep+'__init__.py' if m_path.endswith(os.path.sep+'__init__.py') else '.py')

		m_names, m_dests = get_rebase_paths(m_specs, m_paths, args['rebase.from'], args['rebase.to'])


		rb_path = os.path.abspath(args['rebase.from'])
		for m_path, m_dest,m_name, in zip(m_paths, m_dests, m_names):
			m_path = os.path.abspath(m_path)
			print(f'DEBUG: COPY\n\t{m_name}\n\tFROM \n\t{m_path} \n\tTO \n\t{m_dest}')
			

		for m_path, m_dest in zip(m_paths, m_dests):
			if m_dest is not None:
				folder = os.path.dirname(m_dest)
				if not os.path.exists(folder):
					os.makedirs(folder)
				shutil.copy(m_path, m_dest)


		m_spec_names = [spec.name for spec in m_specs]
		for m_dest in m_dests + args['files']:
			if m_dest is not None:
				print(f'\n################################## {m_dest} ###')
				with open(m_dest, 'r') as f:
					file_contents = f.read()

				# find all "import" statements in the file
				i_names, i_poss, i_types = find_import_statements(m_dest)
				print(len(i_names), len(i_types), len(i_poss))

				for i_name, i_type in zip(i_names, i_types):
					# loop over the import statements, no use passing positions as they will change
					# as we alter the file
					print(i_name)

					# Find the module in our list of modules that should be rebased
					try:
						m_name = m_names[m_spec_names.index(i_name)]
					except ValueError:
						m_name='NOTHING FOUND'

					
					if (i_name in m_spec_names) and (m_name != 'NOTHING FOUND'):
						# if the module name we're operating on is one that should
						# be rebased, then make a regular expression that matches
						# the "import MODULE_NAME" or "from MODULE_NAME import ..." statement
						
						# Desired regex:
						# ^\s*(${IMPORT_TYPE}\s+${IMPORT_NAME})(\s|\n)
						# Explanation:
						# ^ 				match start of line
						# \s* 				match any whitespace indent
						# ( 				start a new group (group 1, group 0 is the whole regex)
						# ${IMPORT_TYPE}	match the string corresponding to the import type, "import" or "from"
						# \s+				match at least 1 bit of whitespace
						# ${IMPORT_NAME}	match the name of the module, need to escape any "." characters as
						# 					otherwise the regex will treat them as matching any character, not 
						#					literal dot.
						# )					end group 1
						# (					start a new group, group 2
						# \s|\n				match either a whitespace character, or a new line (do I need the \n?)
						# ) 				end group 2
						regex_str = r'^\s*(' + i_type + r'\s+' + i_name.replace('.','\\.') + r')(\s|\n)'

						# compile the regex, we're going to want to use this repeatedly
						regex = re.compile(regex_str, flags=re.MULTILINE)

						# create the replacement string, if m_name is None or empty, that means we
						# should not include the import statement in the new version. Only do this
						# when we don't need the import as it will break code otherwise.
						replacement = f"{i_type} {m_name}" if m_name not in (None, '') else ''

						print(repr(regex_str), repr(replacement))
						
						# Try to find a single match for the regex in 'file_contents' from the start
						replacement_substituted_flag = False
						match = regex.search(file_contents)
						while match: # if a regex does not match anything, the return value evaluates to False.
							# get the start and end points of the match for group 1 (the part that matches the
							# import statement)
							span = match.span(1)
							print(match, repr(file_contents[span[0]:span[1]]))
							print(repr(replacement))
							
							if replacement != match.group(1):
								# if our replacement string is different from the one that we matched,
								# replace the matched import statement with the 'replacement' string
								file_contents = file_contents[:span[0]] + replacement + file_contents[span[1]:]
								replacement_substituted_flag = True

							# try to find another match for the regex, start searcing where the
							# previous replacement ends
							match = regex.search(file_contents, span[0]+len(replacement))

						# for some reason, using "re.sub()" failed to work correctly
						# I am not sure why, but I can emulate it's behaviour so it's not important
						#file_contents = re.sub(regex_str, r'\g<1>'+replacement, file_contents, count=0, flags=re.M)

						if replacement_substituted_flag and (replacement != ''):
							# if we performed a replacement, and we didn't just
							# delete the import statement, we should change all the
							# other occurances of the module identifier in the code 
							# as well.
							# NOTE: I may need to switch this to use a regex so
							# 		it will match a whole word and not change sub-strings
							file_contents = file_contents.replace(i_name, m_name)
							
						
						print(i_name, i_name in [spec.name for spec in m_specs], m_name)
						print(i_type)
						print()
				
				# overwrite the file with the altered contents.
				with open(m_dest, 'w') as f:
					f.write(file_contents)



if __name__=='__main__':
	main()


