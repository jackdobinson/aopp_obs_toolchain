#!/usr/bin/env python3

import sys, os
import stat
import datetime as dt
import nemesis.copy_inputs
import nemesis.write
import slurm
import plotutils

# TODO:
# * create a retrieval for each minnaert coefficient lat file
# * for each latitude minnaert coefficients file, create a *.spx file
# * copy other retrieval files from a reference retrieval
# * 



class RetrievalTemplateSet:
	"""
	Holds the details about a set of nemesis retrieval templates
	"""
	def __init__(self, dirs, n_search_parents, runnames):
		self.dirs = dirs
		self.n_search_parents = n_search_parents
		self.runnames = runnames
		self.identifiers = None
		return
		
	def set_identifiers(self, identifiers):
		self.identifiers = identifiers
		return



class RetrievalSet:
	"""
	Class that creates a set of retrievals from a set of templates and a set of input files
	
	You MUST give the class a function using the "set_retrieval_data_function()" method
	"""
	def __init__(self, input_files, dest, dest_runname, dest_overwrite, tags, template_set, slurm_job_name='nemesis'):
		self.data = None
		self.identifiers=None
		self.retrieval_data_function = None
		self.input_files = input_files
		self.template_set = template_set
		self.dest = dest
		self.dest_runname = dest_runname
		self.dest_overwrite = dest_overwrite
		self.tags = tags
		self.slurm_job_name=slurm_job_name
		return
		
	def _create_retrieval_set_from_templates(self):
		slurm_submission_scripts = []
		
		if self.template_set.identifiers is None:
			self.template_set.set_identifiers(get_unique_path_part(self.template_set.dirs))
			
		for i, retrieval_template_dir in enumerate(self.template_set.dirs):
			current_retrieval_dest_dir = os.path.abspath(os.path.join(self.dest, self.template_set.identifiers[i]))
			slurm_job_scripts = []
			
			for j, (r_data, r_dir) in enumerate(zip(self.data, self.identifiers)):
				current_retrieval_dir = os.path.abspath(os.path.join(current_retrieval_dest_dir, r_dir))
				os.makedirs(current_retrieval_dir, exist_ok=True)
				
				nc_args = [	retrieval_template_dir, 
							current_retrieval_dir,
							'--src.runname', self.template_set.runnames[i],
							'--dest.runname', self.dest_runname,
							'--src.n_search_parents', self.template_set.n_search_parents[i],
							'--nemesis.tags', *self.tags
							]
				if not self.dest_overwrite:
					nc_args.append('--dest.no_overwrite')
				
				nemesis.copy_inputs.main(list(map(str,nc_args)))
				
				# overwrite copied template data with our new data
				self._write_retrieval_data(r_data, os.path.join(current_retrieval_dir, self.dest_runname))
				
				# create slurm submission scripts in each directory
				current_slurm_job_script = slurm.create_job_script(current_retrieval_dir, job_name=self.slurm_job_name)
				slurm_job_scripts.append(current_slurm_job_script)
				
				# change permissions of job script to give read,write,execute permission to user,
				# read permission to group and other
				os.chmod(current_slurm_job_script, stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
	
		   
			# create master submission script at top level of directory
			retrieval_set_slurm_submission_script = slurm.create_submission_script(current_retrieval_dest_dir, slurm_job_scripts)
			slurm_submission_scripts.append(retrieval_set_slurm_submission_script)
			os.chmod(retrieval_set_slurm_submission_script, stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
				
		# create a global slurm submission script
		gss = slurm.create_global_script(self.dest, slurm_submission_scripts)
		os.chmod(gss, stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
		
		# create a global output removal script
		gsocs = slurm.create_global_output_clean_script(self.dest)
		os.chmod(gsocs, stat.S_IRWXU|stat.S_IRGRP|stat.S_IROTH)
		
		# create a notes file that contains the inputs for the just created set of retrievals
		notesf = self._create_notes_file()
		return()
	
	
	
	def _write_retrieval_data(self, retrieval_data, runname):
		"""
		Takes a dictionary consisting of key-value pairs that describe a file type (keys) and the data to write to them
		(values). 
		
		Expand this as we want to write more data files
		"""
		rundir = os.path.dirname(runname)
		if 'spx' in retrieval_data:
			nemesis.write.spx(retrieval_data['spx'], runname=runname)
		if 'set' in retrieval_data:
			nemesis.write.set(retrieval_data['set'], runname=runname)
		return
	
	def _create_notes_file(self):
		notesf = os.path.join(self.dest, 'notes.txt')
		# string helper functions
		def str_wrap(astr, wrapsize=79, split_on=(' ',)):
			#print('###')
			split_str = []
			last_space = 0
			last_split = 0
			for i, ch in enumerate(astr):
				#print(i, ch)
				if ch in split_on:
					last_space = i
				if ch=='\n':
					split_str.append(astr[last_split:i])
					last_split = i
				if i - last_split >= wrapsize:
					if i-last_space < wrapsize:
						split_str.append(astr[last_split:last_space])
						last_split = last_space
					else:
						split_str.append(astr[last_split:i])
						last_split = i
			split_str.append(astr[last_split:])
			#print('---')
			return([ss.strip() for ss in split_str]) 
		
		def str_dict(adict, depth=0, indentstr='\t', wrapsize=140, end_new_line=False):
			"""Returns a string that contains the formatted output of a dictionary"""
			keys = adict.keys()
			maxlen_k = max(map(len, map(str, keys)))
			kv_fmt_str = depth*indentstr+'{:'+'{}'.format(maxlen_k)+'} {}\n'
			k_fmt_str = depth*indentstr+'{:'+'{}'.format(maxlen_k)+'}\n'
			astr = ''
			for k,v in sorted(adict.items()):
				if type(v)==dict and len(v)>0:
					astr+=k_fmt_str.format(k)
					astr+= str_dict(v, depth=depth+1)
				elif type(v) in (tuple, list):
					kstr = k_fmt_str.format(k)
					astr+=kstr
					vstr = ''.join([(depth+1)*indentstr+'{}\n'.format(x) for x in v])
					astr+=vstr
				else: 
					astr += kv_fmt_str.format(k, ('\n'+((maxlen_k+1)*' ')).join(str_wrap(f'{v}', wrapsize=wrapsize-(maxlen_k+1))))
			if end_new_line:
				astr += '\n'
			return(astr)
		
		def str_wrap_in_tag(astr, starttag, endtag=False, indent=True, indentstr='\t', numhashes=3):
			"""
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
		
				indentstr='\t'
					string to use for indentation
				
				numhashes=3
					Wraps the start and end tags in 'numhashes' '#' characters? 
		
			RETURNS
				a string wrapped in a starting and ending tag, and possibly indented
		
			EXAMPLE
				>>> str_wrap_in_tag('this string\nwill be\nwrapped', 'WRAPPER')
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
		
		vardict = vars(self)
		vardict['template_set'] = vars(vardict['template_set'])
		vardict.pop('data')
		vardict['retrieval_data_function'] =f"{vardict['retrieval_data_function'].__name__}() from file \"{vardict['retrieval_data_function'].__globals__.get('__file__', '__main__')}\""
		with open(notesf, 'w') as f:
			f.write('This file keeps a record of the command that created this directory structure.\n')
			f.write('Script "{}"\n'.format(__file__))
			f.write('Called at {}\n'.format(dt.datetime.now()))
			f.write('Called using arguments\n')
			f.write(str_wrap_in_tag(str_dict(vardict), 'ARGUMENTS'))
			f.write('\n') # ut.str_wrap_in_tag does not include a final newline character
			f.write('\n')
		return(notesf)
	
	def set_identifiers(self, identifiers):
		self.identifiers = identifiers
		return
	
	def set_retrieval_data_function(self, rdf, rdf_args=[], rdf_kwargs={}):
		"""
		rdf(self.input_files, *rdf_args, **rdf_kwargs) 
			A function that returns a list of dictionaries, each dictionary contains the data
			needed to write a nemesis file using the "nemesis.write" module.
		
		rdf_args
			arguments to be passed to 'retrieval_data_function()'
			
		rdf_kwargs
			keword arguments to be passed to 'retrieval_data_function()'
		"""
		self.retrieval_data_function=rdf
		self.rdf_args = rdf_args
		self.rdf_kwargs = rdf_kwargs
		return
	
	def get_retrieval_data_function(self):
		return(self.retrieval_data_function, self.rdf_args, self.rdf_kwargs)
		
	def create(self, retrieval_data_function=None, rdf_args=[], rdf_kwargs={}):
		"""
		actually creates the retrieval set
		"""
		
		# get a unique name for each input file if it hasn't already been set
		if self.identifiers is None:
			self.set_identifiers(get_unique_shared_substring(self.input_files))
		
		# ensure we have set a function that gets the data for each retrieval
		if self.retrieval_data_function is None:
			print('ERROR: in "RetrievalSet.create()", "self.retrieval_data_function()" must be set. Use "RetrievalSet.set_retrieval_data_function()"')
			sys.exit()
			
		# generate data for each input file
		self.data = self.retrieval_data_function(self.input_files, *self.rdf_args, **self.rdf_kwargs)
		
		# create retrievals in the destination directory
		self._create_retrieval_set_from_templates()
		return
	
	
# Helper functions below this line -------------------------------------------#
def get_unique_path_part(paths):
		"""Get a unique part of a path when the paths passed all share a common prefix or suffix"""
		common_prefix = os.path.commonpath(paths)
		common_suffix = os.path.commonpath([p[::-1] for p in paths])[::-1]
		print(common_prefix)
		print(common_suffix)
		unique_path_part = []
		for p in paths:
			unique_path_part.append('.'+p[len(common_prefix):len(p)-(len(common_suffix))])
		return(unique_path_part)

def get_unique_shared_substring(strs):
	# find shared prefixes
	i = 0
	while all([s[:i]==strs[0][:i] for s in strs]):
		i+=1
	# find shared suffixes
	j = -1
	while all([s[j:]==strs[0][j:] for s in strs]):
		j-=1
	return([s[i:j] for s in strs])

def add_argument_groups(parser):
	import utils as ut
	rts_grp = parser.add_argument_group(title='Retrieval Template Set Arguments', description='Controls which retrievals are used as templates')
	rts_grp.add_argument('--retrieval_template_set.dirs', type=str, nargs='+',
						help='Folders to use as retrieval templates, will copy files from these folders if they are not created by the routine',
						default=[os.path.expanduser('~/scratch/nemesis_run_results/nemesis_run_templates/neptune/2cloud_ch4_imajrefidx_std/fix_iri_larger_haze_err/02/03')]
						)
	rts_grp.add_argument('--retrieval_template_set.runname', type=str, help='runname of the retrieval to copy from the templates', default='neptune')
	rts_grp.add_argument('--retrieval_template_set.n_search_parents', type=int, help='Number of parent directories that are searched if there are required retrieval input files not in the template directories', default=5)
	
	rs_grp = parser.add_argument_group(title='Retrieval Set Arguments', description='Controls how the set of retrievals are treated')
	rs_grp.add_argument('--retrieval_set.overwrite', action=plotutils.ActionTf, prefix='retrieval_set.', help='Should we overwrite files that are already present in the retrieval set destination?')
	rs_grp.add_argument('--retrieval_set.dest', type=str, help='directory to write the set of retrieval to, will give each input file a unique directory within this one', default=os.path.expanduser('~/scratch/slurm/nemesis'))
	rs_grp.add_argument('--retrieval_set.runname', type=str, help='runname to use for each created retrieval in the set', default='neptune')
	
	nemesis_tags = {'LBL':'Layer by layer calcuation',
					'LIMB':'Calculation is of a limb',
					'GP':'Calculation is of a giant planet',
					'REF':'Calcualtion is of a reflectance spectrum',
					'TRA':'Calculation is of a transit spectrum',
					'SCA':'Calculation includes scattering',
					'CHI':'Channel integratons are required',
					'RFL':'Include extra reflecting layer calculations the output',
					'VPF':'Include a file that limits specified gases volume mixing ratio by saturation',
					'RDW':'Includes a file that contains ranked wavelengths, NEMESIS will fit highest ranked wavelengths first, reduces computational time, incompatible with SCA',
					'ZEN':'Includes a file that specifies where the zenith angle is measured from, if not present zenith angle is measured from bottom of deepest layer'
					}
	rs_grp.add_argument('--retrieval_set.tags', type=str, nargs='+', choices=nemesis_tags.keys(), help=f'Sets the type of retrieval for the whole set, different types require different information.\n{ut.str_dict(nemesis_tags)}', default=['SCA', 'REF', 'GP'])
	rs_grp.add_argument('--retrieval_set.slurm_job_name', type=str, help='name to give to retrieval when it is in the Slurm queue, 8 characters max.', default='nemesis')
	
	return(rts_grp, rs_grp)