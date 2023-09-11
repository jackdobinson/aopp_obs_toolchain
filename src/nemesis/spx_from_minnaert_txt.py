#!/usr/bin/env python3

import sys, os
import numpy as np
import glob
import re # regular expressions
import plotutils
import nemesis.copy_inputs
import nemesis.retrieval_set

# TODO:
# * create a retrieval for each minnaert coefficient lat file
# * for each latitude minnaert coefficients file, create a *.spx file
# * copy other retrieval files from a reference retrieval
# * 

def get_retrieval_data_from_minnaert_coefficient_files(minnaert_coefficient_files, spx={}):
	# generate data for each input file
	retrieval_data = []
	
	for i, minnaert_coefficient_file in enumerate(minnaert_coefficient_files):
		print(f'INFO: {minnaert_coefficient_file} {i}')
		# get latitude range from file name
		regex_match_number = '[+-]?\d+(\.\d*)?([eE][+-]?\d+)?'
		regex_pattern = f'lat_(?P<lat_min>{regex_match_number})_to_(?P<lat_max>{regex_match_number})'
		regex_matches = re.compile(regex_pattern).search(minnaert_coefficient_file)
		lat_min = float(regex_matches['lat_min'])
		lat_max = float(regex_matches['lat_max'])
		
		mcf_d = np.load(minnaert_coefficient_file)
		(	wavs, 
			IperF0s, 
			ks, 
			log_IperF0s_var, 
			ks_var, 
			IperF0s_var, 
			n_points, 
			n_used_points
		) = (	mcf_d['wavs'],
				mcf_d['IperF0s'], 
				mcf_d['ks'], 
				mcf_d['log_IperF0s_var'], 
				mcf_d['ks_var'], 
				mcf_d['IperF0s_var'], 
				mcf_d['n_points'], 
				mcf_d['n_used_points']
			)
		
		# wrangle data into *.spx format
		
		zens = (0, 61.45) # zenith angles to calculate minnaert limb darkening at
		solar_zens = (0, 61.45)
		ngeom = len(zens)
		spec_fwhm = spx.get('spectral_fwhm', 0)
		mean_lat = [0.5*(lat_min+lat_max)]*ngeom
		mean_lon = [0]*ngeom
		emission_angle = [180]*ngeom
		fov_weight = [1]*ngeom
		navs = [1]*ngeom
		nconvs = [wavs.size]*ngeom
		fov_averaging_record = []
		spec_record = []
		
		us = np.cos((np.pi/180)*np.array(zens))
		u0s = np.cos((np.pi/180)*np.array(solar_zens))
	
		IperF0s_std = np.sqrt(IperF0s_var)
		ks_std = np.sqrt(ks_var)
		IperFs = np.full((len(us), wavs.size), fill_value=np.nan)
		IperFs_var = np.full((len(us), wavs.size), fill_value=np.nan)
		for j, (u0, u) in enumerate(zip(u0s, us)):
			IperFs[j,:] = calculate_minnaert(IperF0s, ks, u0, u)
			#print(IperF)
			worst_IperFs = np.full((4, wavs.size), fill_value=np.nan)
			IperF0s_lims = np.array([IperF0s+IperF0s_std, IperF0s-IperF0s_std])
			ks_lims = np.array([ks+ks_std, ks-ks_std])
			
			_j = 0
			for _IperF0s in IperF0s_lims:
				for _ks in ks_lims:
					worst_IperFs[_j,...] = calculate_minnaert(_IperF0s, _ks, u0, u)
					_j+=1
			IperFs_var[j,:] = (np.nanmax(worst_IperFs, axis=0) - np.nanmin(worst_IperFs, axis=0))**2
		
		for geom in range(ngeom):
			fov_each_geom = []
			for av in range(navs[geom]):
				fov = 	(
							mean_lat[geom],
							mean_lon[geom],
							solar_zens[geom],
							zens[geom],
							emission_angle[geom],
							fov_weight[geom]
				)
				fov_each_geom.append(fov)
			fov_averaging_record.append(fov_each_geom)
			spec_data = np.full((nconvs[geom], 3), fill_value=np.nan)
			spec_data[:,0] = wavs
			spec_data[:,1] = IperFs[geom]
			spec_data[:,2] = np.sqrt(IperFs_var[geom])
			spec_record.append(spec_data)
		
		
		spx_dict = {
			'fwhm':spec_fwhm,
			'latitude':np.average(mean_lat, weights=fov_weight),
			'longitude':np.average(mean_lon, weights=fov_weight),
			'ngeom':ngeom,
			'nconvs':nconvs,
			'navs':navs,
			'fov_averaging_record':fov_averaging_record,
			'spec_record':spec_record
		}
		
		retrieval_data.append({'spx':spx_dict})
		
		
	return(retrieval_data)

def calculate_minnaert(IperF0, k, u_0, u):
	"""
	Uses (1) to calculate I/F, should work just as well with arrays if they are all
	the same shape.

	# ARGUMENTS #
		IperF0
			<float> Brightness at zenith = 0
		k
			<float> Minnaert parameter that determines limb brightening/darkeing,
			usually between 0 and 1
		u_0
			<float> cos(solar zenith)
		u
			<float> cos(zenith)
	# RETURNS #
		IperF
			<float> Brightness according to (1)
	"""
	return(IperF0*(u_0**k)*(u**(k-1)))


def parse_args(argv):
	"""
	Parses command line arguments, you only need to edit the bottom of this 
	funciton. See https://docs.python.org/3/library/argparse.html
	"""
	import argparse as ap
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	# A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	# see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	# of them do.
	# See https://github.com/python/cpython/blob/ebe20d9e7eb138c053958bc0a3058d34c6e1a679/Lib/argparse.py#L531
	# for source code
	# Quick reference:
	# ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	# ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	# ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	# ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	class RawDefaultTypeFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class RawDefaultFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter, ap.RawDescriptionHelpFormatter):
		def _split_lines(self, text, width):
			#print('-----')
			#print(f"{text[:width]}|{text[width:]}")
			#print(text.split('\n'))
			lines = []
			for aline in text.split('\n'):
				if len(aline)>width:
					for _l in str_wrap(aline, width):
						lines.append(_l)
				else:
					lines.append(aline)
			return(lines)
		def _get_help_string(self, action):
			help = action.help
			help_position = min(self._action_max_length + 2, self._max_help_position)
			width = max(self._width - help_position, 11)
			params = dict(vars(action), prog=self._prog)
			#print('### IN _get_help_string()')
			if '(default' not in action.help:
				if action.default != '==SUPPRESS==':
					defaulting_nargs = ['?', '*']
					if action.option_strings or action.nargs in defaulting_nargs:
						default_str = ''+'\n          '.join(str_wrap(f'{params["default"]}', wrapsize=width-10, split_on=(' ',os.sep)))
						#print(default_str)
						if len(default_str) > width-10:
							help += f'\n(default: {default_str}\n)'
						else:
							help += f'\n(default: {default_str})'
			return help
		def _expand_help(self, action):
			ahelp = super()._expand_help(action)
			return(ahelp)
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	
	# a load of helper string functions
	def str_count_start_list(astr, slist):
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
	def charlen(astring, tabsize=4):
		slen = len(astring)
		ntabs = astring.count('\t')
		charlen = slen - ntabs + ntabs*tabsize
		return(charlen)
	def str_block_indent_raw(astr, from_indent_str=(' ', '\t'), tabsize=8, wrapsize=50):
		"""treat each line as a block of text"""
		bi_string = ''
		for aline in astr.split('\n'):
			indent_str = str_count_start_list(aline, from_indent_str)
			#print(nindent)
			wrapped_string_list = str_wrap(aline.strip(), wrapsize-charlen(indent_str, tabsize=tabsize))
			#print(wrapped_string_list)
			bi_string += '\n'.join([indent_str+ws for ws in wrapped_string_list]) + '\n'
		return(bi_string)
	def str_rationalise_newline_for_wrap(astr):
		"""Remove new lines that have text directly after them, do not remove them if they have whitespace directly after them"""
		if astr is None:
			return('')
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
	def str_dict(adict, depth=0, indentstr='\t', wrapsize=54, end_new_line=False):
		"""Returns a string that contains the formatted output of a dictionary"""
		keys = adict.keys()
		maxlen_k = max(map(len, map(str, keys)))
		kv_fmt_str = depth*indentstr+'{:'+'{}'.format(maxlen_k)+'} {}\n'
		k_fmt_str = depth*indentstr+'{:'+'{}'.format(maxlen_k)+'}\n'
		astr = ''
		for k,v in sorted(adict.items()):
			if type(v)==dict:
				astr+=k_fmt_str.format(k)
				astr+= str_dict(v, depth=depth+1)
			elif type(v) in (tuple, list):
				kstr = k_fmt_str.format(k)
				astr+=kstr
				vstr = ''.join([(depth+1)*indentstr+'{}\n'.format(x) for x in v])
				astr+=vstr
			else: 
				astr += kv_fmt_str.format(k, ('\n'+((maxlen_k+1)*' ')).join(str_wrap(f'{v}', wrapsize=wrapsize-(maxlen_k+1))))
		if not end_new_line:
			astr = astr[:-1]
		return(astr)

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=str_block_indent_raw(
										str_rationalise_newline_for_wrap(__doc__),
										wrapsize=79),
									formatter_class = TextDefaultTypeFormatter,
									epilog=str_block_indent_raw(
										str_rationalise_newline_for_wrap('END OF USAGE'), 
										wrapsize=79)
								)
	# ====================================

	#===================== ARGUMENTS GO BELOW THIS LINE ======================#

	parser.add_argument('minnaert_coefficient_files', type=str, nargs='+', help='A *.npz files that have minnaert coefficients in them')
	
	parser.add_argument('--spx.spectral_fwhm', type=float, help='Spectral full with half maximum. If -ve will use <runname>.fil to specify a filter function, if 0 will use specified values from a k-table, if +ve will use a top-hat function to average over wavelengths around those specified in the *.spx file.', default=0)
	
	retrieval_template_set_group, retrieval_set_group = nemesis.retrieval_set.add_argument_groups(parser)
	
	plotutils.add_plot_arguments(parser)

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)


# ==================== MAIN SCRIPT BELOW THIS LINE ========================== #
if __name__=='__main__':
	if sys.argv[0] == '':
		sys.argv = ['',
					*glob.glob(os.path.expanduser('~/scratch/reduced_images/SINFO.combined/minnaert_clean_20/combined_minnaert_coefficients_lat_*_to_*.npz')),
					'--retrieval_set.dest', os.path.expanduser('~/scratch/nemesis_run_results/slurm/combined_minnaert_clean_20_txt_000'),
					'--retrieval_template_set.dirs', *glob.glob(os.path.expanduser('~/scratch/nemesis_run_results/nemesis_run_templates/neptune/2cloud_ch4_imajrefidx_std/fix_iri_larger_haze_err/0[0,2,4]/0[0,3,5]/'))
					]
		
	args = parse_args(sys.argv[1:])
	print('# ARGUMENTS #')
	for key, value in args.items():
		print(f'\t{key}')
		print(f'\t\t{value}')
	print('#-----------#')
	
	
	n_templates = len(args['retrieval_template_set.dirs'])
	template_set = nemesis.retrieval_set.RetrievalTemplateSet(	args['retrieval_template_set.dirs'], 
													[args['retrieval_template_set.n_search_parents']]*n_templates, 
													[args['retrieval_template_set.runname']]*n_templates
												)
		
	if args['retrieval_set.slurm_job_name'] is None:
		args['retrieval_set.slurm_job_name'] = args['retrieval_set.dest'].split(os.sep)[-1]
		
	retrieval_set = nemesis.retrieval_set.RetrievalSet(args['minnaert_coefficient_files'],
													args['retrieval_set.dest'],
													args['retrieval_set.runname'],
													args['retrieval_set.overwrite'],
													args['retrieval_set.tags'],
													template_set,
													slurm_job_name = args['retrieval_set.slurm_job_name']
												)
	
	retrieval_set.set_identifiers([s[s.find('lat'):s.find('.npz')] for s in retrieval_set.input_files])
	
	# we want to overwrite the template's *.spx file with our own data.
	retrieval_set.set_retrieval_data_function(get_retrieval_data_from_minnaert_coefficient_files, rdf_kwargs={'spx':{'spectral_fwhm':args['spx.spectral_fwhm']}})
	
	# USING THIS TO TRICK THE CODE INTO NOT OVERWRITING ANYTHING
	# set a dummy retrieval data function when we don't want to alter the template
	# must return a list of dictionaries for each input file.
	#retrieval_set.set_retrieval_data_function(lambda x: [{}]*len(x))
	
	retrieval_set.create()
