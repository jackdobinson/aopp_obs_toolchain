#!/usr/bin/env python3
"""
Runs Nemesis with zero iterations to calculate the jacobian (K). Uses K, along with the a-priori covariance
matrix, and the measurement error to tune the forward model error such that (K Sa Kt)/Se = 1

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



def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	ut.pINFO('This script was last modified on {}'.format(os.path.getmtime(__file__)))
	log_fmt = 'ntfme_{}.log'
	input_file='{}.nam'.format(args['runname'])
	error_balance_ratio = -1.0
	nemesis_runtime_list = []
	idx = 0
	while idx < args['max_iter']:
		ut.pINFO('In loop iteration {}'.format(idx))
		logfile = log_fmt.format(idx)

		ut.pINFO('Opening logs and input files')
		nemesis_start = dt.datetime.now()
		if not ((args['debug_skip_initial_loop']) and (idx==0)):
			with open(logfile, 'w') as lf, open(input_file, 'r') as inpf:
				ut.pINFO('Running NEMESIS process')
				if len(nemesis_runtime_list) >0:
					ut.pINFO('The previous NEMESIS runtime was {}'.format(nemesis_runtime_list[-1]))
				nemesis_process = subprocess.run([args['nemesis_command']], stdout=lf, stdin=inpf, 
								stderr=subprocess.STDOUT)
			if nemesis_process.returncode != 0:
				ut.pERROR('\n'.join([	'loop idx {}'.format(idx),
										'input_file {}'.format(input_file),
										'logfile {}'.format(logfile),
										'command {}'.format(args['nemesis_command']),
										'error_balance_ratio {} (starting value -1)'.format(
											error_balance_ratio),
										'Exited with code {}'.format(nemesis_process.returncode)
									]))
				return(nemesis_process.returncode)
		else:
			ut.pWARN('Skipped NEMESIS computation')
		nemesis_end = dt.datetime.now()
		nemesis_runtime_list.append(nemesis_end-nemesis_start)
		ut.pINFO('This runtime was {}'.format(nemesis_runtime_list[-1]))


		ut.pINFO('Calculating error adjustment')
		covd = nemesis.read.cov(args['runname'])
		print(ut.str_wrap_in_tag(ut.str_dict(covd), 'covd'))

		if args['atmospheric_error_adjustment']=='ignore':
			# read forward model error
			fme_d = nemesis.read.forward_model_error(args['fm_err_file'])
			print(ut.str_wrap_in_tag(ut.str_dict(fme_d), 'fme_d'))

			# Calculate trace of a-priori error matrix
			tr_KSaKt = np.trace(np.matmul(np.matmul(covd['kk'], covd['sa']), covd['kt']))
			ut.pINFO('tr_KSaKt {}'.format(tr_KSaKt))
			
			# Calculate sum of absolute values of a-priori error matrix
			sum_abs_KSaKt = np.sum(np.abs(np.matmul(np.matmul(covd['kk'], covd['sa']), covd['kt'])))
			ut.pINFO('sum_abs_KSaKt {}'.format(sum_abs_KSaKt))	
		
			# Find the square of the average of the forward model error and multiply by the number of data points 
			# to get the total squared forward model error
			fme_sq = covd['ny']*np.mean(fme_d['fme_we'][:,1])**2
			ut.pINFO('fme_sq {}'.format(fme_sq))
			
			# Find the squared measurement error from the covariance matrix
			se1 = np.sum(np.abs(covd['se1']))
			ut.pINFO('se1 {}'.format(se1))

			# set Nwavs as the number of wavelengths we care about
			nwavs_in_band = covd['ny']
			wavs_in_band = np.ones(se1.shape)

		elif args['atmospheric_error_adjustment']=='in_band':
			# read forward model error
			fme_d = nemesis.read.forward_model_error(args['fm_err_file'])
			print(ut.str_wrap_in_tag(ut.str_dict(fme_d), 'fme_d'))

			# read *.spx file
			spxd = nemesis.read.spx(args['runname'])
			wavelengths = spxd['spec_record'][0][:,0] # assume only 1 ngeom per *.spx file
			wavs_in_band = nemesis.common.which_band(wavelengths)
			wavs_in_band = np.array([1 if type(b)!=type(None) else 0 for b in wavs_in_band])
			nwavs_in_band = sum(wavs_in_band)
			ut.pINFO('wavs_in_band {}'.format(wavs_in_band))

			# Calculate a-priori error matrix
			KSaKt = np.matmul(np.matmul(covd['kk'], covd['sa']), covd['kt'])
			# Find the trace, with elements that represent out-of-band wavelengths set to zero
			tr_KSaKt = np.sum(np.diag(KSaKt)*wavs_in_band)
			ut.pINFO('tr_KSaKt {}'.format(tr_KSaKt))

			# Find which forward model wavelength points are inside a band
			fme_in_band = nemesis.common.which_band(fme_d['fme_we'][:,0])
			fme_in_band = np.array([1 if type(b)!=type(None) else 0 for b in fme_in_band])
			# only include the errors of the forward model wavelengths that are in a band
			fme_err = np.array([fe for i, fe in enumerate(fme_d['fme_we'][:,1]) if fme_in_band[i]==1])
			fme_sq = np.sum(fme_err**2) #nwavs_in_band*(np.mean(fme_err)**2)
			ut.pINFO('fme_sq {}'.format(fme_sq))

			# Only select the squared measurement error that is from an in-band wavelength
			se1 = np.sum([x for i,x in enumerate(np.abs(covd['se1'])) if wavs_in_band[i]==1])
			ut.pINFO('se1 {}'.format(se1))

		

		elif args['atmospheric_error_adjustment']=='weight_by_inverse':
			ut.pERROR('argument option "--atmospheric_error_adjustment weight_by_inverse" not implemented yet, exiting...')
			sys.exit()

		# at this point, tr_KSaKt, fme_err, fme_sq, se1 are all weighted by whatever atmospheric adjustement function we want to use

		total_m_err = se1 # forward model error already included, not just error from spx file
		total_a_err = tr_KSaKt
		im_err_sq = total_m_err - fme_sq # im_err_sq = initial measurement error squared, 
	
		# Should error_balance_ratio be per degrees of freedom or not?
		#error_balance_ratio = total_m_err/(covd['ny']*total_a_err) # ratio per dof
		error_balance_ratio = total_m_err/(total_a_err) # ratio NOT per dof
		im_err_factor = total_a_err/im_err_sq # number you need to multiply 'im_err_sq' by to get 'total_a_err'
		err_factor = 1.0/error_balance_ratio
		
		ut.pINFO('total_m_err {}'.format(total_m_err))
		ut.pINFO('total_a_err {}'.format(total_a_err))
		ut.pINFO('im_err_sq {}'.format(im_err_sq))
		ut.pINFO('im_err_factor {}'.format(im_err_factor))
		ut.pINFO('error_balance_ratio {}'.format(error_balance_ratio))
		ut.pINFO('err_factor {}'.format(err_factor))

		"""
		Let
			e_m = measurement error (the one in the *.spx file)
			e_f = forward model error
			e^2 = e_m^2 + e_f^2, the square of the quadrature added error on the measurement from all sources, the elements of se1
			e_a = the a-priori error, the elements of Trace(KSaKt)

		NOTE:
			Check e^2 = e_m^2 + e_f^2 by comparing the output of <runname>.cov, <runname>.spx, and 'forwardnoise.dat' for a COMPLETED nemesis run.

		Therefore for an initial nemesis run we have

			SUM(e^2) = SUM(e_m^2 + e_f^2) = SUM(e_m^2) + SUM(e_f^2) = total_m_err
		
		and

			SUM(e_a^2) = total_a_err,
			SUM(e_f^2) = fme_sq,

		We want to adjust e_f in such a way that total_m_err = total_a_err, to do this there are 2 cases

		CASE 1:
			let e_f' = a*e_f, such that

				SUM(e_m^2) + SUM(e_f'^2) = SUM(e_a^2) = total_a_err

			therefore

				SUM(e_m^2) + a^2*SUM(e_f^2) = total_a_err

				SUM(e_m^2) + a^2*SUM(e_f^2) - SUM(e_m^2) - SUM(e_f^2) = total_a_err - total_m_err

				(a^2 - 1)*SUM(e_f^2) = total_a_err - total_m_err

				a^2 = (total_a_err - total_m_err)/fme_sq + 1

		CASE 2:
			let e_f' = a*e_m, such that

				SUM(e_m^2) + SUM(e_f'^2) = SUM(e_a^2) = total_a_err

			therefore

				(a^2 + 1)*SUM(e_m^2) =  total_a_err

				SUM(e_m^2) = total_m_err - fme_sq

			so

				(a^2 + 1)*(total_m_err - fme_sq) = total_a_err

				a^2 = total_a_err/(total_m_err - fme_sq) - 1

		"""

		# calculate new forward model error needed
		if abs(error_balance_ratio - 1.0) < args['stopping_criteria']:
			break
		
		if args['method'] == 'uniform':
			#fme_sq_factor = (total_m_err*(err_factor-1.0)/fme_sq) + err_factor
			#fme_d['fme_we'][:,1] *= np.sqrt(fme_sq_factor)
			#new_fme_sq = (total_a_err - im_err_sq)/wavs_in_band
			new_fme_sq = (total_a_err - total_m_err)/fme_sq + 1.0
			fme_d['fme_we'][:,1] = np.sqrt(new_fme_sq)
	
		elif args['method'] == 'scale_measurement_err':
			#fme_sq_factor = (err_factor-1) + err_factor*fme_sq/total_m_err
			#new_fme_sq = im_err_factor - 1
			spxd = nemesis.read.spx(args['runname'])
			#meas_err_sq = np.sum((spxd['spec_record'][0][:,2]*wavs_in_band)**2)
			#new_fme_sq = (total_a_err/meas_err_sq -1)**2
			new_fme_sq = (total_a_err/(total_m_err - fme_sq)) - 1.0
			fme_we = np.zeros((spxd['nconvs'][0],2))
			fme_we[:,0] = spxd['spec_record'][0][:,0]
			fme_we[:,1] = spxd['spec_record'][0][:,2]*np.sqrt(new_fme_sq)
			fme_d['fme_we'] = fme_we
			fme_d['fme_n'] = fme_we.shape[0]
		else:
			ut.pERROR('Unknown option "{}" to argument --method, exiting...'.format(args['method']))
			sys.exit()

		if new_fme_sq < 0:
			ut.pERROR('A negative "new_fme_sq" means that the measurement error is too high, this cannot be fixed by changing the forward modelling error. Exiting...')	
			sys.exit(1)
		# write to forward model error file
		nemesis.write.forward_model_error(fme_d, filename=args['fm_err_file'])
		
		ut.pINFO('new_fme_sq {}'.format(new_fme_sq))
		ut.pINFO('new_fme {}'.format(np.sqrt(new_fme_sq)))
		
		# DEBUGGING #
		fme_in_band = nemesis.common.which_band(fme_d['fme_we'][:,0])
		fme_in_band = np.array([1 if type(b)!=type(None) else 0 for b in fme_in_band])
		fme_err = np.array([fe for i, fe in enumerate(fme_d['fme_we'][:,1]) if fme_in_band[i]==1])
		fme_sq = nwavs_in_band*(np.mean(fme_err)**2)
		ut.pDEBUG('next fme_sq {}'.format(fme_sq))
		# END DEBUGGING #

		idx += 1
	if idx >=  args['max_iter']:
		ut.pWARN('Maximum number ({}) of iterations reached'.format(args['max_iter']))

	ut.pINFO('Automatic error finding finished')
	nrs = [x.total_seconds() for x in nemesis_runtime_list]
	nrs_sum = dt.timedelta(seconds=sum(nrs))
	nrs_min = dt.timedelta(seconds=min(nrs))
	nrs_max = dt.timedelta(seconds=max(nrs))
	nrs_mean = dt.timedelta(seconds=np.mean(nrs))
	ut.pINFO('Total NEMESIS runtime {}'.format(nrs_sum))
	ut.pINFO('Minimum NEMESIS runtime {}'.format(nrs_min))
	ut.pINFO('Maximum NEMESIS runtime {}'.format(nrs_max))
	ut.pINFO('Mean NEMESIS runtime {}'.format(nrs_mean))
	
	
	return(0)


def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	# A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	# see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	# of them do.
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
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap(__doc__), wrapsize=79),
									formatter_class = RawDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('runname', type=str, help='name of the run to tune forward model error for')
	parser.add_argument('--stopping_criteria', type=float, 
						help='How close to 1.0 we have to get before we stop iterating', default=0.1)
	parser.add_argument('--nemesis_command', help='Nemesis command to execute', type=str,
						default='/home/dobinson/bin/Nemesis')
	parser.add_argument('--fm_err_file', type=str, help='Forward model error file', 
						default='forwardnoise.dat')
	parser.add_argument('--debug_skip_initial_loop', action='store_true', 
						help='Debug argument, enables skipping of the first NEMESIS calculation')
	parser.add_argument('--method', type=str, help='How should we find and describe the forward noise',
						choices=('uniform', 'scale_measurement_err'),
						default='scale_measurement_err')
	parser.add_argument('--max_iter', type=int, help='Maximum number of iterations to perform',
						default=5)
	parser.add_argument('--atmospheric_error_adjustment', type=str, 
						help='How should we deal with large errors from atmospheric absorption that dominate the error sum and make the tuned forward model error too small?',
						choices=('ignore', 'in_band', 'weight_by_inverse'),
						default='in_band')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)


import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
import utils as ut # used for convenience functions
import nemesis.read
import nemesis.write
import nemesis.cfg
import nemesis.common
import subprocess
import numpy as np
import datetime as dt
if __name__=='__main__':
	main(sys.argv[1:])
