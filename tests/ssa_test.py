
import scientest.decorators

import numpy as np
import matplotlib.pyplot as plt

from aopp_deconv_tool.algorithm.interpolate.ssa_interp import ssa_intepolate_at_mask
import aopp_deconv_tool.scipy_helper as scipy_helper
from aopp_deconv_tool.py_ssa import SSA
import aopp_deconv_tool.estimate_noise as estimate_noise
import common_metrics as common_metrics

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')

@scientest.decorators.pass_args(1E-9)
def test_ssa_residuals_are_within_limits(frac_residual_limit):
	import sys
	
	# get example data in order of desirability
	test_data_type_2d = ('fractal','random')
	test_data_type_1d = ('random',)
	
	dataset = []
	
	# find 1d testing data
	for data_type in test_data_type_1d:
		match data_type:
			case 'random':
				np.random.seed(100)
				n = 1000
				w = 10
				data1d = np.convolve(np.random.random((n,)), np.ones((w,)), mode='same')
			case _:
				raiseValueError(f'Unknown type of test data for 1d case {repr(data_type)}')
		dataset.append(('1d_'+data_type, data1d))
		
	# 2d datasets
	for data_type in test_data_type_2d:
		match data_type:
			case 'fractal':
				try:
					import PIL
					data2d = np.asarray(PIL.Image.effect_mandelbrot((60,50),(0,0,1,1),100))
				except ImportError:
					print('WARNING: Could not import PIL, using next most desirable example data')
			case 'random':
				data2d = np.random.random((61,51))
			case _:
				raise ValueError(f'Unknown type of test data for 2d case {repr(data_type)}')
		dataset.append(('2d_'+data_type, data2d))	
	
	
	for data_name, data2d in dataset:
		_lgr.info(f'TESTING: 2d ssa with {data_name} example data')
		_lgr.info(f'{data2d.shape=}')
		window_size = tuple(s//10 for s in data2d.shape)
		data = data2d.astype(np.float64)
		ssa = SSA(
			data, 
			window_size,
			rev_mapping='fft',
			svd_strategy='eigval', # uses less memory and is faster
			grouping={'mode':'elementary'}
		)
		
		noise_estimate = estimate_noise.corners_standard_deviation(data)
		
		reconstruction = np.sum(ssa.X_ssa, axis=0)
		residual = data - reconstruction
		frac_residual = np.sqrt((residual/(data + noise_estimate))**2)
		
		assert not np.any(np.isnan(frac_residual)), f"{data_name}, require no NANs in fractional residual"
		assert not np.any(np.isinf(frac_residual)), f"{data_name}, require no INFs in fractional residual"
		assert np.all(frac_residual < frac_residual_limit), f'{data_name}, require frac_residual < {frac_residual_limit} everywhere but highest value is {np.max(frac_residual)}'




#@scientest.decorators.debug
def ssa_interpolation_test():
	
	
	
	np.random.seed(100)
	a = np.indices((21,33)).astype(float)
	a[0] -= 10
	a[1] -= 15
	a = -np.sum(a*a, axis=0)
	a -= np.min(a)
	a += np.random.normal(a, 10)
	
	n_idxs = 20
	idxs = []
	for s in a.shape:
		idxs.append(np.random.choice(range(s), n_idxs, replace=True))
	
	idxs = tuple(zip(idxs))
	bp_map = np.zeros_like(a, dtype=bool)
	bp_map[idxs] = True
	
	a_original = np.array(a)
	plt.close('all')
	plt.title('original data')
	plt.imshow(a_original)
	plt.show()
	
	a[idxs] = 50
	
	
	
	plt.title('corrupted data')
	plt.imshow(a)
	plt.show()
	
	plt.clf()
	plt.title('bp_map')
	plt.imshow(bp_map)
	plt.show()
	
	ssa = SSA(
		a,
		grouping={'mode':'elementary'},
		#grouping={'mode':'similar_eigenvalues', 'tolerance':0.01},
	)
	
	a_interp = ssa_intepolate_at_mask(
		ssa, 
		bp_map,
		start=0, 
		stop=None, 
		value=0.5, 
		show_plots=0,
	)
	
	b_interp = scipy_helper.interp.interpolate_at_mask(
		a,
		bp_map,
		edges = None
	)
	
	
	ssa_interp_frac_error = common_metrics.FracError(a_interp, a_original)
	normal_interp_frac_error = common_metrics.FracError(b_interp, a_original)
	interp_frac_error_lims = [min(ssa_interp_frac_error.min, normal_interp_frac_error.min), max(ssa_interp_frac_error.max, normal_interp_frac_error.max)]
	
	plt.clf()
	plt.title('ssa interpolated data')
	plt.imshow(a_interp)
	plt.show()
	
	plt.clf()
	plt.title('(ssa_interpolated - original)^2 / original')
	plt.imshow(ssa_interp_frac_error.value, vmin=interp_frac_error_lims[0], vmax=interp_frac_error_lims[1])
	plt.show()
	
	plt.clf()
	plt.title('normal interpolated data')
	plt.imshow(b_interp)
	plt.show()
	
	plt.clf()
	plt.title('(normal interpolated - original)^2 / original')
	plt.imshow(normal_interp_frac_error.value, vmin=interp_frac_error_lims[0], vmax=interp_frac_error_lims[1])
	plt.show()
	
	
	eps = 1E-9
	plt.clf()
	plt.title(f'pdf of interp_frac_error, not including values < {eps}')
	norm_bins, norm_counts = normal_interp_frac_error.pdf
	plt.hist(normal_interp_frac_error.value[normal_interp_frac_error.value > eps], bins=int(np.sqrt(np.count_nonzero(normal_interp_frac_error.value > eps))), label='normal', alpha=0.5)
	plt.hist(ssa_interp_frac_error.value[ssa_interp_frac_error.value > eps], bins=int(np.sqrt(np.count_nonzero(ssa_interp_frac_error.value > eps))), label='ssa', alpha=0.5)
	plt.legend()
	plt.show()
	
	plt.clf()
	plt.title(f'interp_frac_error, not including values < {eps}')
	plt.plot(normal_interp_frac_error.value[normal_interp_frac_error.value > eps], ssa_interp_frac_error.value[ssa_interp_frac_error.value > eps], 'b+', label='frac error data')
	x = np.linspace(0, min(np.max(normal_interp_frac_error.value[normal_interp_frac_error.value > eps]), np.max(ssa_interp_frac_error.value[ssa_interp_frac_error.value > eps])), 10)
	plt.plot(x, x, 'r--', label='x = y line')
	plt.xlabel('normal')
	plt.ylabel('ssa')
	plt.show()
	

	
	
	ts_threshold = 0.1
	ssa_test_statistic = common_metrics.mean_frac_error(a_interp, a_original)
	normal_test_statistic = common_metrics.mean_frac_error(b_interp, a_original)
	
	_lgr.debug(f'{ts_threshold=} {ssa_test_statistic=} {normal_test_statistic=}')
	
	assert ssa_test_statistic < ts_threshold, f'{ssa_test_statistic=}, target is < {ts_threshold}'
	assert normal_test_statistic < ts_threshold, f'{normal_test_statistic=}, target is < {ts_threshold}'
	assert ssa_test_statistic <= normal_test_statistic, f'Want ssa to work as well as or better than normal interpolation, have {ssa_test_statistic=} {normal_test_statistic=}'











