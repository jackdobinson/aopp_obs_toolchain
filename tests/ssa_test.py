
import scientest.decorators

import numpy as np
from py_ssa import SSA
import estimate_noise

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

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