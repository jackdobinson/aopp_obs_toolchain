"""
Routines for identifying and fixing bad pixels in observation data.
"""
import numpy_helper as nph
import numpy_helper.array.mask
import numpy_helper.array.mask.interp

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'INFO')

def get_map(a):
	return nph.array.mask.from_nan_and_inf(a)

def fix(a, bp_map, method, window_shape=3):
	match method:
		case 'simple':
			return nph.array.mask.interp.constant(a,bp_map,0)
		case 'mean':
			return nph.array.mask.interp.mean(a,bp_map,window_shape)
		case 'interp':
			return nph.array.mask.interp.interp(a,bp_map)
		case _:
			raise RuntimeError(f'Unknown case {method}')
			
	
