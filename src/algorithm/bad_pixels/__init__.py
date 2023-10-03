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

def fix(a, bp_map, method, window=1,boundary='reflect',const=0):
	match method:
		case 'simple':
			return nph.array.mask.interp.constant(a,bp_map,const)
		case 'mean':
			return nph.array.mask.interp.mean(a,bp_map,window,boundary,const)
		case 'interp':
			return nph.array.mask.interp.interp(a,bp_map)
		case _:
			raise RuntimeError(f'Unknown case {method}')
			
	
