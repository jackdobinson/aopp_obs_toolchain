"""
Routines for identifying and fixing bad pixels in observation data.
"""
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array.mask
import aopp_deconv_tool.numpy_helper.array.mask.interp

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')

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
			
	
