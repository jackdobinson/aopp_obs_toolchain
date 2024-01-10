
import scientest.check
from optimise_compat import PriorParam, PriorParamSet

def test_prior_params_constructed_correctly():
	prior_param_list = [
		PriorParam('c1',(0,1),True,0),
		PriorParam('v1',(-1,2),False,1),
		PriorParam('c2',(-2,3),True,2),
		PriorParam('v2',(-3,4),False,3)
	]
	
	ppset_expected_name_index_map = {'c1':0,'v1':1,'c2':2,'v2':3}
	
	ppset = PriorParamSet(*prior_param_list)
	
	for k,v in ppset.param_name_index_map.items():
		assert ppset_expected_name_index_map[k] == v, f'{ppset_expected_name_index_map[k]} != {v}, {k=}'
		
def test_callable_wrapper_for_scipy_parameter_order():
	ppset = PriorParamSet(
		PriorParam('c1',(0,1),True,0),
		PriorParam('v1',(-1,2),False,1),
		PriorParam('c2',(-2,3),True,2),
		PriorParam('NotPresent',(-99,-99),True,-99),
		PriorParam('v2',(-3,4),False,3),
	)
	
	def afunc(x1,c2,y1,v2):
		return f'{x1=} {c2=} {y1=} {v2=}'
	
	arg_to_param_name_map_1 = {'x1':'c1', 'y1':'v1'}
	
	new_func, var_params, const_params = ppset.wrap_callable_for_scipy_parameter_order(afunc, arg_to_param_name_map_1)
	
	expected=['v1','v2']
	assert scientest.check.listlike_is_identical(var_params, expected), f'{var_params=} {expected=}'
	
	expected = ['c1','c2']
	assert scientest.check.listlike_is_identical(const_params, expected), f'{const_params=} {expected=}'
	
	expected_result = 'x1=2 c2=3 y1=0 v2=1'
	test_result = new_func((0,1),2,3)
	assert test_result == expected_result, f'{test_result=} {expected_result=}'
	
	
	# test another case
	
	arg_to_param_name_map_2 = {'x1':'v1', 'y1':'c1'}
	
	new_func, var_params, const_params = ppset.wrap_callable_for_scipy_parameter_order(afunc, arg_to_param_name_map_2)
	
	assert scientest.check.listlike_is_identical(var_params, ['v1','v2'])
	assert scientest.check.listlike_is_identical(const_params, ['c1','c2'])
	
	expected_result = 'x1=0 c2=3 y1=2 v2=1'
	test_result = new_func((0,1),2,3)
	assert test_result == expected_result, f'{test_result=} {expected_result=}'
	
def test_callable_wrapper_for_ultranest_parameter_order():
	ppset = PriorParamSet(
		PriorParam('c1',(0,1),True,0),
		PriorParam('v1',(-1,2),False,1),
		PriorParam('c2',(-2,3),True,2),
		PriorParam('v2',(-3,4),False,3)
	)
	
	def afunc(x1,c2,y1,v2):
		return f'{x1=} {c2=} {y1=} {v2=}'
	
	arg_to_param_name_map_1 = {'x1':'c1', 'y1':'v1'}
	
	new_func, params = ppset.wrap_callable_for_ultranest_parameter_order(afunc, arg_to_param_name_map_1)
	
	expected = ['c1','v1','c2','v2']
	assert scientest.check.listlike_is_identical(params, expected), f'{params=} {expected=}'
	
	expected_result = 'x1=0 c2=2 y1=1 v2=3'
	test_result = new_func((0,1,2,3))
	assert test_result == expected_result, f'{test_result=} {expected_result=}'
	
	# test another case
	
	arg_to_param_name_map_1 = {'x1':'v1', 'y1':'c1'}
	
	new_func, params = ppset.wrap_callable_for_ultranest_parameter_order(afunc, arg_to_param_name_map_1)
	
	expected = ['c1','v1','c2','v2']
	assert scientest.check.listlike_is_identical(params, expected), f'{params=} {expected=}'
	
	expected_result = 'x1=1 c2=2 y1=0 v2=3'
	test_result = new_func((0,1,2,3))
	assert test_result == expected_result, f'{test_result=} {expected_result=}'