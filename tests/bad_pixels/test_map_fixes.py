
import numpy as np
import algorithm.bad_pixels as bp
import random

import scientest.decorators


random.seed(0) # ensure repeatable values

# NOTE: Better way to test these functions would be to generate an array of a
#		known function, e.g. (r-r_0)**2, mask out some points with NANs and 
#		INFs, then use the "bp.fix()" function. Compare the result with the
#		input array

arg_set_1 = (
	(
		(0,0),
		(0,1),
		(3,5)
	),
	(
		(1,1),
		(2,2),
		(6,3)
	),
	np.indices((7,7)).sum(axis=0).astype(float)
)

gen_random_arg_set = lambda i, n, ndim=2: (
	((random.randrange(0,i) for _ in range(ndim)) for _ in range(n)),
	((random.randrange(0,i) for _ in range(ndim)) for _ in range(n)),
	np.sqrt((np.indices(tuple(i for _ in range(ndim)))**2).sum(axis=0))
)

@scientest.decorators.pass_args(*arg_set_1)
@scientest.decorators.pass_arg_sets(gen_random_arg_set(i,6) for i in range(7,10))
def test_bp_map_simple_fix(nans, infs, array : np.ndarray[[...],float]):
	a = array
	print(nans)
	print(infs)
	#a = np.ones((7,7))
	print(a)
	expected_result = np.array(a)

	for coord in nans:
		coord = tuple(coord)
		a[coord] = np.nan
		expected_result[coord] = 0.0
	
	for coord in infs:
		coord = tuple(coord)
		a[coord] = np.inf
		expected_result[coord] = 0.0

	bp_map = bp.get_map(a)
	bp.fix(a, bp_map, 'simple')

	assert np.all(np.equal(a, expected_result)), \
		f"Array should have no NAN or INF elements, and should have zeros at {list(tuple(x) for x in np.argwhere(expected_result == 0))}. " \
		+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(a)))}. " if np.any(np.isnan(a)) else "") \
		+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(a)))}. " if np.any(np.isinf(a)) else "") \
		+ f"Has zeros at {list(tuple(x) for x in np.argwhere(a==0))}."



@scientest.decorators.pass_args(*arg_set_1)
@scientest.decorators.pass_arg_sets(gen_random_arg_set(i, 6) for i in range(7,10))
@scientest.decorators.pass_arg_sets(gen_random_arg_set(i, 9, 3) for i in range(4,8))
def test_bp_map_mean_fix(nans, infs, a):
	print(f'{a=} {nans=} {infs=}')
	a_old = np.array(a)
	for coord in nans:
		print(coord)
		a[tuple(coord)] = np.nan
		
	for coord in infs:
		a[tuple(coord)] = np.inf
	
	print(a)
	
	bp_map = bp.get_map(a)
	
	a_results = [
		bp.fix(a, bp_map, 'mean', window=1, boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=np.array([[1,1,1],[0,0,0],[1,1,1]]), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=np.array([1,1,1]), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=(2,1), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=1, boundary='pacman'),
		bp.fix(a, bp_map, 'mean', window=1, boundary='const',const=1),
		#bp.fix(a_old, bp_map, 'mean', window_shape=5) # testing
	]
	
	for i, ar in enumerate(a_results):
		assert np.all(~np.isnan(ar)) and np.all(~np.isinf(ar)), \
			"Array {i} should have no NAN or INF elements. " \
			+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(ar)))}. " if np.any(np.isnan(ar)) else "") \
			+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(ar)))}. " if np.any(np.isinf(ar)) else "") 
	

def create_example_data():
	nans = (
		(0,0,0),
		(0,1,0),
		(0,1,1),
		(3,5,2)
	)
	infs = (
		(1,1,0),
		(2,2,1),
		(6,3,2)
	)

	a = np.indices((7,7,3))
	a = np.sqrt(np.sum(a*a, 0))
	for coord in nans:
		a[coord] = np.nan
		
	for coord in infs:
		a[coord] = np.inf
	return a

def test_bp_map_interp_fix():
	
	a = create_example_data()
	
	a_old = np.array(a)
	
	bp_map = bp.get_map(a)
	
	bp.fix(a, bp_map, 'interp', window=3)
	
	bp.fix(a, bp_map, 'interp', window=5)
	
	"""
	import matplotlib.pyplot as plt
	import plot_helper as ph
	fig, ax = ph.figure_n_subplots(9)
	ax[0].imshow(a_old[:,:,0])
	ax[1].imshow(bp_map[:,:,0])
	ax[2].imshow(a[:,:,0])
	ax[3].imshow(a_old[:,:,1])
	ax[4].imshow(bp_map[:,:,1])
	ax[5].imshow(a[:,:,1])
	ax[6].imshow(a_old[:,:,2])
	ax[7].imshow(bp_map[:,:,2])
	ax[8].imshow(a[:,:,2])
	plt.show()
	"""

	assert np.all(~np.isnan(a)) and np.all(~np.isinf(a)), \
		"Array should have no NAN or INF elements. " \
		+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(a)))}. " if np.any(np.isnan(a)) else "") \
		+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(a)))}. " if np.any(np.isinf(a)) else "") 


@scientest.decorators.debug
def test_bp_map_ssa_sum_prob():
	import algorithm.bad_pixels.ssa_sum_prob
	from py_ssa import SSA
	import matplotlib.pyplot as plt
	import itertools as it
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
	a[idxs] = 50
	
	plt.imshow(a)
	plt.show()
	
	ssa = SSA(
		a,
		grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
	)
	
	#for item in ssa.X_ssa:
	#	plt.imshow(item)
	#	plt.show()
	#	plt.hist(item.flatten(), bins=100)
	#	plt.show()
	
	
	ssa.plot_ssa([ssa.X_ssa.shape[0]//16, ssa.X_ssa.shape[0]//4, 2*ssa.X_ssa.shape[0]//4, 3*ssa.X_ssa.shape[0]//4])
	plt.show()
	
	algorithm.bad_pixels.ssa_sum_prob.ssa2d_sum_prob_map(
		ssa,
		start = 3,
		stop = None,
		value=0.95,
		show_plots=2,
		weight_by_evals=False,
		transform_value_as = ['ppf']
	)
	
