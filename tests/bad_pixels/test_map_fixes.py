
import decorators
import numpy as np
import algorithm.bad_pixels as bp


# NOTE: Better way to test these functions would be to generate an array of a
#		known function, e.g. (r-r_0)**2, mask out some points with NANs and 
#		INFs, then use the "bp.fix()" function. Compare the result with the
#		input array

#@decorators.argument_set(0, (1*np.ones((7,7)), 2*np.ones((7,7))))
#@decorators.pass_args(np.ones((7,7)))
@decorators.pass_args#(2*np.ones((7,7)))
def test_bp_map_simple_fix(a):
	
	nans = (
		(0,0),
		(0,1),
		(3,5)
	)
	infs = (
		(1,1),
		(2,2),
		(6,3)
	)

	#a = np.ones((7,7))
	print(a)
	expected_result = np.array(a)

	for coord in nans:
		a[coord] = np.nan
		expected_result[coord] = 0.0
	
	for coord in infs:
		a[coord] = np.inf
		expected_result[coord] = 0.0

	bp_map = bp.get_map(a)
	bp.fix(a, bp_map, 'simple')

	assert np.all(np.equal(a, expected_result)), \
		f"Array should have no NAN or INF elements, and should have zeros at {list(tuple(x) for x in np.argwhere(expected_result == 0))}. " \
		+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(a)))}. " if np.any(np.isnan(a)) else "") \
		+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(a)))}. " if np.any(np.isinf(a)) else "") \
		+ f"Has zeros at {list(tuple(x) for x in np.argwhere(a==0))}."



@decorators.argument('a', np.indices((7,7)))
def test_bp_map_mean_fix(a):
	nans = (
		(0,0),
		(0,1),
		(3,5)
	)
	infs = (
		(1,1),
		(2,2),
		(6,3)
	)

	#a = np.ones((7,7))
	#a = np.indices((7,7))
	#a = np.array(a[0],float)
	print(f'{a=}')
	a = np.sqrt(np.sum(a*a, 0))
	a_old = np.array(a)
	for coord in nans:
		a[coord] = np.nan
		
	for coord in infs:
		a[coord] = np.inf
		
	bp_map = bp.get_map(a)
	
	#try:
	#	bp.fix(a, bp_map, 'mean', window_shape=3)
	#except RuntimeError:
	#	pass 
	#else:
	#	assert False, "Window shape = 3 should fail as there are two bad pixels next to eachother"
	
	
	a_results = [
		bp.fix(a, bp_map, 'mean', window=1, boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=np.array([[1,1,1],[0,0,0],[1,1,1]]), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=np.array([1,1,1]), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=(2,1), boundary='reflect'),
		bp.fix(a, bp_map, 'mean', window=1, boundary='pacman'),
		bp.fix(a, bp_map, 'mean', window=1, boundary='const',const=1),
		#bp.fix(a_old, bp_map, 'mean', window_shape=5) # testing
	]
	
	#import matplotlib.pyplot as plt
	#import plot_helper as ph
	#fig, ax = ph.figure_n_subplots(3)
	#ax[0].imshow(a_old)
	#ax[1].imshow(bp_map)
	#ax[2].imshow(a)
	#plt.show()
	
	for i, ar in enumerate(a_results):
		assert np.all(~np.isnan(ar)) and np.all(~np.isinf(ar)), \
			"Array {i} should have no NAN or INF elements. " \
			+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(ar)))}. " if np.any(np.isnan(ar)) else "") \
			+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(ar)))}. " if np.any(np.isinf(ar)) else "") 
	
	#assert False, "TESTING"



def test_bp_map_interp_fix():
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
