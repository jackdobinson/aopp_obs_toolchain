

import numpy as np
import algorithm.bad_pixels.simple as simple_bp



def test_simple_map_gets_nans_and_infs():
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

	a = np.ones((7,7))
	expected_result = np.array(a)

	for coord in nans:
		a[coord] = np.nan
		expected_result[coord] = 0.0
	
	for coord in infs:
		a[coord] = np.inf
		expected_result[coord] = 0.0

	bp_map = simple_bp.get_map(a)
	simple_bp.fix(a, bp_map)

	assert np.all(np.equal(a, expected_result)), \
		f"Array should have no NAN or INF elements, and should have zeros at {list(tuple(x) for x in np.argwhere(expected_result == 0))}. " \
		+ (f"Has NANs at {list(tuple(x) for x in np.argwhere(np.isnan(a)))}. " if np.any(np.isnan(a)) else "") \
		+ (f"Has INFs at {list(tuple(x) for x in np.argwhere(np.isinf(a)))}. " if np.any(np.isinf(a)) else "") \
		+ f"Has zeros at {list(tuple(x) for x in np.argwhere(a==0))}."



