

def test_clean_modified_starts():
	import numpy as np
	
	import sys
	print(sys.path)
	
	from algorithm.deconv.clean_modified import CleanModified
	
	n_iter = 10
	
	deconv = CleanModified()
	
	
	result = deconv(np.ones((7,7)), np.ones((3,3)), n_iter=n_iter)
	
	assert result[2] == n_iter, f"Expect {n_iter} iterations of CleanModified, have {result[2]} instead."
	return
	
