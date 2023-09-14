



import numpy as np

import numpy_helper as nph
import numpy_helper.axes
import numpy_helper.slice

def test_axes_to_end():
	
	a = np.ones((2,3,4,5))
	b = np.ones((9,8,33,7))
	
	
	with nph.axes.to_end(a, (0,1)) as aa:
		assert aa.shape == (4,5,2,3), f"Array should be reordered to shape (4,5,2,3) instead have {aa.shape}"
	
	with nph.axes.to_end(b, (2,0)) as bb:
		assert bb.shape == (8,7,33,9), f"Array should be reordered to shape (8,7,33,9) instead have {bb.shape}"


def test_slice_get_indices():
	a = np.ones((6,7,9))
	
	aslicet = (slice(1,5), slice(0,7), slice(1,3))
	
	idxs = nph.slice.get_indices(a, aslicet)
	assert a[idxs].shape == (4,7,2), f"indices of slices should give something the same shape as the slice, gives {a[idxs].shape}"
	
	for x in nph.slice.get_indices(a, aslicet, group=(2,)):
		assert a[x].shape == (4,7), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"
	
	for x in nph.slice.get_indices(a, aslicet, group=(1,)):
		assert a[x].shape == (4,2), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"

	for x in nph.slice.get_indices(a, aslicet, group=(0,)):
		assert a[x].shape == (7,2), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"

	for x in nph.slice.get_indices(a, aslicet, group=(1,2)):
		assert a[x].shape == (4,), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"

	for x in nph.slice.get_indices(a, aslicet, group=(0,1)):
		assert a[x].shape == (2,), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"

	for x in nph.slice.get_indices(a, aslicet, group=(0,2)):
		assert a[x].shape == (7,), f"iterating over grouped indices should give a subset of the slice, gives {a[idxs].shape}"

	for x in nph.slice.get_indices(a, aslicet, group=(0,1,2)):
		assert type(a[x]) == np.float64, f"grouping all the indices should give individual elements {a[x]=}"
	
