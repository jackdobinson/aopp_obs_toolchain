



import numpy as np

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice
import aopp_deconv_tool.numpy_helper.array

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
	print(f'{len(idxs)=}')
	print(f'{idxs}')
	
	assert a[idxs].shape == (4,7,2), f"indices of slices should give something the same shape as the slice, gives {a[idxs].shape}"
	
	for x in nph.slice.iter_indices(a, aslicet, group=(0,1,2)):
		assert a[x].shape == (4,7,2), f"Iterating over the whole slice of indices should not do anything, gives {a[x].shape}"
	
	for x in nph.slice.iter_indices(a, aslicet, group=(0,1)):
		assert a[x].shape == (4,7), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"
	
	for x in nph.slice.iter_indices(a, aslicet, group=(0,2)):
		assert a[x].shape == (4,2), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"

	for x in nph.slice.iter_indices(a, aslicet, group=(1,2)):
		assert a[x].shape == (7,2), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"

	for x in nph.slice.iter_indices(a, aslicet, group=(0,)):
		assert a[x].shape == (4,), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"

	for x in nph.slice.iter_indices(a, aslicet, group=(2,)):
		assert a[x].shape == (2,), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"

	for x in nph.slice.iter_indices(a, aslicet, group=(1,)):
		assert a[x].shape == (7,), f"iterating over grouped indices should give a subset of the slice, gives {a[x].shape}"

	for x in nph.slice.iter_indices(a, aslicet, group=tuple()):
		assert type(a[x]) == np.float64, f"grouping all the indices should give individual elements {a[x]=}"


def test_array_offsets():
	a = np.zeros((7,7))
	expected_result = np.array(a)
	a[5,5] = 1
	expected_result[3,3] = 1

	o = nph.array.get_centre_offset_brightest_pixel(a)
	assert np.all(o == np.array([-2,-2])), f"Should have an offset of [-2,-2], have {o}"

	b = nph.array.apply_offset(a, o)
	assert np.all(b == expected_result), f"Expected {expected_result}, have {b}"

def test_array_ensure_odd():
	a = np.zeros((1,2,3,4,5,6,7,8))
	expected_shape = ((1,1,3,3,5,5,7,7))

	b = nph.array.ensure_odd_shape(a)

	assert len(b.shape) == len(expected_shape), f"Should have {len(expected_shape)} entries, have {len(b.shape)}"
	assert all(s1 == s2 for s1,s2 in zip(b.shape, expected_shape)), f"shapes should be identical, expected {expected_shape}, have {b.shape}"


def test_axes_ordering():
	idxs = tuple(nph.axes.AxesOrdering.range(4))

	i_numpy = tuple(i.numpy for i in idxs)
	e_numpy = (0,1,2,3)
	i_fits = tuple(i.fits for i in idxs)
	e_fits = (4,3,2,1)
	i_fortran = tuple(i.fortran for i in idxs)
	e_fortran = (1,2,3,4)

	print(f'{i_numpy=} {i_fits=} {i_fortran=}')


	assert len(i_numpy) == len(e_numpy)
	assert all(a==b for a,b in zip(i_numpy, e_numpy))

	assert len(i_fits)  == len(e_fits)
	assert all(a==b for a,b in zip(i_fits, e_fits))

	assert len(i_fortran) == len(e_fortran)
	assert all(a==b for a,b in zip(i_fortran,e_fortran))















