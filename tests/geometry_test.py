import numpy as np

import numpy_helper as nph
import numpy_helper.array.mask

import geometry as geo
import geometry.shape

import matplotlib.pyplot as plt


def test_mask_from_shape(plots=False):
	
	circle_1 = geo.shape.Circle.of_radius(1)
	circle_2 = geo.shape.Circle.of_radius(2)
	circle_3 = geo.shape.Circle.of_radius(3)
	
	#circle_1.metric=geo.taxicab_metric
	#circle_2.metric=geo.taxicab_metric
	#circle_3.metric=geo.taxicab_metric
	
	extent = np.array([
		[-3.0,3.0],
		[-3.0,3.0],
	])
	
	
	expected_1 = np.array([
		[0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0],
		[0,0,0,1,0,0,0],
		[0,0,1,1,1,0,0],
		[0,0,0,1,0,0,0],
		[0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0],
	], bool)
	
	expected_2 = np.array([
		[0,0,0,0,0,0,0],
		[0,0,0,2,0,0,0],
		[0,0,2,1,2,0,0],
		[0,2,1,1,1,2,0],
		[0,0,2,1,2,0,0],
		[0,0,0,2,0,0,0],
		[0,0,0,0,0,0,0],
	], bool)
	
	expected_3 = np.array([
		[0,0,0,3,0,0,0],
		[0,3,3,2,3,3,0],
		[0,3,2,1,2,3,0],
		[3,2,1,1,1,2,3],
		[0,3,2,1,2,3,0],
		[0,3,3,2,3,3,0],
		[0,0,0,3,0,0,0],
	], bool)
	
	
	
	mask_1 = nph.array.mask.from_shape(expected_1.shape, circle_1, extent=extent)
	mask_2 = nph.array.mask.from_shape(expected_2.shape, circle_2, extent=extent)
	mask_3 = nph.array.mask.from_shape(expected_3.shape, circle_3, extent=extent)
	
	for circle, mask, expected in zip([circle_1, circle_2, circle_3], [mask_1, mask_2, mask_3], [expected_1,expected_2,expected_3]):
		if plots:
			print(f'{circle=}')
			print(f'{mask.shape=} {mask=}')
			
			data = np.zeros_like(mask, float)
			data[mask] = 1
			data[expected] = 2
			data[mask & expected] = 1.5
			plt.imshow(data)
			plt.show()
			
		assert np.all(mask[expected]) and np.all(expected[mask]), "Should get the expected masks"
		
