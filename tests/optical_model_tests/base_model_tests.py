

from types import SimpleNamespace

import numpy as np

from optical_model.base import BaseModel, Parameter



class EquilateralTriangleModel(BaseModel):
		required_parameters = ('center', 'size')
		provides_results = ('vertices',)
		offsets =  np.array([[0,2/np.sqrt(3)],[1,-1/np.sqrt(3)],[-1,-1/np.sqrt(3)]]) 

		def run(self):
			vertices = self.offsets*self.p.size + self.p.center
			return SimpleNamespace(vertices = vertices)


class RotatedShapeModel(BaseModel):
		required_parameters = ('angle',)
		required_models = ('shape',)
		
		def run(self):
			rot_matrix = np.array([
				[np.cos(self.p.angle), - np.sin(self.p.angle)],
				[np.sin(self.p.angle), np.cos(self.p.angle)]
			])
			vertices = (rot_matrix @ self.m.shape.vertices.T).T
			return SimpleNamespace(vertices = vertices)




def test_base_model_parameters_work():
	
	try:
		model = BaseModel({
			'a' : Parameter(1, 'first param'),
			'b' : Parameter(2.0, 'second param'),
			'c_eff' : Parameter(np.array([3,4,5],dtype=float), 'third param')
		})
		
		model()
	except NotImplementedError:
		return
	assert False, "Model class should work with parameters"


def test_simple_model():
	
	model = EquilateralTriangleModel({
		'center': Parameter(np.array([1,1]), 'center point of triangle'),
		'size': Parameter(1, 'scale of the triangle')
	})
	
	points_1 = model().vertices
	e1 = np.array([[1, 1+2/np.sqrt(3)],[1+1, 1-1/np.sqrt(3)], [1-1, 1-1/np.sqrt(3)]])
	print(f'{points_1=}')
	
	assert np.all(np.abs(points_1 - e1) < 1E-6), "Expect triangle model to line up with triangle points"
	
	points_2 = model(size=Parameter(7,'scale of the triangle')).vertices
	e2 = np.array([[1, 1+7*2/np.sqrt(3)],[1+7*1, 1-7*1/np.sqrt(3)], [1-7*1, 1-7*1/np.sqrt(3)]])
	print(f'{points_2=}')
	
	assert np.all(np.abs(points_2 - e2) < 1E-6), "Expect triangle model to line up with triangle points"
	
	points_3 = model(np.array([0,0]), 1).vertices
	e3 = np.array([[0, 0+1*2/np.sqrt(3)],[0+1*1, 0-1*1/np.sqrt(3)], [0-1*1, 0-1*1/np.sqrt(3)]])
	print(f'{points_3=}')
	
	assert np.all(np.abs(points_3 - e3) < 1E-6), "Expect triangle model to line up with triangle points"
	
	return points_1


def test_model_composition():
	
	eq_tri_model = EquilateralTriangleModel(
		{	'center': Parameter(np.array([1,1]), 'center point of triangle'),
			'size': Parameter(1, 'scale of the triangle')
		}
	)
	
	rot_model = RotatedShapeModel(
		{	'angle' : Parameter(np.pi, 'angle to rotate points by'),
		},
		{	'shape' : eq_tri_model,
		}
	)
		
	
	p1 = rot_model().vertices
	e1 = np.array([[-1, -(1+2/np.sqrt(3))],[-(1+1), -(1-1/np.sqrt(3))], [-(1-1), -(1-1/np.sqrt(3))]])
	print(f'{p1=}')
	
	assert np.all(np.abs(p1 - e1) < 1E-6), "Expect rotated model to line up with rotated points"
	
	return (p1)



if __name__=='__main__':
	test_base_model_parameters_work()
	p1 = test_simple_model()
	p2 = test_model_composition()
	
	import matplotlib.pyplot as plt
	plt.plot(0,0, 'o')
	plt.plot(*p1.T, 'o-')
	plt.plot(*p2.T, 'o-')
	plt.gca().set_aspect('equal')
	plt.show()
