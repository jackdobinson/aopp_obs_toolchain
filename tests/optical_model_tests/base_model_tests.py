

from types import SimpleNamespace

import numpy as np

from optical_model.base import BaseModel, Parameter

import matplotlib.pyplot as plt

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



def test_psf_model():
	from geometry.shape import Circle
	from optical_model.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor,LightBeam,LightBeamSet
	
	
	shape = (101,101)
	expansion_factor = 7
	supersample_factor = 1/7
	
	
	obj_diameter = 8 # meters
	primary_mirror_focal_length = 120 # meters
	primary_mirror_pos = 100 # meters
	primary_mirror_diameter = 8
	secondary_mirror_diameter_frac_of_primary = 2.52/18
	secondary_mirror_dist_from_primary =50 # meters
	secondary_mirror_diameter_meters = primary_mirror_diameter*secondary_mirror_diameter_frac_of_primary
	ocs = OpticalComponentSet.from_components([
		Aperture(
			0, 
			'objective aperture', 
			shape=Circle.of_radius(obj_diameter/2)
		), 
		Obstruction(
			primary_mirror_pos - secondary_mirror_dist_from_primary, 
			'secondary mirror back', 
			shape=Circle.of_radius(secondary_mirror_diameter_meters/2)
		), 
		Refractor(
			primary_mirror_pos, 
			'primary mirror', 
			shape=Circle.of_radius(primary_mirror_diameter/2), 
			focal_length=primary_mirror_focal_length
		),
	])
	
	pupil_function_scale, pupil_function = ocs.pupil_function(
		shape,
		expansion_factor=expansion_factor,
		supersample_factor=supersample_factor
	)
	
	#result
	pf_scale_axes = np.array([np.linspace(-scale/2,scale/2,int(s*self.p.expansion_factor*self.p.supersample_factor)) for scale, s in zip(pupil_function_scale, self.p.shape)])
	
	# calculation
	pf_fft = np.fft.fftshift(np.fft.fftn(pupil_function))
	
	# result
	psf_rho_per_lambda = (np.conj(pf_fft)*pf_fft).real
	rho_per_lambda_axes = np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in pf_scale_axes])
	
	# result
	otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.m.psf.psf_rho_per_lambda)))
	
	#input
	wavelength = 5E-7
	
	#calculation
	rho_axes = self.m.psf.rho_per_lambda_axes*self.p.wavelength
	f_axes = np.array([np.fft.fftshift(np.fft.fftfreq(_x.size, (_x[1]-_x[0])*self.p.supersample_factor)) for _x in rho_axes])
	



	class InstrumentPSFModel(BaseModel):
		required_parameters = ('optical_component_set', 'shape', 'expansion_factor', 'supersample_factor')
		required_models = tuple()
		provides_results = ('psf_rho_per_lambda', 'rho_per_lambda_axes')
		
		
		def run(self):
			pupil_function_scale, pupil_function = self.p.optical_component_set.pupil_function(
				self.p.shape, 
				expansion_factor=self.p.expansion_factor, 
				supersample_factor=self.p.supersample_factor
			)
			pf_scale_axes = np.array([np.linspace(-scale/2,scale/2,int(s*self.p.expansion_factor*self.p.supersample_factor)) for scale, s in zip(pupil_function_scale, self.p.shape)])
			print(f'{pf_scale_axes=}')
			
			
			pf_fft = np.fft.fftshift(np.fft.fftn(pupil_function))
			psf_rho_per_lambda = (np.conj(pf_fft)*pf_fft).real
			
			rho_per_lambda_axes = np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in pf_scale_axes])
	
			return SimpleNamespace(psf_rho_per_lambda=psf_rho_per_lambda, rho_per_lambda_axes=rho_per_lambda_axes)
	
	
	class OTF(BaseModel):
		required_parameters = ('wavelength','supersample_factor')
		required_models = ('psf',)
		provides_results = ('otf', 'f_axes')
		
		def run(self):
			otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.m.psf.psf_rho_per_lambda)))
			rho_axes = self.m.psf.rho_per_lambda_axes*self.p.wavelength

			f_axes = np.array([np.fft.fftshift(np.fft.fftfreq(_x.size, (_x[1]-_x[0])*self.p.supersample_factor)) for _x in rho_axes])
			return SimpleNamespace(otf=otf, f_axes=f_axes)
	
	class PhasePowerSpectrumDistribution(BaseModel):
		required_paramete = tuple('f_ao')
		required_models = ('ao_psd', 'atmosphere_psd')
		provides_results = ('phase_psd')
		
		def run(self):
			
	
	
		
	vlt_psf_model = InstrumentPSFModel(
		{	'optical_component_set' : Parameter(ocs, 'set of optical components that are needed to determine the pupil function of the instrument'),
			'shape' : Parameter((101,101), 'number of pixels in the returned array'),
			'expansion_factor' : Parameter(7, 'factor to expand internal arrays by to get better frequency resolution'),
			'supersample_factor' : Parameter(1/7, 'factor to expand internal arrays by to get better spatial resolution')
		}
	)
		
	vlt_psf = vlt_psf_model()
	plt.imshow(vlt_psf.psf_rho_per_lambda)
	plt.show()
	
	vlt_otf_model = OTF(
		{	'wavelength' : Parameter(5E-7, 'wavelength of light'),
			'supersample_factor' : Parameter(1/7, 'factor to expand internal array sby to get better spatial resolution')
		},
		{	'psf' : vlt_psf_model }
	)
	vlt_otf = vlt_otf_model()
	print(vlt_otf)
	plt.imshow(vlt_otf.otf.real)
	plt.show()
	
	

if __name__=='__main__':
	test_base_model_parameters_work()
	p1 = test_simple_model()
	p2 = test_model_composition()
	
	"""
	import matplotlib.pyplot as plt
	plt.plot(0,0, 'o')
	plt.plot(*p1.T, 'o-')
	plt.plot(*p2.T, 'o-')
	plt.gca().set_aspect('equal')
	plt.show()
	"""
	
	test_psf_model()
