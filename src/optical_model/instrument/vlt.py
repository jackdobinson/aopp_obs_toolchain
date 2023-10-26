"""
Instrument model for VLT/MUSE
"""
import numpy as np

from optical_model.instrument import OpticalInstrumentModel
from optical_model.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor
from geometry.shape import Circle

# Idea: Maybe combine .../instrument package and adaptive_optics.py into a 
# "optical_component" package that separates concerns better. Then, the 
# "instrument" package would combine a set of optical components together
# to represent a specific instrument.



class VLT_MUSE(OpticalInstrumentModel):
	def __init__(self, shape=(101,101), expansion_factor=1):
		obj_diameter = 8 # meters
		primary_mirror_focal_length = 120 # meters
		primary_mirror_pos = 100 # meters
		primary_mirror_diameter = 8
		
		# obstructions in terms of pupil exit diameter, For VLT secondary mirror has a diameter of 2.52/18*pupil_exit_diameter_in_angular_units. See: https://www.researchgate.net/profile/Marcel-Carbillet/publication/226258534_Apodized_Lyot_coronagraph_for_SPHEREVLT_II_Laboratory_tests_and_performance/links/5c66c092a6fdcc404eb2f530/Apodized-Lyot-coronagraph-for-SPHERE-VLT-II-Laboratory-tests-and-performance.pdf?origin=publication_detail
		secondary_mirror_diameter_frac_of_primary = 2.52/18
		
		secondary_mirror_dist_from_primary =50 # meters
		secondary_mirror_diameter_meters = primary_mirror_diameter*secondary_mirror_diameter_frac_of_primary
		n_actuators = 24
		
		self.f_ao = n_actuators / (2*obj_diameter)
		self._optical_component_set = OpticalComponentSet.from_components([
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
			
		self.pupil_function_scale, self.pupil_function = self._optical_component_set.pupil_function(
			shape, 
			expansion_factor=expansion_factor, 
			supersample_factor=1/expansion_factor
		)
		
		pf_fft = np.fft.fftshift(np.fft.fftn(self.pupil_function))
		self.psf_rho_per_lambda = (np.conj(pf_fft)*pf_fft).real
		
		self.rho_per_lambda_axes=np.array([np.linspace(-scale/2,scale/2,s) for scale, s in zip(self.pupil_function_scale, shape)])
		print(f'{self.rho_per_lambda_axes=}')
		

	def get_otf(self, wavelength):
		return np.fft.fftn(np.fft.ifftshift(self.psf_rho_per_lambda)*wavelength)



# TODO: Alter these classes so that they separate their concerns better. Want to
# be able to obviously see how the instrument is constructed.
class Instrument_VLT_MUSE(OpticalInstrumentModel):
	"""
	This instrument model represents the VLT/MUSE instrument.
	as long as we can get an optical transfer function out of this,
	we are good.
	"""
	objective_diameter : float = 8 # meters
	focal_length : float = 120 # meters, Nasmyth focus
	N_actuators : int = 24 # hard to get an accurate count, goes from ~24 up to about 1150, with 150 mentioned too
	image_fov : float = 1.0438 # meters, Nasmyth focus total value (249.6 mm unvinegetted)
	radius_of_image_curvature : float = 2.0896 # meters towards M2 (secondary mirror)
	pupil_obstructions : list[Circle] = [Circle(2.52/18)] # obstructions in terms of pupil exit diameter, For VLT secondary mirror has a diameter of 2.52/18*pupil_exit_diameter_in_angular_units. See: https://www.researchgate.net/profile/Marcel-Carbillet/publication/226258534_Apodized_Lyot_coronagraph_for_SPHEREVLT_II_Laboratory_tests_and_performance/links/5c66c092a6fdcc404eb2f530/Apodized-Lyot-coronagraph-for-SPHERE-VLT-II-Laboratory-tests-and-performance.pdf?origin=publication_detail
	MUSE_NFM_delta_ang : np.ndarray = np.array([0.025*(1/3600)*(np.pi/180)]*2) # radians, (from 0.025 arcsec in Narrow Field Mode)


	def __init__(self, shape=(201,201), supersample_factor=1):
		self.raw_shape = shape
		self.supersample_factor = supersample_factor
	
		self.f_ao = self.N_actuators/(2*self.objective_diameter)
		self.fov_shape = tuple(self.supersample_factor*_s for _s in self.raw_shape)

		self.fov_delta_ang = np.array(self.MUSE_NFM_delta_ang)/self.supersample_factor
		
		self.fov_angular_axes = (np.array([np.linspace(-_s//2, _s//2 + _s%2, _s) for _s in self.fov_shape]).T * self.fov_delta_ang).T
		#self.fov_angular_axes = (np.array([np.linspace(-_s/2, _s/2, _s) for _s in self.fov_shape]).T * self.fov_delta_ang).T

		#print(f'{self.fov_angular_axes.shape=}')
		#print(f'{self.fov_angular_axes=}')
				
		self.fov_angular = np.array(np.meshgrid(*self.fov_angular_axes))
		self.img_plane_dist = np.sqrt(np.sum(self.fov_angular**2, axis=0))

		#plt.imshow(self.img_plane_dist)
		#plt.show()

		self.rho_per_lambda_axes = np.array([np.fft.fftshift(np.fft.fftfreq(_x.size, (_x[1]-_x[0])*self.supersample_factor)) for _x in self.fov_angular_axes])
		return


	def get_pupil_function(self, wavelength):
		self.pupil_exit_diameter = 0.5*wavelength*self.focal_length/(self.objective_diameter*self.supersample_factor)
		pf = np.zeros_like(self.img_plane_dist)
		mask = (self.img_plane_dist < self.pupil_exit_diameter)
		#print(f'DEBUG: {self.fov_shape=} {self.MUSE_NFM_delta_ang=} {self.pupil_exit_diameter=} {self.img_plane_dist.shape=} {wavelength=} {self.focal_length=} {self.objective_diameter=} {self.supersample_factor=}')
		#print(f'DEBUG: {self.MUSE_NFM_delta_ang/self.pupil_exit_diameter=}')
		#print(f'DEBUG: {self.img_plane_dist=}')
		#print(f'DEBUG: {pf=}')
		#print(f'DEBUG: {mask=}')
		for circle in self.pupil_obstructions:
			# assume everything is in fraction of pupil exit diameter
			mask = mask & (~circle.as_mask(self.fov_shape, scale=self.MUSE_NFM_delta_ang/self.pupil_exit_diameter))

		pf[mask] = 1
		if (pf == 0).all():
			raise RuntimeError(f'Pupil function calculated to be all zeros. Check "input.wavelength_factor" argument to ensure that wavelengths passed routines are in base SI units (meters). For reference, currently {wavelength=}.')
		self.pupil_function = pf
		return(pf)


	def get_otf(self, wavelength):
		self.pf_fft = np.fft.fftshift(np.fft.fftn(self.get_pupil_function(wavelength)))
		self.psf = np.conj(self.pf_fft)*self.pf_fft
		self.otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.psf)))
		
		return(self.otf)


class Instrument_VLT_SINFONI(Instrument_VLT_MUSE):
	"""
	This instrument model represents the VLT/SINFONI instrument.
	as long as we can get an optical transfer function out of this,
	we are good.
	"""
	objective_diameter : float = 8 # meters
	focal_length : float = 120 # meters, Nasmyth focus
	N_actuators : int = 24 # hard to get an accurate count, goes from ~24 up to about 1150, with 150 mentioned too
	image_fov : float = 1.0438 # meters, Nasmyth focus total value (249.6 mm unvinegetted)
	radius_of_image_curvature : float = 2.0896 # meters towards M2 (secondary mirror)
	pupil_obstructions : list[Circle] = [Circle(2.52/18)]#[Circle(2.52/18)] # obstructions in terms of pupil exit diameter, For VLT secondary mirror has a diameter of 2.52/18*pupil_exit_diameter_in_angular_units. See: https://www.researchgate.net/profile/Marcel-Carbillet/publication/226258534_Apodized_Lyot_coronagraph_for_SPHEREVLT_II_Laboratory_tests_and_performance/links/5c66c092a6fdcc404eb2f530/Apodized-Lyot-coronagraph-for-SPHERE-VLT-II-Laboratory-tests-and-performance.pdf?origin=publication_detail
	MUSE_NFM_delta_ang : np.ndarray = np.array([0.1*(1/3600)*(np.pi/180)]*2) # radians, (from 0.1 arcsec) # Should really rename this variable. Is the angular resolution of a pixel of the output image.

	def __init__(self, shape=(201,201), supersample_factor=1):
		super().__init__(shape=shape, supersample_factor=supersample_factor)
