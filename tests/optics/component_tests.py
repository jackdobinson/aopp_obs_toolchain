
import numpy as np

from aopp_deconv_tool.geometry.shape import Circle
from aopp_deconv_tool.optics.geometric.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor,LightBeam,LightBeamSet

import matplotlib.pyplot as plt
import aopp_deconv_tool.plot_helper as plot_helper


def print_iterable(iterable, prefix='[', sep='\n\t',suffix='\n]'):
	print(prefix, end='')
	for item in iterable:
		print(f'{sep}{item}',end='')
	print(suffix,end='')

def test_constructing_optical_component_set():
	"""
	Test of creating an optical component set that can be used to get the 
	pupil function, point spread function, an optical transfer function of the
	optical system.
	"""
	
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
	
	lbs = ocs.get_light_beam(LightBeam(0,0,10,0,-1))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,1E-3,-10000))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,0.5,-20))
	#lbs = ocs.get_light_beam(LightBeam(0,-7E-2,10,-7E-2,-1))
	
	print_iterable(ocs._optical_path)
	print_iterable(lbs.light_beams)
	
	
	o = ocs.get_evaluation_positions()
	print(f'{o=}')
	wa = np.zeros_like(o)
	wb = np.zeros_like(o)
	for i, x in enumerate(o):
		wa_x, wb_x = lbs(x)
		wa[i] = wa_x
		wb[i] = wb_x
	
	
	fig, ax = plot_helper.figure_n_subplots(4)
	
	ax[0].set_title('Geometrical optics approximation of light beam')
	ax[0].set_xlabel('optical distance (m)')
	ax[0].set_ylabel('radius from optical axis (m)')
	ax[0].plot(o,wa, label='wa', alpha=0.6)
	ax[0].plot(o,wb, label='wb', alpha=0.6)
	ax[0].fill_between(o, wa, wb, color='green', alpha=0.3, label='beam body')
	ax[0].axhline(0, color='black', linestyle='--', alpha= 0.3)
			
	ax[0].set_ylim(plot_helper.LimSymAroundValue(0,pad=1)(np.array([wa,wb]).flatten()))
	
	
	xmin, xmax = ax[0].get_xlim()
	ymin, ymax = ax[0].get_ylim()
	for oc in ocs:
		ax[0].axvline(oc.position, color='red', linestyle=':', alpha=0.3)
		ax[0].text(oc.position + 0.01*(xmax-xmin), ymin + 0.01*(ymax-ymin), str(oc), rotation='vertical', fontsize='x-small')
	ax[0].legend()
	
	# TODO: Work out how the scale is changed as I alter the expansion and supersample factors
	#       so I can get the PSF scaled correctly.
	
	expansion_factor = 10
	supersample_factor = 1/expansion_factor
	pf = ocs.pupil_function((501,501), expansion_factor=expansion_factor, supersample_factor=supersample_factor)
	pf_scale = ocs.get_pupil_function_scale(expansion_factor)
	
	print(f'{pf_scale=}')
	print(f'{pf.shape=}')
	pf_extent = tuple(((-1)**i)*pf_scale[i//2] for i in range(2*len(pf_scale)))
	ax[1].set_title(f'pupil function {pf_scale} meters')
	ax[1].imshow(pf, extent=pf_extent)
	ax[1].set_xlabel('meters')
	ax[1].set_ylabel('meters')
	
	pf_fft = np.fft.fftshift(np.fft.fftn(pf))
	psf = (np.conj(pf_fft)*pf_fft)
	wavelength = 5E-7 #500 nm
	rad_to_arcsec = (180/np.pi)*3600
	psf_scale = tuple(wavelength*(supersample_factor*s/expansion_factor)*rad_to_arcsec for s in pf_scale)
	#psf_scale = tuple(wavelength*s*rad_to_arcsec for s in pf_scale)
	psf_extent = tuple(((-1)**i)*psf_scale[i//2] for i in range(2*len(psf_scale)))
	
	
	ax[2].set_title(f'point spread function {wavelength=} {psf_scale} arcsec')
	ax[2].imshow(psf.astype(float), extent=psf_extent)
	ax[2].set_xlabel('arcsec')
	ax[2].set_ylabel('arcsec')
	
	otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf)))
	ax[3].set_title(f'optical transfer function {otf.shape=}')
	ax[3].imshow(otf.astype(float))
	
	plt.show()

	assert True, "should always get here if nothing goes wrong"
