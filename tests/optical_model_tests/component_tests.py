
import numpy as np

from geometry.shape import Circle
from optical_model.component import OpticalComponentSet, Aperture, Obstruction,Refractor,LightBeam,LightBeamSet

import matplotlib.pyplot as plt
import plot_helper


def print_iterable(iterable, prefix='[', sep='\n\t',suffix='\n]'):
	print(prefix, end='')
	for item in iterable:
		print(f'{sep}{item}',end='')
	print(suffix,end='')

def test_constructing_optical_component_set():
	
	MUSE_NFM_delta_ang = 0.025*(1/3600)*(np.pi/180) # radians, (from 0.025 arcsec in Narrow Field Mode)
	obj_diameter = 8 # meters
	primary_mirror_focal_length = 120 # meters
	primary_mirror_pos = 100 # meters
	secondary_mirror_diameter_radians = 2.52/18 # radians
	secondary_mirror_dist_from_primary =90 # meters
	secondary_mirror_diameter_meters = (120 - secondary_mirror_dist_from_primary)*secondary_mirror_diameter_radians
	
	ocs = OpticalComponentSet.from_components([
		Aperture(
			0, 
			'objective aperture', 
			Circle.of_radius(obj_diameter/2)
		), 
		Obstruction(
			primary_mirror_pos - secondary_mirror_dist_from_primary, 
			'secondary mirror back', 
			Circle.of_radius(secondary_mirror_diameter_meters/2)
		), 
		Refractor(
			primary_mirror_pos, 
			'primary mirror', 
			Circle.of_radius(7), 
			primary_mirror_focal_length
		),
		Aperture(
			primary_mirror_pos + secondary_mirror_dist_from_primary, 
			'secondary mirror front', 
			Circle.of_radius(secondary_mirror_diameter_meters)
		)
	])
	
	lbs = ocs.get_light_beam(LightBeam(0,0,10,0,-1))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,1E-3,-10000))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,0.5,-20))
	#lbs = ocs.get_light_beam(LightBeam(0,-7E-2,10,-7E-2,-1))
	
	print_iterable(ocs._optical_path)
	print_iterable(lbs.light_beams)
	
	#o_min, o_max = lbs.get_optical_position_range()
	#o = np.linspace(o_min, o_max+70, 200)
	
	o = ocs.get_evaluation_positions()
	print(f'{o=}')
	wa = np.zeros_like(o)
	wb = np.zeros_like(o)
	for i, x in enumerate(o):
		wa_x, wb_x = lbs(x)
		wa[i] = wa_x
		wb[i] = wb_x
		
	#print(o)
	#print(wa)
	#print(wb)
	
	
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
	
	
	pf_scale, pf = ocs.pupil_function((501,501), expansion_factor=10, supersample_factor=0.1)#, (MUSE_NFM_delta_ang*120,MUSE_NFM_delta_ang*120))#(-2E-3,-2E-3))
	print(f'{pf_scale=}')
	print(f'{pf.shape=}')
	ax[1].set_title(f'pupil function {pf_scale} meters')
	ax[1].imshow(pf)
	
	pf_fft = np.fft.fftshift(np.fft.fftn(pf))
	psf = (np.conj(pf_fft)*pf_fft)
	wavelength = 1E-6 #1 um
	rad_to_arcsec = (180/np.pi)*3600
	ax[2].set_title(f'point spread function {wavelength=} {tuple(wavelength*s*rad_to_arcsec for s in pf_scale)} arcsec')
	ax[2].imshow(psf.astype(float))
	
	otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf)))
	ax[3].set_title('optical transfer function')
	ax[3].imshow(otf.astype(float))
	
	plt.show()
	
	
	
	
	
	assert False, 'TESTING'
