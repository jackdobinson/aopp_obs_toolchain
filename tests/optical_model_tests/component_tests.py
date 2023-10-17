
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
	ocs = OpticalComponentSet.from_components([
		Aperture(0, 'objective aperture', Circle.of_radius(7)), 
		Obstruction(50, 'secondary mirror back', Circle.of_radius(2)), 
		Refractor(100, 'primary mirror', Circle.of_radius(7), 120),
		Aperture(150, 'secondary mirror front', Circle.of_radius(2))
	])
	
	lbs = ocs.get_light_beam(LightBeam(0,0,10,0,-1))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,1E-3,-10000))
	#lbs = ocs.get_light_beam(LightBeam(0,0,0,5E-3,-2000))
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
	
	
	fig, ax = plot_helper.figure_n_subplots(1)
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
	
	plt.show()
	
	
	
	
	
	assert False, 'TESTING'
