#!/usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import fitscube.deconvolve.clean

test_image = np.array(sp.misc.ascent()[::-1,:])

#plt.imshow(test_image, origin='lower')
#plt.imshow(test_psf, origin='lower')

sigmas = (100,10,1,)

filter_function = sp.ndimage.gaussian_filter
#filter_function = lambda *args, **kwargs: 10*sp.ndimage.gaussian_filter(*args, **kwargs)
#def filter_function(*args, **kwargs):
#	a = sp.ndimage.gaussian_filter(*args, **kwargs)
#	return(a-0.5*np.nanmean(a))

"""
s, h, f = fitscube.deconvolve.clean.decompose_image(test_image, 
													sigmas,
													calculation_mode='direct',
													filter_function = filter_function,
													verbose = 1,
													h_shape=(50,50)
													)

"""
def plot_decomposed_images(s, h, f, original):
	(nr, nc, size) = (len(s)+2, 2, 6)
	f1 = plt.figure(figsize=[size*x for x in (nc, nr)])
	a1 = f1.subplots(nr,nc,squeeze=False)
	remove_axis = lambda x: (x.xaxis.set_visible(False), x.yaxis.set_visible(False))
	
	for i, (s_i, h_i, f_i) in enumerate(zip(s, h, f)):
		a1[i,0].imshow(s_i, origin='lower')
		a1[i,0].set_title(f'Decomposed Images\ns[{i}] f[{i}] {f[i]:07.3E} sum {np.nansum(s_i):07.3E} argmax {np.unravel_index(np.nanargmax(s_i),s_i.shape)}') if i==0 else a1[i,0].set_title(f's[{i}] f[{i}] {f[i]:07.3E} sum {np.nansum(s_i):07.3E} argmax {np.unravel_index(np.nanargmax(s_i),s_i.shape)}')
		remove_axis(a1[i,0])
		
		a1[i,1].imshow(h_i, origin='lower')
		a1[i,1].set_title(f'Decompositon Filters\nh[{i}] sum {np.nansum(h[i]):07.3E} argmax {np.unravel_index(np.nanargmax(h_i),h_i.shape)}') if i==0 else a1[i,1].set_title(f'h[{i}] sum {np.nansum(h[i]):07.3E} argmax {np.unravel_index(np.nanargmax(h_i),h_i.shape)}')
		remove_axis(a1[i,1])
	
	I_reconstruct = np.nansum(s, axis=0)
	a1[i+1,0].imshow(I_reconstruct, origin='lower')
	remove_axis(a1[i+1,0])
	a1[i+1,0].set_title(f'reconstruction of image sum {np.nansum(I_reconstruct):07.3E}')
	
	delta_reconstruct = np.nansum(h, axis=0)
	a1[i+1,1].imshow(delta_reconstruct, origin='lower')
	remove_axis(a1[i+1,1])
	a1[i+1,1].set_title(f'reconstruction of delta function sum {np.nansum(delta_reconstruct):07.3E}')
	
	img_residual = original - I_reconstruct
	a1[i+2,0].imshow(img_residual, origin='lower')
	remove_axis(a1[i+2,0])
	a1[i+2,0].set_title(f'residual (original-reconstruction) sum {np.nansum(img_residual):07.3E}')
	
	dirac_delta_residual = fitscube.deconvolve.clean.dirac_delta_filter(h[0]) - delta_reconstruct
	a1[i+2,1].imshow(dirac_delta_residual, origin='lower')
	remove_axis(a1[i+2,1])
	a1[i+2,1].set_title(f'residual (delta-reconstruction) sum {np.nansum(dirac_delta_residual):07.3E}')
	
	plt.show()
	return(f1,a1)

# define sigmas
#sigmas = (10,5,4,3)
#sigmas = (20,15)
sigmas=(20,5,)

# define test psf
tps=(301,301)
x,y=(tps[0]//2, tps[1]//2) # center point of PSF
test_psf = np.zeros(tps)
test_psf[x,y]=1
#test_psf[x-4:x+5,y-4:y+5]=0.5
#test_psf[x-9:x+10,y-9:y+10]=0.2
#test_psf[x-20,y-20:y+20]=0.4
test_psf = sp.ndimage.gaussian_filter(test_psf, 20)

# define test dirty image
#dirty_img = sp.signal.fftconvolve(test_image, test_psf, mode='same')
real_img = np.zeros((400,300))
real_img[150:170,40:50] += 1
real_img[270:290,70:250] += 0.5
real_img[130:180,130:180] += 1
#real_img = sp.ndimage.gaussian_filter(dirty_img, 10)
real_img_min, real_img_max = (np.nanmin(real_img), np.nanmax(real_img))
dirty_img = sp.signal.fftconvolve(real_img, test_psf, mode='full')/np.nansum(test_psf) # don't change amount of flux because of PSF
shape_diff = np.array(dirty_img.shape) - np.array(real_img.shape)
np.random.seed(1000) # seed generator for repeatability
for i, s in enumerate(np.logspace(np.log(np.max(real_img.shape)),0,10,base=np.e)):
	print(f'INFO: Adding noise set {i} at scale {s} to dirty image')
	dirty_img[[slice(shape//2,-shape//2) for shape in shape_diff]] += sp.ndimage.gaussian_filter(np.random.normal(0,0.5,real_img.shape), s)

# define a window
window = True
#window = np.zeros_like(dirty_img, dtype=bool)

# decompose so we can see what's working
filter_function = lambda *args, **kwargs: 0.9*sp.ndimage.gaussian_filter(*args, **kwargs) # set this to the same as "clean_mutliresolution()" for testing
"""
def filter_function(x,s):
	xx, yy = np.mgrid[:x.shape[0],:x.shape[1]]
	xx = (1.0/s)*(xx - x.shape[0]//2)
	yy = (1.0/s)*(yy - x.shape[1]//2)
	rr = np.sqrt(xx*xx+yy*yy)
	#sine_integral, ignore = sp.special.sici(np.nanmax(rr))
	sinc = np.sinc(rr)
	#sinc /= np.nansum(sinc)
	#factor =((np.pi/2)/sine_integral)**2 
	#print(s, sine_integral, np.nansum(sinc), factor)
	#sinc *= factor
	#print(np.nansum(sinc))
	return(sp.signal.fftconvolve(x,sinc,mode='same'))
"""
#hs = [filter_function(fitscube.deconvolve.clean.dirac_delta_filter(test_psf), s) for s in list(sigmas)+[0.5*np.min(sigmas)]]
hs=None

s_dirty, h_dirty, f_dirty = fitscube.deconvolve.clean.decompose_image(dirty_img, 
													sigmas,
													calculation_mode='convolution_fft',
													filter_function = filter_function,
													verbose = 1,
													h_shape=test_psf.shape,
													h=hs
													)
s_psf, h_psf, f_psf = fitscube.deconvolve.clean.decompose_image(test_psf, 
													sigmas,
													calculation_mode='convolution_fft',
													filter_function = filter_function,
													verbose = 1,
													h_shape=test_psf.shape,
													h=hs
													)
fdd, add = plot_decomposed_images(s_dirty, h_dirty, f_dirty, dirty_img)
fdd.suptitle('Decomposed dirty image')
fdp, adp = plot_decomposed_images(s_psf, h_psf, f_psf, test_psf)
fdp.suptitle('Decompsed PSF image')

# normalise PSF to see if it fixes "missing flux", it does not
#test_psf /= np.nanmax(test_psf)

(nr, nc, size) = (1, 3, 6)
f2 = plt.figure(figsize=[size*x for x in (nc, nr)])
a2 = f2.subplots(nr,nc,squeeze=False)
a2[0,0].imshow(dirty_img, origin='lower', vmin=real_img_min, vmax=real_img_max)
a2[0,0].set_title(f'dirty_img\nsum {np.nansum(dirty_img)}')
a2[0,1].imshow(test_psf, origin='lower')
a2[0,1].set_title(f'test_psf\nsum {np.nansum(test_psf)} {np.unravel_index(np.nanargmax(test_psf), test_psf.shape)}')
a2[0,2].imshow(real_img, origin='lower', vmin=real_img_min, vmax=real_img_max)
a2[0,2].set_title(f'real_img\nsum {np.nansum(real_img)}')


clean_result = fitscube.deconvolve.clean.clean_multiresolution(dirty_img,
																test_psf,
																window=window,
																sigmas=sigmas,
																reject_decomposition_idxs=[],
																quiet=False,
																show_plots=True,
																max_iter=int(1E3), # for clean_modified() version
																#max_iter=int(1E5), # for clean_hogbom() version
																n_positive_iter=1E3,
																threshold=0.6, # for clean_modified() version, does nothing for hogbom
																loop_gain=0.1, # for clean_modified() version
																#loop_gain=0.1, # for clean_hogbom() version
																rms_frac_threshold=1E-2,
																fabs_frac_threshold=1E-2,
																norm_psf=True,
																calculation_mode='convolution_fft',
																base_clean_algorithm='modified'
																)

residual = clean_result[0]
components = clean_result[1]

(nr, nc, size) = (2, 3, 6)
f3 = plt.figure(figsize=[size*x for x in (nc, nr)])
a3 = f3.subplots(nr,nc,squeeze=False)

a3[0,0].imshow(residual, origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[0,0].set_title(f'residual\nsum {np.nansum(residual):07.2E}')

a3[0,1].imshow(components, origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[0,1].set_title(f'components\nsum {np.nansum(components):07.2E}')

diff_from_real = np.array(components)
diff_from_real[tuple([slice(shape//2,-shape//2) for shape in shape_diff])] -= real_img
a3[0,2].imshow(diff_from_real, origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[0,2].set_title(f'components-real_img\nsum {np.nansum(diff_from_real):07.2E}')

a3[1,0].imshow(sp.ndimage.gaussian_filter(components, np.min(sigmas)/2), origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[1,0].set_title(f'min smoothed components\nsum {np.nansum(sp.ndimage.gaussian_filter(components, np.min(sigmas))):07.2E}')

a3[1,1].imshow(dirty_img - residual, origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[1,1].set_title(f'dirty_img - residual\nsum {np.nansum(dirty_img - residual):07.2E}')

a3[1,2].imshow(sp.ndimage.gaussian_filter(components, np.max(sigmas)), origin='lower', vmin=real_img_min, vmax=real_img_max)
a3[1,2].set_title(f'max smoothed components\nsum {np.nansum(sp.ndimage.gaussian_filter(components, np.max(sigmas))):07.2E}')

plt.show()

