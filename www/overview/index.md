# Overview of aopp_deconv_tool #

The routines in the `aopp_deconv_tool` package are intended to be used when deconvolving observations of extended objects. The following nomenclature is used.


instrument
: The apparatus used to collect data. In this context, normally refers to the specific combination of telescope (that collects light) and the optical instrument(s) (camera, spectrograph, interferometer, etc.) that analyse the light.

observation
: Data collected from an instrument in a continuous chunk of time.

target
: The object an observation is trying to observe. Hopefully the data collected during the observation will tell us something useful about the target.

science target
: A target that is being observed where the direct intent is to get scientifically useful data from the instrument about the target.

calibration target
: A target that is being observed with the intent to calibrate the instrument, the target is **not** being observed primarily to get scientifially useful data.

standard star
: A star that has stable photometry, many observations, and can be easily used as a calibration target.

point source
: An object that, when observed with a telescope, is small enough that the image is diffraction limited or seeing limited.

extended source
: An object that, when observed with a telescope, is large enough that the image is not diffraction limited or seeing limited.

point spread function (PSF)
: For a specific instrument, how the light from a point source is recorded.



## Problem Description ##

When observing an extended target, obtaining accurate photometry of different regions of the target is challenging as the PSF of the instrument spreads light from bright regions to adjacent dark regions. For traditional observations, this results in the smoothing of brightness changes and lowering of contrast between bright and dark regions, especially when the dark regions are of a similar angular size to the PSF. For adapative optics (AO) observations, the situation is in some ways better and in some ways worse. The "central spike" of an AO PSF is very narrow which results in much sharper changes in brightness, however the "halo ring" part of an AO PSF can cause similar lowering of contrast without the tell tale smoothing of traditional observations.

Spectroscopic observations have the same problem. Along with smoothing and reduced contrast, scattering of light from adjacent sections of the sky also changes the wavelength/frequency distribution of a region. When using data from these observations to infer properties of the objects (e.g. inferring chemical and physical makeup from spectra and radiative transfer models) the scattered light can be a large source of error that swamps any signal from regions of interest. Adaptive optics (AO) does not wholly remove this problem.


## Deconvolution Background ##

A when performing a measurement, the "true" signal is adjusted by the response function of the measurement tool this can be described mathematically as a convolution. Adjusting observations to remove the effect of the PSF of an instrument is a process called deconvolution. There are many techniques, but many suffer from problems when working with extended sources.

Performing an observation $O(x)$, of the true signal, $S(x)$, with an instrument that has a response function, $R(x)$, and some noise, $N(x)$ is written as:
<div>
\begin{equation}
	O(x) = R(x) \star S(x) + N(x)
	\label{eq:deconv}
\end{equation}
</div>

Where the symbol $\star$ denotes the convolution between two functions and can be calculated via:

<div>
\begin{align}
	R(x) \star S(x) &= \mathscr{F}^{-1} [\mathscr{F}[R(x)] \times \mathscr{F}[S(x)]] \notag \\
	
	R(x) \star S(x) &= \mathscr{F}^{-1} [\tilde{R}(k) \times \tilde{S}(k)]
	
\end{align}
</div>

where $\tilde{f}(k) = \mathscr{F}[f(x)]$ is the fourier transform of some function $f(x)$, $f(x) = \mathscr{F}^{-1}[\tilde{f}(k)]$ is the inverse fourier transform of $\tilde{f}(k)$, and $k$ is the fourier conjugate variable to $x$.

Direct inversion of \eqref{eq:deconv} leads to

<div>
\begin{align}
	\tilde{O}(x) &= \tilde{R}(x) \times \tilde{S}(x) + \tilde{N}(x) \notag \\
	
	\tilde{S}(x) &= \frac{\tilde{O}(x) - \tilde{N}(x)}{\tilde{R}(x)} \notag \\
	\text{so,} \notag\\
	\hat{S}(x) &= \mathscr{F}^{-1}\left[ \frac{\tilde{O}(x) - \tilde{N}(x)}{\tilde{R}(x)} \right]
	\label{eq:fourier_inversion}
\end{align}
</div>

where $\hat{S}(x)$ is an estimate of $S(x)$. Unfortunately, \eqref{eq:fourier_inversion} is incredibly sensitive to noise and therefore cannot be used in practice.

Other deconvolution methods exist; for example the [Wiener Filter](https://en.wikipedia.org/wiki/Wiener_filter) miminises the mean square error of a solution, and [Lucy-Richardson deconvolution](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution) is found via finding maximum likelihood estimate assuming Poisson noise.

The [original CLEAN algorithm](https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H/abstract) is conceptually very simple:

0) The image is called the "dirty map", and a new empty image is called the "component map".

1) The brightest pixel in the dirty map is found

2) The PSF of an instrument is centered on the brightest pixel, multiplied by a fraction of the brightest pixel, and subtracted from the dirty map.

3) The fraction of the brightest pixel subtracted in the previous step is written to the component map at the location of the brightest pixel.

4) This process repeats from step (1) until a the brightest pixel in the dirty map is below a specified threshold.

5) The component map is convolved with a gaussian to form the "clean map", smoothing out non-physical high frequency noise in the component map. Note: this is a form of [regularisation](https://en.wikipedia.org/wiki/Regularization_(mathematics)).

6) The "residual", the dirty map after all the subtractions, is added to the clean map.

In pseudo-code:
```
// Constants
define array[2] image_size = (X,Y)
define array[2] psf_size = (A,B)
define float loop_gain = 0.01
define stop_factor = 1E-3
define clean_beam_sigma = 3

// Variables
define array[X,Y] component_map
define array[X,Y] dirty_map
define array[A,B] psf_map
define array[X,Y] psf_adjusted_map
define array[X,Y] clean_map
define array[2] brightest_pixel_location
define float psf_sub_multiplier
define float stop_value


dirty_map = load_from_disk(original_image)
psf_map = load_from_disk(psf_image)
stop_value = max(dirty_map) * stop_factor

DO {
	brightest_pixel_location = argmax(dirty_map) // argmax returns the x,y coords of the maximum value in an array
	psf_sub_multiplier = dirty_map[brightest_pixel_location]*loop_gain // get a factor to multiply the PSF by for this iteration
	psf_adjusted_map = copy_array_centered_at(psf_map, brightest_pixel_location) * psf_sub_multiplier // center the PSF on a selected pixel and scale it.
	
	// Subtract the scaled PSF from the dirty map
	FOR pixel_location IN dirty_map {
		dirty_map[pixel_location] -= psf_adjusted_map[pixel_location] 
	}
	
	// Write the value of the subtracted brightest pixel to the component map
	component_map[brightest_pixel_location] += psf_sub_multiplier
	
} WHILE (dirty_map[brightest_pixel_location] > stop_value)

// Convolve the component map with a gaussian to get the clean map
clean_map = convolution(gaussian(clean_beam_sigma), component_map)

```

CLEAN has the nice property that the original image is equal to the compoment map convolved with the original PSF and added to the residual. 

The algorithm in this package is a [modified version of CLEAN algorithm](https://ui.adsabs.harvard.edu/abs/1984A%26A...137..159S/abstract)


## Modified CLEAN ##

Sometimes called the [Steer algorithm](https://www.aanda.org/articles/aa/pdf/2003/49/aa2937.pdf), or [SDI CLEAN](https://www.atnf.csiro.au/computing/software/miriad/userguide/node103.html), MODIFIED CLEAN works with sets of pixels larger than some threshold value rather than only with the brightest pixel as the original CLEAN algorithm does. 


<!--
TODO:
* Why is Modified CLEAN better than other approaches?
* Explain algorithm
-->