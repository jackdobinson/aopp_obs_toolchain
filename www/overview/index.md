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

Adjusting observations to remove the effect of the PSF of an instrument is a process called deconvolution. There are many techniques, but many suffer from problems when working with extended sources.

<div>
\[
	O(x) = R(x) \star S(x) + E(x)
\]
</div>

<!--
TODO: 
* Overview of techniques
* Overview of problems they suffer with extended sources
* Introduce mathematical symbology
-->



## Modified CLEAN ##

<!--
TODO:
* Why is Modified CLEAN better than other approaches?
* Explain algorithm
-->