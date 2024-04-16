# aopp_obs-toolchain #

## aopp_deconv_tool ##

### Examples ###

See the `examples` folder of the github. 

### Deconvolution ###

The main deconvolution routines are imported via

```
from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
from aopp_deconv_tool.algorithm.deconv.lucy_richardson import LucyRichardson
```

They have docstrings available, e.g. `help(CleanModified)` at the Python REPL will
tell you details about how they work.

There is a script `aopp_deconv_tool.deconvolve` that performs deconvolution using CleanModified on
two files passed to it (the first argument is the observation, the second is the PSF). The output
is saved to `./deconv.fits`. Invoke it with `python -m aopp_deconv_tool.deconvolve <OBS> <PSF>`.
By default, it will assume it should use the PRIMARY fits extension, and deconvolve everything.
If you want it to use a different one, pass the files as `'./path/to/file.fits{EXTENSION_NAME_OR_NUMBER}[10:12](1,2)'`.
Where `EXTENSION_NAME_OR_NUMBER` is the name or number of the extension to use, `[10:12]` is an example of
a slice (in Python slice format) of the extension cube to use, and `(1,2)` specifies which axes are the 'image' axes
i.e. RA and DEC (i.e. CELESTIAL) axes. NOTE: the `(1,2)` can be omitted, and it will try and guess the correct ones.

### PSF Fitting ###

The main PSF fitting routines are in `aopp_deconv_tools.psf_model_dependency_injector`, and `aopp_deconv_tools.psf_data_ops`. 
The examples on the github deal with this area. Specifically `<REPO_DIR>/examples/psf_model_example.py` for adaptive optics
instrument fitting.

### SSA Filtering ###

Singular Spectrum Analysis is performed by the `SSA` class in the `aopp_deconv_tools.py_ssa` module. An interactive 
viewer that can show SSA components can be run via `python -m aopp_deoconv_tools.graphical_frontends.ssa_filtering`.
By default it will show some test data, if you pass an **image** file (i.e. not a FITS file, but a `.jpg` etc.) it
will use that image instead of the default one.

The `ssa2d_sub_prob_map` function in the `aopp_deconv_tool.algorithm.bad_pixels.ssa_sub_prob` module attempts to 
make an informed choice of hot/cold pixels for masking purposes. See the docstring for more details.

The `ssa_interpolate_at_mask` function in the `aopp_deconv_tool.algorithm.interpolate.ssa_interp` module attempts
to interpolate data by interpolating between SSA components, only when the value of the component at the point
to be interpolated is not an extreme value. See the docstring for more details.