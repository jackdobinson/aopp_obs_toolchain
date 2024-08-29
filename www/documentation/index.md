# Documentation #

In addition to this overview of the command-line scripts, there is [Doxygen documentation](./doxygen/html/index.html) and [pydoc](./pydoc/index.html) documentation. These go into detail on the functions, classes, members, and methods in the package.

## Command-line Scripts <a id="command-line-scripts"></a> ##

When running command-line scripts, use the `-h` option to see the help message. The appendix has a [overview of help message syntax]({{site.baseurl}}/resources#command-line-script-help-message-syntax).

See the appendix for a [bash script that performs all steps on a given dataset]({{site.baseurl}}/resources#full-deconvolution-process)

### Spectral Rebinning <a id="spectral-rebinning-script"></a> ##

Invoke via `python -m aopp_deconv_tool.spectral_rebin`. 

This routine accepts a FITS file specifier, it will spectrally rebin the fits extension and output a new fits file.

The underlying algorithm does the following:

* We have a 3d dataset. In the FITS file we get points that correspond to each pixel centre, not a bin or region. We assume that the value "belongs" to the centre point of the pixel and is not defined elsewhere. I.e., treat the data like point-cloud data. 

* The data is convolved with a response function (usually triangular for the spectral axis) of some characteristic with `bin_width`, and then sampled every `bin_step` from the start of the axis.

* NOTE: We define three things, and `offset`, a `bin_width`, and a `bin_step`. `offset` is the difference between the starting point of the new bins, `bin_width` is the "width" of the response function (for a histrogram the response function would be a square), `bin_step` is the distance between the start of one bin and the beginning of the next. Most of the time `bin_width` = `bin_step`, but sometimes they do not. A picture of a general example is below

```
	#############################################################
	NOTE: This is a general case and not accurate for a FITS file as we don't have bin-edges, just the centres. Therefore we assume bin_width=bin_step.
	input_grid    : |---^---|   |---^---|   |---^---|   |---^---|
				  :       |---^---|   |---^---|   |---^---|   	
	
	bin_width     : |-------|

	bin_step      : |-----|

	#############################################################

	output_grid   :     |--------^-------|    |--------^-------|
				  :                |--------^-------|
	
	offset        : |---|

	new_bin_width :     |----------------|
	
	new_bin_step  :     |----------|


	#############################################################

	"|" = bin edge

	"^" = bin centre

	width
		The distance between the start and end of a bin
	
	step
		The distance between the start of two consecutive bins

	offset
		The distance between the start/end of the new grid and the start/end of the old grid
```

* We assume the `offset` is the same for the old and new binning.

* When sampling at `bin_step`, we linearly interpolate between the results of the convolution.

* The integral of the response function makes a big difference to the output. If the response function integrates to 1, we have the effect of averaging over the response function. This is appropriate when we are dealing with units that divide out the effect of the physical pixel size e.g., counts/wavelength. However, if we are dealing with raw counts, the response function should not integrate to 1 it's value should relate to the size of the response function relative to the size of the physical pixels.

* `bin_width` is defined differently for different response functions:
  - Triangular response function : `bin_width` is the full width at half maximum (FWHM),
  - Rectangular response function : `bin_width` is still the FWHM, but as it has straight sides this is the same as the full width.

#### Module Arguments ####

{% include module_arguments_note.md %}

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_rebin" appended to the filename.

* `--rebin_operation` : str
  - How are values combined when rebinning is performed
	+ `sum` : sums the old bins into the new bins, use when you have a measurement like "counts"
	+ `mean` : averages the old bins into the new bins, use when you have a measurement like "counts per frequency" (DEFAULT)
	+ `mean_err` : averages the square of the old bins into the new bins then square roots, use when you have standard deviations of a measurement like "counts per frequency"


* One of the two mutually exclusive options:
  * `--rebin_params` : float float
	- Takes two floats, `bin_step` and `bin_width`. Defines the new bin sizes, values are in SI units.

  * `--rebin_preset` : str = `spex` (DEFAULT)
	- Is a named preset that defines `bin_step` and `bin_width`. Presets are:
	  + `spex`: `bin_step` = 1E-9, `bin_width` = 2E-9

#### Examples ####

Using the example data, the `datasets.json` file lists a dataset called "example neptune observation" with a science target observation in the file "MUSE.2019-10-18T00:01:19.521.fits" and a calibration observation in the file "MUSE.2019-10-17T23:46:14.117.fits"

Run the rebinning for each of the files via the command:
* `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117.fits`
* `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521.fits`

By default, files are rebinned by averaging old bins into new bins, and using the `spex` preset for bin width and bin step. The equivalent to the above command is `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117.fits --rebin_operation mean --rebin_preset spex --output_path ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits`

After both commands are complete there should be two new files that contain their output:
* `./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits`
* `./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits`

### Artefact Detection <a id="artefact-detection-script"></a> ###

Invoke via `python -m aopp_deconv_tool.artefact_detection`.

WARNING: The current implementation of artefact detection is tuned for observations of objects of a significant fraction of the field size. Therefore, **it will not give good results for standard star observations**.

Accepts a FITS specifier, uses a singular spectrum analysis (SSA) based algorithm to produce a heuristic `badness_map` that reflects how likely a pixel is to be part of an artefact.

The badness map is calculated as follows for each 2D image in the FITS data:

* SSA components for a 10x10 window are calculated
* A subset of the components, assuming components are ordered by decending magnitude of eigenvalue, is chosen via:
  - Starting with 25% of the way through the components, as the components with the largest eigenvalues are likely to be made up of the main signal
  - Ending with 75% of the way through the components, as the components with the smallest eigenvalues are likely to be made up of noise.
* For each component in the chosen SSA subset, the number of standard deviations a pixel is away from the mean of its region is calculated (called the `component_badness_map`)
  -  The "region" of a pixel is defined in the following way. A pixel's region is its brightness class (a pixel is a member of one of three brightness thresholds (background, midground, foreground)), and all pixels of its brightness class within N/8 from the pixel, where N is the largest x or y dimension of the image.
* The `component_badness_map`s for the chosen subset are averaged together to create the `badness_map` heuristic.

#### Module Arguments ####

NOTE: argument type is specified by a colon (:) following the argument name, multiple accepted types are separated by the pipe (\|) character, some arguments take more than one value these are separated by spaces; arguments with an equals sign (=) take the specified value as a default.

* `-o` or `--output_path` : str
  - Output fits file path. If not specified, it is same as the path to the input file with "_artefactmap" appended to the filename.

* `--strategy` : str = `ssa`
  -  Which strategy to use when detecting artifacts
	+ `ssa` : Uses singular spectrum analysis (SSA) to determine how likely a pixel is to belong to an artefact
* `--ssa.w_shape` : int = 10
  - Shape of the window used for the `ssa` strategy.
* `--ssa.start` : int \| float = -0.25
  - First SSA component to be included in artefact detection calc. Negative numbers are fractions of range.
* `--ssa.stop` : int \| float = -0.75
  - Last SSA component to be included in artefact detection calc. Negative numbers are fractions of range.


#### Examples ####

Using the results from the rebinning example:

* `python -m aopp_deconv_tool.artefact_detection ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits`

### Bad Pixel Mask <a id="bad-pixel-mask-script"></a> ###

Invoke via `python -m aopp_deconv_tool.create_bad_pixel_mask`.

NOTE: By setting the `value` part of `--value_cut_at_index` argument, this script can be used to only apply dynamic and/or constant regions without applying value cuts. This can be useful when manually removing artifacts from standard star datacubes. However, note that the [psf normalisation script](#psf-normalisation-script) does include a routine for removing outliers so creating a bad pixel mask and interpolating is normally not required for a standard star.

Accepts a `badness_map` heuristic, uses a set of value cuts to produce a boolean mask (the `bad_pixel_mask`) that describes which pixels are considered "bad" and should be interpolated over using a different script. Also accepts DS9 region files (in IMAGE coords only at the moment), both constant and dynamic region files can be passed. Constant ones apply across all wavelengths, dynamic ones vary with wavelength.

The `badness_map` is assumed to be a 3D cube, therefore the `bad_pixel_mask` is calculated from a set of (`index`,`value`) pairs. Where `index` is an index into the `badness_map`, and `value` is the value above which a pixel in the `badness_map` is considered "bad". Not all indices have to be specified, and values for unspecified indices will be interpolated (with the values clamped at the LHS and RHS). If no pairs are provided, a value of 3 is assumed for all indices. For each 1 above the cutoff value, a bad pixel is binary dilated. This way "very bad" pixels spread their "badness" to neighbouring pixels.

To get a set of (`index`, `value`) pairs for use with the artefact `badness_map`, the following workflow is suggested:

1. Open the `badness_map` FITS file in a FITS viewer of some sort (e.g., [DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9) or [QFitsView](https://sites.google.com/cfa.harvard.edu/saoimageds9)), you may need to use a logarithmic scale.
2. Open the data the `badness_map` was created from as well so you can compare them.
3. Choose some representative indices (i.e., wavelengths) to work on. For illustrative purposes we will assume indices, (10, 99, 135).
4. In the `badness_map` viewer, alter the minimum value of the data display range (somewhere around 4 or 5 is a good starting point) until the visible pixels select artefacts reliably, but do not select real image features (e.g., the edge of the planetary disk). Once found, record the (`index`,`value`) pair.
5. Repeat (4) for each index you chose in step (3).

To get a set of (`index`, `path`) pairs for use with the `--dynamic_regions` argument, the following workflow is suggested:

* Open the observation FITS file in a FITS viewer that can create [DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9) compatible region files. Note, all keyboard/menu/shortcut instructions will assume DS9 is being used. Here is a quick description of how to use regions in DS9:
  - Switch to "Region Edit" mode via the keyboard shortcut [CTRL]+[R] or via `Edit` -> `Region` on the top menu bar.
  - Switch to the "region" tab, the 9th tab above the image area.
  - Select the region shape via `Region` -> `Shape` on the top menu bar.
  - **Place** a region with a [left-click], or [left-click] + [drag] to set the size at the same time.
  - Once placed, select a region with a [left-click] on its centre.
  - When selected:
	+ **Resize** by [left-click] + [drag] on the corner handles.
	+ **Delete** by pressing the [DELETE] key on the keyboard
	+ **Move** by [left-click] + [drag] when the mouse is over the region.
	+ **Label** the region by clicking the "information" button (1st button above the image area when the "region" tab is selected), filling in the "text" field, and clicking "apply".
	+ NOTE: You may have to move your mouse between operations (e.g. creation, selection, moving, resizing) to let DS9's event loop "catch up" with the new situation.
  - **Save** the current state of regions to a file using the "save" button in the "region" tab. Give the new file a name related to the observation and which wavelength channel the regions belong to (e.g., "neptune_observation_dynamic_000.reg". IMPORTANT: When saving, set the "Coordinate System" to "image" **not** "physical" (the package used to read the region files does not work with "physical" coordinates).
* Scroll through the wavelength channels of the image. For each moving artefact to mask out, do the following as required:
  - Periodically (every 50 or so channels):
	+ Align regions with their associated artefacts (e.g., move artefact-regions over their associated artefacts, change thier shapes as required to cover their associated artefacts) and save the regions to a file.
  - When an artefact appears:
	+ Create a new region for the new artefact, label it, align regions with their associated artefacts, and save the regions to a file.
  - When an artefact disappears:
	+ Delete the region for the artefact that disappeared, align regions with their associated artefacts, and save the regions to a file.
* When all channels have been gone through, close the FITS viewer.

NOTE: The current implementation using regions to mask out moving artefacts does this by associating regions within region-files across multiple (`index`, `path`) pairs by either: the "text" label of the region; or the region's location within a region-file if it has no text label. Therefore, it is recommended to **use text labels for ALL dynamic regions** as otherwise you have to never delete or create a region. Region properties are linearly interpolated between the wavelength channel index of the first region file they are present in, the index of any intermediate region files, and the index of the last region file they are present in. Outside the index of the first and last files a region is in, a region is not present and therefore does not mask any pixels. If you want a region to be present in all wavelength channels, include it in the region files for the first wavelength channel and the last wavelength channel.

#### Module Arguments ####

{% include module_arguments_note.md %}

* `-o` or `--output_path` : str
  - Output fits file path. If not specified, it is same as the path to the input file with "_bpmask" appended to the filename.

* `-x` or `--value_cut_at_index` : int float = 0 3
  - A single pair of `index` `value` numbers. Can be specified multiple times so the value cut can vary with index. Unspecified indices will be interpolated from the given (`index`,`value`) pairs, and `value` is clamped at the lowest and highest `index`. If not present, 3 is assumed for all indices.

* `--const_regions` : str
  - DS9 region files (one or more) that defines regions to be masked. Assumed to not move, and will be applied to all wavelengths. Must use IMAGE coordinates.

* `--dynamic_regions` : int str
  - (`index`, `path`) pair. Defines a set of region files that denote **dynamic** regions that should be masked. `index` denotes the wavelength index the regions in a file apply to, region parameters are interpolated between index values, and are associated by text label. Must use IMAGE coordinates.'

#### Examples ####

Using the results from the rebinning example:

* `python -m aopp_deconv_tool.create_bad_pixel_mask ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_artefactmap.fits`


### Interpolation <a id="interpolation-script"></a> ###

Invoke via `python -m aopp_deconv_tool.interpolate`.

Accepts a FITS file specifier for data to be interpolated and a FITS file specifier for a `bad_pixel_mask` that specifies which pixels to interpolate. At any NAN and INF values are also interpolated over if these are not included in the `bad_pixel_mask`.

The interpolation process uses a [standard interpolation routine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). However, to avoid edge effects the data is:

* embedded in a larger field of zeros
* convolved with a (3x3) kernel
* the centre region of the convolved data is replaced with the original data
* the interpolation is performed
* the centre region is extracted as the interpolation of the original data.

This process removes hard edges and reduces edge effects in a similar way to a "reflect" boundary condition (which the routine does not support at the time of writing) but the value tends towards zero and high-frequency variations have little impact.


#### Module Arguments ####

{% include module_arguments_note.md %}


* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_interp" appended to the filename.

* `--interp_method` : str = `scipy`
  - Selects how interpolation is performed, available options are:
	+ `scipy` :  Uses scipy routines to interpolate over the bad pixels. Uses a convolution technique to assist with edge effect problems.
	+ `sssa` : [EXPERIMENTAL] Calculates singular spectrum analysis (SSA) components, and uses the first 25% of them to fill in masked regions.
	+ `ssa_deviation` : [EXPERIMENTAL] Interpolates over SSA components only where extreme values are present. Testing has shown this to give results more similar to the underlying test data than `scipy`, but is substantially slower and requires parameter fiddling to give any substantial improvement.

#### Examples ####

Using the results from the rebinning example. Interpolation is perfomed via:

* `python -m aopp_deconv_tool.interpolate ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_artefactmap_bpmask.fits`

Unfortunately, the file `./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits` is not quite standard and lists its sky axes as 'PIXEL' axes. Therefore we have to provide the sky axes to the interpolate routine (or alter the FITS file). As axes are denoted using round brackets in a FITS Specifier, we have to wrap the string in single quotes and remove the escaping `\`s from the colons to enable the terminal to understand the string does not contain commands.


### PSF Normalisation <a id="psf-normalisation-script"></a> ###

Invoke via `python -m aopp_deconv_tool.psf_normalise`.

Performs the following operations:
* Ensures image shape is odd, so there is a definite central pixel
* Removes any outliers (based on the `sigma` option)
* Recentres the image around the centre of mass (uses the `threshold` and `n_largest_regions` options)
* Optionally trims the image to a desired shape around the centre of mass to reduce data volume and speed up subsequent steps
* Normalises the image to sum to 1

#### Module Arguments ####

{% include module_arguments_note.md %}

* `-o` or `--output_path` : str
  - Output fits file path. If not specified, it is same as the path to the input file with "_normalised" appended to the filename.

* `--threshold` : float = 1E-2
  - When finding region of interest, only values larger than this fraction of the maximum value are included.

* `--n_largest_regions` : int = 1
  - When finding region of interest, if using a threshold will only select the `n_largest_regions` regions in the calculation. A region is defined as a contiguous area where value >= `threshold` along `axes`. I.e., in a 3D cube, if we recentre about the centre of brightness (COB) on the sky (CELESTIAL) axes the regions will be calculated on the sky, not in the spectral axis (for example)

* `--background_threshold` : float = 1E-3
  - Exclude the largest connected region with values larger than this fraction of the maximum value when finding the background

* `--background_noise_model` : str = `gennorm`
  - Model of background noise to use when subtracting a global offset:
	+ `norm` : Assume background noise is a normal distribution
	+ `gennorm` : Assume background noise is a generalised normal distribution
	+ `none` : Do not use a background noise model and therefore do not subtract a global offset

* `--n_sigma` = 5
  - When finding the outlier mask, the number of standard deviations away from the mean a pixel must be to be considered an outlier

* `--trim_to_shape` : None \| int int = None
  - After centring etc. if not None, will trim data to the specified shape (i.e., rectangle of pixels) around the centre pixel. Used to reduce data volume for faster processing.

#### Examples ####

Using the results from the interpolation example. Normalisation is perfomed via:

* `python -m aopp_deconv_tool.psf_normalise './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin.fits(1,2)'`


### PSF Model Fitting <a id="psf-model-fitting-script"></a> ###

Invoke via `python -m aopp_deconv_tool.fit_psf_model`.

Fits a model, specified by the `--model` option, using a fitting method specified by the `--method` option. Fits each wavelength in turn, records the fitted values in a FITS table extension called "FITTED MODEL PARAMS", and the modelled PSF in the primary extension.

Fitting Methods:

* scipy.minimize
  - A simple gradient descent solver. Fast and useful when the optimal solution is close to the passed starting parameters.

* ultranest
  - Nested sampling. Much slower (but can be sped up, by fiddling with its settings), but works when the optimal solution has local maxima/minima that would trap `scipy.minimize`. Currently the `muse_ao` model only finds a good solution with this method. Setting ultranest to use a low number of points can cause a big speedup but causes the behaviour to resemble monte-carlo-markov-chain (MCMC) in that the probability distribution is not well defined due to the low number of sample points.

#### Module Arguments ####

{% include module_arguments_note.md %}

* `-o` or `--output_path` : str
  - Output fits file path. If not specified, it is same as the path to the input file with "_modelled" appended to the filename.

* `--fit_result_dir` : None \| str = None
  - Directory to store results of PSF fit in. Will create a sub-directory below the given path. If None, will create a sibling folder to the output file (i.e., output file parent directory is used)

* `--model` : str = `radial`
  - Model to fit to PSF data, options are:
	+ `radial` : Models the PSF as a radial histogram with logarithmically spaced bins. Histogram values are found by summing the signal in the relevant PSF annuli.
	+ `gaussian` : Models the PSF as a gaussian + a constant.
	+ `turbulence` : Models the PSF as a simple telescope that looking through an atmosphere that obeys a von-karman turbulence model.
	+ `muse_ao` : Models the PSF using a Moffat function as in [Fetick(2019)](https://doi.org/10.1051/0004-6361/201935830).

* `--method` : str = `scipy.minimize`
  - Method to use when fitting model to PSF data, options are:
	+ `ultranest` : Uses nested sampling to find where the parameters of a model maximise the likelihood.
	+`scipy.minimize` : Uses gradient descent to find the parameters that minimise the negative of the likelihood.

* `--model_help`
  - Shows help information about the selected model. You can specify starting/constant values, the domain over which a parameter can vary, and if the parameter is varied when fitting or held constant.

* `--<param>` : float
  - Sets the constant/starting value of a model parameter, default is model dependent.

* `--<param>_domain` : float float
  - Sets the domain (min, max) of a model parameter, default is model dependent.

* `--variables` : str ...
  - parameter names provided to this argument are varied by the fitting method within the domain set by `--<param>_domain`, others are held constant at the value set in `--<param>`.

#### Examples ####

Using results from the normalisation example, fitting is performed via:

* `python -m aopp_deconv_tool.fit_psf_model './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_normalised.fits(1,2)'`


### Deconvolution <a id="deconvolution-script"></a> ###

Invoke via `python -m aopp_deconv_tool.deconvolve`. Use the `-h` option to see the help message.

Assumes the observation data has no NAN or INF pixels, assumes the PSF data is centreed and sums to 1. 

#### Module Arguments ####

{% include module_arguments_note.md %}

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_deconv" appended to the filename.

* `--plot`
  - If present, will show plots of the progress of the deconvolution

* `--deconv_method` : str = `clean_modified`
  - which method to use for deconvolution, options are:
	+ `clean_modified`
	+ `lucy_richardson`

* `--deconv_method_help`
  - Show help for the selected deconvolution method
  
* `--<param>`
  - Set the value of a parameter of the chosen deconvolution method, use the `--deconv_method_help` option to list all parameters of a method and their defaults.

#### Description: CLEAN MODIFIED ####

At each iteration of the MODIFIED_CLEAN algorithm, the following procedure is performed:
* The current clean map is calculated, by convolving the components map (initially empty) with the PSF
* The residual is calculated by subtracting the current clean map from the original data
* The pixel selection metric is set equal to the absolute value of the residual
* Pixels in the selection metric (specified using the `--threshold` parameter) above a specified threshold create the selection mask. The threshold can be calculated one of two ways:
  - A static threshold, e.g., a fraction of the brightest pixel in the selection metric (range from 0->1, normally 0.3)
  - An adaptive threshold, e.g., the maximum fraction difference Otsu threshold calculated from the selection metric
* The selection mask is applied to the residual, and the selected pixels of the residual are copied into a new array called the current components and multiplied by the loop gain (range from 0->1, normally 0.02)
* The current components are added to the components map
* The current components are convolved with the PSF to create the "current convolved map"
* The current convolved map is subtracted from the residual
* Various statistics are calculated to determine a stopping point, if any of them fall below a user-set threshold the iteration terminates
  - The ratio of the brightest pixel in the residual to the brightest pixel in the observation
  - The ratio of the RMS of the residual to the RMS of the observation
  - The standard deviation of the above two statistics for the last 10 steps

Upon iteration, the components map **may** be convolved with a gaussian to regularise (smooth) it (set by the `--clean_beam_sigma` parameter). The smoothing function is referred to as the "clean beam" in radio astronomy sources. Often, careful choice of the threshold can give a smooth components map that does not require regularising in this way. An adaptive threshold, like the maximum fraction difference Otsu threshold, often performs better than a static threshold.

Use the `--plot` option to see a progress plot that updates every 10 iterations of the MODIFIED_CLEAN algorithm, useful for working out what different parameters do. The plot frames (from left-right, then top-bottom) are:
* A histogram of the residual (blue stepped line), with a vertical red line indicating the current threshold value
* An image of the residual
* An image of the current clean map
* An image of the components map
* An image of the selected pixels. Selected for the current step, magenta indicates pixels that are not selected.
* An image of the pixel choice metric, red is a "high selectability" blue is "low selectability". Exact calculation depends on the `--threshold` parameter, but is usually the absolute value of the pixel.
* A histogram of the pixel choice metric.
* A plot of metrics as a function of iteration number. `fabs value` (blue) is the absolute value of the brightest pixel in the residual. `rms value` (red) is the root-mean-square value of the residual.
* Note: The plots are defined in the `create_plot_set` function of the `deconvolve.py` script.


##### Parameters: CLEAN MODIFIED #####

* `--n_iter` : int = 1000
  - Maximum number of iterations performed
* `--loop_gain` : float = 0.02
  - Fraction of emission that could be accounted for by a PSF added to components each iteration. Higher values are faster, but unstable.
* `--threshold` : float = 0.3
  - Fraction of maximum brightness of residual above which pixels will be included in CLEAN step, if negative will use the maximum fractional difference otsu threshold. 0.3 is a  good default value, if stippling becomes an issue, reduce or set to a negative value. Lower positive numbers will require more iterations, but give a more "accurate" result.
* `--n_positive_iter` : int = 0
  - Number of iterations to do that only "adds" emission, before switching to "adding and subtracting" emission.
* `--noise_std` : float = 1E-1
  - Estimate of the deviation of the noise present in the observation, at the moment only used for calculating generalised least squares statistic.
* `--rms_frac_threshold` : float = 1E-2
  - Fraction of original RMS of residual at which iteration is stopped, lower values continue iteration for longer.
* `--fabs_frac_threshold` : float 1E-2
  - Fraction of original Absolute Brightest Pixel of residual at which iteration is stopped, lower values continue iteration for longer.
* `--max_stat_increase` : float = INF
  - Maximum fractional increase of a statistic before terminating.
* `--min_frac_stat_delta` : float = 1E-3
   - Minimum fractional standard deviation of statistics before assuming no progress is being made and terminating iteration.
* `--give_best_result` : bool = True
  - If True, will return the best (measured by statistics) result instead of final result.
* `--clean_beam_sigma` : float = 0
  - If not zero, will convolve the components with a gaussian with this sigma, useful for regularising the result.

#### Description: LUCY RICHARDSON ####

TODO: Write a description of how LR deconvolution works.

##### Parameters: LUCY RICHARDSON #####

* `--n_iter` : int = 100
  - Maximum number of iterations performed
* `--nudge_factor` : float = 1E-2
  - Fraction of maximum brightness to add to numerator and denominator to try and avoid numerical instability. This value should be in the range [0,1), and will usually be small. Larger values require more steps to give a solution, but suffer less numerical instability.
* `--strength` : float = 1E-1
  - Multiplier to the correction factors, if numerical insability is encountered decrease this. A more crude method of avoiding instability than `nudge_factor`, should be in the range (0,1].
* `--cf_negative_fix` : bool = True
  - Should we change negative correction factors to close-to-zero correction factors? Usually we should as we don\'t want any negative correction factors to flip-flop the end result.
* `--cf_limit` : float = INF
  - End iteration if the correction factors are larger than this limit. Large correction factors are a symptom of numerical instability.
* `--cf_uclip` : float = INF
  - Clip the correction factors to be no larger than this value. A crude method to control numerical instability.
* `--cf_lclip` : float = -INF
  - Clip the correction factors to be no smaller than this value. A crude method to control numerical instability.
* `--offset_obs` : bool = False
  - Should we offset the observation so there are no negative pixels? Enables the algorithm to find -ve values as the offset is reversed at the end.
* `--threshold` : None \| float = None
  - Below this value LR will not be applied to pixels. This is useful as at low brightness LR has a tendency to fit itself to noise. If -ve will use \|threshold\|*brightest_pixel as threshold each step. If zero will use mean and standard deviation to work out a threshold, if None will not be used.
* `--pad_observation` : bool = True
  - Should we pad the input data with extra space to avoid edge effects? Padding will take the form of convolving the observation with the psf, but only keeping the edges of the convolved data. This will hopefully cause a smooth-ish drop-off at the edges instead of a hard cutoff, thus reducing insability.


#### Examples ####

Using results from the previous examples, deconvolution is performed via:

* `python -m aopp_deconv_tool.deconvolve ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_interp.fits './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_normalised_modelled_radial.fits(1,2)' --threshold -1`
