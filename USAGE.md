# aopp_obs_toolchain <a id="aopp_obs_toolchain"></a> #

Eventually this will consist of multiple packages, for now it just consists of aopp_deconv_tool.

See the [github](https://github.com/jackdobinson/aopp_obs_toolchain) for more details about the internal workings.

If you download the repository, there is doxygen documentation available. See the `README.md` file for more information on how it is generated and how to view it.

## TODO <a id="todo"></a>  ##


path_to_fits_file
path_to_fits_file{DATA}[:,10:20,30:50]{CELESTIAL:(1,2),SPECTRAL:(0)}

* Explain format of `python -m <module> ...`. I.e. which command line arguments go to what place.

* SSA filtering example [Half Done in Interpolation Example]

* Add domain to all numerical arguments.
  - deconvolve.py [DONE]

* Update usage line of deconvolve to show whole command when specifying algorithm parameters. [DONE]

* Change sub-command help argument to not require positional arguments to be specified.
  - deconvolve.py [DONE]

* Add which axis a script requires in the help message for the FITS Specifiers of the script. [DONE]

* Explain axis ordering difference between FITS, python, (C and Fortran) [DONE]

* Test if routines work on "single wavelength" FITS files (i.e. only CELESTIAL axes)
 - interpolate.py [WORKS]

* Write a converter from TIFF to FITS format. [DONE]

* Add information on what axes type a script requires. [DONE]

* Alter spectral rebinning to use triangular response function [DONE]

* Alter documentation in this file for spectral rebinning to reflect new algorithm [DONE]

* Split interpolation script into:
  - SSA Filtering/Artifact detection [DONE]
    + Take object FITS file, create heuteristic map
  - Create bad pixel map [DONE]
    + Use heuteristic map + arguments that specify how to classify a heuteristic map as 'bad' or 'good'
	+ returns boolean map of 'bad' pixels
  - Interpolate over a supplied mask [DONE]

## Python Installation and Virtual Environment Setup <a id="python-installation-and-virtual-environment-setup"></a>  ##

As Python is used by many operating systems as part of its tool-chain it's a good idea to avoid
fiddling with the "system" installation (so you don't unintentionally overwrite packages or 
python versions and make it incompatible with your OS). Therefore the recommended way to use Python
is via a *virtual environment*. A *virtual environment* is isolated from your OS's Python installation.
It has its own packages, its own `pip` (stands for "pip install packages", recursive acronyms...),
and its own versions of the other bits it needs. 

This package was developed using Python 3.12.2. Therefore, you need a Python 3.12.2 (or later)
installation, and ideally a *virtual environment* to run it in.

### Installing Python <a id="installing-python"></a>  ###

It is recommended that you **do not add the installed python to your path**. If you do, the operating system may/will find the installed version
before the operating system's expected version. And as our new installation almost certainly doesn't have the packages the operating system requires,
and may be an incompatible version, annoying things can happen. Instead, install Python in an obvious place that you can access easily. 

Suggested installation locations:

* Windows : `C:\Python\Python3.12`

* Unix/Linux/Mac: `${HOME}/python/python3.12`

Installation instructions for [windows, mac](#windows/mac-installation-instructions), [unix and linux](#unix/linux-installation-instructions) are slightly
different, so please refer to the appropriate section below.

Once installed, if using the suggested installation location, the actual Python interpreter executable will be at one of the following
locations or an equivalent relative location if not using the suggested install location:

* Windows: `C:\Python\Python3.12\bin\python3.exe`

* Unix/Linux/Mac: `${HOME}/python/python3.12/bin/python3`

* NOTE: if using *Anaconda*, you will have a `conda` command that manages the installation location of Python for you.

NOTE: I will assume a linux installation in this guide, so the executable will be at `${HOME}/python/python3.12/bin/python3`
in all code snippets. Alter this appropriately if using windows or a non-suggested installation location.

#### Windows/Mac Installation Instructions <a id="windows/mac-installation-instructions"></a> ####

* Download and run an installer from [the official Python site](https://www.python.org/downloads/).


#### Unix/Linux Installation Instructions <a id="unix/linux-installation-instructions"></a> ####

* **IF** you have `sudo` access [see the appendix for a test for sudo access](#sudo-access-test), try one of the following:

  - Install the desired version of Python via the Package Manager included in your operating system

  - Build and [install python from source](https://docs.python.org/3/using/unix.html).

    + NOTE: Building from source can be a little fiddly, but there are [online tools to help with building from source](https://www.build-python-from-source.com/).
      There is also a [python installation script in the appendix](#linux-installation-bash-script) that will fetch the python 
      source code, install it, and create a virtual environment.

* **OTHERWISE**, if you don't have `sudo` access, [anaconda python](https://docs.anaconda.com/free/miniconda/index.html#quick-command-line-install)
  is probably the easiest way as it does not require `sudo`. I recommend the `miniconda` version (linked above), as the main version installs many
  packages you may not need. You can always install other packages later.

  - NOTE: The main problem is that installing dependencies requires `sudo` access and, while there 
    [are ways around `sudo`](https://askubuntu.com/questions/339/how-can-i-install-a-package-without-root-access), 
    they are fiddly and annoying to use as you can quickly end up in [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell).





### Creating and Activating a Virtual Environment <a id="creating-and-activating-a-virtual-environment"></a> ###

A virtual environment isolates the packages you are using for a project from your normal environment and other virtual environments.
Generally they are created in a directory which we will call `<VENV_DIR>`, and then activated and deactivated as required. NOTE:
*anaconda python* has slightly different commands for managing virtual environments, and uses **names** of virtual environments instead
of **directories**, however the concept and the idea of activating and deactivating them remains the same dispite the slightly different
technical details.

NOTE: For the rest of this guide, *python* refers to a manual Python installation and *anaconda python* to Python provided by Anaconda.
NOTE: I will assume python version `3.12.2` for the rest of this guide but this will also work for different versions as long as the
      version number is changed appropriately.

#### Check Python Installation <a id="check-python-installation"></a> ####

With [Python installed](#installing-python), make sure you have the correct version via `${HOME}/python/python3.12/bin/python3 --version`. The command should print `Python 3.12.2`, or whichever version you expect.

#### Check Anaconda Python Installation <a id="check-anaconda-python-installation"></a> ####

If using *anaconda python* check that everything is installed correctly by using the command `conda --version`. This should print
a string like `conda X.Y.Z`, where X,Y,Z are the version number of anaconda.

#### Creating a Python Virtual Environment <a id="creating-a-python-virtual-environment"></a> ####

To create a virtual environment use the command `${HOME}/python/python3.12/bin/python3 -m venv <VENV_DIR>`, where `<VENV_DIR>` is the directory
you want the virtual environment to be in. E.g. `${HOME}/python/python3.12/bin/python3 -m venv .venv_3.12.2` will create the virtual
environment in the directory `.venv_3.12.2` in the current folder (NOTE: the `.` infront of the directory
name will make it hidden by default).

#### Creating an Anaconda Python Virtual Environment <a id="creating-an-anaconca-python-virtual-environment"></a> ####

*Anaconda Python* manages many of the background details for you. Use the command `conda create -n <VENV_NAME> python=3.12.2`, where
`<VENV_NAME>` is the name of the virtual environment to create. E.g. `conda create -n venv_3.12.2 python=3.12.2`


#### Activating and Deactivating a Python Virtual Environment <a id="activating-and-deactivating-a-python-virtual-environment"></a> ####

The process of activating the virtual environment varies depending on the terminal shell you are using.
On the command line, use one of the following commands:

* cmd.exe (Windows): `<VENV_DIR>\Scripts\activate.bat` 

* PowerShell (Windows, maybe Linux): `<VENV_DIR>/bin/Activate.ps1`

* bash|zsh (Linux, Mac): `source <VENV_DIR>/bin/activate`

* fish (Linux, Mac): `source <VENV_DIR>/bin/activate.fish`

* csh|tcsh (Linux, Mac): `source <VENV_DIR>/bin/activate.csh`

Once activated, your command line prompt should change to have something like `(.venv_3.12.2)` infront of it.

To check everything is working, enter the following commands (NOTE: the full path is not required as we are now using the virtual environment):

* `python --version`
  - Should output the version you expect, e.g. `Python 3.12.2`

* `python -c 'import sys; print(sys.prefix != sys.base_prefix)'`
  - Should output `True` if you are in a virtual environment or `False` if you are not.

To deactivate the environment, use the command `deactivate`. Your prompt should return to normal.

#### Activating and Deactivating an Anaconda Python Virtual Environment <a id="activating-and-deactivating-an-anaconda-python-virtual-environment"></a> ####

*Anaconda python* has a simpler way of activating a virtual environment. Use the command `conda activate <VENV_NAME>`, your prompt
should change to have something like `(<VENV_NAME>)` infront of it. Use `python --version` to check that the activated environment
contains the expected python version.

To deactivate the environment, use the command `conda deactivate`. Your prompt should return to normal.



## Installing the Package via Pip <a id="installing-the-package-via-pip"></a> ##

NOTE: If using *anaconda python* you **may** be able to use `conda install` instead of `pip` but I have not tested this. Conda [should behave well](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages) when using packages installed via `pip`.

Once you have [installed Python 3.12.2 or higher](#installing-python), [created a virtual environment](#creating-and-activating-a-virtual-environment), and [activated
the virtual environment](#activating-and-deactivating-a-python-virtual-environment). Use the following command to install the package:

* `python -m pip install --upgrade pip`
  - This updates pip to its latest version

* `python -m pip install aopp_deconv_tool`
  - This actually installs the package.

NOTE: We are using `python -m pip` instead of just `pip` incase the `pip` command does not point to the virtual environment's `pip` executable. You can run
`pip --version` to see which version of python it is for and where the pip executable is located if you want. As explanation, `python -m pip` means "use python
to run its pip module", whereas `pip` means "look on my path for the first executable called 'pip' and run it". Usually they are the same, but not always.


To update the package to it's newest version use:

* `python -m pip install --upgrade aopp_deconv_tool`


# aopp_deconv_tool <a id="aopp_deconv_tool"></a> #

This tool provides deconvolution, psf fitting, and ssa filtering routines.

NOTE: It can be useful to look through the source files, see the appendix for how to find [the package's source files location](#location-of-package-source-files)

## Examples <a id="examples"></a> ##

See the `examples` folder of the github. 

## FITS Specifier <a id="fits-specifier"></a> ##

When operating on a FITS file there are often multiple extensions, the axes ordering is unknown, and you may only need a subset of the data. Therefore, where possible scripts accept a *fits specifier* instead of a path. The format is a follows:

A string that describes which FITS file to load, the extension (i.e backplane) name or number to use enclosed in curly brackets, the slices (i.e. sub-regions) that should be operated upon in [python slice syntax](#python-slice-syntax), and the data axes to operate on as a [tuple](#python-tuple-syntax) or as a [python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) with strings as keys and [tuples](#python-tuple-syntax) as values.

See the appendix for a [quick introduction to the FITS format](#fits-file-format-information) for a description of why this information is needed, and why [axis numbers are different between FITS and Python](#fits-data-order).



```
Format:

	path_to_fits_file{ext}[slice0,slice1,...,sliceM](ax1,ax2,...,axL)

	OR

	path_to_fits_file{ext}[slice0,slice1,...,sliceM]{axes_type_1:(ax11,ax12,...,ax1L),...,axes_type_N:(axN1,axN2,...,axNL)}

	NOTE: everything that is not "path_to_fits_file" is optional and will use default values if not present.

	NOTE: FITS uses Column-major storage and 1-indexing (like Fortran). However, Python
		uses Row-major storage and 0-indexing (like C).	This has the effect of reversing
		the python numbering and adding one everywhere. I.e. In Python, the axes named 
		(0, 1, 2), as we have done above, a FITS file calls (3, 2, 1). Therefore, in the 
		above format, "slice0" operates on axis M+1 in FITS numbering, and "sliceM" 
		operates on axis 1 in FITS numbering. These specifiers use the PYTHON convention,
		so the LEFT axis is 0 and the numbers increment to the right. The FITS convention
		is that the LEFT axis is N and the numbers decrement to the right.

	Where:
		path_to_fits_file : str
			A path to a FITS file. Must be present.
		ext : str | int = 0
			Fits extension, either a string or an index. If not present, will assume PRIMARY hdu, (hdu index 0)
		sliceM : str = `:`
			Slice of Mth axis in the normal python `start:stop:step` format. If not present use all of an axis.
		axes_type_N : axes_type = inferred from context (sometimes not possible)
			Type of the Nth axes set. Axes types are detailed below. They are used to tell a program what to do with
			an axis when normal FITS methods of description fail (or as a backup). Note, the type (and enclosing curly backets
			`{{}}` can be ommited if a program only accepts one or two axes types. In that case, the specified axes will be
			assumed to be of the first type specified in the documentation, and the remaining axes of the second type (if there
			is a second type).
		axNL : int = inferred from headers in FITS file (sometimes not possible)
			Axes of the extension that are of the specified type. If not present, will assume all axes are of the first type
			specified in the documentation.

	Accepted axes_type:
		SPECTRAL
			wavelength/frequency varies along this axis
		CELESTIAL
			sky position varies along this axis
		POLARISATION
			polarisation varies along this axis, could be linear or circular
		TIME
			time varies along this axis

	NOTE: Not all scripts require all axes types to be specified. In fact almost all of them just require the CELESTIAL axes.
		And even then, they can often infer the correct values. The help information for a script should say which axes types
		it needs specified.
		
	NOTE: As the format for a FITS specifier uses characters that a terminal application may interpret as special characters,
		e.g. square/curly/round brackets, and colons. It can be better to wrap specifiers in quotes or single quotes. When
		doing this, it is important to un-escape any previously escaped characters.
		For example, specifies with timestamps in them would normally have the colons escaped, but when wrapped in quotes
		this is not required. E.g. the specifier ./example_data/MUSE.2019-10-17T23\:46\:14.117_normalised.fits(1,2) will not
		play nice with the bash shell due to the brackets. However, wrapping it in single quotes and removing the escaping
		slashes from the colons means it will work. E.g. './example_data/MUSE.2019-10-17T23:46:14.117_normalised.fits(1,2)'

	Examples:
		~/home/datasets/MUSE/neptune_obs_1.fits{DATA}[100:200,:,:](1,2)
			Selects the "DATA" extension, slices the 0th axis from 100->200 leaving the others untouched, and passes axes 1 and 2. For example, the script may require the celestial axes, and in this case axes 1 and 2 are the RA and DEC axes.
		
		~/home/datasets/MUSE/neptune_obs_1.fits{DATA}[100:200](1,2)
			Does the same thing as above, but omits un-needed slice specifiers.
		
		~/home/datasets/MUSE/neptune_obs_1.fits{DATA}[100:200]{CELESTIAL:(1,2)}
			Again, same as above, but adds explicit axes type.
```

## Commandline Scripts <a id="commandline-scripts"></a> ##

When running commandline scripts, use the `-h` option to see the help message. The appendix has a [overview of help message syntax](#overview-of-help-message-syntax).

The examples in this section use [example data stored on an external site](TODO: ADD LINK TO EXAMPLE DATA).

See the appendix for a [bash script that performs all steps on a given dataset](#whole-process-bash-script)

### Spectral Rebinning <a id="spectral-rebinning-script"></a> ##

Invoke via `python -m aopp_deconv_tool.spectral_rebin`. 

This routine accepts a FITS file specifier, it will spectrally rebin the fits extension and output a new fits file.

The underlying algorithm does the following:

* We have a 3d dataset. In the FITS file we get points that correspond to each pixel center, not a bin or region. We assume that the value "belongs" to the center point of the pixel and is not defined elsewhere. I.e. treat the data like point-cloud data. 

* The data is convolved with a response function (usually triangular for the spectral axis) of some characteristic with `bin_width`, and then sampled every `bin_step` from the start of the axis.

* NOTE: We define three things, and `offset`, a `bin_width`, and a `bin_step`. `offset` is the difference between the starting point of the new bins, `bin_width` is the "width" of the response function (for a histrogram the response function would be a square), `bin_step` is the distance between the start of one bin and the beginning of the next. Most of the time `bin_width` = `bin_step`, but sometimes they do not. A picture of a general example is below

```
	#############################################################
	NOTE: This is a general case and not accurate for a FITS file as we don't have bin-edges, just the centers. Therefore we assume bin_width=bin_step.
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

	"^" = bin center

	width
		The distance between the start and end of a bin
	
	step
		The distance between the start of two consencutive bins

	offset
		The distance between the start/end of the new grid and the start/end of the old grid
```

* We assume the `offset` is the same for the old and new binning.

* When sampling at `bin_step`, we linearly interpolate between the results of the convolution.

* The integral of the response function makes a big difference to the output. If there response function integrates to 1, we have the effect of averaging over the response function. This is appropriate when we are dealing with units that divide out the effect of the physical pixel size e.g. counts/wavelength. However, if we are dealing with raw counts, the response function should not integrate to 1 it's value should relate to the size of the response function relative to the size of the physical pixels.

* `bin_width` is defined differently for different response functions:
  - Triangular response function : `bin_width` is the full width at half maximum (FWHM),
  - Rectangular response function : `bin_width` is still the FWHM, but as it has straight sides this is the same as the full width.

#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_rebin" appended to the filename.

* `--rebin_operation`
  - `sum` sums the old bins into the new bins, use when you have a measurement like "counts"
  - `mean` averages the old bins into the new bins, use when you have a measurement like "counts per frequency" (DEFAULT)
  - `mean_err` averages the square of the old bins into the new bins then square roots, use when you have standard deviations of a measurement like "counts per frequency"


* One of the two mutually exclusive options:
  * `--rebin_params`
    - Takes two floats, `bin_step` and `bin_width`. Defines the new bin sizes, values are in SI units.

  * `--rebin_preset` (DEFAULT)
    - Is a named preset that defines `bin_step` and `bin_width`. Presets are:
      + `spex`: `bin_step` = 1E-9, `bin_width` = 2E-9 (DEFAULT)

#### Examples ####

Using the example data, the `datasets.json` file lists a dataset called "example neptune observation" with a science target observation in the file "MUSE.2019-10-18T00:01:19.521.fits" and a calibration observation in the file "MUSE.2019-10-17T23:46:14.117.fits"

Run the rebinning for each of the files via the command:
* `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117.fits`
* `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521.fits`

By default, files are rebinned by averaging old bins into new bins, and using the `spex` preset for bin width and bin step. The equivalent to the above command is `python -m aopp_deconv_tool.spectral_rebin ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117.fits --rebin_operation mean --rebin_preset spex --output_path ./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits`

After both command are complete there should be two new files that contain their output:
* `./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits`
* `./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits`

### Artifact Detection <a id="artifact-detection-script"></a> ###

Invoke via `python -m aopp_deconv_tool.artifact_detection`.

NOTE: The current implementataion of artifact detection is tuned for observations of objects of a significant fraction of the field size. Therefore it will not give good results for standard star observations.

Accepts a FITS specifier, uses a singular spectrum analysis (SSA) based algorithm to produce a heuteristic `badness_map` that reflects how likely a pixel is to be part of an artifact.

The badness map is calculated as follows for each 2D image in the FITS data:

* SSA components for a 10x10 window are calculated
* A subset of the components, assuming components are ordered by decending magnitude of eigenvalue, is chosen via:
  - Starting with 25% of the way through the components, as the components with the largest eigenvalues are likely to be made up of the main signal
  - Ending with 75% of the way through the components, as the components with the smallest eigenvalues are likely to be made up of noise.
* For each component in the chosen SSA subset, the number of standard deviations a pixel is away from the mean of it's region is calculated (called the `component_badness_map`)
  -  The "region" of a pixel is defined in the following way. A pixel's region is it's brightness class (a pixel is a member of one of three brightness thresholds (background, midground, foreground)), and all pixels of it's brightness class within N/8 from the pixel, where N is the largest x or y dimension of the image.
* The `component_badness_map`s for the chosen subset are averaged together to create the `badness_map` heuteristic.

#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_artifactmap" appended to the filename.

* `--strategy`
  - `ssa` (DEFAULT)
    + Uses singular spectrum analysis (SSA) to determine how likely a pixel is to belong to an artifact
* `--ssa.w_shape`
  - Shape of the window used for the `ssa` strategy (default : 10)
* `--ssa.start`
  - First SSA component to be included in artifact detection calc. Negative numbers are fractions of range
* `--ssa.stop`
  - Last SSA component to be included in artifact detection calc. Negative numbers are fractions of range


#### Examples ####

Using the results from the rebinning example:

* `python -m aopp_deconv_tool.artifact_detection './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin.fits(1,2)'`
* `python -m aopp_deconv_tool.artifact_detection ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits`

### Bad Pixel Mask <a id="bad-pixel-mask-script"></a> ###

Invoke via `python -m aopp_deconv_tool.create_bad_pixel_mask`.

Accepts a `badness_map` heuteristic, uses a set of value cuts to produce a boolean mask (the `bad_pixel_mask`) that describes which pixels are considered "bad" and should be interpolated over using a different script.

The `badness_map` is assumed to be a 3D cube, therefore the `bad_pixel_mask` is calculated from a set of (`index`,`value`) pairs. Where `index` is an index into the `badness_map`, and `value` is the value above which a pixel in the `badness_map` is considered "bad". Not all indices have to be specified, and values for unspecified indices will be interpolated (with the values clamped at the LHS and RHS). If no pairs are provided, a value of 3 is assumed for all indices. For each 1 above the cutoff value, a bad pixel is binary dilated. This way "very bad" pixels spread their "badness" to neighbouring pixels.

To get a set of (`index`, `value`) pairs, the following workflow is suggested:

1. Open the `badness_map` FTIS file in a FITS viewer of some sort (e.g. [DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9) or [QFitsView](https://sites.google.com/cfa.harvard.edu/saoimageds9)), you may need to use a logarithmic scale.
2. Open the data the `badness_map` was created from as well so you can compare them.
3. Choose some representative indices (i.e. wavelengths) to work on. For illustrative purposes we will assume indices, (10, 99, 135).
4. In the `badness_map` viewer, alter the minimum value of the data display range (somewhere around 4 or 5 is a good starting point) until the visible pixels select artifacts reliably, but do not select real image features (e.g. the edge of the planetary disk). Once found, record the (`index`,`value`) pair.
5. Repeat (4) for each index you chose in step (3).

#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_bpmask" appended to the filename.

* `-x` or `--value_cut_at_index`
  - A single pair of `index` `value` numbers. Can be specified multiple times so the value cut can vary with index. Unspecified indices will be interpolated from the given (`index`,`value`) pairs, and `value` is clamped at the lowest and highest `index`. If not present, 3 is assumed for all indices.

#### Examples ####

Using the results from the rebinning example:

* `python -m aopp_deconv_tool.create_bad_pixel_mask './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_artifactmap.fits(1,2)'`
* `python -m aopp_deconv_tool.create_bad_pixel_mask ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_artifactmap.fits`


### Interpolation <a id="interpolation-script"></a> ###

Invoke via `python -m aopp_deconv_tool.interpolate`.

Accepts a FTIS file specifier for data to be interpolated and a FITS file specifier for a `bad_pixel_mask` that specifies which pixels to interpolate. At any NAN and INF values are also interpolated over if these are not included in the `bad_pixel_mask`.

The interpolation process uses a [standard interpolation routine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html). However, to avoid edge effects the data is:

* embedded in a larger field of zeros
* convolved with a (3x3) kernel
* the center region of the convolved data is replaced with the original data
* the interpolation is performed
* the center region is extracted as the interpolation of the original data.

This process removes hard edges and reduces edge effects in a similar way to a "reflect" boundary condition (which the routine does not support at the time of writing) but the value tends towards zero and high-frequency variations have little impact.


#### Module Arguments ####


* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_interp" appended to the filename.

* `--interp_method` : selects how interpolation is performed:
  - `scipy` (DEFAULT)
    + Uses scipy routines to interpolate over the bad pixels. Uses a convolution technique to assist with edge effect problems.
  - `ssa`
    + [EXPERIMENTAL] Calculates singular spectrum analysis (SSA) components, and uses the first 25% of them to fill in masked regions.
  - `ssa_deviation`
    + [EXPERIMENTAL] Interpolates over SSA components only where extreme values are present. Testing has shown this to give results more similar to the underlying test data than `scipy`, but is substantially slower and requires parameter fiddling to give any substantial improvement.

#### Examples ####

Using the results from the rebinning example. Interpolation is perfomed via:

* `python -m aopp_deconv_tool.interpolate './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin.fits(1,2)' './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_artifactmap_bpmask.fits(1,2)'`
* `python -m aopp_deconv_tool.interpolate ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin.fits ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_artifactmap_bpmask.fits`

Unfortunately, the file `./example_data/ifu_observation_datasets/MUSE.2019-10-17T23\:46\:14.117_rebin.fits` is not quite standard and lists it's sky axes as 'PIXEL' axes. Therefore we have to provide the sky axes to the interpolate routine (or alter the FITS file). As axes are denoted using round brackets in a FITS Specifier, we have to wrap the string in single quotes and remove the escaping `\`s from the colons to enable the terminal to understand the string does not contain commands.


### PSF Normalisation <a id="psf-normalisation-script"></a> ###

Invoke via `python -m aopp_deconv_tool.psf_normalise`.

Peforms the following operations:
* Ensures image shape is odd, so there is a definite central pixel
* Removes any outliers (based on the `sigma` option)
* Recenters the image around the center of mass (uses the `threshold` and `n_largest_regions` options)
* Optionally trims the image to a desired shape around the center of mass to reduce data volume and speed up subsequent steps
* Normalises the image to sum to 1

#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_normalised" appended to the filename.

* `--threshold` = 1E-2
  - When finding region of interest, only values larger than this fraction of the maximum value are included.

* `--n_largest_regions` = 1
  - When finding region of interest, if using a threshold will only the n_largest_regions in the calculation. A region is defined as a contiguous area where value >= `threshold` along `axes`. I.e. in a 3D cube, if we recenter about the Center of mass (COM) on the sky (CELESTIAL) axes the regions will be calculated on the sky, not in the spectral axis (for example)

* `--background_threshold` = 1E-3
  - Exclude the largest connected region with values larger than this fraction of the maximum value when finding the background

* `--background_noise_model` : model of background noise to use when subtracting a global offset:
  - `norm`
    + Assume background noise is a normal distribution
  - `gennorm`
    + Assume background noise is a generalised normal distribution
  - `none` (DEFAULT)
    + Do not use a background noise model and therefore do not subtract a global offset

* `--n_sigma` = 5
  - When finding the outlier mask, the number of standard deviations away from the mean a pixel must be to be considered an outlier

* `--trim_to_shape` = None
  - After centering etc. if not None, will trim data to this shape around the center pixel. Used to reduce data volume for faster processing.

#### Examples ####

Using the results from the interpolation example. Normalisation is perfomed via:

* `python -m aopp_deconv_tool.psf_normalise './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_interp.fits(1,2)'`


### PSF Model Fitting <a id="psf-model-fitting-script"></a> ###

Invoke via `python -m aopp_deconv_tool.fit_psf_model`.

Fits a model, specified by the `--model` option, using a fitting method specified by the `--method` option. Fits each wavelength in turn, records the fitted values in a FITS table extension called "FITTED MODEL PARAMS", and the modelled PSF in the primary extension.

Fitting Methods:

* scipy.minimize
  - A simple gradient descent solver. Fast and useful when the optimal solution is close to the passed starting parameters.

* ultranest
  - Nested sampling. Much slower (but can be sped up, by fiddling with its settings), but works when the optimal solution has local maxima/minima that would trap `scipy.minimize`. Currently the `muse_ao` model only finds a good solution with this method. Setting ultranest to use a low number of points can cause a big speedup but causes the behaviour to resemble monte-carlo-markov-chain (MCMC) in that the probability distribution is not well defined due to the low number of sample points.

#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_modelled" appended to the filename.

* `--fit_result_dir`
  - Directory to store results of PSF fit in. Will create a sub-directory below the given path. If None, will create a sibling folder to the output file (i.e. output file parent directory is used)

* `--model` : Model to fit to PSF data:
  - `radial` (DEFAULT)
    + Models the PSF as a radial histogram with logarithmically spaced bins. Histogram values are found by summing the signal in the relevant PSF annuli.
  - `gaussian`
    + Models the PSF as a gaussian + a constant.
  - `turbulence`
    + Models the PSF as a simple telescope that looking through an atmosphere that obeys a von-karman turbulence model.
  - `muse_ao`
    + Models the PSF using a moffat function as in Fetick(2019).

* `--method` : Method to use when fitting model to PSF data:
  - `ultranest`
    + Uses nested sampling to find where the parameters of a model maximise the likelihood.
  - `scipy.minimize` (DEFAULT)
    + Uses gradient descent to find the parameters that minimise the negative of the likelihood.

* `--model_help`
  - Shows help information about the selected model. You can specify starting/constant values, the domain over which a parameter can vary, and if the parameter is varied when fitting or held constant.

* `--<param>`
  - Sets the constant/starting value of a model parameter, default is model dependent.

* `--<param>_domain`
  - Sets the domain (min, max) of a model parameter, default is model dependent.

* `--variables`
  - parameter names provided to this argument are varied by the fitting method within the domain set by `--<param>_domain`, others are held constant at the value set in `--<param>`.

#### Examples ####

Using results from the normalisation example, fitting is performed via:

* `python -m aopp_deconv_tool.fit_psf_model './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_interp_normalised.fits(1,2)'`

### Deconvolution <a id="deconvolution-script"></a> ###

Invoke via `python -m aopp_deconv_tool.deconvolve`. Use the `-h` option to see the help message.

Assumes the observation data has no NAN or INF pixels, assumes the PSF data is centered and sums to 1. Use the `--plot` option
to see an progress plot that updates every 10 iterations of the MODIFIED_CLEAN algorithm, useful for working out what different
parameters do.

At each iteration of the MODIFIED_CLEAN algorithm, the following procedure is performed:
* The clean map is calculated, by convolving the component map (initially empty) with the PSF
* The residual is calculated by subtracting the clean map from the original data
* The pixel selection metric is set equal to the absolute value of the residual
* Pixels in the selection metric above a specified threshold create the selection mask. The threshold can be calculated one of two ways
  - A static threshold, e.g. a fraction of the brightest pixel in the selection metric (range from 0->1, normally 0.3)
  - An adaptive threshold, e.g. the maximum fraction difference Otsu threshold calculated from the selection metric
* The selection mask is applied to the residual, and the selected pixels of the residual are copied into a new array called the current components and multiplied by the loop gain (range from 0->1, normally 0.02)
* The current components are added to the component map
* The current components are conolved with the PSF to create the "current convolved map"
* The current convolved map is subtracted from the residual
* Various statistics are calculated to determine a stopping point, if any of them fall below a user-set threshold the iteration terminates
  - The ratio of the brightest pixel in the residual to the brightest pixel in the observation
  - The ratio of the RMS of the residual to the RMS of the observation
  - The standard deviation of the above two statistics for the last 10 steps

Upon iteration, the component map **may** be convolved with a gaussian to regularise (smooth) it. This is referred to as the "clean beam". Often, careful choice of the threshold can give a smooth component map that does not require regularising in this way. An adaptive threshold, like the maximum fraction difference Otsu threshold, often performs better than a static threshold


#### Module Arguments ####

* `-o` or `--output_path`
  - Output fits file path. If not specified, it is same as the path to the input file with "_deconv" appended to the filename.

* `--plot`
  - If present, will show plots of the progress of the deconvolution

* `--deconv_method` : which method to use for deconvoltion:
  - `clean_modified` (DEFAULT)
  - `lucy_richardson`

* `--deconv_method_help`
  - Show help for the selected deconvolution method
  
* `--<param>`
  - Set the value of a parameter of the chosen deconvolution method, use the `--deconv_method_help` option to list all parameters of a method and their defaults.

#### Examples ####

Using results from the previous examples, deconvolution is performed via:

* `python -m aopp_deconv_tool.deconvolve ./example_data/ifu_observation_datasets/MUSE.2019-10-18T00\:01\:19.521_rebin_interp.fits './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117_rebin_interp_normalised_modelled_radial.fits(1,2)' --threshold -1`


## Using the Package in Code <a id="using-the-package-in-code"></a> ##

The commandline scripts provide a blueprint of how the various routines can be used. However sometimes more customisation is
needed. Below is a quick overview of the main routines and how they function. 

NOTE: You can always [get help within python](#getting-documentation-from-within-python) as the classes and functions have docstrings.


### Deconvolution <a id="deconvolution-code"></a> ###

The main deconvolution routines are imported via

```
from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
from aopp_deconv_tool.algorithm.deconv.lucy_richardson import LucyRichardson
```

`CleanModified` is the class implementeing the MODIFIED_CLEAN algorithm, the `LucyRichardson` class implements the Lucy-Richardson algorithm.

### PSF Fitting <a id="psf-fitting-code"></a> ###

The main PSF fitting routines are in `aopp_deconv_tools.psf_model_dependency_injector`, and `aopp_deconv_tools.psf_data_ops`. 
The examples on the github deal with this area. Specifically `<REPO_DIR>/examples/psf_model_example.py` for adaptive optics
instrument fitting.

### SSA Filtering <a id="ssa-filtering-code"></a> ###

Singular Spectrum Analysis is performed by the `SSA` class in the `aopp_deconv_tools.py_ssa` module. An interactive 
viewer that can show SSA components can be run via `python -m aopp_deconv_tool.graphical_frontends.ssa_filtering`.
By default it will show some test data, if you pass an **image** file (i.e. not a FITS file, but a `.jpg` etc.) it
will use that image instead of the default one.

The `ssa2d_sub_prob_map` function in the `aopp_deconv_tool.algorithm.bad_pixels.ssa_sub_prob` module attempts to 
make an informed choice of hot/cold pixels for masking purposes. See the docstring for more details.

The `ssa_interpolate_at_mask` function in the `aopp_deconv_tool.algorithm.interpolate.ssa_interp` module attempts
to interpolate data by interpolating between SSA components, only when the value of the component at the point
to be interpolated is not an extreme value. See the docstring for more details.


# APPENDICES <a id="appendices"></a> #

## APPENDIX: Supplimentary Information <a id="appendix:-supplimentary-information"></a> ##

### Overview of Help Message Syntax <a id="overview-of-help-message-syntax"></a> ###

Python scripts use the [argparse](https://docs.python.org/3/library/argparse.html) standard library module to parse commandline arguments. This generates help messages that follow a fairly standard syntax. Unfortunately there is no accepted standard, but [some style guides are available](https://docs.oracle.com/cd/E19455-01/806-2914/6jc3mhd5q/index.html).

The help message consists of the following sections:
* A "usage" line
* A quick description of the script, possibly with an example invocation.
* Positional and Keyword argument descriptions
* Any extra information that would be useful to the user.

#### The Usage Line ####

Contains a short description of the invocation syntax. 

* Starts with `usage: <script_name>` where `<script_name>` is the name of the script being invoked. 
* Then keyword arguments are listed, followed by positional arguments (sometimes these are reversed). 
  - There are two types of keyword arguments, *short* and *long*, usually to save space only the *short* name is in the usage line.
    + *short* denoted by a single dash followed by a single letter, e.g. `-h`.
    + *long* denoted by two dashes followed by a string, e.g. `--help`.
* Optional arguments are denoted by square brackets `[]`.
* A set of choices of arguments that are mutually exclusive are separated by a pipe `|`
* A grouping of arguments (e.g. for required arguments that are mutually exclusive) is done with round brackets `()`
* A keyword argument that requires a value will have one of the following after it:
  - An uppercase name that denotes the kind of value e.g. `NAME` 
  - A data type that denotes the value e.g. `float` or `int` or `string`
  - A set of choices of literal values (one of which must be chosen) is denoted by curly brackets `{}`, e.g. `{1,2,3}` or `{choice1,choice2}`.

Example: `usage: deconvolve.py [-h] [-o OUTPUT_PATH] [--plot] [--deconv_method {clean_modified,lucy_richardson}] obs_fits_spec psf_fits_spec`

* The script file is `deconvolve.py`
* The `-h` `-o` `--plot` `--deconv_method` keyword arguments are optional
* The `--deconv_method` argument accepts one of the values `clean_modified` or `lucy_richardson`
* The keyword arguments are `obs_fits_spec` and `psf_fits_spec`

Example: `usage: spectral_rebin.py [-h] [-o OUTPUT_PATH] [--rebin_operation {sum,mean,mean_err}] [--rebin_preset {spex} | --rebin_params float float] fits_spec`

* The script file is `spectral_rebin.py`
* The all of the keyword arguments `-h` `-o` `--rebin_operation` `--rebin_preset` `-rebin_params` are optional
* The single positional argument is `fits_spec` 
* The `--rebin_operation` argument has a choice of values `sum`,`mean`,`mean_err`
* The `--rebin_preset` and `--rebin_params` arguments are mutually exclusive, only one of them can be passed
* The `--rebin_preset` argument only accepts one value `spex`
* The `--rebin_params` argument accepts two values that should be floats


#### The Quick Description ####

The goal of the quick description is to summarise the intent of the program/script, and possibly give some guidance on how to invoke it.

#### Argument Descriptions ####

Usually these are grouped into two sections `positional arguments` and `options` (personally I would call them keyword arguments, as they can sometimes be required) but they follow the same syntax.

* The argument name, or names if it has more than one.
* Any parameters to the argument (for keyword arguments).
* A description of the argument, and ideally the default value of the argument.

Example:

```
--rebin_operation {sum,mean,mean_err}
                        Operation to perform when binning.
```
* Argument is a keyword argument (starts with `--`)
* Argument name is 'rebin_operation'
* Accepts one of `sum`,`mean`,`mean_err`
* Description is `Operation to perform when binning.`

A full argument description looks like this:
```
positional arguments:
  obs_fits_spec         The observation's (i.e. science target) FITS SPECIFIER, see the end of the help message for more information
  psf_fits_spec         The psf's (i.e. calibration target) FITS SPECIFIER, see the end of the help message for more information

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output fits file path. By default is same as the `fits_spec` path with "_deconv" appended to the filename (default: None)
  --plot                If present will show progress plots of the deconvolution (default: False)
  --deconv_method {clean_modified,lucy_richardson}
                        Which method to use for deconvolution. For more information, pass the deconvolution method and the "--info" argument. (default: clean_modified)
```

#### Extra Information ####

Information listed at the end is usually clarification about the formatting of string-based arguments and/or any other information that would be required to use the script/program. For example, in the above fill argument description example the FITS SPECIFIER format information is added to the end as extra information.



### FITS File Format Information <a id="fits-file-format-information"></a> ###

Documentation for the Flexible Image Transport System (FITS) file format is hosted at [NASA's Goddard Space Flight Center](https://fits.gsfc.nasa.gov/fits_standard.html), please refer to that as the authoratative source of information. What follows is a brief description of the format to aid understanding, see below for a [schematic of a fits file](#fits-file-schematic).

A FITS file consists of one or more *header data units" (HDUs). An HDU contains header and (optionally) data information. The first HDU in a file is the "primary" HDU, and others are "extension" HDUs. The primary HDU always holds image data, extension HDUs can hold other types of data (not just images, but tables and anything else specified by the standard). An HDU always has a number which describes it's order in the FITS file, and can optionally have a name. Naming an HDU is always a good idea as it helps users navigate the file. NOTE: The terms "extension", "HDU", and "backplane" are used fairly interchangably to mean HDU

Within each HDU there is header-data and (optionally) binary-data. The header-data consists of keys and values stored as restricted [ASCII](https://en.wikipedia.org/wiki/ASCII) strings of 80 characters in total. I.e. the whole key+value string must be 80 characters, they can be padded with spaces on the right. Practically, you can have as many header key-value entries as you have memory for. There are some reserved keys that define how the binary data of the HDU is to be interpreted. NOTE: Keys can only consist of uppercase latin letters, underscores, dashes, and numerals. The binary-data of an HDU is stored bin-endian, and intended to be read as a byte stream. The header-data descibes how to read the binary-data, the most common data is image data and tabular data.

Fits image HDUs (and the primary HDU) define the image data via the following header keywords.

BITPIX
: The magnitude is the number of bits in each pixel value. Negative values are floats, positive values are integers.

NAXIS
: The number of axes the image data has, from 0->999 (inclusive).

NAXISn
: The number of elements along axis "n" of the image.

Relating an axis to a coordinate system is done via more keywords that define a world coordinate system (WCS), that maps integer pixel indices to floating-point coordinates in, for example, time, sky position, spectral frequency, etc. The specifications for this are suggestions rather than rules, and non-conforming FITS files are not too hard to find. As details are what the spec is for, here is a high-level overview. Pixel indices are linearly transformed into "intermediate pixel coordinates", which are rescaled to physical units as "intermediate world coordinates", which are then projected/offset/have some (possibly non-linear) function applied to get them to "world coordinates". The CTYPE keyword for an axis describes what kind of axis it is, i.e. sky-position, spectral frequency, time, etc.

Therefore, when using a FITS file it is important to specify which HDU (extension) to use, which axes of an image correspond to what physical coordinates, and sometimes what subset of the binary-data we want to operate upon.

#### FITS Data Order <a id="fits-data-order"></a> ####

FITS files use the FORTRAN convention of **column-major ordering**, whereas Python uses **row-major ordering** (sometimes called "C" ordering). For example, if we have an N-dimensional matrix, then we can specify a number in that matrix by it's indices (e_1, e_2, e_3, ..., e_N). FITS files store data so that the **left** most index changes the fastest, i.e. in memory the data is stored {a_11, a_21, a_31, ..., a_M1, a_M2, ..., a_ML}. However, Python stores it's data where the **right** most index changes the fastest, i.e. data is stored in memory as {a_11, a_12, a_13, ..., a_1L, a_2L, a_3L, ..., a_NL}. Also, just to make things more difficult FITS (and Fortran) start indices at 1 whereas Python (and C) start indices at 0. The upshot of all of this is that if you have data in a FITS file, the axis numbers are related via the equation 

N - f_i = p_i

where N is the number of dimensions, f_i is the FITS/FOTRAN axis number, p_i is the Python/C axis number.

The upshot is that even though both FITS and Python label the axes of an array from left-to-right, (the "leftmost" axis being 0 for python, 1 for FITS), the ordering of the data in memory means that when reading a FITS array in Python, the axes are reversed.

Example:
```

Let `x` be a 3 by 4 matrix, it has two axes.
The FOTRAN convention is they are labelled 1 and 2,
the C convention is that they are labelled 0 and 1.
To make it obvious when we are talking about axes numbers
in c or fortran I will use (f1, f2) for fortran, and
(c0, c1) for c.

    / a b c d \
x = | e f g h |
    \ i j k l /

Assume `x` is stored as a 2 dimensional array.

In the FORTRAN convention, the matrix is stored in memory as
{a e i b f j c g k d h l}, i.e. the COLUMNS vary the fastest

|offset from start | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|
|------------------|--|--|--|--|--|--|--|--|--|--|--|--|
|value             | a| e| i| b| f| j| c| g| k| d| h| l|

Therefore, the memory offset of an element from the start of
memory is 
m_i = (row-1) + number_of_rows * (column-1)

In FOTRAN, the left most index varies the fastest, and indices
start from 1 so to extract a single number from x we index it via
x[row, column]
e.g. x[2,3] is an offset of (2-1)+3*(3-1) = 7, which selects 'g'.

In the C convention, the matrix is stored in memory as
{a b c d e f g h i j k l}, i.e. the ROWS vary the fastest

|offset from start | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|
|------------------|--|--|--|--|--|--|--|--|--|--|--|--|
|value             | a| b| c| d| e| f| g| h| i| j| k| l|

Therefore, the memory offset of an element from the start of
memory is 
m_i = number_of_columns * (row) + column

In C, the right most index varies the fastest, and indices
start at 0 so to extract a single number from x we index it via
x[row, column]
E.g. x[1,2] is an offset of 4*1+2 = 6, which selects 'g' also.

Wait, these are the same (except the offset of 1)?! That is because
we just looked at NATIVE data ordering. I.e. when FOTRAN and C have
data they have written themselves.

What if we get C to read a FORTRAN written array?

The data in memory is stored as
{a e i b f j c g k d h l}

If we index this the same way we did before, using x[row, columm]
we will have a problem.
E.g. x[1,2] is an offset of 4*1+2 = 6, which selects 'c', not 'g'!

This is happening because C assumes the axis that varies the fastest
is the RIGHT-MOST axis. But this data is written so the fastest
varing axis is the LEFT-MOST axis. So we should swap them around
and use x[column, row].
E.g. 
For C; m_i = number_of_columns * (row) + column. Therefore,
x[2,1] is an offset of 4*2+1 = 9, but that selects 'd', not 'g'!?

This fails because we forgot to swap around the formula for the
memory offset. The better way of writing the memory offset formula
is

m_i = number_of_entries_in_fastest_varying_axis * (slowest_varying_axis_index) 
      + fastest_varying_axis_index

And we know that for the data being written this way, the fastest
varying axis has 3 entries not 4.

E.g. x[2,1] is an offset of 3*2+1 = 7, which that selects 'g', success!

What we have actually done is just make sure that we read the data
the way it was written, in C the fastest varying axis is the RIGHT-MOST
axis when indexing, so we have to reverse the indices AND the lengths
of the axes. Therefore, data written in FORTRAN with axes (f1, f2, ... fN) and
axis lengths (K, L, ..., M), should be read in C with axes ordered as
(c0 = fN, c1 = fN-1, ..., cN-2 = f2, cN-1 = f1) and lengths (M, ..., L, K).

I.e. The fastest varying axis should always go at the correct position
LEFT in FORTRAN and RIGHT in C.

Confusion happens because:

* FORTRAN and C order axes from left-to-right
  - FORTRAN starts at 1
  - C starts at 0

* The **data order** is different between FOTRAN and C
  - FORTRAN writes data by varying the left-most axis the fastest
  - C writes data by varying the right-most axis the fastest
  
* FORTRAN and C index an N-dimensional **native data order** array the
  same way e.g. x[row, column]
  - Axes numbers are based on the **order in the written expression**
  - FORTRAN numbers these axes as row=1, column=2
  - C numbers these axes as row=0, column=1

* When reading data, it should always be read how it is ordered **in memory**
  for native data order this is as above, but for **non native data order**
  the axes order is reversed as it **requires the axis speeds to be the same**.
  - C should read data as x[slowest, fastest]
  - FORTRAN should read data as x[fastest, slowest]
  - So x[row, column] written by FORTRAN is read as x[column, row] by C
    FORTRAN numbers them row=1, column=2; C numbers them column=0, row=1.
  - In FOTRAN **low** axis numbers are fastest
  - In C **high** axis numbers are fastest


NOTE: axis numbers are always from the left hand side.
--------------------------------------
|FORTRAN     | In memory     |C axis |
|axis number | varying speed |number |
|------------|---------------|-------|
|     N      |   slow        |   0   |
|    N-1     |   slower      |   1   |
|    ...     |   ...         |  ...  |
|     2      |   fast        |  N-2  |
|     1      |   fastest     |  N-1  |
--------------------------------------

```




#### Fits File Schematic <a id="fits-file-schematic"></a> ####
```
FITS FILE
|- Primary HDU
|  |- header data
|  |- binary data (optional)
|- Extension HDU (optional)
|  |- header data
|  |- binary data (optional)
|- Extension HDU (optional)
|  |- header data
|  |- binary data (optional)
.
.
.
```

## APPENDIX: Snippets <a id="appendix:-snippets"></a> ##

### Sudo Access Test <a id="sudo-access-test"></a>  ###

Enter the following commands at the command line:

* `ls`
* `sudo ls` 

If, after entering your password, you see the same output for both commands, you have `sudo` access. Otherwise, you do not.

### Location of package source files <a id="location-of-package-source-files"></a> ### 

To find the location of the package's files, run the following command:

* `python -c 'import site; print(site.getsitepackages())'`

This will output the *site packages* directory for the python executable. The package's
files will be in the `aopp_deconv_tool` subdirectory.

### Getting documentation from within python <a id="getting-documentation-from-within-python"></a> ###

Python's command line, often called the "Read-Evaluate-Print-Loop (REPL)", has a built-in help system. 

To get the help information for a class, function, or object use the following code. Note, `>>>` denotes the
python REPL (i.e. the command line you get when you type the `python` command), and `$` denotes the shell 
commandline. 

This example is for the built-in `os` module, but should work with any python object.

```
$ python
... # prints information about the python version etc. here
>>> import os
>>> help(os)
... # prints out the docstring of the 'os' module
```

### Pyhton tuple syntax <a id="python-tuple-syntax"></a> ###

Tuples are ordered collections of hetrogenuous items. They are denoted by separating each element with a comma and enclosing the whole thing in round brackets. Tuples can be nested.

Examples:
* `(1,2,3)`
* `('cat', 7, 'dog', 9)`
* `(5.55, ('x', 'y', 0), 888, 'abc', 'def')`

### Python slice syntax <a id="python-slice-syntax"></a> ###

When specifying subsets of datacubes, it is useful to be able to select a N-square (i.e., square, cube, tesseract) 
region to operate upon to reduce data volume and therefore processing time. [Python and numpy's slicing syntax](https://www.w3schools.com/python/numpy/numpy_array_slicing.asp)
is a nice way to represent these operations. A quick explanation of the synatx follows.

#### Important Points ####

* Python arrays are zero-indexed. I.e the first element of an array has the index `0`.

* Negative indices count backwards from the end of the array. If you have `N` entries in an array, `-1` becomes `N-1`, so the slice 
  `0:-1` selects the whole array except the last element.

* Python slices have the format `start:stop:step` when they are used to index arrays via square brackets, e.g. `a[5:25:2]`
  returns a slice of the object `a`. Slices can also be defined by the `slice` object via `slice(start,stop,step)`, e.g.
  `a[slice(5,25,2)]` returns the same slice of object `a` as the last example.

* The `start`, `stop`, and `step` parameters generate indices of an array by iteratively adding `step` to `start` until the
  value is equal to or greater than `stop`. I.e. `selected_indices = start + step * i` where `i = {0, 1, ..., N-1}`, 
  `N = CEILING[(stop - start) / step]`.
  
* Mathematically, a slice specifies a [half-open interval](https://en.wikipedia.org/wiki/Interval_(mathematics)),
  the `step` of a slice selects every `step` entry of that interval. I.e. they include their start point but not their end point, 
  and only select every `step` elements of the interval. E.g. `5:25:2` selects the elements at the indices {5,7,9,11,13,15,17,19,21,23}

* By default `start` is zero, `stop` is the number of elements in the array, and `step` is one.

* Only the first colon in a slice is required to define it. The slice `:` is equivalent to the slice `0:N:1`, where `N` is the number
  of elements in the object being sliced. E.g. `a[:]` selects all the elements of `a`.

* Negative parameters to `start` and `stop` work the same way as negative indices. Negative values to `step` reverse the default values
  of `start` and `stop`, but otherwise work in the same way as positive values.
  
* A slice never extends beyond the beginning or end of an array. I.e. even though negative numbers are valid parameters, a slice that
  would result in an index *before* it's `start` will be empty (i.e. select no elements of the array). E.g. If we have an array with 5
  entries, the slice `1:3:-1` **does not** select {1, 0, 4}, it is empty.

#### Details ####

Let `a` be a 1 dimensional array, such that `a = np.array([10,11,15,16,19,20])`, selecting an element of the array
is done via square brackets `a[1]` is the 1^th element, and as python is 0-indexed is equal to 11 for our example array.

Slicing is also done via square brackets, instead of a number (that would select and element), we pass a slice. Slices are
defined via the format `<start>:<stop>:<step>`. Where `<start>` is the first index to include in the slice, `<stop>` is the
first index to **not** include in the slice, and `<step>` is what to add to the previously included index to get the next
included index.

E.g. 
* `2:5:1` includes the indices 2,3,4 in the slice. So `a[2:5:1]` would select 15,16,19.
* `0:4:2` includes the indices 0,2 in the slice. So `a[0:4:2]` would select 10,15.

Eveything in a slice is optional exept the first colon. The defaults of everything are as follows:
* `<start>` defaults to 0
* `<stop>` defaults to the last + 1 index in the array. As python supports negative indexing (-ve indices "wrap around", with 
  -1 being the index **after** the largest index) this is often called the -1^th index, or that `<stop>` defaults to -1.
* `<step>` defaults to 1

Therefore, the slice `:` selects all of the array, and the slice `::-1` selects all of the array, but reverses the ordering.

When dealing with N-dimensional arrays, indexing accepts a tuple. 
E.g. for a 2-dimensional array `b=np.array([[10,11,15],[16,19,20],[33,35,36]])`, 
* `b[1,2]` is equal to 20
* `b[2,1]` is equal to 35

Similarly, slicing an N-dimensional array uses tuples of slices. E.g.
```
>>> b
array([[10, 11, 15],
       [16, 19, 20],
       [33, 35, 36]])
>>> b[::-1,:]
array([[33, 35, 36],
       [16, 19, 20],
       [10, 11, 15]])
>>> b[1:2,::-1]
array([[20, 19, 16]])
```

Slices and indices can be mixed, so you can slice one dimension and select an index from another. E.g.
```
>>> b[:1, 2]
array([15])
>>> b[::-1, 0]
array([33, 16, 10])
>>> b[0,::-1]
array([15, 11, 10])
```

There is a `slice` object in Python that can be used to programatically create slices, it's prototype is `slice(start,stop,step)`,
but only `stop` is required, and if `stop=None` the slice will continue until the end of the array dimension. Slice objects
are almost interchangeable with the slice syntax. E.g.
```
>>> s = slice(2)
>>> b[s,0]
array([10, 16])
>>> b[:2,0]
array([10, 16])
```


## APPENDIX: Scripts <a id="appendix:-scripts"></a> ##

### Whole Process Bash Script <a id="whole-process-bash-script"></a>  ###

Below is an example *bash* script for performing every step of the deconvolution process on an observation and standard star file

```
#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Constants
THIS_SCRIPT=$0
USAGE="USAGE: whole_process.sh [-hr] <obs_fits:path> <std_fits:path> [slice:str] [spectral_axes:str] [celestial_axes:str]"

# Functions
exit_with_msg() { echo "${@:2}"; exit $1; }
arg_error() { echo "${THIS_SCRIPT} ERROR: ${1}"; echo "${USAGE}"; exit 1; }

# START Parse Arguments
# NOTE: We just send the filenames, we rely on the defaults of the FITS Specifiers to handle extension and slice information for us
# Option defaults
RECALC=0

# let positional arguments and optional arguments be intermixed
# Therfore, must do this without useing "getopts"
N_REQUIRED_POS_ARGS=2
N_OPTIONAL_POS_ARGS=3
N_MAX_POS_ARGS=$((${N_REQUIRED_POS_ARGS}+${N_OPTIONAL_POS_ARGS}))
# echo "N_REQUIRED_POS_ARGS=${N_REQUIRED_POS_ARGS}"
# echo "N_OPTIONAL_POS_ARGS=${N_OPTIONAL_POS_ARGS}"
# echo "N_MAX_POS_ARGS=${N_MAX_POS_ARGS}"

declare -a POS_ARGS=()
ARGS=($@)
for ARG_IDX in ${!ARGS[@]}; do
	#echo "Processing argument at index ${ARG_IDX}"
	ARG=${ARGS[${ARG_IDX}]}
	#echo "    ${ARG}"
	#echo "#POS_ARGS[@]=${#POS_ARGS[@]}"
	case $ARG in
		-h)
			exit_with_msg 0 ${USAGE}
			;;
		-r)
			RECALC=1
			;;
		*)
			if [[ ${#POS_ARGS[@]} -lt ${N_MAX_POS_ARGS} ]]; then
				POS_ARGS+=(${ARG})
			else
				arg_error "Maximum of ${N_MAX_POS_ARGS} positional arguments supported. Argument \"${ARG}\" is not an option or a positional."
			fi
			;;
	esac
done
if [[ ${#POS_ARGS[@]} -lt ${N_REQUIRED_POS_ARGS} ]]; then
	arg_error "Only ${#POS_ARGS[@]} positional arguments were specified, but ${N_REQUIRED_POS_ARGS} are required."
fi

#echo "POS_ARGS=${POS_ARGS[@]}"

# Get observation and standard star as 1st and 2nd argument to this script
FITS_OBS=${POS_ARGS[0]}
FITS_STD=${POS_ARGS[1]}
# Get slices, spectral axes, celestial axes as arguments 3,4,5
SLICE=${POS_ARGS[2]:-'[:]'}
SPECTRAL_AXES=${POS_ARGS[3]:-'(0)'}
CELESTIAL_AXES=${POS_ARGS[4]:-'(1,2)'}



# Output argument values for user information
echo "FITS_OBS=${FITS_OBS}"
echo "FITS_STD=${FITS_STD}"
echo "SLICE=${SLICE}"
echo "SPECTRAL_AXES=${SPECTRAL_AXES}"
echo "CELESTIAL_AXES=${CELESTIAL_AXES}"
echo "RECALC=${RECALC}"
# END Parse Arguments

# Set parameter constants

PSF_MODEL_STR="radial" # "radial" is the current default, it influences the name of the one of the output files
FITS_VIEWERS=("QFitsView" "ds9")

# Create output filenames for each step of the process that mirror the default output filenames

FITS_OBS_REBIN="${FITS_OBS%.fits}_rebin.fits"
FITS_OBS_REBIN_ARTIFACT="${FITS_OBS%.fits}_rebin_artifactmap.fits"
FITS_OBS_REBIN_ARTIFACT_BPMASK="${FITS_OBS%.fits}_rebin_artifactmap_bpmask.fits"
FITS_OBS_REBIN_INTERP="${FITS_OBS%.fits}_rebin_interp.fits"
FITS_OBS_REBIN_INTERP_DECONV="${FITS_OBS%.fits}_rebin_interp_deconv.fits"

FITS_STD_REBIN="${FITS_STD%.fits}_rebin.fits"
FITS_STD_REBIN_NORM="${FITS_STD%.fits}_rebin_normalised.fits"
FITS_STD_REBIN_NORM_MODEL="${FITS_STD%.fits}_rebin_normalised_modelled_${PSF_MODEL_STR}.fits"


ALL_FITS_FILES=(
	${FITS_OBS_REBIN} 
	${FITS_STD_REBIN}
	${FITS_OBS_REBIN_ARTIFACT}
	${FITS_OBS_REBIN_ARTIFACT_BPMASK}
	${FITS_OBS_REBIN_INTERP}
	${FITS_STD_REBIN_NORM}
	${FITS_STD_REBIN_NORM_MODEL}
	${FITS_OBS_REBIN_INTERP_DECONV}
)

# Perform each stage in turn

echo "Performing spectral rebinning"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN} ]]; then
	python -m aopp_deconv_tool.spectral_rebin "${FITS_OBS}${SLICE}${SPECTRAL_AXES}"
fi
if [[ ${RECALC} == 1 || ! -f ${FITS_STD_REBIN} ]]; then
	python -m aopp_deconv_tool.spectral_rebin "${FITS_STD}${SLICE}${SPECTRAL_AXES}"
fi

echo "Performing artifact detection"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_ARTIFACT} ]]; then
	python -m aopp_deconv_tool.artifact_detection "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}"
fi

echo "Creating bad pixel mask"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_ARTIFACT_BPMASK} ]]; then
	python -m aopp_deconv_tool.create_bad_pixel_mask "${FITS_OBS_REBIN_ARTIFACT}${SLICE}${CELESTIAL_AXES}"
fi

echo "Interpolating at bad pixel mask"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_INTERP} ]]; then
	python -m aopp_deconv_tool.interpolate "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}" "${FITS_OBS_REBIN_ARTIFACT_BPMASK}${SLICE}${CELESTIAL_AXES}"
fi

echo "Normalising PSF"
if [[ ${RECALC} == 1 || ! -f ${FITS_STD_REBIN_NORM} ]]; then
	python -m aopp_deconv_tool.psf_normalise "${FITS_STD_REBIN}${SLICE}${CELESTIAL_AXES}"
fi

echo "Modelling PSF"
if [[ ${RECALC} == 1 || ! -f ${FITS_STD_REBIN_NORM_MODEL} ]]; then
	python -m aopp_deconv_tool.fit_psf_model "${FITS_STD_REBIN_NORM}${SLICE}${CELESTIAL_AXES}" --model "${PSF_MODEL_STR}"
fi

echo "Performing deconvolution"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_INTERP_DECONV} ]]; then
	python -m aopp_deconv_tool.deconvolve "${FITS_OBS_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}" "${FITS_STD_REBIN_NORM_MODEL}${SLICE}${CELESTIAL_AXES}"
fi

echo "Deconvolved file is ${FITS_OBS_REBIN_INTERP_DECONV}"

# Open all products in the first viewer available
for FITS_VIEWER in ${FITS_VIEWERS[@]}; do
	if command -v ${FITS_VIEWER} &> /dev/null; then
		${FITS_VIEWER} ${ALL_FITS_FILES[@]} &
		break
	fi
done
```

### Linux Installation Bash Script <a id="linux-installation-bash-script"></a>  ###

Below is an example *bash* script for building python from source and configuring a virtual environment.
Use it via copying the code into a file (recommended name `install_python.sh`). If Python's dependencies
are not already installed, you will need `sudo` access so the script can install them.

* Make the script executable : `chmod u+x install_python.sh`

* Get help on the scripts options with: `./install_python.sh -h`

* Run the script with : `./install_python.sh`


```
#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Remember values of environment variables as we enter the script
OLD_IFS=$IFS 
INITIAL_PWD=${PWD}



############################################################################################
##############                    PROCESS ARGUMENTS                         ################
############################################################################################

# Set default parameters
PYTHON_VERSION=(3 12 2)
PYTHON_INSTALL_DIRECTORY="${HOME:?}/python/python_versions"
VENV_PREFIX=".venv_"
VENV_DIR="${PWD}"

# Get the usage string with the default values of everything
usage(){
	echo "install_python.sh [-v INT.INT.INT] [-i PATH] [-p STR] [-d PATH] [-l PATH] [-h]"
	echo "    -v : Python version to install. Default = ${PYTHON_VERSION[0]}.${PYTHON_VERSION[1]}.${PYTHON_VERSION[2]}"
	echo "    -i : Path to install python to. Default = '${PYTHON_INSTALL_DIRECTORY}'"
	echo "    -p : Prefix for virtual environment (will have python version added as a suffix). Default = ${VENV_PREFIX}"
	echo "    -d : Directory to create virtual envronment. Default = '${VENV_DIR}'"
	echo "    -h : display this help message"

}
USAGE=$(usage)

# Parse input arguments
while getopts "v:i:p:d:h" OPT; do
	case $OPT in
		v)
			IFS="."
			PYTHON_VERSION=(${OPTARG})
			IFS=$OLD_IFS
			;;
		i)
			PYTHON_INSTALL_DIRECTORY=${OPTARG}
			;;
		p)
			VENV_PREFIX=${OPTARG}
			;;
		d)
			VENV_DIR=${OPTARG}
			;;
		*)
			echo "${USAGE}"
			exit 0
			;;
	esac
done

# Perform argument processing
PYTHON_VERSION_STR="${PYTHON_VERSION[0]}.${PYTHON_VERSION[1]}.${PYTHON_VERSION[2]}"

# Print parameters to user so they know what's going on
echo "Parameters:"
echo "    -v : PYTHON_VERSION=${PYTHON_VERSION_STR}"
echo "    -i : PYTHON_INSTALL_DIRECTORY=${PYTHON_INSTALL_DIRECTORY}"
echo "    -p : VENV_PREFIX=${VENV_PREFIX}"
echo "    -d : VENV_DIR=${VENV_DIR}"


############################################################################################
##############                     DEFINE FUNCTIONS                         ################
############################################################################################


function install_pkg_if_not_present(){

	# Turn on "strict" mode
	set -o errexit -o nounset -o pipefail
	REQUIRES_INSTALL=()

	for PKG in ${@}; do
		# We want the command to fail when a package is not installed, therefore unset errexit
		set +o errexit 
			DPKG_RCRD=$(dpkg-query -l ${PKG} 2> /dev/null | grep "^.i.[[:space:]]${PKG}\(:\|[[:space:]]\)")
			INSTALLED=$?
		set -o errexit
		
		if [ ${INSTALLED} -eq 0 ]; then
			echo "${PKG} is installed"	
		else
			echo "${PKG} is NOT installed"
			REQUIRES_INSTALL[${#REQUIRES_INSTALL[@]}]=${PKG}
		fi

	done


	if [ ${#REQUIRES_INSTALL[@]} -ne 0 ]; then


		UNFOUND_PKGS=()
		for PKG in ${REQUIRES_INSTALL[@]}; do
			# We want the command to fail when a package is not installed, therefore unset errexit
			set +o errexit 
				apt-cache showpkg ${PKG} | grep '^Package: ${PKG}$'
				PKG_FOUND=$?
			set -o errexit

			if [ $PKG_FOUND -ne 0 ]; then
				UNFOUND_PKGS[${#UNFOUND_PKGS[@]}]=${PKG}
			fi
		done

		if [ ${#UNFOUND_PKGS[@]} -ne 0 ]; then 
			echo "ERROR: Cannot install. Could not find the following packages in apt: ${UNFOUND_PKGS[@]}"
			return 1
		fi

		echo "Installing packages: ${REQUIRES_INSTALL[@]}"
		sudo apt-get install -y ${REQUIRES_INSTALL[@]}
	else
		echo "All required packages are installed"
	fi
}


############################################################################################
##############                       START SCRIPT                           ################
############################################################################################

# Define the dependencies that python requires for installation
PYTHON_DEPENDENCIES=(   \
	curl                \
	gcc                 \
	libbz2-dev          \
	libev-dev           \
	libffi-dev          \
	libgdbm-dev         \
	liblzma-dev         \
	libncurses-dev      \
	libreadline-dev     \
	libsqlite3-dev      \
	libssl-dev          \
	make                \
	tk-dev              \
	wget                \
	zlib1g-dev          \
)

# Get a temporary directory and make sure it's cleaned up when the script exits
TEMP_WORKSPACE=$(mktemp -d -t py_build_src.XXXXXXXX)
cleanup(){
	echo "Cleaning up on exit..."
	echo "Removing ${TEMP_WORKSPACE}"
	rm -rf ${TEMP_WORKSPACE:?}
}
trap cleanup EXIT

# If there is an error, make sure we print the usage string with default parameter values
error_message(){
	echo "${USAGE}"
}
trap error_message ERR


# Define variables

PYTHON_VERSION_INSTALL_DIR="${PYTHON_INSTALL_DIRECTORY}/${PYTHON_VERSION_STR}"
VENV_PATH="${VENV_DIR}/${VENV_PREFIX}${PYTHON_VERSION_STR}"
PYTHON_VERSION_SOURCE_URL="https://www.python.org/ftp/python/${PYTHON_VERSION_STR}/Python-${PYTHON_VERSION_STR}.tgz"

PY_SRC_DIR="${TEMP_WORKSPACE}/Python-${PYTHON_VERSION_STR}"
PY_SRC_FILE="${PY_SRC_DIR}.tgz"


# Perform actions

echo "Checking python dependencies and installing if required..."
install_pkg_if_not_present ${PYTHON_DEPENDENCIES}

echo "Downloading python source code to '${PY_SRC_FILE}'..."
curl ${PYTHON_VERSION_SOURCE_URL} --output ${PY_SRC_FILE}

echo "Extracting source file..."
mkdir ${PY_SRC_DIR}
tar -xvzf ${PY_SRC_FILE} -C ${TEMP_WORKSPACE}


cd ${PY_SRC_DIR}
echo "Configuring python installation..."
./configure                                  \
	--prefix=${PYTHON_VERSION_INSTALL_DIR:?} \
	--enable-optimizations                   \
	--with-lto                               \
	--enable-ipv6

echo "Running makefile..."
make

echo "Created ${PYTHON_VERSION_INSTALL_DIR}"
mkdir -p ${PYTHON_VERSION_INSTALL_DIR}

echo "Performing installation"
make install

cd ${INITIAL_PWD}

echo "Creating virtual environment..."
${PYTHON_VERSION_INSTALL_DIR}/bin/python3 -m venv ${VENV_PATH}

echo "Virtual environment created at ${VENV_PATH}"


# Output information to user
echo ""
echo "Activate the virtual environment with the following command:"
echo "    source ${VENV_PATH}/bin/activate"
```