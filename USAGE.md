# aopp_obs-toolchain #

Eventually this will consist of multiple packages, for now it just consists of aopp_deconv_tool.

## TODO ##

* Python virtual environment setup guide [DONE]

* Add instructions for non-sudo access installation [DONE]

* Add instrucitons for CONDA install + virtual environment [DONE]

* Add instruction for updating to latest versions [DONE]

* Add how to find package source files [DONE]

* Get some **small** example files and add them to the package to be used with example deconvolution script.
  - Look up python package index's policy on hosting example data files
  - Zenodo - Site for uploading and sharing data.
  - Also try using the shared storage

* Deconvolution example code + files

* PSF fitting example + files

* SSA filtering example

## Python Installation and Virtual Environment Setup ##

As Python is used by many operating systems as part of its tool-chain it's a good idea to avoid
fiddling with the "system" installation (so you don't unintentionally overwrite packages or 
python versions and make it incompatible with your OS). Therefore the recommended way to use Python
is via a *virtual environment*. A *virtual environment* is isolated from your OS's Python installation.
It has its own packages, its own `pip` (stands for "pip install packages", recursive acronyms...),
and its own versions of the other bits it needs. 

This package was developed using Python 3.12.2. Therefore, you need a Python 3.12.2 (or later)
installation, and ideally a *virtual environment* to run it in.

### Installing Python ###

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

#### Windows/Mac Installation Instructions ####

* Download and run an installer from [the official Python site](https://www.python.org/downloads/).


#### Unix/Linux Installation Instructions ####

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





### Creating and Activating a Virtual Environment ###

A virtual environment isolates the packages you are using form a project from your normal environment and other virtual environments.
Generally they are created in a directory which we will call `<VENV_DIR>`, and then activated and deactivated as required. NOTE:
*anaconda python* has slightly different commands for managing virtual environments, and uses **names** of virtual environments instead
of **directories**, however the concept and the idea of activating and deactivating them remains the same dispite the slightly different
technical details.

NOTE: For the rest of this guide, *python* refers to a manual Python installation and *anaconda python* to Python provided by Anaconda.
NOTE: I will assume python version `3.12.2` for the rest of this guide but this will also work for different versions as long as the
      version number is changed appropriately.

#### Check Python Installation ####

With [Python installed](#installing-python), make sure you have the correct version via `${HOME}/python/python3.12/bin/python3 --version`. The command should
not print `Python 3.12.2`, or whichever version you expect.

#### Check Anaconda Python Installation ####

If using *anaconda python* check that everything is installed correctly by using the command `conda --version`. This should print
a string like `conda X.Y.Z`, where X,Y,Z are the version number of anaconda.

#### Creating a Python Virtual Environment ####

To create a virtual environment use the command `${HOME}/python/python3.12/bin/python3 -m venv <VENV_DIR>`, where `<VENV_DIR>` is the directory
you want the virtual environment to be in. E.g. `${HOME}/python/python3.12/bin/python3 -m venv .venv_3.12.2` will create the virtual
environment in the directory `.venv_3.12.2` in the current folder (NOTE: the `.` infront of the directory
name will make it hidden by default).

#### Creating an Anaconda Python Virtual Environment ####

*Anaconda Python* manages many of the background details for you. Use the command `conda create -n <VENV_NAME> python=3.12.2`, where
`<VENV_NAME>` is the name of the virtual environment to create. E.g. `conda create -n venv_3.12.2 python=3.12.2`


#### Activating and Deactivating a Python Virtual Environment ####

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

#### Activating and Deactivating an Anaconda Python Virtual Environment ####

*Anaconda python* has a simpler way of activating a virtual environment. Use the command `conda activate <VENV_NAME>`, your prompt
should change to have something like `(<VENV_NAME>)` infront of it. Use `python --version` to check that the activated environment
contains the expected python version.

To deactivate the environment, use the command `conda deactivate`. Your prompt should return to normal.



## Installing the Package via Pip ##

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


# aopp_deconv_tool #

This tool provides deconvolution, psf fitting, and ssa filtering routines.

NOTE: It can be useful to look through the source files, see the appendix for how to find [the package's source files location](#location-of-package-source-files)

## Examples ##

See the `examples` folder of the github. 

## Commandline Scripts ##

### Spectral Rebinning ##

Invoke via `python -m aopp_deconv_tool.spectral_rebin`. Use the `-h` option to see the help message.

This routine accepts a FITS file specifier, it will spectrally rebin the fits extension and output a new fits file.

### Interpolation ##

Invoke via `python -m aopp_deconv_tool.interpolate`. Use the `-h` option to see the help message.

Accepts a FTIS file specifier, will find bad pixels and interpolate over them. The strategies used are
dependent on the options given to the program.

bad pixel strategies:
	ssa
		Uses singular spectrum analysis to determine bad pixels. Useful for situations where artifacts are not
		seperable from the science data via a simple brightness threshold. Also interpolates over INF and NAN pixels.
	simple
		Only interpolates over INF and NAN pixels

interpolation strategies:
	scipy
		Uses scipy routines to interpolate over the bad pixels. Uses a convolution technique to assist with edge effect problems.
	ssa
		[EXPERIMENTAL] Interpolates over SSA components only where extreme values are present. Testing has shown this to give
		results more similar to the underlying test data than `scipy`, but is substantially slower and requires parameter
		fiddling to give any substantial improvement.

### PSF Normalisation ###

Invoke via `python -m aopp_deconv_tool.psf_normalise`. Use the `-h` option to see the help message.

Peforms the following operations:
* Ensures image shape is odd, so there is a definite central pixel
* Removes any outliers (based on the `sigma` option)
* Recenters the image around the center of mass (uses the `threshold` and `n_largest_regions` options)
* Optionally trims the image to a desired shape around the center of mass to reduce data volume and speed up subsequent steps
* Normalises the image to sum to 1


### PSF Model Fitting ###

Invoke via `python -m aopp_deconv_tool.psf_normalise`. Use the `-h` option to see the help message.
NOTE: The `--model` option sets the model to fit. To see which parameters a model accepts use the `--model_help` option [NOTE: CHECK THIS WORKS]

Specifying the `--method` option sets the routine used for fitting. Two are available `scipy.minimize` (default) and `ultranest`.

scipy.minimize
	A simple gradient descent solver. Fast and useful when the optimal solution is close to the passed starting parameters.

ultranest
	Nested sampling. Much slower (but can be sped up), but works when the optimal solution has local maxima/minima that
	would trap `scipy.minimize`. Currently the `muse_ao` model only finds a good solution with this method.

### Deconvolution ###

Invoke via `python -m aopp_deconv_tool.deconvolve`. Use the `-h` option to see the help message.
NOTE: the `--parameter_help` option will show the help message for the deconvolution parameters [NOTE: CHECK THIS WORKS]

Assumes the observation data has no NAN or INF pixels, assumes the PSF data is centered and sums to 1. Use the `--plot` option
to see an progress plot that updates every 10 iterations of the MODIFIED_CLEAN algorithm, useful for working out what different
parameters do.


## Deconvolution ##

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

## PSF Fitting ##

The main PSF fitting routines are in `aopp_deconv_tools.psf_model_dependency_injector`, and `aopp_deconv_tools.psf_data_ops`. 
The examples on the github deal with this area. Specifically `<REPO_DIR>/examples/psf_model_example.py` for adaptive optics
instrument fitting.

## SSA Filtering ##

Singular Spectrum Analysis is performed by the `SSA` class in the `aopp_deconv_tools.py_ssa` module. An interactive 
viewer that can show SSA components can be run via `python -m aopp_deconv_tool.graphical_frontends.ssa_filtering`.
By default it will show some test data, if you pass an **image** file (i.e. not a FITS file, but a `.jpg` etc.) it
will use that image instead of the default one.

The `ssa2d_sub_prob_map` function in the `aopp_deconv_tool.algorithm.bad_pixels.ssa_sub_prob` module attempts to 
make an informed choice of hot/cold pixels for masking purposes. See the docstring for more details.

The `ssa_interpolate_at_mask` function in the `aopp_deconv_tool.algorithm.interpolate.ssa_interp` module attempts
to interpolate data by interpolating between SSA components, only when the value of the component at the point
to be interpolated is not an extreme value. See the docstring for more details.


# APPENDICES #

## APPENDIX: Snippets ##

### Sudo Access Test ###

Enter the following commands at the command line:

* `ls`
* `sudo ls` 

If, after entering your password, you see the same output for both commands, you have `sudo` access. Otherwise, you do not.

### Location of package source files ###

To find the location of the package's files, run the following command:

* `python -c 'import site; print(site.getsitepackages())'`

This will output the *site packages* directory for the python executable. The package's
files will be in the `aopp_deconv_tool` subdirectory.

## APPENDIX: Scripts ##

### Linux Installation Bash Script ###

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