# aopp_obs-toolchain #

Eventually this will consist of multiple packages, for now it just consists of aopp_deconv_tool.

## TODO ##

* Python virtual environment setup guide [DONE]

* Deconvolution example code + files

* PSF fitting example + files

* SSA filtering example

## Python Installation and Virtual Environment Setup ##

As Python is used by many operating systems as part of it's tool-chain it is a good idea to avoid
fiddling with the "system" installation (so you don't unintentionally overwrite packages or 
python versions and make it incompatible with your OS). Therefore the recommended way to use Python
is via a *virtual environment*. A *virtual environment* is isolated from your OS's Python installation.
It has its own packages, it's own `pip` (stands for "pip install packages", recursive acronyms...),
and its own versions of the other bits it needs. 

This package was developed using Python 3.12.2. Therefore, you need a Python 3.12.2 (or later) virtual
installation, and ideally a *virtual environment* to run it in.

### Installing Python ###

If using Windows or mac-OS, you can download an installer from [the official Python site](https://www.python.org/downloads/),
if using Unix/Linux you can install Python via the Package Manager included in your operating system, or
[build python from source](https://docs.python.org/3/using/unix.html). Building from source can be a little
fiddly, but there are [online tools to help with building from source](https://www.build-python-from-source.com/).
There is also an example script below that will fetch the python source code, install it, and create a virtual environment.

### Creating and Activating a Virtual Environment ###

With Python installed, make sure you have the correct version via `python --version`. If that command does
not print `Python 3.12.2` (or whichever version you expect), you need to find the full path to the Python
executable you just installed.

To create a virtual environment use the command `python -m venv <VENV_DIR>`, where `<VENV_DIR>` is the directory
you want the virtual environment to be in. E.g. `python -m venv .venv_3.12.2` will create the virtual
environment in the directory `.venv_3.12.2` in the current folder (NOTE: the `.` infront of the directory
name will make it hidden by default).

The process of activating the virtual environment varies depending on the terminal shell you are using.
On the command line, use one of the following commands:

* cmd.exe (Windows): `<VENV_DIR>\Scripts\activate.bat` 

* PowerShell (Windows, maybe Linux): `<VENV_DIR>/bin/Activate.ps1`

* bash|zsh (Linux, Mac): `source <VENV_DIR>/bin/activate`

* fish (Linux, Mac): `source <VENV_DIR>/bin/activate.fish`

* csh|tcsh (Linux, Mac): `source <VENV_DIR>/bin/activate.csh`

Once activated, your command line prompt should change to have something like `(.venv_3.12.2)` infront of it.

To check everything is working, enter the following commands:

* `python --version`
  - Should output the version you expect, e.g. `Python 3.12.2`

* `python -c 'import sys; print(sys.prefix != sys.base_prefix)'`
  - Should output `True` if you are in a virtual environment or `False` if you are not


### Linux Installation Bash Script ###

Below is an example *bash* script for building python from source and configuring a virtual environment.
Use it via copying the code into a file (recommended name `install_python.sh`). You will need `sudo` access
to use the script.

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
##############                       START SCRIPT                           ################
############################################################################################

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

echo "Installing python dependencies..."
sudo apt-get install -y \
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
	zlib1g-dev

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

## Installing the Package via Pip ##

Once you have installed Python 3.12.2 or higher, created a virtual environment, and activated
the virtual environment. Install the package via `pip install aopp_deconv_tool`



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
viewer that can show SSA components can be run via `python -m aopp_deconv_tool.graphical_frontends.ssa_filtering`.
By default it will show some test data, if you pass an **image** file (i.e. not a FITS file, but a `.jpg` etc.) it
will use that image instead of the default one.

The `ssa2d_sub_prob_map` function in the `aopp_deconv_tool.algorithm.bad_pixels.ssa_sub_prob` module attempts to 
make an informed choice of hot/cold pixels for masking purposes. See the docstring for more details.

The `ssa_interpolate_at_mask` function in the `aopp_deconv_tool.algorithm.interpolate.ssa_interp` module attempts
to interpolate data by interpolating between SSA components, only when the value of the component at the point
to be interpolated is not an extreme value. See the docstring for more details.