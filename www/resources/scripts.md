## Bash Scripts ##

### Full Deconvolution Process <a id="whole-process-bash-script"></a>  ###

Below is an example *bash* script for performing every step of the deconvolution process on an observation and standard star file

```bash
#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Constants
THIS_SCRIPT=$0
USAGE="USAGE: whole_process.sh [-hr] <obs_fits:path> <std_fits:path> [slice:str] [spectral_axes:str] [celestial_axes:str]

Performs the entire deconvolution process from start to finish. Acts as a 
test-bed, an example bash script, and a way to use the tool without 
babysitting it.


# ARGUMENTS #

  obs_fits : path
	Path to the FITS file of the science observation to use, it will be 
	deconvolved at the end of this process.
  std_fits : path
	Path to the FITS file of the standard star observation to use, \`obs_fits\`
	will be deconvolved using (a model of) this as the PSF.


# OPTIONS #

  -h
	Display this help message
  -r
	Recalculate all products
  slice : str
	Python-style slice notation that will be applied to obs_fits and std_fits 
	data, often used to focus on specific spectral slice of data
  spectral_axes : str
	Axis number of spectral axis, enclosed in brackets e.g., '(0)'. Will be 
	automatically calculated if not present.
  celestial_axes : str
	Axis numbers of celestial axes, enclosed in brakcets e.g., '(1,2)'. Will be
	automatically calculated if not present.
"

# Functions
exit_with_msg() { echo "${@:2}"; exit $1; }
arg_error() { echo "${THIS_SCRIPT} ERROR: ${1}"; echo "${USAGE}"; exit 1; }

# START Parse Arguments
# NOTE: We just send the filenames, we rely on the defaults of the FITS Specifiers to handle extension and slice information for us
# Option defaults
RECALC=0

# let positional arguments and optional arguments be intermixed
# Therefore, must do this without useing "getopts"
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
			exit_with_msg 0 "${USAGE}"
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
FITS_OBS_REBIN_artefact="${FITS_OBS%.fits}_rebin_artefactmap.fits"
FITS_OBS_REBIN_artefact_BPMASK="${FITS_OBS%.fits}_rebin_artefactmap_bpmask.fits"
FITS_OBS_REBIN_INTERP="${FITS_OBS%.fits}_rebin_interp.fits"
FITS_OBS_REBIN_INTERP_DECONV="${FITS_OBS%.fits}_rebin_interp_deconv.fits"

FITS_STD_REBIN="${FITS_STD%.fits}_rebin.fits"
FITS_STD_REBIN_NORM="${FITS_STD%.fits}_rebin_normalised.fits"
FITS_STD_REBIN_NORM_MODEL="${FITS_STD%.fits}_rebin_normalised_modelled_${PSF_MODEL_STR}.fits"


ALL_FITS_FILES=(
	${FITS_OBS_REBIN} 
	${FITS_STD_REBIN}
	${FITS_OBS_REBIN_artefact}
	${FITS_OBS_REBIN_artefact_BPMASK}
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

echo "Performing artefact detection"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_artefact} ]]; then
	python -m aopp_deconv_tool.artefact_detection "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}"
fi

echo "Creating bad pixel mask"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_artefact_BPMASK} ]]; then
	python -m aopp_deconv_tool.create_bad_pixel_mask "${FITS_OBS_REBIN_artefact}${SLICE}${CELESTIAL_AXES}"
fi

echo "Interpolating at bad pixel mask"
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN_INTERP} ]]; then
	python -m aopp_deconv_tool.interpolate "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}" "${FITS_OBS_REBIN_artefact_BPMASK}${SLICE}${CELESTIAL_AXES}"
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

### Linux Python Installation <a id="python-installation-on-linux-bash-script"></a>  ###

Below is an example *bash* script for building python from source and configuring a virtual environment.
Use it via copying the code into a file (recommended name `install_python.sh`). If Python's dependencies
are not already installed, you will need `sudo` access so the script can install them.

* Make the script executable : `chmod u+x install_python.sh`

* Get help on the scripts options with: `./install_python.sh -h`

* Run the script with : `./install_python.sh`


```bash
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