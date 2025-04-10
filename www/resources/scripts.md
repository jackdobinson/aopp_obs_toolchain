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

## Remember values of environment variables as we enter the script
OLD_IFS=$IFS 
INITIAL_PWD=${PWD}

## Define Constants
TRUE=0
FALSE=1

## Define Variables that don't depend on arguments
# Size of subarea to display output of long-running commands, set to 0 to disable.
TERMINAL_SUBAREA_SIZE=20
# Flag to set terminal back to previous state
TERM_DIRTY_FLAG=${FALSE}
# For keeping track of when we should exit.
EXIT_FLAG=${FALSE}


############################################################################################
##############                    PROCESS ARGUMENTS                         ################
############################################################################################

# Set default parameters
PYTHON_VERSION=(3 12 2)
PYTHON_INSTALL_DIRECTORY="${HOME:?}/.local/.python"
VENV_PREFIX=".venv_"
VENV_DIR="${PWD}"
LOG_FILE="${PYTHON_INSTALL_DIRECTORY}/install_<python_version>.log"
PYTHON_SOURCE_DIR=""

# Flags for default parameters
LOG_FILE_SET=${FALSE}
SHOW_HELP=${FALSE}
SHOW_CV_HELP=${FALSE}

# Get the usage string with the default values of everything
usage(){
        local param_value_type_string=${1:-Current Value}
        {
                echo "install_python.sh [-v INT.INT.INT] [-i PATH] [-p STR] [-d PATH] [-l PATH] [-s PATH] [-h] [-H]"
                echo ""
                print_param_info "${param_value_type_string}"
        } | text_frame_around '-' '|' 'USAGE'
}

text_n_str(){
        local n=$1
        local str=${2}

        for ((i=0; i<n; i++)); do echo -n "${str}"; done
}

text_frame_around() {
        local hchar=${1:--}
        local vchar=${2:-|}
        local title=${3:-}

        local ifs_old="${IFS}"

        if [ -n "${title}" ]; then
                title=" ${title} "
        fi

        local n_title=$(echo "${title}" | wc -L)
        local text_str="$(cat -)"
        local lmax=$(wc -L <<<"${text_str}")

        if [ $((lmax+4)) -lt ${n_title} ]; then
                lmax=$((n_title+2))
        else
                lmax=$((lmax + 4))
        fi

        local remainder=${lmax}

        # First line
        echo -n "${hchar}"
        echo -n "${title}"
        text_n_str $((lmax - 1 - n_title)) ${hchar}
        echo ""

        # Empty line
        echo -n ${vchar}
        text_n_str $((lmax -2)) ' '
        echo ${vchar}

        # Content
        while read; do
                remainder=$(wc -L <<<"$REPLY")
                remainder=$((lmax -2 - remainder))
                echo -n "${vchar} "
                echo -n "$REPLY"
                text_n_str $((remainder -1)) ' '
                echo "${vchar}"

        done <<<"$text_str"

        # Empty line
        echo -n ${vchar}
        text_n_str $((lmax -2)) ' '
        echo ${vchar}

        # Final line
        text_n_str ${lmax} ${hchar}
        echo ""

}

print_param_info() {
        local param_value_type_string=${1:-Current Value}

        echo "    -v : PYTHON_VERSION <INT.INT.INT>"
        echo "         Python version to install."
        echo "         ${param_value_type_string} = ${PYTHON_VERSION[0]}.${PYTHON_VERSION[1]}.${PYTHON_VERSION[2]}"
        echo ""
        echo "    -i : PYTHON_INSTALL_DIRECTORY <PATH>"
        echo "         Path to install python to."
        echo "         ${param_value_type_string} = '${PYTHON_INSTALL_DIRECTORY}'"
        echo ""
        echo "    -p : VENV_PREFIX <STR>"
        echo "         Prefix for virtual environment (will have python version added as a suffix)."
        echo "         ${param_value_type_string} = '${VENV_PREFIX}'"
        echo ""
        echo "    -d : VENV_DIR <PATH>"
        echo "         Directory to create virtual envronment in, if empty will not create a virtual environment."
        echo "         ${param_value_type_string} = '${VENV_DIR}'"
        echo ""
        echo "    -l : LOG_FILE <PATH>"
        echo "         File to copy output to. If empty will only output to stdout (the terminal)."
        echo "         ${param_value_type_string} = '${LOG_FILE}'"
        echo ""
        echo "    -s : PYTHON_SOURCE_DIR <PATH>"
        echo "         Directory to download source to, will be a temp directory if not set. "
        echo "         ${param_value_type_string} = '${PYTHON_SOURCE_DIR}'"
        echo ""
        echo "    -h : Display this help message with default parameter values"
        echo ""
        echo "    -H : Display this help message with passed parameter values"
        echo ""
}


USAGE=$(usage "Default")

# Parse input arguments
while getopts "v:i:p:d:l:s:hH" OPT; do
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
                l)
                        LOG_FILE=${OPTARG}
                        LOG_FILE_SET=${TRUE}
                        ;;
                s)
                        PYTHON_SOURCE_DIR=${OPTARG}
                        ;;
                H)
                        SHOW_CV_HELP=${TRUE}
                        ;;
                *)
                        SHOW_HELP=${TRUE}
                        ;;
        esac
done

## Perform argument processing

PYTHON_VERSION_STR="${PYTHON_VERSION[0]}.${PYTHON_VERSION[1]}.${PYTHON_VERSION[2]}"

# If the log file was not set on the command-line, use a default location
if [ ${LOG_FILE_SET} -eq ${FALSE} ]; then
        LOG_FILE="${PYTHON_INSTALL_DIRECTORY}/install_${PYTHON_VERSION_STR}.log"
fi

# If the python source directory is not specified, use a temporary directory
if [ -z "${PYTHON_SOURCE_DIR}" ]; then
        TEMP_WORKSPACE=$(mktemp -d -t py_build_src.XXXXXXXX)
        PYTHON_SOURCE_DIR="${TEMP_WORKSPACE}"
else
        TEMP_WORKSPACE=""
fi

# After any processing, show help message if required.
if [ ${SHOW_HELP} -eq ${TRUE} ]; then
        echo "${USAGE}"
        EXIT_FLAG=${TRUE}
fi
if [ ${SHOW_CV_HELP} -eq ${TRUE} ]; then
        echo "$(usage)"
        EXIT_FLAG=${TRUE}
fi
if [ ${EXIT_FLAG} -eq ${TRUE} ]; then
        exit 0
fi

# Print parameters to user so they know what's going on
echo "Parameters:"
print_param_info


############################################################################################
##############                     DEFINE FUNCTIONS                         ################
############################################################################################


install_pkg_if_not_present(){

        # Turn on "strict" mode
        set -o errexit -o nounset -o pipefail
        REQUIRES_INSTALL=()

        for PKG in "$@"; do
                # We want the command to fail when a package is not installed, therefore unset errexit
                set +o errexit 
                        DPKG_RCRD=$(dpkg-query -l ${PKG} 2> /dev/null | grep "^.i.[[:space:]]${PKG}\(:\|[[:space:]]\)")
                        INSTALLED=$?
                set -o errexit

                if [ ${INSTALLED} -eq 0 ]; then
                        echo "  ${PKG} is installed"
                else
                        echo "  ${PKG} is NOT installed"
                        REQUIRES_INSTALL[${#REQUIRES_INSTALL[@]}]=${PKG}
                fi

        done


        if [ ${#REQUIRES_INSTALL[@]} -ne 0 ]; then


                UNFOUND_PKGS=()
                for PKG in ${REQUIRES_INSTALL[@]}; do
                        # We want the command to fail when a package is not installed, therefore unset errexit
                        set +o errexit 
                                apt-cache showpkg ${PKG} | grep -E "^Package: ${PKG}"
                                PKG_FOUND=$?
                        set -o errexit

                        if [ $PKG_FOUND -ne 0 ]; then
                                echo "Could not find package '${PKG}' using 'apt-cache showpkg'"
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

set_term_scroll_region(){
        TERM_DIRTY_FLAG=${TRUE}

        local n_lines=$(tput lines)
        local n_scroll=${1:-${TERMINAL_SUBAREA_SIZE:-$((lines/5))}}
        local n_after=2
        local scroll_to=$((n_lines - 1 - n_after))
        local scroll_from=$((scroll_to - n_scroll))

        for ((i=0; i<(n_scroll+1+n_after); i++)); do echo ""; done

        tput cup $((scroll_from -1)) 0
        echo "-------------------------------------------------------------------------"
        tput cup $((scroll_to +1)) 0
        echo "-------------------------------------------------------------------------"

        tput csr $scroll_from $scroll_to

        tput cup $scroll_from 0

}

unset_term_scroll_region(){
        n_lines=$(tput lines)
        #tput smcup
        tput csr 0 $((n_lines -1))
        #tput rmcup
        tput cup $n_lines
        TERM_DIRTY_FLAG=${FALSE}
}

term_subarea(){
        set_term_scroll_region
        cat -
        unset_term_scroll_region
}

term_remove_ctrl_chars(){
        sed -u 's/\x1B[@A-Z\\\]^_]\|\x1B\[[0-9:;<=>?]*[-!"#$%&'"'"'()*+,.\/]*[][\\@A-Z^_`a-z{|}~]//g'
}


install_python(){

        echo "Checking python dependencies and installing if required..."
        install_pkg_if_not_present ${PYTHON_DEPENDENCIES[@]} | term_subarea

        if [ -f "${PY_SRC_FILE}" ]; then
                echo "Source for python version ${PYTHON_VERSION_STR} already exists at ${PY_SRC_FILE}"
        else
                echo "Downloading python source code to '${PY_SRC_FILE}'..."
                mkdir -p ${PYTHON_SOURCE_DIR}
                curl ${PYTHON_VERSION_SOURCE_URL} --output ${PY_SRC_FILE}
        fi

        if [ -f "${PY_SRC_DIR}/configure" ]; then
                echo "Python source already extracted to ${PY_SRC_DIR}"
        else
                echo "Extracting source file..."
                mkdir -p ${PY_SRC_DIR}
                tar -xvzf ${PY_SRC_FILE} -C ${PYTHON_SOURCE_DIR} | term_subarea
        fi

        cd ${PY_SRC_DIR}
        echo "Configuring python installation..."
        ./configure                                  \
                --prefix=${PYTHON_VERSION_INSTALL_DIR:?} \
                --enable-optimizations                   \
                --with-lto                               \
                --enable-ipv6                            \
                --with-computed-gotos                    \
                --with-system-ffi                        \
                --disable-test-modules                   \
                --with-ensurepip=install                 \
                | term_subarea



        echo "Running makefile..."
        if command -v nproc &> /dev/null; then
                # Spread across all processors if possible
                make -j $(($(nproc)-1)) | term_subarea
        else
                make | term_subarea
        fi

        echo "Created ${PYTHON_VERSION_INSTALL_DIR}"
        mkdir -p ${PYTHON_VERSION_INSTALL_DIR}

        echo "Performing installation..."
        if [ -e "${PYTHON_VERSION_INSTALL_DIR}/bin/python3" ]; then
                echo "NOTE: '${PYTHON_VERSION_INSTALL_DIR}/bin/python3' already exists, using 'altinstall'"
                echo "----:  to ensure we do not overwrite it as it may be the system Python"
                make altinstall                              \
                        | term_subarea
        else
                make install                                 \
                        | term_subarea
        fi
        cd ${INITIAL_PWD}
}

create_virtual_environment() {

        if [ -z "${VENV_DIR}" ]; then
                echo "No virtual environment created."
                return
        fi

        echo "Creating virtual environment..."
        ${PYTHON_VERSION_INSTALL_DIR}/bin/python3 -m venv ${VENV_PATH}

        echo "Virtual environment created at ${VENV_PATH}"


        # Output information to user
        echo ""
        echo "Activate the virtual environment with the following command:"
        echo "    source ${VENV_PATH}/bin/activate"
} 


############################################################################################
##############                       START SCRIPT                           ################
############################################################################################

# Define the dependencies that python requires for installation
PYTHON_DEPENDENCIES=(   \
        curl                \
        gcc                 \
        make                \
        wget                \
        xz-utils            \
        tar                 \
        llvm                \
        libbz2-dev          \
        libev-dev           \
        libffi-dev          \
        libgdbm-dev         \
        liblzma-dev         \
        libncurses-dev      \
        libreadline-dev     \
        libsqlite3-dev      \
        libssl-dev          \
        tk-dev              \
        zlib1g-dev          \
)

if commmand -v apt &> /dev/null; then
        {
                echo "Could not install Python for this machine.                       "
                echo "                                                                 "
                echo "This script is created for Linux distributions that use the 'apt'"
                echo "package distribution program, i.e.. Debian based distributions   "
                echo "such as Ubuntu.                                                  "
                echo "                                                                 "
                echo "Distributions with other package managers should use a different "
                echo "method of installing alternate Python versions.                  "
                echo "                                                                 "
                echo "For Example:                                                     "
                echo "                                                                 "
                echo "  * The official site (https://www.python.org/downloads/) has    "
                echo "    installers for Windows and MacOs                             "
                echo "                                                                 "
                echo "  * This article (https://realpython.com/installing-python/) has "
                echo "    instructions for Windows, MacOs. and Linux distributions     "
                echo "                                                                 "
                echo "  * This site (https://www.build-python-from-source.com/) has    "
                echo "    installation scripts for a variety of Linux distributions.   "
        } | text_frame_around '#' '#' 'WARNING'
        exit ${FALSE}
fi


# Get a temporary directory and make sure it's cleaned up when the script exits
cleanup(){
        echo "Cleaning up on exit..."

        echo ""
        unset_term_scroll_region

        if [ -n "${TEMP_WORKSPACE}" ] && [ -e "${TEMP_WORKSPACE}" ]; then
                echo "Removing ${TEMP_WORKSPACE} ..."
                rm -rf ${TEMP_WORKSPACE:?}
        fi
}
trap cleanup EXIT SIGTERM

# If there is an error, make sure we print the usage string with default parameter values
error_message(){
        echo "${USAGE}"
}
trap error_message ERR


# Define variables

PYTHON_VERSION_INSTALL_DIR="${PYTHON_INSTALL_DIRECTORY}/${PYTHON_VERSION_STR}"
VENV_PATH="${VENV_DIR}/${VENV_PREFIX}${PYTHON_VERSION_STR}"
PYTHON_VERSION_SOURCE_URL="https://www.python.org/ftp/python/${PYTHON_VERSION_STR}/Python-${PYTHON_VERSION_STR}.tgz"

PY_SRC_DIR="${PYTHON_SOURCE_DIR}/Python-${PYTHON_VERSION_STR}"
PY_SRC_FILE="${PY_SRC_DIR}.tgz"

# Create directories if required
mkdir -p "${PYTHON_INSTALL_DIR}"

# Perform actions
install_python | tee >(term_remove_ctrl_chars > ${LOG_FILE:-/dev/null})
create_virtual_environment | tee >(term_remove_ctrl_chars > ${LOG_FILE:-/dev/null})


```