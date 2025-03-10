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
  NOTE: Enter an empty string ('') to use default values of an option
  
  -h
    Display this help message
  -r
    Recalculate all products
  slice : str
    Python-style slice notation that will be applied to obs_fits and std_fits 
    data, often used to focus on specific spectral slice of data.
  spectral_axes : str
    Axis number of spectral axis, enclosed in brackets e.g., '(0)'. Will be 
    automatically calculated if not present.
  celestial_axes : str
    Axis numbers of celestial axes, enclosed in brakcets e.g., '(1,2)'. Will be
    automatically calculated if not present.
  spectral_rebin_slice : str
    Python-style slice notation that will be applied to obs_fits and std_fits 
    data, when spectrally rebinning data.
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
N_OPTIONAL_POS_ARGS=4
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
SPECTRAL_REBIN_SLICE=${POS_ARGS[5]:-'[:]'}



# Output argument values for user information
echo "FITS_OBS=${FITS_OBS}"
echo "FITS_STD=${FITS_STD}"
echo "SLICE=${SLICE}"
echo "SPECTRAL_AXES=${SPECTRAL_AXES}"
echo "CELESTIAL_AXES=${CELESTIAL_AXES}"
echo "SPECTRAL_REBIN_SLICE=${SPECTRAL_REBIN_SLICE}"
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
# TODO: The slice here breaks slicing further on as the shape of the cubes change. Use a different slicing argument here.
if [[ ${RECALC} == 1 || ! -f ${FITS_OBS_REBIN} ]]; then
	python -m aopp_deconv_tool.spectral_rebin "${FITS_OBS}${SPECTRAL_REBIN_SLICE}${SPECTRAL_AXES}"
fi
if [[ ${RECALC} == 1 || ! -f ${FITS_STD_REBIN} ]]; then
	python -m aopp_deconv_tool.spectral_rebin "${FITS_STD}${SPECTRAL_REBIN_SLICE}${SPECTRAL_AXES}"
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
	python -m aopp_deconv_tool.deconvolve                       \
		"${FITS_OBS_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}"     \
		"${FITS_STD_REBIN_NORM_MODEL}${SLICE}${CELESTIAL_AXES}" \
		--threshold -1                                          \
		--progress 10
fi

echo ""
echo "## PROCESS COMPLETE ##"
echo ""
echo "Deconvolved file is ${FITS_OBS_REBIN_INTERP_DECONV}"
echo ""
echo "The deconvolution was generic, so it may not be the best result."
echo "To fine tune the deconvolution, use the following command to display"
echo "plots of the deconvolution in progress to more easily tweak variables."
echo ""
{
echo "python -m aopp_deconv_tool.deconvolve                   \\"
echo "  ${FITS_OBS_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}     \\"
echo "  ${FITS_STD_REBIN_NORM_MODEL}${SLICE}${CELESTIAL_AXES} \\" 
echo "  --threshold -1                                        \\"
echo "  --plot                                                \\"
echo "  --progress 10                                           "
echo ""
echo "######################"
} | sed -E 's/([][():])/\\\1/g' # escape the characters the shell has a problem with.


DISPLAY_ALL_FILES=1

# Open all products in the first viewer available
for FITS_VIEWER in ${FITS_VIEWERS[@]}; do
	if command -v ${FITS_VIEWER} &> /dev/null; then
	
		# Ask user if we should display files
		while true; do
			read -p "Display all files in ${FITS_VIEWER}? (Y/n)" REPLY
			if [ -n "${REPLY}" ]; then
				case "$REPLY" in
					[Yy]* )
						DISPLAY_ALL_FILES=1
						break
						;;
					[Nn]* )
						DISPLAY_ALL_FILES=0
						break
						;;
					* ) 
						echo "Unrecognised response '${REPLY}'. Please answer 'yes' or 'no' (default is 'yes')."
						;;
				esac
			else
				# If no input given, assume answer is yes
				break
			fi
		done
		
		# Communicate FITS viewer being used and files being displayed.
		if [ ${DISPLAY_ALL_FILES} -eq 1 ]; then
			echo "Fits viewer '${FITS_VIEWER}' displaying ${#ALL_FITS_FILES[@]} files:"
			for INDEX in "${!ALL_FITS_FILES[@]}"; do
				printf "  %3d) %s\n"  $((INDEX+1))  ${ALL_FITS_FILES[$INDEX]}
			done
			${FITS_VIEWER} ${ALL_FITS_FILES[@]} &
		fi
		break
	fi
done