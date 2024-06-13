#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# START Parse Arguments

# NOTE: We just send the filenames, we rely on the defaults of the FITS Specifiers to handle extension and slice information for us

# Get observation and standard star as 1st and 2nd argument to this script
FITS_OBS=${1}
FITS_STD=${2}
# Get slices, spectral axes, celestial axes as arguments 3,4,5
SLICE=${3:-'[:]'}
SPECTRAL_AXES=${4:-'(0)'}
CELESTIAL_AXES=${5:-'(1,2)'}

# Output argument values for user information
echo "FITS_OBS=${FITS_OBS}"
echo "FITS_STD=${FITS_STD}"
echo "SLICE=${SLICE}"
echo "SPECTRAL_AXES=${SPECTRAL_AXES}"
echo "CELESTIAL_AXES=${CELESTIAL_AXES}"
# END Parse Arguments

# Set parameter constants
PSF_MODEL_STR="radial" # "radial" is the current default, it influences the name of the one of the output files
FITS_VIEWERS=("QFitsView" "ds9")

# Create output filenames for each step of the process that mirror the default output filenames

FITS_OBS_REBIN="${FITS_OBS%.fits}_rebin.fits"
FITS_STD_REBIN="${FITS_STD%.fits}_rebin.fits"

FITS_OBS_REBIN_ARTIFACT="${FITS_OBS%.fits}_rebin_artifactmap.fits"

FITS_OBS_REBIN_ARTIFACT_BPMASK="${FITS_OBS%.fits}_rebin_artifactmap_bpmask.fits"

FITS_OBS_REBIN_INTERP="${FITS_OBS%.fits}_rebin_interp.fits"

FITS_STD_REBIN_NORM="${FITS_STD%.fits}_rebin_normalised.fits"

FITS_STD_REBIN_NORM_MODEL="${FITS_STD%.fits}_rebin_normalised_modelled_${PSF_MODEL_STR}.fits"

FITS_OBS_REBIN_INTERP_DECONV="${FITS_OBS%.fits}_rebin_interp_deconv.fits"

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
python -m aopp_deconv_tool.spectral_rebin "${FITS_OBS}${SLICE}${SPECTRAL_AXES}"
python -m aopp_deconv_tool.spectral_rebin "${FITS_STD}${SLICE}${SPECTRAL_AXES}"

echo "Performing artifact detection"
python -m aopp_deconv_tool.artifact_detection "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}"

echo "Creating bad pixel mask"
python -m aopp_deconv_tool.create_bad_pixel_mask "${FITS_OBS_REBIN_ARTIFACT}${SLICE}${CELESTIAL_AXES}"

echo "Interpolating at bad pixel mask"
python -m aopp_deconv_tool.interpolate "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}" "${FITS_OBS_REBIN_ARTIFACT_BPMASK}${SLICE}${CELESTIAL_AXES}"

echo "Normalising PSF"
python -m aopp_deconv_tool.psf_normalise "${FITS_STD_REBIN}${SLICE}${CELESTIAL_AXES}"

echo "Modelling PSF"
python -m aopp_deconv_tool.fit_psf_model "${FITS_STD_REBIN_NORM}${SLICE}${CELESTIAL_AXES}" --model ${PSF_MODEL_STR}

echo "Performing deconvolution"
python -m aopp_deconv_tool.deconvolve "${FITS_OBS_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}" "${FITS_STD_REBIN_NORM_MODEL}${SLICE}${CELESTIAL_AXES}"

echo "Deconvolved file is ${FITS_OBS_REBIN_INTERP_DECONV}"

# Open all products in the first viewer available
for FITS_VIEWER in ${FITS_VIEWERS[@]}; do
	if command -v ${FITS_VIEWER} &> /dev/null; then
		${FITS_VIEWER} ${ALL_FITS_FILES[@]} &
		break
	fi
done