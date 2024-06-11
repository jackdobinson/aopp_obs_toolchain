#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# NOTE: We just send the filenames, we rely on the defaults of the FITS Specifiers to handle extension and slice information for us

# Get observation and standard star as 1st and 2nd argument to this script
FITS_OBS=$1
FITS_STD=$2
# Get slices, spectral axes, celestial axes as arguments 3,4,5
SLICE=${3:+}
SPECTRAL_AXES=${4:+(0)}
CELESTIAL_AXES=${5:+(1,2)}

# Create output filenames for each step of the process that mirror the default output filenames

FITS_OBS_REBIN="${FITS_OBS%.fits}_rebin.fits"
FITS_STD_REBIN="${FITS_STD%.fits}_rebin.fits"

FITS_OBS_REBIN_ARTIFACT="${FITS_OBS%.fits}_rebin_artifactmap.fits"
FITS_STD_REBIN_ARTIFACT="${FITS_STD%.fits}_rebin_artifactmap.fits"

FITS_OBS_REBIN_ARTIFACT_BPMASK="${FITS_OBS%.fits}_rebin_artifactmap_bpmask.fits"
FITS_STD_REBIN_ARTIFACT_BPMASK="${FITS_STD%.fits}_rebin_artifactmap_bpmask.fits"

FITS_OBS_REBIN_INTERP="${FITS_OBS%.fits}_rebin_interp.fits"
FITS_STD_REBIN_INTERP="${FITS_STD%.fits}_rebin_interp.fits"

FITS_STD_REBIN_INTERP_NORM="${FITS_STD%.fits}_rebin_interp_normalised.fits"

FITS_STD_REBIN_INTERP_NORM_MODEL="${FITS_STD%.fits}_rebin_interp_normalised_modelled.fits"

FITS_OBS_REBIN_INTERP_DECONV="${FITS_OBS%.fits}_rebin_interp_deconv.fits"


# Perform each stage in turn

echo "Performin spectral rebinning"
python -m aopp_deconv_tool.spectral_rebin "${FITS_OBS}${SLICE}${SPECTRAL_AXES}"
python -m aopp_deconv_tool.spectral_rebin "${FITS_STD}${SLICE}${SPECTRAL_AXES}"

echo "Performing artifact detection"
python -m aopp_deconv_tool.artifact_detection "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}"
python -m aopp_deconv_tool.artifact_detection "${FITS_STD_REBIN}${SLICE}${CELESTIAL_AXES}"

echo "Creating bad pixel mask"
python -m aopp_deconv_tool.create_bad_pixel_mask "${FITS_OBS_REBIN_ARTIFACT}${SLICE}${CELESTIAL_AXES}"
python -m aopp_deconv_tool.create_bad_pixel_mask "${FITS_STD_REBIN_ARTIFACT}${SLICE}${CELESTIAL_AXES}"

echo "Interpolating at bad pixel mask"
python -m aopp_deconv_tool.interpolate "${FITS_OBS_REBIN}${SLICE}${CELESTIAL_AXES}" "${FITS_OBS_REBIN_ARTIFACT_BPMASK}${SLICE}${CELESTIAL_AXES}"
python -m aopp_deconv_tool.interpolate "${FITS_STD_REBIN}${SLICE}${CELESTIAL_AXES}" "${FITS_STD_REBIN_ARTIFACT_BPMASK}${SLICE}${CELESTIAL_AXES}"

echo "Normalising PSF"
python -m aopp_deconv_tool.psf_normalise "${FITS_STD_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}"

echo "Modelling PSF"
python -m aopp_deconv_tool.fit_psf_model "${FITS_STD_REBIN_INTERP_NORM}${SLICE}${CELESTIAL_AXES}"

echo "Performing deconvolution"
python -m aopp_deconv_tool.deconvolve "${FITS_OBS_REBIN_INTERP}${SLICE}${CELESTIAL_AXES}" "${FITS_STD_REBIN_INTERP_NORM_MODEL}${SLICE}${CELESTIAL_AXES}"

echo "Deconvolved file is ${FITS_OBS_REBIN_INTERP_DECONV}"