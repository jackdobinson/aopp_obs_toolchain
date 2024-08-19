#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

SCRIPT_DIR=${0%/*}
EXAMPLE_DIR="${SCRIPT_DIR}/../example_data/ifu_observation_datasets/"

SCI_FILE="${EXAMPLE_DIR}/single_wavelength_example_sci.fits"
STD_FILE="${EXAMPLE_DIR}/single_wavelength_example_std.fits"

echo "SCI_FILE=$SCI_FILE"
echo "STD_FILE=$STD_FILE"


STD_FILE_NORM="${STD_FILE%.*}_normalised.fits"
SCI_ARTEFACT_FILE="${SCI_FILE%.*}_artefactmap.fits"
SCI_ARTEFACT_MASK_FILE="${SCI_FILE%.*}_artefactmap_bpmask.fits"
SCI_INTERP_FILE="${SCI_FILE%.*}_interp.fits"
DECONV_FILE="${SCI_FILE%.*}_deconv.fits"

# First we must normalise the PSF, must be centered and sum to 1
python -m aopp_deconv_tool.psf_normalise ${STD_FILE} -o ${STD_FILE_NORM}


python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM}
# Initial deconvolution: Result looks nice, but there is still lots of signal in the residual. Looking at the header,
# the iteration stopped due to a conservative cutoff. We can increase that.


python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} --min_frac_stat_delta 1E-5
# Cutoff is reduced, and now runs to full 1000 iterations, but still signal in the residual. We can increase the
# number of iterations.


python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} --min_frac_stat_delta 1E-5 --n_iter 5000
# Iterations increased. Oh no! The result looks really bad. This is because
# there are some artifacts in the observation that the deconvolution algorithm does not cope with well
# the artifacts are smaller than the PSF, so Modified CLEAN cannot account for them
#
# We need to process the science observaton to remove the artefacts. We can do this manually in this case
# but for large amounts of images and IFU images that have multiple wavelengths per FITS file this would
# be a labourious process. Therefore we have an automated tool. The purpose of the tool is to "fix"
# artefacts enough that they will no longer affect the deconvolution process, not to completely remove them.
#
# There are three steps to removing artefacts. 1) Identification. 2) Masking. 3) Interpolating.

python -m aopp_deconv_tool.artefact_detection ${SCI_FILE} -o ${SCI_ARTEFACT_FILE}
# A file is created that can be thought of as a "heat map" of pixels that are likely to be artefacts and 
# cause problems during deconvolution. We now take this heat map and select all the pixels above a certain
# cutoff to create a "bad pixel" mask.

python -m aopp_deconv_tool.create_bad_pixel_mask ${SCI_ARTEFACT_FILE} -o ${SCI_ARTEFACT_MASK_FILE}
# Now we have a file that tells us which pixels we want to "fix". The next step is to interpolate over
# these pixels so we have an image that can be deconvolved successfully

python -m aopp_deconv_tool.interpolate ${SCI_FILE} ${SCI_ARTEFACT_MASK_FILE} -o ${SCI_INTERP_FILE}
# The resulting file should be very similar to the original file, but with artefacts removed (or at least reduced).
# We can now use this new file in place of the old one when deconvolving.
# Instead of re-running the previous command, which took a while, we will introduce another option.
# The `--loop_gain` option sets how aggressive Modified CLEAN is each iteration. The default value is a conservative
# setting at 0.02. We can increase this (but not too much) to speed things up. Setting it to 0.2 will speed thins up 
# so we only need about 200 iterations.

python -m aopp_deconv_tool.deconvolve ${SCI_INTERP_FILE} ${STD_FILE_NORM} --min_frac_stat_delta 1E-5 --loop_gain 0.2 --n_iter 200
# Something is still wrong. We don't have trouble with the artefacts any more, and nearly all of the signal is in
# the components not the residual, but the deconvolved image is all speckly! This is a problem with the threshold
# value of Modified CLEAN. When the set of spread of pixel values chosen by the threshold is of a similar magnitude
# to the noise of the image, the noise tends to reinforce itself and shows up as regular specking. There is a
# "smart" threshold setting that aims to reduce the specking problem by heuristically choosing a threshold. This mode
# is enabled when the `--threshold` option is a negative number. Automatic threshold selection is slower than
# a constant value.

python -m aopp_deconv_tool.deconvolve ${SCI_INTERP_FILE} ${STD_FILE_NORM} --min_frac_stat_delta 1E-5 --loop_gain 0.2 --n_iter 200 --threshold -1
# The end result isn't too bad. You can tell that almost all of the signal is now in the deconvolved image and not
# the residual. There is still a bit of speckling, but it's much better than it was. There are even fairly obvious
# artefacts present (which didn't break the deconvolution process) due to the optical setup of the telescope used
# to create this image.