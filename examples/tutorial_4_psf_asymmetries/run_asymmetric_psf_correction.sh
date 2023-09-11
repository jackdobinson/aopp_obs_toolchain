#!/bin/bash

echo -e "This script does not seem to work, please use \"./run_asymmetric_psf_correction.py\" instead."
exit

# "asymmetric_psf_correction.py" is actuall quite well set up to run from the
# command line so that's what I'll show.

# Get the parent folder of this script
FILE_DIR=$(dirname $(realpath $0))

# Get the folder that contains the scripts common to the tutorial files
SCRIPTS_DIR=$(realpath "${FILE_DIR}/../scripts")

# Get the path to the data files
OBS_FILE=$(realpath "${FILE_DIR}/../data/test_rebin.fits")
PSF_FILE=$(realpath "${FILE_DIR}/../data/test_standard_star.fits")

# Get the file to output results to
OUTPUT_FILE=$(realpath "${FILE_DIR}/output/asym_corrected.fits")

# Get the location of the asymmetric_psf_correction.py script
SCRIPT=$(realpath -s "${FILE_DIR}/../scripts/fitscube/deconvolve/asymmetric_psf_correction.py")

# Build command up as a string
CMD="${SCRIPT} ${OBS_FILE} --ext_obs 1 --file_psf ${PSF_FILE} --ext_psf 1 --file_out ${OUTPUT_FILE} --window_shape 51 --n_subdivisions 64"

# Prepend folder that contains tutorial scripts to PYTHONPATH, that way
# any name collisions will be resolved from the tutorial scripts first.
PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH}"

echo -e "# INFORMATION #"
echo -e "Setting up variables:"
echo -e "\tFILE_DIR    = ${FILE_DIR}"
echo -e "\tSCRIPTS_DIR = ${SCRIPTS_DIR}"
echo -e "\tOBS_FILE    = ${OBS_FILE}"
echo -e "\tPSF_FILE    = ${PSF_FILE}"
echo -e "\tSCRIPT      = ${SCRIPT}"
echo -e "\tOUTPUT_FILE = ${OUTPUT_FILE}"
echo -e "Prepending tutorial scripts folder to PYTHONPATH:"
echo -e "\tPYTHONPATH  = ${PYTHONPATH}"
echo -e "Command to run:"
echo -e "\tCMD         = ${CMD}"

#exit # DEBUGGING
echo -e "# RUNNING COMMAND #"

# Technically I should not do this as exec is really unsafe, but if you can execute
# random python scripts on a machine, you can do horrible things already and the
# "correct" way to do things is very long-winded.
eval ${CMD}


