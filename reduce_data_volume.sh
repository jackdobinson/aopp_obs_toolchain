#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Data files to use
#./example_data/ifu_observation_datasets/ADP.2021-10-25T05\:14\:56.504_NFM-AO-N_OBJ.fits
#./example_data/ifu_observation_datasets/MUSE.2021-10-25T06\:56\:42.763.fits

THIS_SCRIPT=$0

REDUCED_VOLUME_NAME="reduced_data_volume"

COUNT=0
for FILE in ${@:1}; do
	DIR=${FILE%/*}
	FPATH=${FILE%.*}
	EXT=${FILE##*.}
	
	RV1=${FPATH}_${REDUCED_VOLUME_NAME}_spatial.${EXT}
	RV2=${FPATH}_${REDUCED_VOLUME_NAME}_spatial_spectral.${EXT}
	
	#RV1=${FPATH}_${REDUCED_VOLUME_NAME}_spectral.${EXT}
	#RV2=${FPATH}_${REDUCED_VOLUME_NAME}_spectral_spatial.${EXT}
	
	echo "COUNT=$COUNT"
	echo "FPATH=$FPATH"
	echo "EXT=$EXT"
	echo "RV1=$RV1"
	echo "RV2=$RV2"
	
	
	python -m aopp_deconv_tool.spatial_rebin ${FILE} -o ${RV1} --rebin_step 3E-5 3E-5
	python -m aopp_deconv_tool.spectral_rebin ${RV1} -o ${RV2} --rebin_params 1E-8 2E-8
	
	#python -m aopp_deconv_tool.spectral_rebin ${FILE} -o ${RV1} --rebin_params 1E-8 2E-8
	#python -m aopp_deconv_tool.spatial_rebin ${RV1} -o ${RV2} --rebin_step 3E-5 3E-5
	
	# ONLY WHEN PASSING A SCI AND PSF OBSERVATION
	if [[ $COUNT < 2 ]]; then
		if [[ $COUNT == 0 ]]; then
			echo "SCI"
			mv "${RV2}" "${DIR}/${REDUCED_VOLUME_NAME}_1_sci.fits"
			rm "${RV1}"
		fi
		if [[ $COUNT == 1 ]]; then
			echo "STD"
			mv "${RV2}" "${DIR}/${REDUCED_VOLUME_NAME}_1_std.fits"
			rm "${RV1}"
		fi
	fi
	
	
	COUNT=$((COUNT + 1))
done