#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Data files to use
#./example_data/ifu_observation_datasets/ADP.2021-10-25T05\:14\:56.504_NFM-AO-N_OBJ.fits[3000:3300]
#./example_data/ifu_observation_datasets/MUSE.2021-10-25T06\:56\:42.763.fits[3000:3300]

THIS_SCRIPT=$0
source "${THIS_SCRIPT%/*}/argparse.sh"

set_usage_info ${THIS_SCRIPT##*/} 'Reduces data volume of input datafiles'
add_argument SCI_FILE path "Science observation to reduce volume of"
add_argument STD_FILE path "Standard star observation for the science observation"
add_argument --slice str "Python slice synatx to apply to both files" '' '[3000:3300]'
add_argument --output_name str "Name used for reduced volume datafiles" '' 'reduced_data_volume'
add_argument --all_axes str "All axes to use for python routines" '' ''
add_argument --celestial_axes str "Celestial axes to use for python routines" '' ''
add_argument -h flag "Show usage message" print_usage_and_exit

# If something goes wrong then pring the usage message
trap "echo 'ERROR: Something went wrong'; print_usage_and_exit" EXIT

argparse "$@"

REDUCED_VOLUME_NAME="${ARGS['--output_name']}"

SLICE="${ARGS[--slice]}"
ALL_AXES="${ARGS[--all_axes]}"
CELESTIAL_AXES="${ARGS[--celestial_axes]}"



echo "SLICE=$SLICE"
echo "ALL_AXES=$ALL_AXES"
echo "CELESTIAL_AXES=$CELESTIAL_AXES"


COUNT=0
FILES=(
	${ARGS['SCI_FILE']}
	${ARGS['STD_FILE']}
)
for FILE in ${FILES[@]}; do
	DIR=${FILE%/*}
	FPATH=${FILE%.*}
	EXT=${FILE##*.}
	
	RV1=${FPATH}_${REDUCED_VOLUME_NAME}_sliced.${EXT}
	RV2=${FPATH}_${REDUCED_VOLUME_NAME}_sliced_spatial.${EXT}
	
	#RV1=${FPATH}_${REDUCED_VOLUME_NAME}_spatial.${EXT}
	#RV2=${FPATH}_${REDUCED_VOLUME_NAME}_spatial_spectral.${EXT}
	
	#RV1=${FPATH}_${REDUCED_VOLUME_NAME}_spectral.${EXT}
	#RV2=${FPATH}_${REDUCED_VOLUME_NAME}_spectral_spatial.${EXT}
	
	echo "COUNT=$COUNT"
	echo "FPATH=$FPATH"
	echo "EXT=$EXT"
	echo "RV1=$RV1"
	echo "RV2=$RV2"
	
	python -m aopp_deconv_tool.fits_slice "${FILE}${SLICE}${ALL_AXES}" -o ${RV1}
	python -m aopp_deconv_tool.spatial_rebin "${RV1}${CELESTIAL_AXES}" -o ${RV2} --rebin_step 3E-5 3E-5
	
	#python -m aopp_deconv_tool.spatial_rebin ${FILE} -o ${RV1} --rebin_step 3E-5 3E-5
	#python -m aopp_deconv_tool.spectral_rebin ${RV1} -o ${RV2} --rebin_params 1E-8 2E-8
	
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

# Remove trap for successful exit
trap - EXIT