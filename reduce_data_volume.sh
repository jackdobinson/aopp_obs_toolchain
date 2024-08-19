#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail

# Data files to use
#./example_data/ifu_observation_datasets/ADP.2021-10-25T05\:14\:56.504_NFM-AO-N_OBJ.fits[3000:3300]
#./example_data/ifu_observation_datasets/MUSE.2021-10-25T06\:56\:42.763.fits[3000:3300]

# Example command
# `bash ./reduce_data_volume.sh ./example_data/ifu_observation_datasets/ADP.2021-10-25T05\:14\:56.504_NFM-AO-N_OBJ.fits ./example_data/ifu_observation_datasets/MUSE.2021-10-25T06\:56\:42.763.fits --slice [3000] --output_name single_wavelength_example --new_spatial_resolution none`

THIS_SCRIPT=$0
source "${THIS_SCRIPT%/*}/argparse.sh"

set_usage_info ${THIS_SCRIPT##*/} 'Reduces data volume of input datafiles'
add_argument SCI_FILE path "Science observation to reduce volume of"
add_argument STD_FILE path "Standard star observation for the science observation"
add_argument --slice str "Python slice syntax to apply to both files, note if only 1 element is present in an axis that axis will be removed from the final cube" '' '[3000:3300]'
add_argument --output_name str "Name used for reduced volume datafiles" '' 'reduced_data_volume'
add_argument --all_axes str "All axes to use for python routines" '' ''
add_argument --celestial_axes str "Celestial axes to use for python routines" '' ''
add_argument --new_spatial_resolution float "New resolution to use along spatial axes (use 'none' to disable), in the same units the input FITS file uses." '' '3E-5'
add_argument -h flag "Show usage message" print_usage_and_exit

# If something goes wrong then pring the usage message
trap "echo 'ERROR: Something went wrong'; print_usage_and_exit" EXIT

argparse "$@"

REDUCED_VOLUME_NAME="${ARGS['--output_name']}"

SLICE="${ARGS[--slice]}"
ALL_AXES="${ARGS[--all_axes]}"
CELESTIAL_AXES="${ARGS[--celestial_axes]}"
SPATIAL_RES="${ARGS[--new_spatial_resolution]}"
SQUEEZE_FLAG="${ARGS[--squeeze]-''}"


echo "SLICE=$SLICE"
echo "ALL_AXES=$ALL_AXES"
echo "CELESTIAL_AXES=$CELESTIAL_AXES"
echo "SPATIAL_RES=$SPATIAL_RES"


INPUT_FILES=(
	${ARGS['SCI_FILE']}
	${ARGS['STD_FILE']}
)

OUTPUT_FILE_CATEGORIES=(
	"sci"
	"std"
)

for IDX in "${!INPUT_FILES[@]}"; do
	FILE=${INPUT_FILES[${IDX}]}
	DIR=${FILE%/*}
	FPATH=${FILE%.*}
	EXT=${FILE##*.}
	
	OUTPUT_FILE="${DIR}/${REDUCED_VOLUME_NAME}_${OUTPUT_FILE_CATEGORIES[${IDX}]}.fits"
	
	TEMP1=${FPATH}_${REDUCED_VOLUME_NAME}_sliced.${EXT}
	TEMP2=${FPATH}_${REDUCED_VOLUME_NAME}_sliced_spatial.${EXT}
	TEMP_FINAL="TEMP_NOT_PRESENT"
	
	TEMP_FILES=(
		"${TEMP1}"
		"${TEMP2}"
	)
	
	echo "FPATH=$FPATH"
	echo "EXT=$EXT"
	echo "TEMP1=$TEMP1"
	echo "TEMP2=$TEMP2"
	
	CMD=(python -m aopp_deconv_tool.fits_slice "${FILE}${SLICE}${ALL_AXES}" -o "${TEMP1}")
	echo "Running command:"
	echo "    ${CMD[@]}"
	"${CMD[@]}"

	if [ "${SPATIAL_RES}" != "none" ]; then
		CMD=(python -m aopp_deconv_tool.spatial_rebin "${TEMP1}${CELESTIAL_AXES}" -o "${TEMP2}" --rebin_step "${SPATIAL_RES}" "${SPATIAL_RES}")
		echo "Running command:"
		echo "    ${CMD[@]}"
		"${CMD[@]}"
		TEMP_FINAL="TEMP2"
	else
		TEMP_FINAL="TEMP1"
	fi
		
	
	echo "Outputting file"
	echo "    category \"${OUTPUT_FILE_CATEGORIES[${IDX}]}\""
	echo "    path \"${OUTPUT_FILE}\""
	
	cp "${!TEMP_FINAL}" "${OUTPUT_FILE}"
	
	for TEMP_FILE in "${TEMP_FILES[@]}"; do 
		if [ -e "${TEMP_FILE}" ]; then
			rm "${TEMP_FILE}"
		fi
	done
	
done

# Remove trap for successful exit
trap - EXIT