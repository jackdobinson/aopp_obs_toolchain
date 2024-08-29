#!/usr/bin/env bash

# Use strict mode
set -o errexit -o nounset -o pipefail

shopt -s nullglob

THIS_DIR=$(readlink -f $(dirname $0))
CWD=${PWD}

EXAMPLE_FILE_SET=(
	${THIS_DIR}/example_*/bash/example.sh
	${THIS_DIR}/example_*/jupyter/example.ipynb
)


echo "Looping over fileset..."
for FILE in "${EXAMPLE_FILE_SET[@]}"; do
	echo "--------------------------------------------------------"
	echo "FILE=${FILE}"
	
	# Make sure we always start from the same directory
	cd ${CWD}
	
	DIR=${FILE%/*}
	NAME=${FILE##*/}

	case "${FILE}" in
		*.sh)
			echo "FILE detected as BASH file."
			CMD=(
				${THIS_DIR}/bash_example_to_html.sh 
				${FILE}
			)
			;;
		*.ipynb)
			echo "FILE detected as JYPYTER-NOTEBOOK file."
			CMD=(
				${THIS_DIR}/ipynb_to_html.sh
				${FILE}
			)
			;;
		*)
			echo "ERROR: File '${FILE}' has unknown file type '${FILE##*.}', cannot convert to HTML"
			continue
			;;
	esac
	
	# Change to the same directory as the file.
	cd ${DIR}
	# Run the command
	${CMD[@]}

done
echo "--------------------------------------------------------"
echo "fileset loop complete."