#!/usr/bin/env bash

# Use strict mode
set -o errexit -o nounset -o pipefail

shopt -s nullglob

THIS_DIR=$(readlink -f $(dirname $0))
CWD=${PWD}

echo "THIS_DIR=${THIS_DIR}"
echo "CWD=${CWD}"


EXAMPLE_FILE_SET=(
	${THIS_DIR}/example_*/bash/example.sh
	${THIS_DIR}/example_*/jupyter/example.ipynb
)

for FILE in "${EXAMPLE_FILE_SET[@]}"; do
	cd ${CWD}
	
	echo "FILE=${FILE}"
	
	DIR=${FILE%/*}
	NAME=${FILE##*/}

	case "${FILE}" in
		*.sh)
			CMD=(
				${THIS_DIR}/bash_example_to_html.sh 
				${FILE}
			)
			;;
		*.ipynb)
			CMD=(
				${THIS_DIR}/ipynb_to_html.sh
				${FILE}
			)
			;;
		*)
			echo "ERROR: File '${FILE}' has unknown file type '${FILE##*.}', cannot convert to HTML"
			;;
	esac
	
	cd ${DIR}
	${CMD[@]}
	
	# newlines to separate output
	echo "" 
	echo ""

done