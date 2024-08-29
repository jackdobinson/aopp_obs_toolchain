#!/usr/bin/env bash
#
# Convert jupyter notebook to HTML
#

# Alter styling with file "~/.jupyter/custom/custom.css"

IPYNB_INPUT="$1"
HTML_OUTPUT="${IPYNB_INPUT%.ipynb}.html"

echo "Processing: ${IPYNB_INPUT}"
echo "Producing: ${HTML_OUTPUT}."
if [ ${IPYNB_INPUT} -nt ${HTML_OUTPUT} ]; then
	echo "Process file is newer than product file."
else
	echo "Process file is older than product file."
	echo "Not rebuilding product."
	echo "Exiting..."
	exit 0
fi

echo "Rebuilding product..."



jupyter-nbconvert $1 --to html --template lab
