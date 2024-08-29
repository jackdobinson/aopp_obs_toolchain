#!/usr/bin/env bash
#
# Convert jupyter notebook to HTML
#

# Alter styling with file "~/.jupyter/custom/custom.css"

IPYNB_INPUT="$1"
HTML_OUTPUT="${IPYNB_INPUT%.ipynb}.html"


echo "Processing ${IPYNB_INPUT} will produce ${HTML_OUTPUT}."
if [ ${IPYNB_INPUT} -nt ${HTML_OUTPUT} ]; then
	echo "${IPYNB_INPUT} is newer than ${HTML_OUTPUT}. Rebuilding dependent file..."
else 
	echo "${IPYNB_INPUT} is older than ${HTML_OUTPUT}. Dependent file should be newest version, no rebuilding will be performed."
	exit 0
fi


jupyter-nbconvert $1 --to html --template lab
