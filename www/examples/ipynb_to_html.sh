#!/usr/bin/env bash
#
# Convert jupyter notebook to HTML
#

# Alter styling with file "~/.jupyter/custom/custom.css"

jupyter-nbconvert $1 --to html --template lab
