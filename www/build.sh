#!/usr/bin/env bash

# Use strict mode
set -o errexit -o nounset -o pipefail

# Builds all parts of the website that come from other places

THIS_DIR=$(readlink -f $(dirname $0))
CWD=${PWD}

bash ${THIS_DIR}/examples/all_examples_to_html.sh
bash ${THIS_DIR}/webapp/fetch_webapp_files.sh


