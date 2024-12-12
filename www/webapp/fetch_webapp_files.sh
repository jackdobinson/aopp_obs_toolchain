#!/usr/bin/env bash

# Use strict mode
set -o errexit -o nounset -o pipefail

# Fetches the webapp files from the "../../emscripten_test" submodule

THIS_DIR=$(readlink -f $(dirname $0))
CWD=${PWD}

WEBAPP_DIR=${THIS_DIR}/../../emscripten_test/deconv_testing


###############################
# Copy files to this location #
###############################

# Copy main page
cp ${WEBAPP_DIR}/minimal.html ${THIS_DIR}/index.html

# Copy web assembly module
cp ${WEBAPP_DIR}/deconv.wasm ${THIS_DIR}/

# Copy CSS file
cp ${WEBAPP_DIR}/minimal.css ${THIS_DIR}/

# Copy all javascript to here
cp -r ${WEBAPP_DIR}/*.js ${WEBAPP_DIR}/js_modules ${THIS_DIR}/

