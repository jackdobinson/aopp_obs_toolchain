#!/usr/bin/env bash

# Use strict mode
set -o errexit -o nounset -o pipefail

# Fetches the webapp files from the "../../emscripten_test" submodule

THIS_DIR=$(readlink -f $(dirname $0))
CWD=${PWD}

WEBAPP_DIR=${THIS_DIR}/../../emscripten_test/deconv_testing
echo "Fetching webapp files from ${WEBAPP_DIR} copying to ${THIS_DIR}"

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


###############################################################
# Update parts of HTML file to overwrite placeholder elements #
###############################################################

# Add link back to home page

find_1='<a id="link-to-home"></a>'
replace_1='<nav class="top-menu"><a id="link-to-home" href="https://jackdobinson.github.io/aopp_obs_toolchain/index.html">Home</a></nav><h1>Deconv Tool Web Application</h1>'

sed -i -e "s#${find_1}#${replace_1}#g" ${THIS_DIR}/index.html


# Add attributations

find_1='<div id="attributation"></div>'
replace_1=$(cat << "END"
<footer>
	<p class="title">Attributation</p>
	<div id="attributation">
		<figure id="university_of_oxford">
			<img src="https://jackdobinson.github.io/aopp_obs_toolchain/assets/imgs/university_of_oxford.png" alt="University of oxford logo" />
			<figcaption>Thanks to the University of Oxford for their continued support.</figcaption>
		</figure>
		
		<figure id="leverhulme_trust">
			<img src="https://jackdobinson.github.io/aopp_obs_toolchain/assets/imgs/Leverhulme_Trust_RGB_white.png" alt="Leverhulme Trust Logo"/>
			<figcaption>This work was made possible by the generous funding of the Leverhulme Trust for project RPG-2023-028</figcaption>
		</figure>
	</div>
</footer>
END
)


#echo "find_1=${find_1}"
#echo "replace_1=${replace_1}"

find_escaped=$(printf "%q" "${find_1}") # for some reason this fails when finding the thing to replace
replace_escaped=$(printf "%q" "${replace_1}") # for some reason this adds some extra characters we have to remove

#echo "find_escaped=${find_escaped}"
#echo "replace_escaped=${replace_escaped:2:-1}"
#echo "s@${find_1}@${replace_1}@g"

sed -i -e "s@${find_1}@${replace_escaped:2:-1}@g" ${THIS_DIR}/index.html

# Update stylesheet
find_1='<link rel="stylesheet" href="minimal.css" />'
replace_1='<link rel="stylesheet" href="https://jackdobinson.github.io/aopp_obs_toolchain/assets/css/style.css" /><link rel="stylesheet" href="minimal.css" />'

sed -i -e "s@${find_1}@${replace_1}@g" ${THIS_DIR}/index.html

#####################################################################
# Update JAVASCRIPT that is different between repos for some reason #
#####################################################################

echo "Patching javascript that behaves differently between repos"

find='(this.body_rect.top < 0 ? this.body_rect.top : 0)'
replace='this.body_rect.top'
sed -i -e "s@${find}@${replace}@g" ${THIS_DIR}/status_kv.js


find='(this.body_rect.left < 0 ? this.body_rect.left : 0)'
replace='this.body_rect.left'
sed -i -e "s@${find}@${replace}@g" ${THIS_DIR}/status_kv.js
