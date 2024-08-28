#!/usr/bin/env bash
#:HIDE
BUILD_LOG_FILE="./build_log.txt"; echo "" > ${BUILD_LOG_FILE}

#:begin{CELL}

# Use strict mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

SCRIPT="$(realpath -e ${0})"
SCRIPT_DIR=${SCRIPT%/*}
EXAMPLE_DIR=$(realpath -e "${SCRIPT_DIR}/../../../../example_data/ifu_observation_datasets/")

echo "EXAMPLE_DIR=${EXAMPLE_DIR}"

SCI_FILE="${EXAMPLE_DIR}/single_wavelength_example_sci.fits"
STD_FILE="${EXAMPLE_DIR}/single_wavelength_example_std.fits"

STD_FILE_NORM="${STD_FILE%.*}_normalised.fits"
SCI_ARTEFACT_FILE="${SCI_FILE%.*}_artefactmap.fits"
SCI_ARTEFACT_MASK_FILE="${SCI_FILE%.*}_artefactmap_bpmask.fits"
SCI_INTERP_FILE="${SCI_FILE%.*}_interp.fits"
DECONV_FILE="${SCI_FILE%.*}_deconv.fits"

#:end{CELL}

#:begin{HIDE}

screenshot_process(){
	# NOTE: Can use a window title instead of a PID as we are just using `grep` to find the window id.
	set +o nounset
	if [ -z "$2" ]; then
		local PID="$!"
	else
		local PID=$2
	fi
	set -o nounset
	#V=$(wmctrl -l -p | grep "${PID}")
	#echo "V=$V"
	local TMP_LAST_PID_WINDOW_ID=$(wmctrl -l -p | grep "${PID}" | sed 's/ .*//')
	if [ -z "${TMP_LAST_PID_WINDOW_ID}" ]; then
		echo "ERROR: Cannot find window associated with process ${PID}."
		return 1
	fi
	import -window ${TMP_LAST_PID_WINDOW_ID} $1
}
{
	mkdir -p ./figures
	mkdir -p ./logs

	# Start DS9 instance and get it's XPA server
	# will use the instance throughout the script
	ds9 &
	set +o errexit
	xpaget xpans
	while [[ "$?" != "0" ]]; do
		sleep 5
		xpaget xpans
	done
set -o errexit 
} &>> ${BUILD_LOG_FILE}
#:end{HIDE}


#:begin{CELL}

#:HIDE
{ xpaset -p ds9 fits ${SCI_FILE}; xpaset -p ds9 export png ./figures/sci-file.png;} &>> ${BUILD_LOG_FILE}
#:HIDE
{ xpaset -p ds9 fits ${STD_FILE}; xpaset -p ds9 export png ./figures/std-file.png;} &>> ${BUILD_LOG_FILE}

#:end{CELL}



#:begin{CELL}

#:begin{HIDE}
{
	xpaset -p ds9 fits ${STD_FILE}
	xpaset -p ds9 scale log
	xpaset -p ds9 crosshair 200 200 physical
	screenshot_process ./figures/std-file-pixel.png "SAOImage"
	xpaset -p ds9 header save ./figures/std-file-header.txt 
}&>> ${BUILD_LOG_FILE}
#:end{HIDE}

#:end{CELL}


#:begin{CELL}
python -m aopp_deconv_tool.psf_normalise ${STD_FILE} -o ${STD_FILE_NORM} &> ./logs/psf_normalise_log.txt


#:begin{HIDE}
{
	xpaset -p ds9 fits ${STD_FILE_NORM}
	xpaset -p ds9 scale log
	xpaset -p ds9 crosshair 200 200 physical
	screenshot_process ./figures/std-norm-file-pixel.png "SAOImage"
	xpaset -p ds9 header save ./figures/std-norm-file-header.txt 
} &>> ${BUILD_LOG_FILE}
#:end{HIDE}

#:end{CELL}



#:begin{CELL}
python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} -o ${DECONV_FILE} &> ./logs/deconv-1-log.txt
#:end{CELL}


#:begin{CELL}

#:begin{HIDE}
{
	xpaset -p ds9 scale linear 
	xpaset -p ds9 scale limits 0 2.5E4
	xpaset -p ds9 fits ${SCI_FILE}
	xpaset -p ds9 export png ./figures/sci-file-ds9.png
	xpaset -p ds9 fits ${DECONV_FILE}
	xpaset -p ds9 scale linear 
	xpaset -p ds9 scale mode minmax
	xpaset -p ds9 export png ./figures/deconv-primary-1.png
	xpaset -p ds9 fits ${DECONV_FILE}[RESIDUAL]
	xpaset -p ds9 scale linear 
	xpaset -p ds9 scale mode minmax
	xpaset -p ds9 export png ./figures/deconv-residual-1.png
} &>> ${BUILD_LOG_FILE}

#:end{HIDE}

python -c 'import sys; import numpy as np; from astropy.io import fits; original=fits.getdata(sys.argv[1]); deconv=fits.getdata(sys.argv[2]); residual=fits.getdata(sys.argv[2],ext=1); print(f"signal fraction in deconvolved components {np.nansum(deconv)/np.nansum(original)}"); print(f"signal fraction in residual {np.nansum(residual)/np.nansum(original)}")' ${SCI_FILE} ${DECONV_FILE}
#:end{CELL}


#:begin{CELL}

#:HIDE
{ xpaset -p ds9 fits ${DECONV_FILE}; xpaset -p ds9 header save ./figures/deconv-file-header-1.txt; } &>> ${BUILD_LOG_FILE}

grep -E 'PKEY*|PVAL*|CONTINUE' ./figures/deconv-file-header-1.txt
#:end{CELL}


#:begin{CELL}
python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} -o ${DECONV_FILE} --min_frac_stat_delta 1E-5 &> ./logs/deconv-2-log.txt
#:end{CELL}


#:begin{CELL}

#:begin{HIDE}
{
	xpaset -p ds9 scale linear 
	xpaset -p ds9 scale limits 0 2.5E4
	xpaset -p ds9 fits ${DECONV_FILE}
	xpaset -p ds9 export png ./figures/deconv-primary-2.png
	xpaset -p ds9 fits ${DECONV_FILE}[RESIDUAL]
	xpaset -p ds9 export png ./figures/deconv-residual-2.png
	xpaset -p ds9 header save ./figures/deconv-file-header-2.txt
	xpaset -p ds9 scale mode minmax
} &>> ${BUILD_LOG_FILE}

#:end{HIDE}

grep -E 'PKEY*|PVAL*|CONTINUE' ./figures/deconv-file-header-2.txt

echo "" # Empty line for formatting
python -c 'import sys; import numpy as np; from astropy.io import fits; original=fits.getdata(sys.argv[1]); deconv=fits.getdata(sys.argv[2]); residual=fits.getdata(sys.argv[2],ext=1); print(f"signal fraction in deconvolved components {np.nansum(deconv)/np.nansum(original)}"); print(f"signal fraction in residual {np.nansum(residual)/np.nansum(original)}")' ${SCI_FILE} ${DECONV_FILE}
#:end{CELL}


#:begin{CELL}
python -m aopp_deconv_tool.artefact_detection $SCI_FILE -o $SCI_ARTEFACT_FILE &> ./logs/sci-artefact-1-log.txt
python -m aopp_deconv_tool.create_bad_pixel_mask $SCI_ARTEFACT_FILE -o $SCI_ARTEFACT_MASK_FILE &> ./logs/sci-artefact-mask-1-log.txt
python -m aopp_deconv_tool.interpolate $SCI_FILE $SCI_ARTEFACT_MASK_FILE -o $SCI_INTERP_FILE &> ./logs/sci-interp-1-log.txt


#:begin{HIDE}
{
	xpaset -p ds9 fits ${SCI_ARTEFACT_FILE}
	xpaset -p ds9 export png ./figures/sci-artefact-file-1.png
	xpaset -p ds9 fits ${SCI_ARTEFACT_MASK_FILE}
	xpaset -p ds9 export png ./figures/sci-artefact-mask-file-1.png
	xpaset -p ds9 fits ${SCI_INTERP_FILE}
	xpaset -p ds9 export png ./figures/sci-interp-file-1.png
} &>> ${BUILD_LOG_FILE}
#:end{HIDE}
#:end{CELL}




#:begin{CELL}
python -m aopp_deconv_tool.deconvolve $SCI_INTERP_FILE $STD_FILE -o $DECONV_FILE --min_frac_stat_delta 1E-5 --loop_gain 0.2 --n_iter 200 --threshold -1 --rms_frac_threshold 1E-3 &> ./logs/deconv-3-log.txt

#:begin{HIDE}
{
	xpaset -p ds9 scale linear 
	xpaset -p ds9 scale limits -1.0E4 4.0E4
	xpaset -p ds9 fits ${SCI_FILE}
	xpaset -p ds9 export png ./figures/sci-file-2.png
	xpaset -p ds9 fits ${DECONV_FILE}
	xpaset -p ds9 export png ./figures/deconv-primary-3.png
	xpaset -p ds9 scale limits -100 100
	xpaset -p ds9 fits ${DECONV_FILE}[RESIDUAL]
	xpaset -p ds9 export png ./figures/deconv-residual-3.png
	xpaset -p ds9 scale mode minmax
} &>> ${BUILD_LOG_FILE}

#:end{HIDE}

echo "" # Empty line for formatting
python -c 'import sys; import numpy as np; from astropy.io import fits; original=fits.getdata(sys.argv[1]); deconv=fits.getdata(sys.argv[2]); residual=fits.getdata(sys.argv[2],ext=1); print(f"signal fraction in deconvolved components {np.nansum(deconv)/np.nansum(original)}"); print(f"signal fraction in residual {np.nansum(residual)/np.nansum(original)}")' ${SCI_FILE} ${DECONV_FILE}
#:end{CELL}






#:HIDE
xpaset -p ds9 exit &>> ${BUILD_LOG_FILE}


