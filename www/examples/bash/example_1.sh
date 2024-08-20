
: << ---MD

# Deconvolution of a single wavelength #

This example uses BASH scripting to invoke *aopp_deconv_tool*. To show graphical output we are using two FITS file viewers; [QFitsView](https://www.mpe.mpg.de/~ott/QFitsView/), and [DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9). Each of these has their strenghts so it makes sense to use both when required.

## Setup ##

The first thing to do in define some constants for later use. We will also enable "strict mode" to make it easier to debug. However, if you are following along in an interactive session, it is suggested to **not** enable "strict mode".

---MD
#:begin{CELL}

# Use strict mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

SCRIPT_DIR=${0%/*}
#EXAMPLE_DIR="${SCRIPT_DIR}/../../../example_data/ifu_observation_datasets/"
EXAMPLE_DIR=${1}

SCI_FILE="${EXAMPLE_DIR}/single_wavelength_example_sci.fits"
STD_FILE="${EXAMPLE_DIR}/single_wavelength_example_std.fits"

echo "SCI_FILE=$SCI_FILE"
echo "STD_FILE=$STD_FILE"

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

mkdir -p ./figures

#:end{HIDE}

: << ---MD
With that out of the way, we can move on to something more interesting.

## Looking at the data ##

To ensure that everything is set up correctly. Lets open the FITS files and ensure that they are what we expect.

NOTE: FITS files can have multiple extensions, if \`QFitsView\` is passed a file with multiple extensions it will ask you to select the one you want. To select the extension from the command-line, append \`[<int>]\` or \`[<str>]\` to the path. The first version uses the extension number, the second uses the extension name. Generally we will use the extension name. The first extension in a FITS file is always called PRIMARY.
---MD

#:begin{CELL}
QFitsView ${SCI_FILE}[PRIMARY] &
#:HIDE
sleep 2; screenshot_process ./figures/sci-file.png; kill $!

QFitsView ${STD_FILE}[PRIMARY] &
#:HIDE
sleep 2; screenshot_process ./figures/std-file.png; kill $!

#:end{CELL}

: << ---MD
science observation                 | standard star observation
![sci-file](./figures/sci-file.png) | ![std-file](./figures/std-file.png)

The above images show that we at least have the correct input files. That's a good start. The deconvolution script \`aopp_deconv_tool.deconvolve\` requires a science observation and a file for the *point spread function* (PSF) of the observation. We are going to  use the standard star as our PSF. This is not optimal, but good enough for demonstration purposes. However, we must first get the standard star observation into the correct format.

## Normalising the PSF ##

The deconvolution script requires the PSF meets some basic requirements:

1. The PSF has an odd number of pixels in each direction.
2. The PSF is centered, meaning the brightest pixel is coincident with the center pixel of the image.
3. The brightness of the PSF sums to one.

We know that (2) is not met from the figures we created above. Using DS9 to grab the information we want
We can see from the header data that the standard star observation does not meet criteria (1), 


---MD

#:begin{CELL}
ds9 ${STD_FILE} -scale log -crosshair 200 200 physical &

#:HIDE
sleep 2; screenshot_process ./figures/std-file-pixel.png "SAOImage"; kill $!

#:HIDE
ds9 -fits ${STD_FILE} -header save ./figures/std-file-header.txt -exit

#:end{CELL}

: << ---MD
Pixel value of standard star observation
![std-file-pixel](./figures/std-file-pixel.png)

Standard star observation header data excerpt
\`\`\`bash
$(head -10 ./figures/std-file-header.txt)
\`\`\`

The above shows that the standard star observation does not meet the PSF requirements. The image shows that a single pixel has a value larger than one, and the \`NAXIS1\` and \`NAXIS2\` keys in the header information show that at least one of the dimensions is not odd.

## Normalising the PSF ##

Thankfully, there is a script included in the package that will normalise a PSF so it obeys the constraints we require. The script is the \`aopp_deconv_tool.psf_normalise module\`. We can invoke it on the command-line, we will redirect the output into a log file to avoid cluttering the terminal.
---MD

#:begin{CELL}
python -m aopp_deconv_tool.psf_normalise ${STD_FILE} -o ${STD_FILE_NORM} &> ./log.txt
#:end{CELL}
