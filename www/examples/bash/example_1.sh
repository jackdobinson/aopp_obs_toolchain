#!/usr/bin/env bash
#:HIDE
BUILD_LOG_FILE="./build_log.txt"; echo "" > ${BUILD_LOG_FILE}

: << ---MD
---
layout: bare
---

# Deconvolution of a single wavelength #

This example uses BASH scripting to invoke *aopp_deconv_tool*. To show graphical output we are using two FITS file viewers; [QFitsView](https://www.mpe.mpg.de/~ott/QFitsView/), and [DS9](https://sites.google.com/cfa.harvard.edu/saoimageds9). Each of these has their strenghts so it makes sense to use both when required.

## Setup ##

The first thing to do in define some constants for later use. We will also enable "strict mode" to make it easier to debug. However, if you are following along in an interactive session, it is suggested to **not** enable "strict mode".

---MD
#:begin{CELL}

# Use strict mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

SCRIPT="$(realpath -e ${0})"
SCRIPT_DIR=${SCRIPT%/*}
EXAMPLE_DIR="${SCRIPT_DIR}/../../../example_data/ifu_observation_datasets/"

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

: << ---MD
With that out of the way, we can move on to something more interesting.

## Looking at the data ##

To ensure that everything is set up correctly. Lets open the FITS files and ensure that they are what we expect.

NOTE: FITS files can have multiple extensions, if \`QFitsView\` is passed a file with multiple extensions it will ask you to select the one you want. To select the extension from the command-line, append \`[<int>]\` or \`[<str>]\` to the path. The first version uses the extension number, the second uses the extension name. Generally we will use the extension name. The first extension in a FITS file is always called PRIMARY.
---MD

#:begin{CELL}
#:DUMMY
ds9 ${SCI_FILE} &
#:DUMMY
ds9 ${STD_FILE} &

#:HIDE
{ xpaset -p ds9 fits ${SCI_FILE}; xpaset -p ds9 export png ./figures/sci-file.png;} &>> ${BUILD_LOG_FILE}
#:HIDE
{ xpaset -p ds9 fits ${STD_FILE}; xpaset -p ds9 export png ./figures/std-file.png;} &>> ${BUILD_LOG_FILE}

#:end{CELL}


: << ---MD
|science observation                 | standard star observation           |
|------------------------------------|-------------------------------------|
|![sci-file](./figures/sci-file.png) | ![std-file](./figures/std-file.png) |

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
#:DUMMY
ds9 ${STD_FILE} -scale log -crosshair 200 200 physical &

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

: << ---MD
Pixel value of standard star observation

![std-file-pixel](./figures/std-file-pixel.png)

Standard star observation header data excerpt
\`\`\`bash
$(head -10 ./figures/std-file-header.txt)
\`\`\`

The above shows that the standard star observation does not meet the PSF requirements. The image shows that a single pixel has a value larger than one, and the \`NAXIS1\` and \`NAXIS2\` keys in the header information show that at least one of the dimensions is not odd.

Thankfully, there is a script included in the package that will normalise a PSF so it obeys the constraints we require. The script is the \`aopp_deconv_tool.psf_normalise module\`. We can invoke it on the command-line, we will redirect the output into a log file to avoid cluttering the terminal.
---MD

#:begin{CELL}
python -m aopp_deconv_tool.psf_normalise ${STD_FILE} -o ${STD_FILE_NORM} &> ./logs/psf_normalise_log.txt

#:DUMMY
ds9 ${STD_FILE_NORM} -scale log -crosshair 200 200 physical &

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

: << ---MD
Looking at the result, we can see that the standard star observation is not normalised and can be used as a PSF.

<table>
<tr>
<th>Normalised Image</th><th>Header excerpt</th>
</tr>
<tr>
<td>
<img src="./figures/std-norm-file-pixel.png">
</td>
<td>
<pre>
$(head -10 ./figures/std-norm-file-header.txt)
</pre>
</td>
</tr>
</table>


## Deconvolving the image ##

As we have a normalised PSF, we now have everything we need to deconvolve the image. We use the command-line python script \`aopp_deconv_tool.deconvolve\`.

---MD


#:begin{CELL}
python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} -o ${DECONV_FILE} &> ./logs/deconv-1-log.txt
#:end{CELL}

: << ---MD
## Comparing the result with the original ##

The resulting file has two extensions, a primary extension and another one that holds the residual. You can access a FITS file extension via its index, or via a name (if one was defined for it). In our case we know the name of the second extension is 'RESIDUAL' so we use that.

We can tell how well the deconvolution has gone by comparison between the original image, the deconvolved image, and the residual. Ideally, we are looking for the residual to be indistinguishable from background noise, and the deconvolved image to be an obviously "higher resolution" version of the original image.

We will also use a small python script to get the fraction of original signal in the deconvolved components and the residual.

---MD

#:begin{CELL}
#:DUMMY
ds9 ${SCI_FILE} ${DECONV_FILE} ${DECONV_FILE}[RESIDUAL]

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

: << ---MD

| Original Image | Deconvolved Image | Residual |
|----------------|-------------------|----------|
|![original](./figures/sci-file-ds9.png) | ![deconv](./figures/deconv-primary-1.png) | ![original](./figures/deconv-residual-1.png) |


From the above results, a couple of things are apparent:

1. Both the deconvolved image and the residual look very similar.
2. There is still a large amount of signal left in the residual, approximately 9% if you compare the pixel sums of each image.

This is indicative of not deconvolving the science image for long enough.


## Image is not completely deconvolved ##

The FITS file that holds the deconvolved image also has some useful data about the deconvolution, including why the deconvolution stopped, in its header (this information is also in the command-line output but we are not showing that for brevity). We can access the header of the FITS file and decide upon how to proceed depending upon why the deconvolution stopped.

FITS headers can hold key-value pairs in pairs of (PKEYn, PVALn), where n is a number, header entries. PKEYn holds the name of the key, and PVALn contains the value of that key. I know the information we are looking for the deconv.progress_string key so we will search for that in the FITS file, we will also search for the deconv.n_iter key which will tell us the maximum number of iterations that could have been performed. The simplest way is to save the header to a file, then use the \`grep\` program to search for the key-value header keys.
---MD

#:begin{CELL}
#:DUMMY
ds9 ${DECONV_FILE} -header save ./figures/deconv-file-header-1.txt -exit

#:HIDE
{ xpaset -p ds9 fits ${DECONV_FILE}; xpaset -p ds9 header save ./figures/deconv-file-header-1.txt; } &>> ${BUILD_LOG_FILE}

grep -E 'PKEY*|PVAL*|CONTINUE' ./figures/deconv-file-header-1.txt
#:end{CELL}

: << ---MD
The "deconv.progress_string" key holds a message that tells us why the deconvolution ended, and after how many iterations. The "deconv.n_iter" key tells us the maximum number of iterations.

Therefore,
\`\`\`
...
PKEY9   = 'deconv.n_iter'                                                       
PVAL9   = '1000    '                                                            
PKEY10  = 'deconv.progress_string'                                              
PVAL10  = 'Ended at 192 iterations: Standard deviation of statistics in last &' 
CONTINUE  '10 steps are all below minimum fraction as defined in &'             
CONTINUE  '\`min_frac_stat_delta\` parameter.'                                    
...
PKEY18  = 'deconv.min_frac_stat_delta'                                          
PVAL18  = '0.001   '                                                            
...
\`\`\`
tells us that we stopped at 192 iterations out of a possible 1000 because one of the stopping criteria of the algorithm was tripped. I know that the stopping criteria that was tripped is min_frac_stat_delta. It stops the iteration if the standard deviation of the brightest pixel of the residual and the RMS of the residual is lower than its value in the last 10 iterations.

The \`deconv.min_frac_stat_delta parameter\` is what stopped the algorithm, and is set to a conservative value, 1E-3, by default. If we decrease this value, our deconvolution will run for longer and move more signal from the residual to the deconvolved image.

## Re-run command with a smaller \`min_frac_stat_delta\` parameter ##

Let's re-run the command with \`min_frac_stat_delta\` set to a much smaller value, 1E-5 should be small enough.
---MD

#:begin{CELL}
python -m aopp_deconv_tool.deconvolve ${SCI_FILE} ${STD_FILE_NORM} -o ${DECONV_FILE} --min_frac_stat_delta 1E-5 &> ./logs/deconv-2-log.txt
#:end{CELL}

: << ---MD
## Comparing the new result with the original ##

Just as before, we can compare the new result, we are looking for the same things. Ideally, the residual to be indistinguishable from background noise, and the deconvolved image to be an obviously "higher resolution" version of the original image. We will also print out the deconvolution key-pair values, and set the same colourscale for all images.
---MD

#:begin{CELL}
#:DUMMY
ds9 -scale limits 0 2.5E4 ${SCI_FILE} ${DECONV_FILE} ${DECONV_FILE}[RESIDUAL]
#:DUMMY
ds9 ${DECONV_FILE} -header save ./figures/deconv-file-header-2.txt -exit

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

: << ---MD

| Original Image | Deconvolved Image | Residual |
|----------------|-------------------|----------|
|![original](./figures/sci-file-ds9.png) | ![deconv](./figures/deconv-primary-2.png) | ![original](./figures/deconv-residual-2.png) |


A few things are going on here. Firstly, deconv.progress_string shows we completed the full 1000 iterations. Good, that is what we wanted. Also, we have reduced the amout of signal in the residual, but only to 8%. The result is not quite what we hoped for. There are artefacts in this image that cause problems with the deconvolution (see the bottom left quarter of the planet in the middle image).

There is also another problem, this result is a bit speckly. There are discrete pixels that are noticably brighter than their neighbours. There is no physical reason for this, it is unfortunately one of the problems with the CLEAN algorithm but it can be mitigated.

## Artefact Reduction ##

An artefact is any feature on an image that is due to flaws in the instrumentation and/or processing. We are going to restrict ourselves to worrying about the subset artefacts that cause a deconvolution to fail. In the case of CLEAN algorithms, these are artefacts that are smaller that the PSF used for the deconvolution. Instead of solving the hard problem of identifying artefacts in general, we are going to work on the much easier problem of identifying artefacts that are a problem for the deconvolution algorithm, and reducing their influence such that the deconvolution will complete.

There are a few scripts provided in aopp_deconv_tool to help with this.

aopp_deconv_tool.artefact_detection
: Creates a map of the image that rates each pixel by how likely it is to be a problematic artefact
aopp_deconv_tool.create_bad_pixel_mask
: Takes the result of aopp_deconv_tool.artefact_detection as a FITS file and applies a threshold. Can also accept DS9 region files to manually designate artefacts to be masked. Creates a true/false mask where true values indicate a pixel is part of (of influenced by) an artefact.
aopp_deconv_tool.interpolate
: Accepts an observation FITS file and a boolean mask FITS file (the output of aopp_deconv_tool.create_bad_pixel_mask), and interpolates over the observations where the mask is true. Creates a FITS file of the result that is the same as the input observation FITS file except fot the interpolation.

We will apply these three steps, and show the results of each of them.



---MD

#:begin{CELL}
python -m aopp_deconv_tool.artefact_detection $SCI_FILE -o $SCI_ARTEFACT_FILE &> ./logs/sci-artefact-1-log.txt
python -m aopp_deconv_tool.create_bad_pixel_mask $SCI_ARTEFACT_FILE -o $SCI_ARTEFACT_MASK_FILE &> ./logs/sci-artefact-mask-1-log.txt
python -m aopp_deconv_tool.interpolate $SCI_FILE $SCI_ARTEFACT_MASK_FILE -o $SCI_INTERP_FILE &> ./logs/sci-interp-1-log.txt

#:DUMMY
ds9 ${SCI_ARTEFACT_FILE} ${SCI_ARTEFACT_MASK_FILE} ${SCI_INTERP_FILE}

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



: << ---MD
| Artefact Map | Artefact Mask | Interpolated Image |
|----------------|-------------------|----------|
|![original](./figures/sci-artefact-file-1.png) | ![deconv](./figures/sci-artefact-mask-file-1.png) | ![original](./figures/sci-interp-file-1.png) |

Artefact Map
: The bright regions in the image are pixels that the script thinks are influenced by artefacts.

Artefact Mask
: The pixels are selected by aopp_deconv_tool.create_bad_pixel_mask by a threshold value which is 3 by default. The resulting binary image then undergoes dilation based on the value of the pixel, i.e. a pixel with a value of 5 would be dilated twice so it would mask its 2-step away neighbours. This process ensures that artefact edge effects are caught.

Interpolated Image
: The plot shows the missing data in the original image has successfully been filled in. Interestingly, the dark dot area on the right of the disk has not been touched very much. This is because the threshold we have used for aopp_deconv_tool.create_bad_pixel_mask was the default of 3, we could lower it but for our purposes that dark dot doesn't stop the deconvolution process. We can now use the interpolated FITS file in place of the original FITS file.


## Removing Speckles and Faster deconvolution by tweaking parameters ##

Instead of re-running the original command, which can take a lot of time. We can increase the loop_gain parameter. This parameter controls the fraction of brightness of a pixel transferred from the residual to the components upon each iteration. The current value is very conservative, so it can be safely increased in most cases. The relationship between loop_gain and the maximum number of iterationsn_iter is not linear, so increasing to 0.2 means we only need about 200 interations.

Also, if speckles are a problem altering the threshold value of Modified CLEAN can reduce their influence. At the start of an iteration, Modified CLEAN chooses pixels to operate on for that iteration. When the spread of pixel values chosen is of a similar magnitude to the noise of the image, the noise tends to reinforce itself and shows up as regular specking. There is a "smart" threshold setting that aims to reduce the specking problem by heuristically choosing a threshold. This mode is enabled when the --threshold parameter is a negative number, usually -1. By default --threshold is set to 0.3 which means that pixels will be chosen that are brighter than 0.3 times the brightest pixel in the residual. Automatic threshold selection is slower than a constant value, but it tries to account for the source of speckling and reduce it if possible.

We will also reduce the \`rms_frac_threshold\` parameter, another stopping parameter, so we can be sure the iterations will complete.

---MD

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

: << ---MD
| Original Image | Deconvolved Image | Residual |
|----------------|-------------------|----------|
|![original](./figures/sci-file-2.png) | ![deconv](./figures/deconv-primary-3.png) | ![original](./figures/deconv-residual-3.png) |


Note that the original and deconvolved image are on the same colour scale, however the residual image is scaled at +/- 100. Otherwise the residual would not be visible at all.

Another thing to note in the above result is that the fraction of signal in the residual has become negative. This is a fairly common occurance, in this case a negative of approximately 0.2%. Ideally the fraction of signal in the residual should be around zero, within the noise level of the data, which is about 1-2% in this case so our residual is within acceptable limits.

We have been quite aggressive in deconvolving the data in this case, and I've set the colour scales so we can see more detail and compare between images more easily. There are some interesting features in the above plots. 

Comparing the original image with the deconvolved image we can see the deconvolved image is much better defined, smudges in the original have become distinct features, it is even possible to see the planet's ring in the lower half of the image (about 1 planet radius away from the disk). 

You can also see many more artefacts in the deconvolved image:
* There are some straight lines running from top-left to bottom-right
* Lines with what looks like shadowing going from bottom-left to top-right
* The large dot we identified earlier
* Many smaller dots, mostly on the left hand side of the disk.

These are features of the detector, the instrument used to obtain this image was an [*integral field unit*](https://www.eso.org/public/teles-instr/technology/ifu/). There is a lot of manipulation of the light as it travels through the optical system, and some of these artefacts are the result, others are the result of defects in the CCD.

Looking at the residual, there is a very apparent rectangle set into some "cloudyness". This is due to how the deconvolution algorithm handles edge-effects. The "cloudy" region normally contains no data (NANs), but was filled with a value to avoid edge effects during deconvolution and is therefore not real signal. The bottom left to top-right ridges are still visible, and the planet disk is still visible as well, along with some of the planet's ring. However, comparing histograms between the images, it is obvious that the vast majority of emission is explained by the deconvolved image.

Further deconvolution would not help in this case as problems with speckling can continue to arise even when using the "smart" threshold setting (there is a limit to how smart it is). If we need to, we can do some [regularisation](https://en.wikipedia.org/wiki/Regularization_(mathematics)) to push our result to be something more physically plausible. However, you can start to lose some of the nice properties that Modified CLEAN has (e.g. flux conservation).

There is another source of noise we have not discussed in this example. The PSF itself is noisy, and will suffer from the same artefacts that the science observation suffers from. *aopp_deconv_tool* does include a way to model a PSF, please see the relevant example.

## Conclusion ##

We started with a science observation and a standard star observation. Normalising the standard star so it could be used as a PSF we deconvolved the science observation with default settings. Then we introduced different problems and solutions to those problems as they arised.

1) The initial deconvolution stopped before the vast majority of the original observation was explained by the deconvolved image, this was fixed by altering some parameters so the deconvolution continued.
2) Artefacts stopped the image deconvolving beyond a certain point, as they were smaller than the PSF the deconvolution process gets 'stuck' as it cannot account for such small features. We used the artefact reduction tools in aopp_deconv_tool to reduce the effect of the artefacts and let the deconvolution process continue.
3) We removed speckling due to deconvolution proceeding close to the noise level of the observations. This can be corrected for in multiple ways (including reducing the --threshold parameter as the as deconvolution process continues), we chose to use the "smart threshold" (--threshold set to -1). This does effectively reduce the speckling, but cannot eliminate it entirely. However it is often good enough to get to the noise level of an image.

Finally we examined the last deconvolved image, compared it to the original, pointed out features of the deconvolved image and the residual, and finally touched upon the topics of regularisation and PSF modelling.

---MD




#:HIDE
xpaset -p ds9 exit &>> ${BUILD_LOG_FILE}


