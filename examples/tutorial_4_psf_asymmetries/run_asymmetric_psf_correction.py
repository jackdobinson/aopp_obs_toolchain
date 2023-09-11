#!/usr/bin/env python3
"""
This script sets up a modified python environment and runs 
"""
import sys, os


# Global variables hold locations of data, I would not normally do this but for
# illustrative purposes it's good enough.
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
TUTORIAL_PATH =  os.path.abspath(os.path.join(THIS_DIR,"../"))
SCRIPTS_PATH = os.path.abspath(os.path.join(THIS_DIR,"../scripts"))
EXAMPLE_INPUT_PSF_DATA = os.path.join(TUTORIAL_PATH, "./data/test_standard_star.fits")
EXAMPLE_INPUT_OBS_DATA = os.path.join(TUTORIAL_PATH, "./data/test_rebin_deconv_bad.fits")
OUTPUT_PATH = os.path.join(THIS_DIR, './output/asym_corrected.fits')

print(f'{THIS_DIR = }')
print(f'{TUTORIAL_PATH = }')
print(f'{SCRIPTS_PATH = }')
print(f'{EXAMPLE_INPUT_OBS_DATA = }')
print(f'{EXAMPLE_INPUT_PSF_DATA = }')
print(f'{OUTPUT_PATH = }')

# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
sys.path.insert(0,SCRIPTS_PATH)
print(f'Adding SCRIPTS_PATH to locations python searches for modules:')
print(f'\tsys.path = [')
for apath in sys.path:
	print(f'\t\t{apath},')
print(f'\t]')

# import the script we actually want to execute
import fitscube.deconvolve.asymmetric_psf_correction

if __name__ == '__main__':
	# create a list of arguments to the script
	argv = (f"{EXAMPLE_INPUT_OBS_DATA} --ext_obs 1 --file_psf {EXAMPLE_INPUT_PSF_DATA} --ext_psf 1 --file_out {OUTPUT_PATH} --window_shape 51 --n_subdivisions 64").split()

	# Parse the arguments into a dictionary
	args = fitscube.deconvolve.asymmetric_psf_correction.parse_args(argv)

	# Call the entrypoint of the script with our desired arguments
	fitscube.deconvolve.asymmetric_psf_correction.main(args)

