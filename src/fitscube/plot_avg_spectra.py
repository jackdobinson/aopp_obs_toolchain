#!/usr/bin/env python3
"""
Investigate properties of observation spectra
"""

import sys, os
import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt

#fits_files = glob.glob(os.path.expanduser("~/scratch/reduced_images/SINFO.*/*/analysis/*_cleanMR2_200*.fits"))
fits_files = glob.glob(os.path.expanduser("~/scratch/reduced_images/SINFO.*/**/analysis/*_renormed.fits"))

norm_to = 1

min_idx, max_idx = (200,500)

for afile in fits_files:
	with fits.open(afile) as hdul:
		factor = 1.0/np.nanmean(hdul['PRIMARY'].data[min_idx:max_idx])
		if '0.1' in afile:
			plt.plot(np.nanmean(hdul['PRIMARY'].data*factor, axis=(1,2)), label=afile, ls=':', lw=0.5)
		else:
			plt.plot(np.nanmean(hdul['PRIMARY'].data*factor, axis=(1,2)), label=afile, lw=0.5)
#plt.legend()
#plt.xlim(0,200)
#plt.ylim(0, 3E-7)
plt.ylim(0, 2)
plt.axvline(min_idx)
plt.axvline(max_idx)
plt.show()
