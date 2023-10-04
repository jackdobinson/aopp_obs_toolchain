import os
from types import SimpleNamespace
import copy

import numpy as np
from astropy.io import fits
import astropy_helper as aph
import astropy_helper.fits.header
from astropy_helper.fits.specifier import FitsSpecifier
import numpy_helper as nph
import numpy_helper.axes
import numpy_helper.slice
import numpy_helper.array
from algorithm.deconv.lucy_richardson import LucyRichardson
import algorithm.bad_pixels

import test_data
import decorators

algo_basic_test_data = SimpleNamespace(
	n_iter = 10,
	test_obs = np.ones((7,7)),
	test_psf = np.ones((7,7))
)


def test_runs_for_basic_data():
	deconv = LucyRichardson(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf)
	
	assert result[2] == deconv.n_iter, f"Expect {deconv.n_iter} iterations of LucyRichardson, have {result[2]} instead."


def test_call_altered_instantiated_parameters():
	n_iter_overwrite=15
	assert n_iter_overwrite != algo_basic_test_data.n_iter, "Must have an overwrite value that is not the same as the original value"
	
	deconv = LucyRichardson(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf, n_iter=n_iter_overwrite)
	
	assert result[2] == n_iter_overwrite, f"Expect {n_iter_overwrite} iterations of LucyRichardson, have {result[2]} instead."
 
