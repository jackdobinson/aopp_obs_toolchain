
from types import SimpleNamespace
import numpy as np

from algorithm.deconv.clean_modified import CleanModified

algo_basic_test_data = SimpleNamespace(
	n_iter = 10,
	test_obs = np.ones((7,7)),
	test_psf = np.ones((7,7))
)

def test_clean_modified_runs_for_basic_data():
	
	deconv = CleanModified(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf)
	
	assert result[2] == deconv.n_iter, f"Expect {deconv.n_iter} iterations of CleanModified, have {result[2]} instead."


def test_clean_modified_call_altered_instantiated_parameters():
	
	n_iter_overwrite=15
	assert n_iter_overwrite != algo_basic_test_data.n_iter, "Must have an overwrite value that is not the same as the original value"
	
	deconv = CleanModified(n_iter=algo_basic_test_data.n_iter)
	result = deconv(algo_basic_test_data.test_obs, algo_basic_test_data.test_psf, n_iter=n_iter_overwrite)
	
	assert result[2] == n_iter_overwrite, f"Expect {n_iter_overwrite} iterations of CleanModified, have {result[2]} instead."
