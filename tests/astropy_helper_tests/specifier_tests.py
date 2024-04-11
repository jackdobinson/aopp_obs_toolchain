


import os

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier

import test_data

import scientest.decorators

@scientest.decorators.skip(
	predicate = lambda : not os.path.exists(test_data.example_fits_file),
	message=f"Skipped as '{test_data.example_fits_file}' cannot be found"
)
def test_specifier_completes_parsing():
	examples = (
		f"{test_data.example_fits_file}"+"{PRIMARY}[100:200](1,2)",
		f"{test_data.example_fits_file}"+"{DATA}[:,100,:]{CELESTIAL:(0,2)}",
		f"{test_data.example_fits_file}"+"{1}[:,:]",
		f"{test_data.example_fits_file}"
	)

	results = []
	for x in examples:
		results.append(aph.fits.specifier.parse(x, ["CELESTIAL"]))
	
	assert results[0].slices[0] == slice(100,200)
	assert results[0].axes == {'CELESTIAL':(1,2)}

	assert results[1].slices[0] == slice(None)
	assert results[1].axes == {'CELESTIAL':(0,2)}

	assert results[2].slices[0] == slice(None)
	assert results[2].axes == {'CELESTIAL':(1,2)}

	assert results[3].slices[0] == slice(None)
	assert results[3].axes == {'CELESTIAL':(1,2)}



















