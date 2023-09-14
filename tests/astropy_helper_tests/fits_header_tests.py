
import astropy_helper as aph
import astropy_helper.fits.header

def test_aph_fits_header_gives_correct_standard_output_format():
	
	
	params_1 = {
		"key_1" : {"child_key_1" : "X1", "child_key_2": "X2"},
		"key_2" : "value_2",
		"key_3" : {"child_2_key_1" : "Y1", "child_2_key_2" : {"child_1_key_1" : "Z1"}}
	}
	
	expected_output=[
		('PKEY0   ', 'key_1.child_key_1'),
		('PVAL0   ', 'X1'),
		('PKEY1   ', 'key_1.child_key_2'),
		('PVAL1   ', 'X2'),
		('PKEY2   ', 'key_2'),
		('PVAL2   ', 'value_2'),
		('PKEY3   ', 'key_3.child_2_key_1'),
		('PVAL3   ', 'Y1'),
		('PKEY4   ', 'key_3.child_2_key_2.child_1_key_1'),
		('PVAL4   ', 'Z1')
	]

	output = []
	for item in aph.fits.header.DictReader(params_1, 'standard'):
		output.append(item)

	assert len(output) == len(expected_output), "dictionary reader should have same output length as expected output"
	assert all(x==y for x,y in zip(output, expected_output)), "dictionary reader should give standard fits output as expected"
		

