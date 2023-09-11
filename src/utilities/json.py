#!/usr/bin/env python3
"""
Helper functions for interacting with the python JSON dictionary
"""
import json
import numpy as np


def json_to_array_dict(json_files : str, json_to_arrays_map : dict) -> dict:
	"""
	Converts JSON files to array of values
	
	json_files
		Path to files to convert to arrays
	json_to_arrays_map
		Mapping from the JSON structure to the arrays. Conforms to the following
		structure:
			{
				json_field_name : {
					'key'            : typing.Hashable
					'type_converter' : typing.Callable(json_value) -> python_value
					'json_fetcher'   : typing.Callable(json_obj, json_field_name) -> json_value
					'dtype'          : typing.Type
				},
				.
				.
				.
			}
			
			where
				json_field_name [REQUIRED]
					The name of the field in the JSON file
				'key'
					Holds the key that will reference the data in the array_dict
				'type_converter'
					Holds a function that converts between JSON representation and
					python type. Default is identity operation, uses the result
					of the 'json_fetcher' function as it's argument.
				'json_fetcher'
					Holds a function that gets the json_value from the JSON 
					object. Uses the current JSON Object and the "json_field_name"
					as it's arguments. Default is to use the "json_field_name" field
					of the JSON object as the json_value.
				'dtype'
					The type to store the python_value in the numpy
					array as. Default is to let numpy infer the type.
			
			Example:
				json_to_arrays_map = {
					'dateTime' : {
						'dtype'          : np.datetime64, 
						'type_converter' : lambda json_value : datetime.datetime.strptime(json_value, "%m/%d/%y %H:%M:%S"), 
						'json_fetcher'   : lambda json_obj, json_field_name : json_obj[json_field_name],
					},
					'bpm' : {
						'key'            : 'heartRate', 
						'dtype'          : float, 
						'json_fetcher'   : lambda json_obj, json_field_name : json_obj['value'][json_field_name],
					},
					'confidence' : {
						'key'            : 'heartRateErr', 
						'dtype'          : float, 
						'json_fetcher'   : lambda json_obj, json_field_name : json_obj['value'][json_field_name],
					},
				}
	"""
	type_converter_default = lambda x: x
	json_fetcher_default = lambda jsob_obj, k: json_obj[k]
	dtype_default = None
	
	
	tmp = {}
	for k,v in json_to_arrays_map.items():
		tmp[json_to_arrays_map[k].get('key',k)] = []
	
	# loop over files that hold the data we are interested in and fill up our tmp data holder
	for afile in json_files:
		with open(afile,'r') as f:
			json_obj_list = json.load(f)
		for k in json_to_arrays_map.keys():
			#print(k)
			#print(json_to_arrays_map[k])
			tc_func = json_to_arrays_map[k].get('type_converter', type_converter_default)
			jf_func = json_to_arrays_map[k].get('json_fetcher', json_fetcher_default)
			dt_val = json_to_arrays_map[k].get('dtype', dtype_default)
			tmp[json_to_arrays_map[k].get('key',k)].append(
				np.array(
					[tc_func(jf_func(_x,k)) for _x in json_obj_list],
					dtype = dt_val
				)
			)
	
	for k,v in tmp.items():
		tmp[k] = np.concatenate(v, axis=0)
		
	return(tmp)

