#!/usr/bin/env python3
"""
Generic functions that implement a certain type of flow control
"""



def mapping_lookup_with_keys(m_obj, keys, fail_states=(None,), mutators=lambda x: x):
	"""
	Applies each key from "keys" in turn to "m_obj", when a key is found that does
	not give a result in "fail_states", return that result. If no key is found
	that gives a result not in "fail_states", will return the first element of "fail_states".

	# ARGUMENTS #
		m_obj
			Some mapping object (i.e. something that has the ".get()" method)
		keys
			A sequence of keys to try on the mapping object
		fail_states
			A sequence of objects that denote a failed lookup of the key in the
			mapping object. The first object in the sequence will be used as the
			default argument to "m_obj.get(key,default)". It's recommended that
			the first object is therefore "None" or at least lightweight.
		mutators
			A callable (or tuple of callables) that takes the result of a key lookup
			in m_obj and alters it in some way. Useful for standardising case of strings
			(e.g. "mutators=lambda x: x.lower()"). If the tuple is smaller than the number of 
			keys, the last element will be repeated. If is a callable, it will be applied
			to the result of all key lookups.

	# RETURNS #
		result
			The result of the first key from "keys" lookup in "m_obj" that is not in "fail_states" after
			the application of any "mutators". If no such key exists, is the first element of "fail_states".
	"""
	if type(mutators) is tuple:
		mutators = tuple(list(mutators)+[mutators[-1]]*(len(keys)-len(mutators)))
	elif callable(mutators):
		mutators = [mutators]*len(keys)
	else:
		raise TypeError(f'Argument "mutators" must be a callable or tuple of callables that takes one parameter.')


	_first_fail_state = fail_states[0]
	for _key, _mutator in zip(keys, mutators):
		result = _mutator(m_obj.get(_key, _first_fail_state))
		if result not in fail_states:
			break
	return(result if result not in fail_states else _first_fail_state)


