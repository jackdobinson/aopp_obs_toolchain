#!/usr/bin/env python3
"""
Utilities for dictionary operations

"""
import utilities.cfg

def to_str(adict, indentstr=utilities.cfg.indent_str, depth=0):
	"""Returns a string that contains the formatted output of a dictionary"""
	keys = adict.keys()
	maxlen_k = max(map(len, map(str, keys)))
	kv_fmt_str = depth*indentstr+'{:'+f'{maxlen_k}'+'} {}\n'
	k_fmt_str = depth*indentstr+'{:'+f'{maxlen_k}'+'}\n'
	astr = ''
	for k,v in sorted(adict.items()):
		if type(v)==dict:
			astr+=k_fmt_str.format(k)
			astr+= to_str(v, indentstr=indentstr, depth=depth+1)
		elif type(v) in (tuple, list):
			kstr = k_fmt_str.format(k)
			astr+=kstr
			vstr = ''.join([(depth+1)*indentstr+'{}\n'.format(x) for x in v])
			astr+=vstr
		else: 
			astr += kv_fmt_str.format(k, v)
	return(astr)

def create_rand(n=10, consecutive_keys=False, unique_vals=True, key_pool=None,
				val_pool=None, seed=None, DEBUG=False):
	'''
	Creates a randomly populated dictionary for testing purposes.
	consecutive keys - should keys be consecutive values (of alphabet or key_pool)?
	unique_vals - should values be unique or possibly repeating?
	key_pool - if present, will choose keys from here instead of randomly generated words
	val_pool - if present, will choose values from here instead of randomly generated numbers
	seed - seed value for RNG
	DEBGUG - switch debug mode on to see
	'''
	import numpy as np
	import math as m
	import itertools as it
	import random as r
	if DEBUG: print('='*30 + ' ENTERING ' + '='*30)
	alphabet = utilities.cfg.alphabet
	np.random.seed(seed)
	r.seed(seed)

	wordlen = int(m.ceil(n/float(len(alphabet))))
	
	dict = {}
	if key_pool==None:
		iterator = it.combinations(alphabet, wordlen)
	else:
		iterator = it.chain(key_pool)
	if DEBUG:
		astr = 'n {} consecutive_keys {} unique_vals {} key_pool {} val_pool {} seed {}'
		print(astr.format(	n, 
							consecutive_keys, 
							unique_vals, 
							False if key_pool==None else True, 
							False if val_pool==None else True, 
							seed
							)
						)
	if consecutive_keys:
		if unique_vals:
			if val_pool != None:
				values = np.random.choice(val_pool, size=n, replace=False)
				for i, v in zip(range(n), values):
					dict[''.join(next(iterator))] = v
			else:
				for i in range(n):
					dict[''.join(next(iterator))] = i
		else:
			if val_pool != None:
				values = np.random.choice(val_pool, size=n)
				for i, v in zip(range(n), values):
					dict[''.join(next(iterator))] = v
			else:
				for i in range(n):
					dict[''.join(next(iterator))] = r.randint(0, n)
	else:
		keys = np.random.choice([''.join(el) for el in iterator], size=n, replace=False)
		if unique_vals:
			if val_pool != None:
				values = np.random.choice(val_pool, size=n, replace=False)
				for i, k, v in zip(range(n), keys, values):
					dict[k] = v
			else:
				for i, k in zip(range(n), keys):
					dict[k] = i
		else:
			if val_pool != None:
				values = np.random.choice(val_pool, size=n)
				for i, k, v in zip(range(n), keys, values):
					dict[k] = v
			else:
				for i, k in zip(range(n), keys):
					dict[k] = r.randint(0,n)
	if DEBUG: print('='*30 + ' LEAVING ' + '='*30)
	return(dict)

def take_if_present(adict, keys):
	rdict = {}
	for k in keys:
		try:
			rdict[k] = adict[k]
		except KeyError:
			pass
	return(rdict)


def rename(adict, name_map):
	# works with ordered dicts too
	renamed_dict = type(adict)()
	for k, v in adict.items():
		renamed_dict[name_map[k]] = v
	return(renamed_dict)

if __name__=='__main__':
	# Run test cases

	test_dict = create_rand()
	print(test_dict)

	test_dict['fds'] = create_rand()
	test_dict['ythf'] = [123,34,567,'dht']
	print(to_str(test_dict))
