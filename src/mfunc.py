"""
Module containing small mathematical functions that are useful in lots of contexts
"""
import numpy as np

def logistic_function(x, left_limit=0, right_limit=1, transition_scale=1, center=0):
	return (right_limit-left_limit)/(1+np.exp(-(np.e/transition_scale)*(x-center))) + left_limit
