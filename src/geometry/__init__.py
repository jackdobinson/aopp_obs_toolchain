"""
Routines useful for geometry
"""
import dataclasses as dc

import numpy as np

def taxicab_metric(p1 : np.ndarray, p2 : np.ndarray, axis : float = -1):
	return np.abs(p1-p2).sum(axis=axis)

def euclidean_metric(p1 : np.ndarray, p2 : np.ndarray, axis : float = -1):
	print(f'{p1.shape} {p2.shape}')
	print(f'{np.sqrt(((p1-p2)**2).sum(axis=axis))}')
	return np.sqrt(((p1-p2)**2).sum(axis=axis))
