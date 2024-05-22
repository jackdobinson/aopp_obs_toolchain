
import numpy as np

import matplotlib.pyplot as plt

def keep(label_map, n_labels, to_keep_labels):
	for label in range(1, n_labels+1):
		if label not in to_keep_labels:
			label_map[label_map==label] = 0
	return label_map, to_keep_labels

def remove(label_map, to_remove_labels):
	for label in to_remove_labels:
		label_map[label_map == label] = 0
	return label_map, tuple(x for x in range(1, np.max(label_map)+1) if x not in to_remove_labels)

def relabel(label_map, n_old_labels, new_labels):
	for i, label in enumerate(new_labels):
		label_map[label_map == label] = n_old_labels + i + 1
	label_map[label_map > 0] -= n_old_labels
	return label_map, len(new_labels)

def keep_n_largest(label_map : np.ndarray, n_labels : int, n_to_keep : int):
	# order labels by number of pixels they contain
					
	ordered_labels = list(range(1,n_labels+1)) # label 0 is background
	ordered_labels.sort(key = lambda x: np.count_nonzero(label_map == x), reverse=True)
	
	if n_labels > n_to_keep:
		label_map, _ = keep(label_map, n_labels, ordered_labels[:n_to_keep])
		
		# relabel regions so they are in descending size order
		label_map, n_labels = relabel(label_map, n_labels, ordered_labels[:n_to_keep])
	
	return label_map, n_labels

def keep_larger_than(label_map : np.ndarray, n_labels : int, max_remove_size : int):
	kept_labels = []
	for i in range(1, n_labels+1):
		lbl_mask = label_map==i
		n_pixels = np.count_nonzero(lbl_mask)
		if n_pixels <= max_remove_size:
			label_map[lbl_mask] = 0 # set too-small regions to background
		else:
			kept_labels.append(i)
	
	label_map, n_labels = relabel(label_map, n_labels, kept_labels)
	return label_map, n_labels