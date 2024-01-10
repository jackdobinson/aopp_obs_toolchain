"""
Functions that perform checks on variables, they should return true when the check is satisfied
"""

from typing import Any


def listlike_is_identical(test_list, example_list) -> bool:
	check_ok = True
	check_ok &= len(test_list) == len(example_list)
	for test_item, example_item in zip(test_list, example_list):
		check_ok &= test_item == example_item
	return check_ok