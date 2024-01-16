


from __future__ import annotations

import os
from pathlib import Path
import glob

example_scripts_dir = Path(__file__).parent
example_data_dir = example_scripts_dir.parent / "example_data"


example_fits_file = example_data_dir / "test_rebin.fits"
example_standard_star_file = example_data_dir / "test_standard_star.fits"


def get_amateur_data_set_directory(index : int):
	return example_data_dir / "amateur_data" / f"set_{index}"

def get_amateur_data_set(index : int):
	return glob.glob(str(example_data_dir / "amateur_data" / f"set_{index}" / "*.tif"))