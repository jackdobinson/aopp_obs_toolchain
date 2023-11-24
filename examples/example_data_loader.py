import os
from pathlib import Path

example_scripts_dir = Path(__file__).parent
example_data_dir = example_scripts_dir.parent / "example_data"


example_fits_file = example_data_dir / "test_rebin.fits"
example_standard_star_file = example_data_dir / "test_standard_star.fits"
