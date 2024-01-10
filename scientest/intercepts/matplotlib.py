"""
Very hacky way to get matplotlib to write plots to the test folder instead of wherever it would normally do it.

"""
import os

if os.environ['DISPLAY'] is None or os.environ['DISPLAY'] == '':
	raise RuntimeError('DISPLAY environment variable not set, if you are using WSL have you started an X-server? If not, start one and restart the terminal.')

from pathlib import Path

import matplotlib.pyplot as plt

import scientest

# TODO: Ideally I would like to make all output go to a "<PATH>/test_output/test_<DATE>/" folder.
# Make matplotlib.show() function not block
plt._old_show = plt.show
plt._old_savefig = plt.savefig

def matplotlib_plt_show_intercept(*args, **kwargs):
	scientest.test_output_dir.ensure_dir()
	
	# generate a attribute name to store a counter in
	counter_attr_str = 'mpl_show_intercept_counter'
	
	n = scientest.test_output_dir.__dict__.get(counter_attr_str, 0)
	scientest.test_output_dir.__dict__[counter_attr_str] = n+1
	
	if (len(plt.get_fignums()) == 0):
		_lgr.warn('plt.show(...) called, but no figure is ready to be displayed')
		return
		
	plot_file_name = f"show_plot_{n}.png"
	
	plt._old_savefig(scientest.test_output_dir / plot_file_name)

def matplotlib_savefig_intercept(fname, **kwargs):
	scientest.test_output_dir.ensure_dir()
	
	# generate a attribute name to store a counter in
	counter_attr_str = 'mpl_show_intercept_counter'
	
	n = scientest.test_output_dir.__dict__.get(counter_attr_str, 0)
	scientest.test_output_dir.__dict__[counter_attr_str] = n+1
	
	if (len(plt.get_fignums()) == 0):
		_lgr.warn('plt.show(...) called, but no figure is ready to be displayed')
		return
	
	plot_file_name = f"{fname}_plot_{n}.png"
	
	plt._old_savefig(scientest.test_output_dir / plot_file_name)



plt.show = matplotlib_plt_show_intercept
plt.savefig = matplotlib_savefig_intercept
