"""
Contains classes and functions that aid compatibility with the ultranest package
"""
import sys
import json
from typing import ParamSpec, TypeVar, Callable, Any
from pathlib import Path

import numpy as np

import plot_helper
import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

from optimise_compat import PriorParamSet

import ultranest

T = TypeVar('T')
P = ParamSpec('P')

def fitting_function_factory(
		reactive_nested_sampler_kwargs : dict[str,Any] = {}, 
		sampler_run_kwargs : dict[str,Any] = {}
	) -> Callable[[PriorParamSet, Callable[...,float], list[str]|tuple[str], list[str]|tuple[str]], [...]]:
	
	reactive_nested_sampler_kwargs_defaults = dict(
		resume = 'subfolder',
		run_num = 0
	)
	reactive_nested_sampler_kwargs_defaults.update(reactive_nested_sampler_kwargs)
	
	reactive_nested_sampler_kwargs = reactive_nested_sampler_kwargs_defaults
	
	sampler_run_kwargs_defaults = dict(
		max_iters=2000, #500, # 2000,
		max_ncalls=10000, #5000
		frac_remain=1E-2,
		Lepsilon = 1E-1,
		min_num_live_points=200, #20, #80
		cluster_num_live_points=40, #1, #40
		dlogz=100,
		min_ess=200, #1, #40
		update_interval_volume_fraction=0.99, #0.8
		max_num_improvement_loops=3,
		widen_before_initial_plateau_num_warn = 1.5*200, #*min_live_points,
		widen_before_initial_plateau_num_max = 2*200 #*min_live_points
	)
	
	sampler_run_kwargs_defaults.update(sampler_run_kwargs)
	sampler_run_kwargs = sampler_run_kwargs_defaults
	
	
	
	_lgr.debug(f'{reactive_nested_sampler_kwargs=}')
	
	if 'log_dir' not in reactive_nested_sampler_kwargs:
		raise RuntimeError(f'Need to pass "log_dir" as one of "reactive_nested_sampler_kwargs" to tell ultranest where to store results')
	
	
	def fitting_function(params, objective_function, var_param_name_order, const_param_name_order): # should return fitted parameters
		sampler = ultranest.ReactiveNestedSampler(
			var_param_name_order, 
			objective_function,
			params.get_linear_transform_to_domain(var_param_name_order, (0,1)),
			**reactive_nested_sampler_kwargs
		)
		
		final_result = None
		
		for result in sampler.run_iter(**sampler_run_kwargs):
			sampler.print_results()
			sampler.plot()
			final_result = result
	
		return final_result['maximum_likelihood']['point']
		
	return fitting_function

class UltranestResultSet:
	
	metadata_file : str = 'result_set_metadata.json'
	
	def __init__(self, result_set_directory : Path | str):
		self.directory = Path(result_set_directory)
		self.metadata = dict()
		self.metadata_path = self.directory / self.metadata_file
		if self.metadata_path.exists():
			self.load_metadata()
	
	def __repr__(self):
		return f'UltransetResultSet({self.directory.absolute()})'
	
	def load_metadata(self):
		with open(self.metadata_path, 'r') as f:
			self.metadata.update(json.load(f))
	
	def save_metadata(self, make_parent_dirs=True):
		if make_parent_dirs:
			self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
		with open(self.metadata_path, 'w') as f:
			json.dump(self.metadata, f)

	def clear_metadata(self):
		self.metadata = dict()

	def get_result_data_path(self, idx):
		"""
		Ultranest has "result_set_directory/run[INT]" to hold data for each run
		"""
		return self.directory / f'run{idx}'
		
	def get_result_data_from_path(self, result_data_path : Path):
		fname = result_data_path / 'info' / 'results.json'
		
		with open(fname, 'r') as f:
			rdata = json.load(f)
		
		return {
			'param_names': rdata['paramnames'],
			'best_point' : rdata['maximum_likelihood']['point'],
			'stats' : rdata['posterior']
		}

	def get_run_indices(self) -> tuple[int,...] | list[int,...]:
		run_indices = []
		for p in self.directory.iterdir():
			if p.is_dir() and p.stem.startswith('run'):
				run_indices.append(int(p.stem[3:]))
		return run_indices


	def get_params_vs_run_index(self, metadata_keys_to_include=[], sort_by=None) -> tuple[list[int,...], dict[str,np.array], dict[str,np.array]]:
		"""
		sort_by : None | str = None
			Metadata or parameter name to sort by. If `None`, sort by run index
		"""
		run_indices = self.get_run_indices()
		n_runs_indices = len(run_indices)
		
		metadata_values = {}
		param_values = {}
		
		
		# use numpy arrays
		for i, idx in enumerate(run_indices):
			
			for key in metadata_keys_to_include:
				if key not in metadata_values:
					metadata_values[key] = np.full((n_runs_indices,), np.nan)
				metadata_values[key][i] = self.metadata[k][i] 
		
		
			result = self.get_result_data_from_path(self.get_result_data_path(idx))
			for j, pname in enumerate(result['param_names']):
				if pname not in param_values:
					param_values[pname] = np.full((n_runs_indices,), np.nan)
				param_values[pname][i] = result['best_point'][j]
		
		sort_indices = np.argsort(run_indices if sort_by not in param_values else (param_values[sort_by] if sort_by not in metadata_values else metadata_values[sort_by]))
		metadata_values = dict((k,v[sort_indices]) for k,v in metadata_values.items())
		param_values = dict((k,v[sort_indices]) for k,v in param_values.items())
		
		_lgr.debug(f'{metadata_values=} {param_values=}')
		
		return run_indices, metadata_values, param_values

	def plot_params_vs_run_index(self, x_var=None, show=False, save=True, **kwargs):
		"""
		x_var : str | None
			Variable to display along x-axis. If `None`, uses run index.
		kwargs : dict[str,Any]
			Forwarded to `self.get_params_vs_run_index(...)`
		"""
		run_indices, metadata_values, param_values = self.get_params_vs_run_index(**kwargs)
		
		x_label = 'run index' if x_var is None else x_var
		x_values = run_indices if x_var is None else (param_values[x_var] if (x_var not in metadata_values) else metadata_values[x_var])
		
		fname = f'params_vs_{x_label}.png'
		
		f, a = plot_helper.figure_n_subplots(len(param_values)+len(metadata_values))
		#f.tight_layout(pad=16, w_pad=8, h_pad=4)
		f.set_layout_engine('constrained')
		
		f.suptitle(f'params and metadata vs {x_label}')
		for i, pname in enumerate(param_values):
			a[i].set_title(pname)
			a[i].set_xlabel(x_label)
			a[i].set_ylabel(pname)
			a[i].plot(x_values, param_values[pname], 'bo-')
		
		for i, mname in enumerate(metadata_values):
			a[i].set_title(mname)
			a[i].set_xlabel(x_label)
			a[i].set_ylabel(mname)
			a[i].plot(x_values, metadata_values[mname], 'bo-')
		
		
		plot_helper.output(
			show, 
			None if save is None else self.directory / fname
		)
	
	@property
	def plot_directory(self) -> Path:
		return self.directory / "resultset_plots" 
	
	def plot_results(self, 
			result_providers : list[Callable[tuple[float,...], np.ndarray]], # List of functions that produce results for run "N" 
			ref_data : np.ndarray, 
			show=False, 
			save=True
		):
		log_plot_fname_fmt = 'log_result_{idx}.png'
		linear_plot_fname_fmt = 'linear_result_{idx}.png'
		
		run_indices, metadata_values, param_values = self.get_params_vs_run_index()
		
		# Ensure plot directory exists if we are going to use it
		if save:
			self.plot_directory.mkdir(parents=True, exist_ok=True)
		
		# Loop over run indices and plot all the things we need
		for i, idx in enumerate(run_indices):
			_lgr.debug(f'{i=} {idx=}')
			
			params = tuple(param_values[pname][i] for pname in param_values)
			_lgr.debug(f'{params=}')
			result = result_providers[i](params)
			
			
			data = ref_data[idx]
			log_data = np.log(data)
			log_data[np.isinf(log_data)] = np.nan
			
			
			
			
			# plot log of result vs reference data
			f, a = plot_helper.figure_n_subplots(4)
			f.set_layout_engine('constrained')
			f.suptitle(f'log results {idx=}')
			vmin, vmax = np.nanmin(log_data), np.nanmax(log_data)
			
			a[0].set_title(f'log data [{vmin}, {vmax}]')
			a[0].imshow(log_data, vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'log result [{vmin}, {vmax}]')
			a[1].imshow(np.log(result), vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'log residual [{vmin}, {vmax}]')
			a[2].imshow(np.log(data-result), vmin=vmin, vmax=vmax)
			
			log_abs_residual = np.log(np.abs(data-result))
			a[3].set_title(f'log abs residual [{np.nanmin(log_abs_residual)}, {np.nanmax(log_abs_residual)}]')
			a[3].imshow(log_abs_residual)
			
			plot_helper.output(
				show, 
				None if save is None else self.plot_directory / log_plot_fname_fmt.format(idx=idx)
			)
			
			
			# plot result vs reference data
			f, a = plot_helper.figure_n_subplots(4)
			f.set_layout_engine('constrained')
			f.suptitle(f'linear results {idx=}')
			
			vmin, vmax = np.nanmin(data), np.nanmax(data)
			
			a[0].set_title(f'data [{vmin}, {vmax}]')
			a[0].imshow(data, vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'result [{vmin}, {vmax}]')
			a[1].imshow(result, vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'residual [{vmin}, {vmax}]')
			a[2].imshow(data-result, vmin=vmin, vmax=vmax)
			
			frac_residual = np.abs(data-result)/data
			fr_sorted = np.sort(frac_residual.flatten())
			vmin=fr_sorted[fr_sorted.size//4]
			vmax = fr_sorted[3*fr_sorted.size//4]
			a[3].set_title(f'frac residual [{vmin}, {vmax}]')
			a[3].imshow(frac_residual, vmin=vmin, vmax=vmax)
			
			
			
			plot_helper.output(
				show, 
				None if save is None else self.plot_directory / linear_plot_fname_fmt.format(idx=idx)
			)
