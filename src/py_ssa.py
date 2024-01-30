#!/usr/bin/python3
"""
Implementation of Singular Spectrum Analysis.

See https://arxiv.org/pdf/1309.5050.pdf for details.
"""
from typing import Any

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.gridspec
import utilities as ut
import utilities.np 
import utilities.plt
import py_svd as py_svd
import typing

class SSA:
	def __init__(self, 
			a, 
			w_shape=None, 
			svd_strategy='numpy', 
			rev_mapping='fft', 
			grouping : dict[str, Any] = {'mode':'elementary'}, 
			n_eigen_values=None
		):
		self.grouping = grouping
		
		self.a = a
		
		self.n = (*a.shape,)
		self.nx = self.n[0]
		
		self.l = (tuple(nx//4 for nx in self.n) if w_shape is None 
			else ((w_shape for nx in self.n) if type(w_shape) is int
				else w_shape
			)
		)
		self.L = np.product(self.l)
		self.lx = self.l[0]
		
		self.k = tuple(nx - lx + 1 for nx, lx in zip(self.n, self.l))
		self.K = np.product(self.k)
		self.kx = self.k[0]

		self.X = self.embed(self.a)
		
		if svd_strategy == 'numpy':
			# numpy way
			# get svd of trajectory matrix
			self.u, self.s, self.v_star = np.linalg.svd(self.X)
			# make sure we have the full eigenvalue matrix, not just the diagonal
			self.s = py_svd.rect_diag(self.s, (self.lx,self.kx))
		
		if svd_strategy == 'eigval':
			# this strategy is faster
			evals, self.u = np.linalg.eig(self.X @ self.X.T)
			# get eigenvectors and eigen values into decending order
			evals_decending_idxs = np.argsort(evals)[::-1][:n_eigen_values]
			evals = evals[evals_decending_idxs]
			self.u = self.u.T[evals_decending_idxs].T
			v = self.X.T @ (self.u / np.sqrt(evals))
			self.v_star = v.T
			self.s = np.sqrt(np.diag(evals))
		
		_lgr.info('decomposing trajectories')
		# get the decomposed trajectories
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star)
		self.d = self.X_decomp.shape[0]
		
		_lgr.info('determining trajectory groupings')
		# determine optimal grouping
		self.grouping = self.get_grouping(**self.grouping)
		self.m = len(self.grouping)
		
		_lgr.info('grouping trajectories')
		# group trajectory components
		self.X_g, self.u_g, self.v_star_g, self.s_g = self.group()
		
		_lgr.info('reversing mapping')
		if rev_mapping == 'fft':
			self.X_ssa = self.quasi_hankelisation()
		elif rev_mapping=='direct':
			self.X_ssa = self.reverse_mapping()
		else:
			raise ValueError(f'Unknown rev_mapping {repr(rev_mapping)}')
		
		_lgr.info('Single spectrum analysis complete')
		return

	def embed(self, a):
		ndim = len(self.k)
		k_steps = np.cumprod((1,*self.k[:-1])).astype(int)[::-1]
		_lgr.debug(f'{ndim=} {k_steps=} {self.L=} {self.K=}')
		X = np.zeros((self.L, self.K))
		_lgr.debug(f'{X.shape=}')
		_lgr.debug(f'{self.n=} {self.k=} {self.l=}')
		kn = np.zeros((len(self.n)),dtype=int)
		for k in range(self.K):
			for i in range(len(kn)):
				kn[i] = (k-np.sum(kn[:i]*k_steps[:i]))//k_steps[i]
			slices = tuple(slice(_k, _k+_l) for _k, _l in zip(kn[::-1], self.l))
			#X[:, k] = vectorise_mat(a[k:k+self.lx])
			X[:, k] = vectorise_mat(a[slices])
		return(X)
	
	def get_grouping(self, mode='elementary', **kwargs):
		"""
		Define how the eigentriples (evals, evecs, fvecs) should be grouped.
		"""
		def ensure_grouping_parameter(x): 
			assert x in kwargs, f'Grouping {mode=} requires grouping parameter "{x}", which is not found in {kwargs}'
			return kwargs[x]
		
		# simplest method, we don't bother grouping.
		match mode:
			case 'elementary':
				grouping = [[i] for i in range(self.d)]
				return(grouping)
			# test grouping op
			case 'pairs':
				grouping = [[_x,_x+1] for _x in range(0,self.d-1,2)]
				return(grouping)
			# test better grouping op
			case 'pairs_after_first':
				z = 1-self.d%2
				grouping = [
					np.array([0]), 
					*list(zip(*np.stack([np.arange(1,self.d-z,2),np.arange(2,self.d,2)]))),
					np.arange(self.d-z, self.d)
				]
				return(grouping)
			case 'similar_eigenvalues':
				tolerance = ensure_grouping_parameter('tolerance')
				grouping = []
				last_ev = 0
				for i in range(self.d):
					this_ev = self.s[i,i]
					if abs(last_ev - this_ev) / last_ev < tolerance:
						grouping[-1].append(i)
					else:
						grouping.append([i])
						last_ev = this_ev
				return grouping
			case 'blocks_of_n':
				n = ensure_grouping_parameter('n')
				grouping = [list(range(j,j+kwargs['n'] if j+kwargs['n'] <= self.d else self.d)) for j in range(0,self.d,kwargs['n'])]
				return grouping
			case _:
				raise NotImplementedError(f'Unknown grouping {mode=}, {kwargs} for {self}')

	def group(self):
		# don't bother grouping if we have nothing to group
		if self.m == self.d:
			# if we have as many groups as we have decomposed elements, then we didn't actually group any, so return decomposition.
			return(self.X_decomp, self.u, self.v_star, self.s)
		_lgr.debug(f'{self.grouping=}')
		_lgr.debug(f'{self.X_decomp.shape=}')
		_lgr.debug(f'{self.u.shape=}')
		_lgr.debug(f'{self.v_star.shape=}')
		_lgr.debug(f'{self.s.shape=}')
		
		
		n_groups = len(self.grouping)
		X_g = np.zeros((n_groups,*self.X_decomp.shape[1:]))
		u_g = np.zeros((*self.u.shape[1:], n_groups))
		v_star_g = np.zeros((n_groups,*self.v_star.shape[1:]))
		s_g = np.zeros((n_groups,n_groups))
		
		for I, idxs in enumerate(self.grouping):
			ss = np.sum(self.s[idxs,idxs])
			for j in idxs:
				s_fac = self.s[j,j]/ss # add components in porportion
				X_g[I] += self.X_decomp[j]
				u_g[:,I] += self.u[:,j]
				v_star_g[I,:] += self.v_star[j,:]
				s_g[I,I] += self.s[j,j]*s_fac
				
		return(X_g, u_g, v_star_g, s_g)
	
	
	def reverse_mapping_fft(self):
		#print('DEBUGGING: IN "reverse_mapping_fft()"')
		self.X_ssa = np.zeros((self.m, self.nx))
		#self.X_dash = np.zeros_like(self.X_ssa)
		
		
		# extend u and v_star to size nx with zeros
		u_dash = np.zeros((self.nx,))
		v_dash = np.zeros((self.nx,))
		
		self.u_dash = np.zeros((self.m,self.nx,))
		self.v_dash = np.zeros((self.m,self.nx,))
		
		l_star = min(self.lx, self.kx)
		W = np.convolve(np.ones((self.kx)), np.ones((self.lx,)), mode='full')
		"""
		W = np.zeros((self.nx,))
		W[l_star-1:-(l_star-1)] = l_star
		W[:l_star-1] = np.arange(1,l_star)
		W[-(l_star-1):] = np.arange(1,l_star)[::-1]
		"""
		#self.W = W
		
		for g in range(self.m):
			u_dash[:self.lx] = self.u_g[:,g]
			v_dash[:self.kx] = self.v_star_g[g,:]
			s_dash = self.s[g,g]
			self.u_dash[g] = u_dash
			self.v_dash[g] = v_dash
			u_dash_fft = np.fft.fft(u_dash)
			v_dash_fft = np.fft.fft(v_dash)
			X_dash = np.fft.ifft(u_dash_fft*v_dash_fft*s_dash)
			#self.X_dash[g] = X_dash
			self.X_ssa[g] = (X_dash/W).real # only need the real part of this
		return
	
	def reverse_mapping(self):
		#print('DEBUGGING: IN "reverse_mapping()"')
		E = np.zeros((self.nx,))
		T_E = np.zeros((self.lx, self.kx))
		self.X_ssa = np.zeros((self.m, self.nx))
		
		#self.f_ip_vec = np.zeros_like(self.X_ssa)
		#self.f_norm_sq_vec = np.zeros_like(self.X_ssa)
		#self.T_E_vec = np.zeros((self.m, self.nx, self.lx, self.kx))
		for g in range(self.m):
			print(g)
			for n in range(self.nx):
				#print('\t',k)
				E[n] = 1
				T_E[...] = self.embed(E)
				self.T_E_vec[g,n,...] = T_E
				f_ip = frobenius_inner_prod(self.X_g[g], T_E)
				f_norm_sq = frobenius_norm(T_E)**2
				#self.f_ip_vec[g,n] = f_ip
				#self.f_norm_sq_vec[g,n] = f_norm_sq
				#self.X_ssa[g,n] = f_ip/f_norm_sq
				E[n] = 0
		return
	
	@staticmethod
	def diagsums(a, b, mode='full', fac=None):
		"""
		This is practically the same as convolution using FFT
		"""
		_lgr.debug(f'{a.shape=} {b.shape=}')
		shape_full = [s1+s2-1 for s1,s2 in zip(a.shape,b.shape)]
		fac = fac if fac is not None else 1
		a_fft = np.fft.fftn(a, shape_full)
		b_fft = np.fft.fftn(b, shape_full)
		conv = np.fft.ifftn(a_fft*b_fft*fac, shape_full).real
		if mode=='full':
			return(conv)
		if mode=='same':
			slices = tuple([slice((sf-sa)//2, (sa-sf)//2) for sa, sf in zip(a.shape, shape_full)])
			return(conv[slices])
		if mode=='valid':
			slices = tuple([slice((sa-sb)//2, (sa-sb)//2) for sa, sb in zip(a.shape, b.shape)])
			return(conv[slices])
	
	def quasi_hankelisation(self):
		# reverse embedding
		X_ssa = np.zeros((self.m, *self.n))
		X_dash = np.zeros(self.n)
		
		order = 'F' # order to reshape arrays with
		
		W = self.diagsums(np.ones(self.k), np.ones(self.l), mode='full')
		for j in range(self.m):
			X_dash = self.diagsums(
				np.squeeze(unvectorise_mat(self.u_g[:,j], self.l[0])),
				np.squeeze(unvectorise_mat(self.v_star_g[j,:], self.k[0])),
				mode='full',
				fac=self.s_g[j,j]
			)
			X_ssa[j] = X_dash/W
			
		return(X_ssa)
	
	def plot_all(self, n_max=36):
		self.plot_svd()
		self.plot_eigenvectors(n_max)
		self.plot_factorvectors(n_max)
		self.plot_trajectory_decomp(n_max)
		self.plot_trajectory_groups(n_max)
		self.plot_ssa(n_max)
		return

	def plot_svd(self, recomp_n=None):
		if recomp_n is None:
			recomp_n = self.X_decomp.shape[0]
		py_svd.plot(self.u, self.s, self.v_star, self.X, self.X_decomp, recomp_n=recomp_n)
		return
	
	def plot_eigenvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		# plot eigenvectors and factor vectors
		n = min(self.u.shape[0], n_max if n_max is not None else self.u.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Eigenvectors of X (of {self.u.shape[0]})')
		ax_iter=iter(a1)
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'i = {i} eigenval = {self.s[i,i]:07.2E}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(np.reshape(self.u[i,:],(self.lx,self.ly)).T)
		return
	
	def plot_factorvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		n = min(self.v_star.shape[0], n_max if n_max is not None else self.v_star.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Factorvectors of X (of {self.v_star.shape[0]})')
		ax_iter = iter(a1)
		for j in range(n):
			ax=next(ax_iter)
			ax.set_title(f'j = {j}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(flip_ravel(np.reshape(self.v_star[j,:],(self.kx,self.ky)).T))
			
	def plot_trajectory_decomp(self, n_max=None):	
		# Plot components of image decomposition
		n = min(self.X_decomp.shape[0], n_max if n_max is not None else self.X_decomp.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix components X_i [M = sum(X_i)]')
		ax_iter = iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_decomp[i], origin='lower', aspect='auto')
		return
		
		
		
	def plot_trajectory_groups(self, n_max=None):
		# plot elements of X_g
		n = min(self.X_g.shape[0], n_max if n_max is not None else self.X_g.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix groups X_g [X = sum(X_g_i)]')
		ax_iter=iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_g[i], origin='lower', aspect='auto')
		return
		
	def plot_ssa(self, n=4, noise_estimate=None):
	
		_lgr.debug(f'Plotting SSA data')
	
		match len(self.n):
			case 1:
				def plot_callable(ax, data, title_fmt, **kwargs):
					vmin, vmax = np.nanmin(data), np.nanmax(data)
					ax.set_title(title_fmt.format(data_limits=f'[{vmin:07.2E}, {vmax:07.2E}]'))
					result = ax.plot(np.arange(len(data)), data, ls='-', marker='')[0]
					return result
			case 2:
				def plot_callable(ax, data, title_fmt, **kwargs): 
					vmin, vmax = np.nanmin(data), np.nanmax(data)
					ax.set_title(title_fmt.format(data_limits=f'[{vmin:07.2E}, {vmax:07.2E}]'))
					result = ax.imshow(data, vmin=vmin, vmax=vmax)
					ax.xaxis.set_visible(False)
					ax.yaxis.set_visible(False)
					return result
			case _:
				raise RuntimeError(f'No plotting callable for {len(self.n)}d SSA')
	
		n = list(range(n)) if type(n) is int else n
		n = list(range(self.X_ssa.shape[0])) if n is None else n
		n = [x if x < self.X_ssa.shape[0] else self.X_ssa.shape[0]-1 for x in n]
	
		n_component_plots = len(n)
		noise_estimate = noise_estimate if noise_estimate is not None else np.std(self.a[tuple(slice(0,s//10) for s in self.a.shape)])
	
		reconstruction = lambda x=None: np.sum(self.X_ssa[:x], axis=0)
		residual = lambda x = None: self.a - reconstruction(x)
		residual_log_likelihood = lambda x = None : -0.5*np.log(np.sum((residual(x)/(self.a + noise_estimate))**2))
	
		print(f'{residual()=}')
		print(f'{np.sum(residual())=}')
	
		# plot SSA of image
		mpl.rcParams['lines.linewidth'] = 1
		mpl.rcParams['font.size'] = 8
		mpl.rcParams['lines.markersize'] = 2
		
		gridspec = mpl.gridspec.GridSpec(4,1)
		fig = plt.gcf()
		fig.set(figwidth=12, figheight=8)
		
		
		f0 = fig.add_subfigure(mpl.gridspec.SubplotSpec(gridspec, 0))
		a0 = f0.subplots(1,4, squeeze=False, gridspec_kw={'top':0.5})
		a0 = a0.flatten()
		f0.suptitle(f'{n_component_plots} ssa images of obs (of {self.X_ssa.shape[0]})')
		ax_iter=iter(a0)
		
		
		ax = next(ax_iter)
		plot_callable(ax,
			self.a,
			'Original\n{data_limits}'
		)
		window_rect = mpl.patches.Rectangle((0,0),*(self.l if len(self.l) == 2 else (self.l[0],1)),color='red',fill=False,ls='-')
		ax.add_patch(window_rect)
		
		ax = next(ax_iter)
		plot_callable(ax,
			reconstruction(),
			'Reconstruction\n{data_limits}'
		)
		#ax.set_title(f'Reconstruction\nclim [{o_clim[0]:07.2E} {o_clim[1]:07.2E}]')
		#plot_callable(ax, reconstruction(), vmin=o_clim[0], vmax=o_clim[1])
		#ut.plt.remove_axes_ticks_and_labels(ax)
		
		ax = next(ax_iter)
		#im = plot_callable(ax,residual())
		#ax.set_title(f'Residual {residual_log_likelihood():07.2E}\nmean {np.mean(residual()):07.2E}\nclim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]')
		#ut.plt.remove_axes_ticks_and_labels(ax)
		plot_callable(ax,
			residual(),
			'\n'.join((
				'Residual\n{data_limits}',
				f'mean {np.mean(residual()):07.2E}',
				f'log_likelihood {residual_log_likelihood():07.2E}'
			))
		)
		
		ax = next(ax_iter)
		ax.set_title('Eigenvalues')
		ax.plot(range(self.m), np.diag(self.s_g))
		ax.plot(n, np.diag(self.s_g)[n], color='red', marker='.', linestyle='none', label='plotted components')
		ax.set_yscale('log')
		ax.set_ylabel('Eigenvalue')
		ax.set_xlabel('Component number')
		ax.legend()
		
		
		
		
		f1 = fig.add_subfigure(mpl.gridspec.SubplotSpec(gridspec, 1,3))
		f1, a1 = ut.plt.figure_n_subplots(3*n_component_plots, figure=f1, sp_kwargs={'gridspec_kw':{'top':0.85, 'hspace':1}})
		a1=a1.flatten()
		
		ax_iter=iter(a1)
		
		
		for i in n:
			ax=next(ax_iter)
			plot_callable(ax,
				self.X_ssa[i],
				'\n'.join((
					f'X_ssa[{i}]',
					'lim {data_limits}',
					f'eigenvalue {self.s_g[i,i]:07.2E}',
					f'eigen_frac {self.s_g[i,i]/np.sum(self.s_g):07.2E}',
					f'sig_frac {np.sqrt(np.sum(self.X_ssa[i]**2)/(np.sum(self.a**2))):07.2E}',
				))
			)
			#im = ax.imshow(self.X_ssa[i])
			#title = '\n'.join((
			#	f'X_ssa[{i}]',
			#	f'eigenvalue {self.s_g[i,i]:07.2E}',
			#	f'eigen_frac {self.s_g[i,i]/np.sum(self.s_g):07.2E}',
			#	f'sig_frac {np.sqrt(np.sum(self.X_ssa[i]**2)/(np.sum(self.a**2))):07.2E}',
			#	f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			#))
			
			#ax.set_title(title)
			#ut.plt.remove_axes_ticks_and_labels(ax)
		
		for j in n:
			i = j+1
			ax=next(ax_iter)
			_data = reconstruction(i)
			plot_callable(ax,
				_data,
				'\n'.join((
					f'sum(X_ssa[:{i}])',
					'lim {data_limits}',
					f'eigen_frac {np.sum(np.diag(self.s_g)[:i])/np.sum(self.s_g):07.2E}',
					f'sig_remain {1 - np.sqrt(np.sum(_data**2)/(np.sum(self.a**2))):07.2E}',
				))
			)
			
			
			#im = plot_callable(ax,_data)
			#title = '\n'.join((
			#	f'sum(X_ssa[:{i}])',
			#	f'eigen_frac {np.sum(np.diag(self.s_g)[:i])/np.sum(self.s_g):07.2E}',
			#	f'sig_remain {1 - np.sqrt(np.sum(_data**2)/(np.sum(self.a**2))):07.2E}',
			#	f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			#))
			#
			#ax.set_title(title)
			#ut.plt.remove_axes_ticks_and_labels(ax)
		
		for j in n:
			i = j+1
			ax=next(ax_iter)
			plot_callable(ax,
				residual(i),
				'\n'.join((
					f'residual sum(X_ssa[:{i}])',
					'lim {data_limits}',
					f'mean {np.mean(residual(i)):07.2E}',
					f'log_likelihood {residual_log_likelihood(i):07.2E}'
				))
			)
			
			
			#im = plot_callable(ax,residual(i))
			#title = '\n'.join((
			#	f'residual sum(X_ssa[:{i}]) {residual_log_likelihood(i):07.2E}',
			#	f'mean {np.mean(residual(i)):07.2E}',
			#	f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			#))
			#
			#ax.set_title(title)
			#ut.plt.remove_axes_ticks_and_labels(ax)
		
		
		return
			



#%%
# TODO:
# * Create low-memory mode by only bothering with u and v which create a square s-matrix,
#   can also try truncating those matrices to only include the first N terms
class SSA2D:	
	def __init__(self, 
			a, 
			w_shape=None, 
			svd_strategy='eigval', 
			rev_mapping='fft', 
			grouping : dict[str,Any] = {'mode':'elementary'}, 
			n_eigen_values=None
		):
		"""
		Set up initial values of useful constants
		
		# ARGUMENTS #
		a [nx,ny]
			Array to operate on, should be a type that numba recognises. If in
			doubt, use np.float64.
		w_shape [2]
			<int,int> Window shape to use for SSA, no array is actually created
			from this shape, it's used as indices to loops etc. If not given,
			will use a.shape//4 as window size
		svd_strategy : 'eigval' | 'numpy'
			How should we calculate single value decomposition, 'eigval' is
			fastest
		rev_mapping : 'fft' | 'direct'
			How should we reverse the embedding? 'fft' is fastest
		grouping : dict[str, Any] {'mode':'elementary' | 'pairs' | 'pairs_after_first' | 'similar_eigenvalues' | 'blocks_of_n', **}
			How should we group the tarjectory matricies? Values other than
			'elementary' do not give exact results
			
		# RETURNS #
			Nothing, but the object will hold the single spectrum analysis of the
			array 'a' in self.X_ssa
			
		"""
		_lgr.info('in SSA2D, initialising attributes')
		self.a = a
		self.nx, self.ny = a.shape
		if w_shape is None:
			self.lx, self.ly = self.nx//4, self.ny//4
		else:
			self.lx, self.ly = w_shape
		self.grouping = grouping
			
		self.kx, self.ky = self.nx-self.lx+1, self.ny-self.ly+1
		self.lxly = self.lx*self.ly
		self.kxky = self.kx*self.ky
		self.nxny = self.nx*self.ny
		
		self.X = self.embed(self.a)
		
		_lgr.info('computing single value decomposition')
		if svd_strategy == 'numpy':
			# get svd of trajectory matrix
			self.u, self.s, self.v_star = np.linalg.svd(self.X)
			# make sure we have the full eigenvalue matrix, not just the diagonal
			self.s = py_svd.rect_diag(self.s, (self.lxly,self.kxky))
		
		elif svd_strategy=='eigval':
			# this version is much faster
			# L = full size of window
			# N = full size of input data
			# K = N - L + 1
			# E = number of eigen values, at most = L
			# X is shape (L,K)
			# X X.T is shape (L,L)
			# evals is shape (E,), self.u is shape (L,E)
			# v is shape (K,L)@(L,E) -> (K,E)
			evals, self.u = np.linalg.eig(self.X @ self.X.T)
			# get eigenvectors and eigen values into decending order
			evals_decending_idxs = np.argsort(evals)[::-1][:n_eigen_values]
			evals = evals[evals_decending_idxs]
			self.u = self.u.T[evals_decending_idxs].T
			v = self.X.T @ (self.u / np.sqrt(evals))
			self.v_star = v.T
			self.s = np.sqrt(np.diag(evals))
			
		else:
			raise ValueError(f'Unknown svd_strategy {repr(svd_strategy)}')
		
		_lgr.info('decomposing trajectories')
		# get the decomposed trajectories
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star)
		self.d = self.X_decomp.shape[0]
		
		_lgr.info('determining trajectory groupings')
		# determine optimal grouping
		self.grouping = self.get_grouping(**self.grouping)
		self.m = len(self.grouping)
		
		_lgr.info('grouping trajectories')
		# group trajectory components
		self.X_g, self.u_g, self.v_star_g, self.s_g = self.group()
		
		_lgr.info('reversing mapping')
		if rev_mapping == 'fft':
			self.X_ssa = self.quasi_hankelisation()
		elif rev_mapping=='direct':
			self.X_ssa = self.reverse_mapping()
		else:
			raise ValueError(f'Unknown rev_mapping {repr(rev_mapping)}')
		
		_lgr.info('Single spectrum analysis complete')
		return
		
	def embed(self, a):
		X = np.zeros((self.lxly, self.kxky))
		for k1 in range(self.kx):
			for k2 in range(self.ky):
				X[:,k1+k2*self.kx] = vectorise_mat(a[k1:k1+self.lx, k2:k2+self.ly])
		return(X)
	
	
	
	def get_grouping(self, mode='elementary', **kwargs):
		"""
		Define how the eigentriples (evals, evecs, fvecs) should be grouped.
		"""
		def ensure_grouping_parameter(x): 
			assert x in kwargs, f'Grouping {mode=} requires grouping parameter "{x}", which is not found in {kwargs}'
			return kwargs[x]
		
		# simplest method, we don't bother grouping.
		match mode:
			case 'elementary':
				grouping = [[i] for i in range(self.d)]
				return(grouping)
			# test grouping op
			case 'pairs':
				grouping = [[_x,_x+1] for _x in range(0,self.d-1,2)]
				return(grouping)
			# test better grouping op
			case 'pairs_after_first':
				z = 1-self.d%2
				grouping = [
					np.array([0]), 
					*list(zip(*np.stack([np.arange(1,self.d-z,2),np.arange(2,self.d,2)]))),
					np.arange(self.d-z, self.d)
				]
				return(grouping)
			case 'similar_eigenvalues':
				tolerance = ensure_grouping_parameter('tolerance')
				grouping = []
				last_ev = 0
				for i in range(self.d):
					this_ev = self.s[i,i]
					if abs(last_ev - this_ev) / last_ev < tolerance:
						grouping[-1].append(i)
					else:
						grouping.append([i])
						last_ev = this_ev
				return grouping
			case 'blocks_of_n':
				n = ensure_grouping_parameter('n')
				grouping = [list(range(j,j+kwargs['n'] if j+kwargs['n'] <= self.d else self.d)) for j in range(0,self.d,kwargs['n'])]
				return grouping
			case _:
				raise NotImplementedError(f'Unknown grouping {mode=}, {kwargs} for {self}')

	
	
	def group(self):
		# don't bother grouping if we have nothing to group
		if self.m == self.d:
			# if we have as many groups as we have decomposed elements, then we didn't actually group any, so return decomposition.
			return(self.X_decomp, self.u, self.v_star, self.s)
		_lgr.debug(f'{self.grouping=}')
		_lgr.debug(f'{self.X_decomp.shape=}')
		_lgr.debug(f'{self.u.shape=}')
		_lgr.debug(f'{self.v_star.shape=}')
		_lgr.debug(f'{self.s.shape=}')
		
		
		n_groups = len(self.grouping)
		X_g = np.zeros((n_groups,*self.X_decomp.shape[1:]))
		u_g = np.zeros((*self.u.shape[1:], n_groups))
		v_star_g = np.zeros((n_groups,*self.v_star.shape[1:]))
		s_g = np.zeros((n_groups,n_groups))
		
		for I, idxs in enumerate(self.grouping):
			ss = np.sum(self.s[idxs,idxs])
			for j in idxs:
				s_fac = self.s[j,j]/ss # add components in porportion
				X_g[I] += self.X_decomp[j]
				u_g[:,I] += self.u[:,j]
				v_star_g[I,:] += self.v_star[j,:]
				s_g[I,I] += self.s[j,j]*s_fac
				
		return(X_g, u_g, v_star_g, s_g)
		
	def reverse_mapping(self) -> np.ndarray :
		"""
		* faster without JIT
		
		Reverses the embedding map
		shape should be (nx,ny)
		"""
		
		E = np.zeros((self.nx,self.ny))
		T_E = np.zeros((self.lxly, self.kxky))
		X_ssa = np.zeros((self.m,self.nx,self.ny))
		
		for g in range(self.m):
			for k in range(self.nx):
				for l in range(self.ny):
					E[k,l] = 1
					T_E[...] = self.embed(E)
					X_ssa[g,k,l] = frobenius_inner_prod(self.X_g[g], T_E)/(frobenius_norm(T_E)**2)
					E[k,l] = 0
		return(X_ssa)
	
	@staticmethod
	def diagsums(a, b, mode='full', fac=None):
		"""
		This is practically the same as convolution using FFT
		"""
		shape_full = [s1+s2-1 for s1,s2 in zip(a.shape,b.shape)]
		fac = fac if fac is not None else 1
		a_fft = np.fft.fft2(a, shape_full)
		b_fft = np.fft.fft2(b, shape_full)
		conv = np.fft.ifft2(a_fft*b_fft*fac, shape_full).real
		if mode=='full':
			return(conv)
		if mode=='same':
			slices = tuple([slice((sf-sa)//2, (sa-sf)//2) for sa, sf in zip(a.shape, shape_full)])
			return(conv[slices])
		if mode=='valid':
			slices = tuple([slice((sa-sb)//2, (sa-sb)//2) for sa, sb in zip(a.shape, b.shape)])
			return(conv[slices])
	
	def quasi_hankelisation(self):
		# reverse embedding
		X_ssa = np.zeros((self.m, self.nx, self.ny))
		X_dash = np.zeros((self.nx,self.ny))
		
		order = 'F' # order to reshape arrays with
		
		W = self.diagsums(np.ones((self.kx,self.ky)), np.ones((self.lx,self.ly)), mode='full')
		for j in range(self.m):
			X_dash = self.diagsums(
				unvectorise_mat(self.u_g[:,j], self.lx),
				unvectorise_mat(self.v_star_g[j,:], self.kx),
				mode='full',
				fac=self.s_g[j,j]
			)
			X_ssa[j] = X_dash/W
			
		return(X_ssa)
	
	def plot_all(self, n_max=36):
		self.plot_svd()
		self.plot_eigenvectors(n_max)
		self.plot_factorvectors(n_max)
		self.plot_trajectory_decomp(n_max)
		self.plot_trajectory_groups(n_max)
		self.plot_ssa(n_max)
		return

	def plot_svd(self, recomp_n=None):
		if recomp_n is None:
			recomp_n = self.X_decomp.shape[0]
		py_svd.plot(self.u, self.s, self.v_star, self.X, self.X_decomp, recomp_n=recomp_n)
		return
	
	def plot_eigenvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		# plot eigenvectors and factor vectors
		n = min(self.u.shape[0], n_max if n_max is not None else self.u.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Eigenvectors of X (of {self.u.shape[0]})')
		ax_iter=iter(a1)
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'i = {i} eigenval = {self.s[i,i]:07.2E}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(np.reshape(self.u[i,:],(self.lx,self.ly)).T)
		return
	
	def plot_factorvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		n = min(self.v_star.shape[0], n_max if n_max is not None else self.v_star.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Factorvectors of X (of {self.v_star.shape[0]})')
		ax_iter = iter(a1)
		for j in range(n):
			ax=next(ax_iter)
			ax.set_title(f'j = {j}')
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(flip_ravel(np.reshape(self.v_star[j,:],(self.kx,self.ky)).T))
			
	def plot_trajectory_decomp(self, n_max=None):	
		# Plot components of image decomposition
		n = min(self.X_decomp.shape[0], n_max if n_max is not None else self.X_decomp.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix components X_i [M = sum(X_i)]')
		ax_iter = iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_decomp[i], origin='lower', aspect='auto')
		return
		
		
		
	def plot_trajectory_groups(self, n_max=None):
		# plot elements of X_g
		n = min(self.X_g.shape[0], n_max if n_max is not None else self.X_g.shape[0])
		f1, a1 = ut.plt.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix groups X_g [X = sum(X_g_i)]')
		ax_iter=iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			ut.plt.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_g[i], origin='lower', aspect='auto')
		return
		
	def plot_ssa(self, n=4, noise_estimate=None):
	
		n = list(range(n)) if type(n) is int else n
		n = list(range(self.X_ssa.shape[0])) if n is None else n
		n = [x if x < self.X_ssa.shape[0] else self.X_ssa.shape[0]-1 for x in n]
	
		n_component_plots = len(n)
		noise_estimate = noise_estimate if noise_estimate is not None else np.std(self.a[tuple(slice(0,s//10) for s in self.a.shape)])
	
		reconstruction = lambda x=None: np.sum(self.X_ssa[:x], axis=0)
		residual = lambda x = None: self.a - reconstruction(x)
		residual_log_likelihood = lambda x = None : -0.5*np.log(np.sum((residual(x)/(self.a + noise_estimate))**2))
	
		print(f'{residual()=}')
		print(f'{np.sum(residual())=}')
	
		# plot SSA of image
		mpl.rcParams['lines.linewidth'] = 1
		mpl.rcParams['font.size'] = 8
		mpl.rcParams['lines.markersize'] = 2
		
		f1, a1 = ut.plt.figure_n_subplots(3*n_component_plots+4)
		a1=a1.flatten()
		f1.suptitle(f'{n_component_plots} ssa images of obs (of {self.X_ssa.shape[0]})')
		ax_iter=iter(a1)
		
		ax = next(ax_iter)
		o_clim = (np.nanmin(self.a), np.nanmax(self.a))
		ax.set_title(f'Original\nclim [{o_clim[0]:07.2E} {o_clim[1]:07.2E}]')
		ax.imshow(self.a, vmin=o_clim[0], vmax=o_clim[1])
		ut.plt.remove_axes_ticks_and_labels(ax)
		window_rect = mpl.patches.Rectangle((0,0),self.lx, self.ly,color='red',fill=False,ls='-')
		ax.add_patch(window_rect)
		
		ax = next(ax_iter)
		ax.set_title(f'Reconstruction\nclim [{o_clim[0]:07.2E} {o_clim[1]:07.2E}]')
		ax.imshow(reconstruction(), vmin=o_clim[0], vmax=o_clim[1])
		ut.plt.remove_axes_ticks_and_labels(ax)
		
		ax = next(ax_iter)
		im = ax.imshow(residual())
		ax.set_title(f'Residual {residual_log_likelihood():07.2E}\nmean {np.mean(residual()):07.2E}\nclim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]')
		ut.plt.remove_axes_ticks_and_labels(ax)
		
		ax = next(ax_iter)
		ax.set_title('Eigenvalues')
		ax.plot(range(self.m), np.diag(self.s_g))
		ax.plot(n, np.diag(self.s_g)[n], color='red', marker='.', linestyle='none', label='plotted components')
		ax.set_yscale('log')
		ax.set_ylabel('Eigenvalue')
		ax.set_xlabel('Component number')
		ax.legend()
		
		for i in n:
			ax=next(ax_iter)
			im = ax.imshow(self.X_ssa[i])
			title = '\n'.join((
				f'X_ssa[{i}]',
				f'eigenvalue {self.s_g[i,i]:07.2E}',
				f'eigen_frac {self.s_g[i,i]/np.sum(self.s_g):07.2E}',
				f'sig_frac {np.sqrt(np.sum(self.X_ssa[i]**2)/(np.sum(self.a**2))):07.2E}',
				f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			))
			
			ax.set_title(title)
			ut.plt.remove_axes_ticks_and_labels(ax)
		
		for j in n:
			i = j+1
			ax=next(ax_iter)
			_data = reconstruction(i)
			im = ax.imshow(_data)
			title = '\n'.join((
				f'sum(X_ssa[:{i}])',
				f'eigen_frac {np.sum(np.diag(self.s_g)[:i])/np.sum(self.s_g):07.2E}',
				f'sig_remain {1 - np.sqrt(np.sum(_data**2)/(np.sum(self.a**2))):07.2E}',
				f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			))
			
			ax.set_title(title)
			ut.plt.remove_axes_ticks_and_labels(ax)
		
		for j in n:
			i = j+1
			ax=next(ax_iter)
			im = ax.imshow(residual(i))
			title = '\n'.join((
				f'residual sum(X_ssa[:{i}]) {residual_log_likelihood(i):07.2E}',
				f'mean {np.mean(residual(i)):07.2E}',
				f'clim [{im.get_clim()[0]:07.2E} {im.get_clim()[1]:07.2E}]',
			))
			
			ax.set_title(title)
			ut.plt.remove_axes_ticks_and_labels(ax)
		
		
		return
	
	
def vectorise_mat(a : np.ndarray) -> np.ndarray:
	return(a.ravel(order='F'))

def unvectorise_mat(a, m):
	return(np.reshape(a, (m,a.size//m), order='F'))

def frobenius_norm(A):
	"""
	Root of the sum of the elementwise squares
	"""
	return(np.sqrt(np.sum(A*A)))

def frobenius_inner_prod(A,B):
	"""
	Sum of the elementwise multiplication
	"""
	return(np.sum(A*B))

def mult_by_quasi_hankel(A, b, bx, cxcy):
	"""
	For use with SSA2D, not actually using it right now
	"""
	nx, ny = A.shape
	b_dash = unvectorise(b, bx)
	A_fft = np.fft.fft2(A, (nx,ny))
	b_dash_fft = np.fft.fft2(b_dash, (nx,ny))
	c_dash = np.fft.ifft2(A_fft*np.conjugate(b_dash_fft), (nx,ny)).T
	return
	
#%%
if __name__=='__main__':
	import sys
	
	# get example data in order of desirability
	test_data_type_2d = ('fitscube','fractal','random')
	test_data_type_1d = ('random',)
	
	
	n_set = [0, 4, 12, 24]
	
	# find 1d testing data
	for data_type in test_data_type_1d:
		if data_type == 'random':
			np.random.seed(100)
			n = 1000
			w = 10
			data1d = np.convolve(np.random.random((n,)), np.ones((w,)), mode='same')
		else:
			raiseValueError(f'Unknown type of test data for 1d case {repr(data_type)}')
			
	_lgr.info(f'TESTING: 1d ssa with {data_type} example data')
	
	ssa = SSA(
		data1d, 
		svd_strategy='eigval', 
		rev_mapping='fft', 
		grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
	)
	ssa.plot_ssa(n_set)
	plt.show()
	
	
	
	dataset = []
	
	
	if len(sys.argv) > 1:
		for item in sys.argv[1:]:
			if item.endswith('.tif'):
				import PIL
				with PIL.Image.open(item) as image:
					dataset.append((item,np.array(image)[160:440, 350:630]))
	else:
		# find 2d testing data
		for data_type in test_data_type_2d:
			if data_type == 'fitscube':
				try:
					import fitscube.deconvolve.helpers
					data2d, psf = fitscube.deconvolve.helpers.get_test_data()
					del psf
				except ImportError:
					print('WARNING: Could not import "fitscube.deconvolve.helpers", use builtins instead')
				else:
					break
				
			elif data_type == 'fractal':
				try:
					import PIL
					data2d = np.asarray(PIL.Image.effect_mandelbrot((60,50),(0,0,1,1),100))
				except ImportError:
					print('WARNING: Could not import PIL, using next most desirable example data')
				else:
					break
				
			elif data_type == 'random':
				data2d = np.random.random((60,50))
				
			else:
				raise ValueError(f'Unknown type of test data for 2d case {repr(data_type)}')
		dataset.append((data_type, data2d))
	
	for data_name, data2d in dataset:
		_lgr.info(f'TESTING: 2d ssa with {data_name} example data')
		_lgr.info(f'{data2d.shape=}')
		window_size = tuple(s//10 for s in data2d.shape)
		ssa2d = SSA(
			data2d.astype(np.float64), 
			window_size, 
			svd_strategy='eigval', # uses less memory and is faster
			# svd_strategy='numpy', # uses more memory and is slower
			#grouping={'mode':'elementary'}
			grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
		)
		ssa2d.plot_ssa(n_set)
		plt.show()
	
	
	
