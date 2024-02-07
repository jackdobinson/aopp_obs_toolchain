#!/usr/bin/python3
"""
Implementation of Singular Spectrum Analysis.

See https://arxiv.org/pdf/1309.5050.pdf for details.
"""
from typing import Any, TypeVar, TypeVarTuple, ParamSpec, Generic, Literal
import weakref


import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.gridspec
import plot_helper
import py_svd as py_svd


T = TypeVar('T')
N = TypeVar('N',bound=int)
M = TypeVar('M',bound=int)





Array_1D = np.ndarray[[N],T]
Array_2D = np.ndarray[[N,M],T]


Grouping_Mode_Literals = Literal['elementary'] | Literal['pairs'] | Literal['pairs_after_first'] | Literal['similar_eigenvalues'] | Literal['blocks_of_n']


#%%
# TODO:
# * Create low-memory mode by only bothering with u and v which create a square s-matrix,
#   can also try truncating those matrices to only include the first N terms
class SSA:
	def __init__(self, 
			a : Array_1D | Array_2D, 
			w_shape : None | int | tuple[N] | tuple[N,M] = None, 
			svd_strategy: Literal['numpy'] | Literal['eigval'] = 'eigval', 
			rev_mapping : Literal['fft'] | Literal['direct'] ='fft', 
			grouping : dict[str, Any] = {'mode':'elementary'}, 
			n_eigen_values : None | int = None
		):
		"""
		Set up initial values of useful constants
		
		# ARGUMENTS #
			
			a : [nx] | [nx,ny]
				Array to operate on, should be 1D or 2D. 
			
			w_shape : None | int | [wx] | [wx, wy]
				Window shape to use for SSA, no array is actually created
				from this shape, it's used as indices to loops etc. If not given,
				will use a.shape//4 as window size
			
			svd_strategy : 'eigval' | 'numpy'
				How should we calculate single value decomposition, 'eigval' is
				fastest and uses least memory so it's normally the best choice.
			
			rev_mapping : 'fft' | 'direct'
				How should we reverse the embedding? 'fft' is fastest so normally best.
			
			grouping : dict[str, Any]
				How should we group the tarjectory matricies? Values other than
				'elementary' do not give exact results. Some modes require extra arguments
				detailed below:
				
				## Required Keys ##
					'mode' : 'elementary' 
								| 'pairs' 
								| 'pairs_after_first' 
								| 'similar_eigenvalues' 
								| 'blocks_of_n'
						Required to set the mode, default is 'elementary'. Values other than
						'elementary' do not give exact results. Some values require extra
						keys.
					
					### Required for 'mode'='similar_eigenvalues' ###
						'tolerance' : float
							Maximum fractional difference that two eigenvalues can have
							if their eigenvectors are to be grouped into one eigenvector.
					
					### Required for 'mode'='blocks_of_n' ###
						'n' : int
							Number of eigenvectors to be grouped together in each group.
							
			n_eigen_values : None | int
				Number of eigenvalues/vectors that should be used in calculations. `None` means
				to use all of them.
		
			
		# RETURNS #
		
			Nothing, but the object will hold the single spectrum analysis of the
			array 'a' in self.X_ssa
			
		"""
		
		_lgr.info('in SSA, initialising attributes')
		
		self.a = a
		_lgr.debug(f'{self.a.shape=}')
		
		self.grouping = grouping
		_lgr.debug(f'{self.grouping=}')
		
		self.ndim = len(a.shape)
		
		# get nx, ny, ... from input array shape
		self.n = (*a.shape,)
		assert len(self.n) == self.ndim, f"{a.shape=} should have {self.ndim=} entries"
		# self.N is the size of the input array
		self.N = np.prod(self.n)
		
		# self.l is the shape of the window
		self.l = (tuple(nx//4 for nx in self.n) if w_shape is None 
			else ((w_shape for nx in self.n) if type(w_shape) is int
				else w_shape
			)
		)
		assert len(self.l) == self.ndim, f"{self.l=} should have {self.ndim=} entries"
		# self.L is the size of the window
		self.L = np.prod(self.l)
		
		# self.k is the difference in shape between the window and input array
		self.k = tuple(nx - lx + 1 for nx, lx in zip(self.n, self.l))
		assert len(self.k) == self.ndim, f"{self.k=} should have {self.ndim=} entries"
		# self.K is the size of the difference in shape between the window and input array
		self.K = np.prod(self.k)

		# Perfom vector embedding of input data
		self.X = self.embed(self.a)
		
		
		
		# calculate singular-value-decomposition of self.X
		match svd_strategy:
			case 'numpy':
				# get svd of trajectory matrix
				self.u, self.s, self.v_star = np.linalg.svd(self.X)
				# make sure we have the full eigenvalue matrix, not just the diagonal
				self.s = py_svd.rect_diag(self.s, (self.L,self.K))
			
			case 'eigval':
				# this strategy is faster
				
				self.u, self.s, self.v_star = self.eigval_svd(self.X, n_eigen_values)
			
			case _:
				raise RuntimeError(f'Unknown singular value decomposition strategy {svd_strategy}')
		
		_lgr.info('decomposing trajectories')
		# get the decomposed trajectories
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star, small = rev_mapping in ('fft',))
		self.d = self.X_decomp.shape[0]
		
		_lgr.info('determining trajectory groupings')
		# determine optimal grouping
		self.grouping = self.get_grouping(**self.grouping)
		self.m = len(self.grouping)
		
		_lgr.info('grouping trajectories')
		# group trajectory components
		self.X_g, self.u_g, self.v_star_g, self.s_g = self.group()
		
		_lgr.info('reversing mapping')
		match rev_mapping:
			case 'fft':
				self.X_ssa = self.quasi_hankelisation()
			case 'direct':
				self.X_ssa = self.reverse_mapping()
			case _:
				raise ValueError(f'Unknown reverse mapping {repr(rev_mapping)}')
		
		_lgr.info('Single spectrum analysis complete')
		return

	@staticmethod
	def eigval_svd(X : Array_2D, n_eigen_values : None | int = None):
		"""
		Calculate single value decomposition from eigenvalues
		"""
		# L = full size of window
		# N = full size of input data
		# K = N - L + 1
		# E = number of eigen values, at most = L
		# X is shape (L,K)
		# @ is matrix multiplication
		# X @ X.T is shape (L,L)
		# evals is shape (E,), self.u is shape (L,E)
		# v is shape (K,L)@(L,E) -> (K,E)
		evals, u = np.linalg.eig(X @ X.T)
		# get eigenvectors and eigen values into decending order
		evals_decending_idxs = np.argsort(evals)[::-1][:n_eigen_values]
		evals = evals[evals_decending_idxs]
		u = u.T[evals_decending_idxs].T
		v = X.T @ (u / np.sqrt(evals))
		v_star = v.T
		s = np.sqrt(np.diag(evals))
		return u, s, v_star

	def embed(self, a : Array_1D | Array_2D):
		"""
		Peform embedding of our data, a, into the trajectory matrix, self.X.
		"""
		X = np.zeros((self.L, self.K))
		a_slices= tuple(slice(_k) for _k in self.k)

		a_it = np.nditer(
			a[a_slices], 
			flags=['f_index', 'multi_index'], 
			order='F'
		)
		for item in a_it:
			idx, coord = a_it.index, a_it.multi_index
			slices = tuple(slice(_k, _k+_l) for _k, _l in zip(coord, self.l))
			X[:, idx] = vectorise_mat(a[slices])
			
		return X
		
	
	def get_grouping(self, 
			mode : Grouping_Mode_Literals = 'elementary', 
			**kwargs
		):
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
			
			case 'pairs':
				grouping = [[_x,_x+1] for _x in range(0,self.d-1,2)]
				return(grouping)
			
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
		"""
		Perform the actual grouping
		"""
		# self.d - number of matrices the trajectory matrix has been decomposed into
		# self.m - number of matrices we will have after grouping the decomposed matrices
		
		# if we have as many groups as we have decomposed elements, 
		# then we didn't actually group any, so return the decomposed elements
		if self.m == self.d:
			return(self.X_decomp, self.u, self.v_star, self.s)
			
		# Otherwise we should actually group the matrices as we said we would
		_lgr.debug(f'{self.grouping=}')
		_lgr.debug(f'{self.X_decomp.shape=}')
		_lgr.debug(f'{self.u.shape=}')
		_lgr.debug(f'{self.v_star.shape=}')
		_lgr.debug(f'{self.s.shape=}')
		
		
		X_g = np.zeros((self.m,*self.X_decomp.shape[1:]))
		u_g = np.zeros((*self.u.shape[1:], self.m))
		v_star_g = np.zeros((self.m,*self.v_star.shape[1:]))
		s_g = np.zeros((self.m,self.m))
		
		for I, idxs in enumerate(self.grouping):
			# For the I^{th} group of matrices,
			# sum up the singular values. We will add members of the
			# group in proportion to their contribution to the singular value
			ss = np.sum(self.s[idxs,idxs]) 
			
			# Add group members together
			for j in idxs:
				s_fac = self.s[j,j]/ss # add components in porportion
				X_g[I] += self.X_decomp[j]
				u_g[:,I] += self.u[:,j]
				v_star_g[I,:] += self.v_star[j,:]
				s_g[I,I] += self.s[j,j]*s_fac
				
		return(X_g, u_g, v_star_g, s_g)
	
	def reverse_mapping(self) -> np.ndarray :
		"""		
		Reverses the embedding map. This version is slow but more accurate, use quasi_hankelisation instead in most applications
		"""
		
		E = np.zeros((*self.n,))
		T_E = np.zeros((self.L, self.K))
		X_ssa = np.zeros((self.m,*self.n))
		
		_lgr.debug(f'{E.shape=} {T_E.shape=} {self.X.shape=} {self.X_g.shape=} {X_ssa.shape=}')
		
		
		_last_coord_0 = -1
		X_ssa_iter = X_ssa.flat
		idx, coord = X_ssa_iter.index, X_ssa_iter.coords
		for item in X_ssa_iter:
			E[coord[1:]] = 1
			T_E[...] = self.embed(E)
			X_ssa[coord] = frobenius_inner_prod(self.X_g[coord[0]], T_E)/(frobenius_norm(T_E)**2)
			
			if coord[0] != _last_coord_0:
				_lgr.debug(f'Reversing mapping of group {coord[0]}/{self.X_g.shape[0]} ...')
				_last_coord_0 = coord[0]
			
			
			E[coord[1:]] = 0
			
			# Fix for unfortunate numpy bug
			idx, coord = X_ssa_iter.index, X_ssa_iter.coords
		
		_lgr.debug(f'Reversing mapping finished.')
		
		return X_ssa
	
	@staticmethod
	def diagsums(a, b, mode='full', fac=None):
		"""
		This is practically the same as convolution using FFT
		"""
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
		"""
		Reverses the embedding map, faster than the direct method but less accurate. Is almost always good enough.
		"""
		X_ssa = np.zeros((self.m, *self.n))
		X_dash = np.zeros(self.n)
		
		order = 'F' # order to reshape arrays with
		
		W = self.diagsums(np.ones(self.k), np.ones(self.l), mode='full')
		for j in range(self.m):
			X_dash = self.diagsums(
				np.squeeze(unvectorise_mat(self.u_g[:,j], self.l[:-1])),
				np.squeeze(unvectorise_mat(self.v_star_g[j,:], self.k[:-1])),
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
		f1, a1 = plot_helper.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Eigenvectors of X (of {self.u.shape[0]})')
		ax_iter=iter(a1)
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'i = {i} eigenval = {self.s[i,i]:07.2E}')
			plot_helper.remove_axes_ticks_and_labels(ax)
			ax.imshow(np.reshape(self.u[i,:],(self.lx,self.ly)).T)
		return
	
	def plot_factorvectors(self, n_max=None):
		flip_ravel = lambda x: np.reshape(x.ravel(order='F'), x.shape)
		n = min(self.v_star.shape[0], n_max if n_max is not None else self.v_star.shape[0])
		f1, a1 = plot_helper.figure_n_subplots(n)
		a1=a1.flatten()
		f1.suptitle(f'First {n} Factorvectors of X (of {self.v_star.shape[0]})')
		ax_iter = iter(a1)
		for j in range(n):
			ax=next(ax_iter)
			ax.set_title(f'j = {j}')
			plot_helper.remove_axes_ticks_and_labels(ax)
			ax.imshow(flip_ravel(np.reshape(self.v_star[j,:],(self.kx,self.ky)).T))
			
	def plot_trajectory_decomp(self, n_max=None):	
		# Plot components of image decomposition
		n = min(self.X_decomp.shape[0], n_max if n_max is not None else self.X_decomp.shape[0])
		f1, a1 = plot_helper.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix components X_i [M = sum(X_i)]')
		ax_iter = iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			plot_helper.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_decomp[i], origin='lower', aspect='auto')
		return
		
	def plot_trajectory_groups(self, n_max=None):
		# plot elements of X_g
		n = min(self.X_g.shape[0], n_max if n_max is not None else self.X_g.shape[0])
		f1, a1 = plot_helper.figure_n_subplots(n)
		a1 = a1.ravel()
		f1.suptitle('Trajectory matrix groups X_g [X = sum(X_g_i)]')
		ax_iter=iter(a1)
		for i in range(n):
			ax = next(ax_iter)
			ax.set_title(f'i = {i}', y=0.9)
			plot_helper.remove_axes_ticks_and_labels(ax)
			ax.imshow(self.X_g[i], origin='lower', aspect='auto')
		return
		
	def plot_ssa(self, n=4, noise_estimate=None):
		import estimate_noise
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
		noise_estimate = noise_estimate if noise_estimate is not None else estimate_noise.corners_standard_deviation(self.a) 
	
		reconstruction = lambda x=None: np.sum(self.X_ssa[:x], axis=0)
		residual = lambda x = None: self.a - reconstruction(x)
		residual_log_likelihood = lambda x = None : -0.5*np.log(np.nansum((residual(x)/(np.nan_to_num(self.a) + noise_estimate))**2))

	
		print(f'{residual()=}')
		print(f'{np.sum(residual())=}')
	
		# plot SSA of image
		mpl.rcParams['lines.linewidth'] = 1
		mpl.rcParams['font.size'] = 8
		mpl.rcParams['lines.markersize'] = 2
		
		gridspec = mpl.gridspec.GridSpec(4,1)
		fig = plt.gcf()
		fig.set(figwidth=12, figheight=8)
		
		
		# Plot summary plots at the top
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
		
		ax = next(ax_iter)
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
		
		
		# Plot individual component plots at the bottom
		f1 = fig.add_subfigure(mpl.gridspec.SubplotSpec(gridspec, 1,3))
		f1, a1 = plot_helper.figure_n_subplots(3*n_component_plots, figure=f1, sp_kwargs={'gridspec_kw':{'top':0.85, 'hspace':1}})
		a1=a1.flatten()
		
		ax_iter=iter(a1)
		
		
		for i in n:
			if i >= self.X_ssa.shape[0]: 
				continue
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
		
		for j in n:
			if j >= self.X_ssa.shape[0]: 
				continue
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
			
		for j in n:
			if j >= self.X_ssa.shape[0]: 
				continue
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
			
	
		return
			





	
def vectorise_mat(a : np.ndarray) -> np.ndarray:
	return(a.ravel(order='F'))

def unvectorise_mat(a : np.ndarray, m : tuple[int,]) -> np.ndarray:
	return(np.reshape(a, (*m,-1), order='F'))


def frobenius_norm(A : np.ndarray) -> np.ndarray:
	"""
	Root of the sum of the elementwise squares
	"""
	return(np.sqrt(np.sum(A*A)))

def frobenius_inner_prod(A : np.ndarray,B : np.ndarray) -> np.ndarray:
	"""
	Sum of the elementwise multiplication
	"""
	return(np.sum(A*B))

	

if __name__=='__main__':
	import sys
	
	n_set = [0, 5, 50, 90]
	dataset = []
	
	if len(sys.argv) > 1:
		for item in sys.argv[1:]:
			if item.endswith('.tif'):
				import PIL
				with PIL.Image.open(item) as image:
					dataset.append((item,np.array(image)[160:440, 350:630]))
				
		for data_name, data2d in dataset:
			_lgr.info(f'TESTING: 2d ssa with {data_name} example data')
			_lgr.info(f'{data2d.shape=}')
			window_size = tuple(s//10 for s in data2d.shape)
			_data = data2d.astype(np.float64)
			ssa2d = SSA(
				_data, 
				window_size,
				rev_mapping='fft',
				svd_strategy='eigval', # uses less memory and is faster
				#svd_strategy='numpy', # uses more memory and is slower
				#grouping={'mode':'elementary'}
				grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
			)
			ssa2d.plot_ssa(n_set)
			plt.show()
	
	
