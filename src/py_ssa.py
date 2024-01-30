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
import utilities as ut
import utilities.np 
import utilities.plt
import py_svd as py_svd
import typing

class SSA:
	def __init__(self, a, w_shape=None, svd_strategy='numpy', rev_mapping='fft', grouping_method='elementary', x_axis=None):
		self.x_axis = x_axis if x_axis is not None else np.arange(0, a.size)
		self.grouping_method = grouping_method
		
		self.a = a
		self.nx = self.a.size
		self.lx = self.nx//4 if w_shape is None else w_shape
		self.kx = self.nx - self.lx + 1

		self.X = self.embed(self.a)
		
		if svd_strategy == 'numpy':
			# numpy way
			# get svd of trajectory matrix
			self.u, self.s, self.v_star = np.linalg.svd(self.X)
			# make sure we have the full eigenvalue matrix, not just the diagonal
			self.s = py_svd.rect_diag(self.s, (self.lx,self.kx))
		
		if svd_strategy == 'eigval':
			# this strategy is faster
			S = self.X @ self.X.T
			s, self.u = np.linalg.eig(S)
			v = (self.X.T @ self.u)/np.sqrt(s)
			self.v_star = np.zeros((self.kx,self.kx))
			self.v_star[:self.lx,:self.kx] = v.T
			self.s = np.zeros((self.lx, self.kx))
			self.s[:self.u.shape[0], :self.u.shape[1]] = np.sqrt(np.diag(s))
		
		self.X_decomp = py_svd.decompose_to_matricies(self.u,self.s,self.v_star)
		self.d = self.X_decomp.shape[0]
		
		self.grouping =  self.get_grouping(grouping_method)
		
		self.group()
		
		if rev_mapping=='fft':
			self.reverse_mapping_fft()
		else:
			self.reverse_mapping()
		
		self.X_reconstructed = np.sum(self.X_ssa, axis=0)
		
		return

	def embed(self, a):
		X = np.zeros((self.lx, self.kx))
		for k in range(self.kx):
				X[:, k] = a[k:k+self.lx]
		return(X)
	
	def get_grouping(self, grouping_method='elementary'):
		# no-op for now
		if grouping_method == 'elementary':
			grouping = [[_x] for _x in range(0,self.d,1)]
			return(grouping)
		# test grouping op
		elif grouping_method == 'pairs':
			grouping = [[_x,_x+1] for _x in range(0,self.d,2)]
			return(grouping)
		# test better grouping op
		elif grouping_method == 'pairs_after_first':
			z = 1-self.d%2
			grouping = [
				np.array([0]), 
				*list(zip(*np.stack([np.arange(1,self.d-z,2),np.arange(2,self.d,2)]))),
				np.arange(self.d-z, self.d)
			]
			return(grouping)
	
		else:
			raise NotImplementedError('Grouping method "{grouping_method}" not defined/implemented')
	
	def group(self):
		# set up constants
		self.m = len(self.grouping)
		#self.m, self.I_max = self.grouping.shape
		# don't group if we don't have to
		
		s_dash = np.zeros((self.kx,self.kx))
		z = min(self.lx,self.kx)
		s_dash[:z,:z] = self.s[:z,:z]
		
		
		if np.all(self.grouping == np.arange(0,self.X_decomp.shape[0])[:,None]):
			#print('DEBUGGING: no op grouping')
			self.X_g = self.X_decomp
			self.u_g = self.u
			self.v_star_g = self.v_star
			self.s_g = self.s
			return
		
		
		#print('DEBUGGING: pay attention to groups')
		self.X_g = np.zeros((self.m, self.lx, self.kx))
		self.u_g = np.zeros((self.lx, self.m))
		self.v_star_g = np.zeros((self.m, self.kx))
		self.s_g = np.zeros((self.m,self.m))
		for _m, I in enumerate(self.grouping):
			ss = np.sum(self.s[I,I])
			for j in I:
				s_fac = self.s[j,j]/ss
				self.X_g[_m] += self.X_decomp[j]
				self.s_g[_m,_m] += self.s[j,j]
				self.u_g[:,_m] += self.u[:,j]*s_fac
				self.v_star_g[_m,:] += self.v_star[j,:]*s_fac

		return
	
	
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
	
	
	def plot_svd(self, recomp_n=None):
		if recomp_n is None:
			recomp_n = self.X_decomp.shape[0]
		py_svd.plot(self.u, self.s, self.v_star, self.X, self.X_decomp, recomp_n=recomp_n)
		return
	
	
	def plot_ssa(self, n_max=6):
		# Matplotlib parameters
		mpl.rcParams['lines.linewidth'] = 1
		mpl.rcParams['font.size'] = 8
		mpl.rcParams['lines.markersize'] = 2

		n = min(self.m, n_max if n_max is not None else self.m)
		f1, a1 = ut.plt.figure_n_subplots(n+4)
		a1=a1.flatten()
		f1.suptitle(f'First {n} ssa components of data (of {self.m})')
		ax_iter=iter(a1)
		
		
		ax = next(ax_iter)
		ax.set_title('Original and Reconstruction')
		ax.plot(self.x_axis, self.a, label='data',ls='-',marker='.')
		ax.plot(self.x_axis, self.X_reconstructed, label='sum(X_ssa)', ls='--', alpha=0.8)
		ax.axvspan(self.x_axis[0], self.x_axis[self.lx], 0, 1, color='tab:gray', alpha=0.1, label='Window Size')
		original_ylim = ax.get_ylim()
		ax.legend()
		
		ax = next(ax_iter)
		ax.set_title('Residual')
		ax.plot(self.x_axis, self.a - self.X_reconstructed, ls='-', marker='')
		
		ax = next(ax_iter)
		ax.set_title('Ratio (original/reconstruction)')
		ax.plot(self.x_axis, self.a/self.X_reconstructed, ls='-', marker='')
		
		ax = next(ax_iter)
		ax.set_title('Eigenvalues of components')
		ii = list(range(self.m))
		ax.plot(ii, [self.s_g[i,i] for i in ii], label='eigenvalues')
		ax.set_ylabel('eigenvalue')
		ax.set_xlabel('component number')
		ax.set_yscale('log')
		
		for i in range(n):
			ax=next(ax_iter)
			ax.set_title(f'X_ssa[{i}]\neigenvalue {self.s_g[i,i]:07.2E}')
			#ut.plt.remove_axes_ticks_and_labels(ax)
			ax.plot(self.x_axis, self.X_ssa[i], color='tab:blue', label='component')
			ax.set_ylabel('component')
			
			original_yrange = max(original_ylim) - min(original_ylim)
			y_mean = np.nanmean(self.X_ssa[i])
			new_ylim = (y_mean - original_yrange/2, y_mean + original_yrange/2)
			ax.set_ylim(new_ylim)
			
			ax2 = ax.twinx()
			ax2.plot(self.x_axis, np.sum(self.X_ssa[:i+1], axis=0), color='tab:orange', label='component sum', ls='--')
			ax2.set_ylim(original_ylim)
			ax2.set_ylabel('component Sum')
			
			ut.plt.set_legend(ax, ax2)
		return(f1,a1)
			



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
			"""
			s, self.u = np.linalg.eig(self.X@self.X.T)
			v = np.zeros((self.kxky, self.kxky))
			v[:self.kxky, :self.lxly] = (self.X.T@self.u / np.sqrt(s))
			self.v_star = v.T
			self.s = np.zeros((self.lxly, self.kxky))
			self.s[:self.u.shape[0], :self.u.shape[1]] =np.sqrt(np.diag(s))
			"""
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
		#self.X_decomp = np.einsum('ii,jk,kl->ijl', self.s, self.u, self.v_star)
		#self.X_decomp = np.diag(self.s)[:,None,None]*(self.u @ self.v_star)[None,:,:]
		
		#self.X_decomp = np.diag(self.s)[:,None,None]*(self.u.T[:,:,None] @ self.v_star[:,None,:])
		self.d = self.X_decomp.shape[0]
		
		_lgr.info('determinng trajectory groupings')
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
					_lgr.debug(f'{abs(last_ev - this_ev) / last_ev=}')
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
		if self.m==self.d:
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
			#ss = np.sum(np.sqrt(self.s[idxs,idxs]))
			for j in idxs:
				s_fac = 1
				s_fac = self.s[j,j]/ss # add components in porportion
				#s_fac = np.sqrt(self.s[j,j])/ss # add components in porportion
				#_lgr.debug(f'{X_g[I]=} {self.X_decomp[j]=}')
				X_g[I] += self.X_decomp[j]
				#u_g[:,I] += self.u[:,j]*s_fac
				u_g[:,I] += self.u[:,j]
				#v_star_g[I,:] += self.v_star[j,:]*s_fac
				v_star_g[I,:] += self.v_star[j,:]
				#s_g[I,I] = max(s_g[I,I],self.s[j,j])
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
	
	ssa = SSA(data1d, svd_strategy='numpy', rev_mapping='fft', grouping_method='elementary')
	ssa.plot_ssa()
	plt.show()
	
	
	
	dataset = []
	n_set = [0, 4, 12, 24]
	
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
		ssa2d = SSA2D(
			data2d.astype(np.float64), 
			window_size, 
			svd_strategy='eigval', # uses less memory and is faster
			# svd_strategy='numpy', # uses more memory and is slower
			#grouping={'mode':'elementary'}
			grouping={'mode':'similar_eigenvalues', 'tolerance':0.01}
		)
		ssa2d.plot_ssa(n_set)
		plt.show()
	
	
	
