#!/usr/bin/env python3
import utils as ut
import numpy as np
import nemesis.common
import nemesis.read

def cov(runname):
	import matplotlib.pyplot as plt
	"""
	Analyses the information in the <runname>.cov file

	According to the manual, when Trace(K Sa Kt)~Trace(Measurement Error) there should be a good
	balance of constraints. Strangely enough this does not use Sm (the measurement covariance) but
	se1 as the measurement error. Are these not supposed to be related?
	"""
	covd = nemesis.read.cov(runname)

	cmap = ('viridis', 'jet', 'gist_heat')[1]
	
	#print(covd['sa'])
	#print(covd['st'])
	#print(covd['sm'])
	#print(covd['sn'])
	#print(covd['aa'])
	#print(covd['dd'])
	#print(covd['kk'])
	#print(covd['kt'])
	#print(covd['se1'])
	

	KSaKt = np.matmul(np.matmul(covd['kk'], covd['sa']), covd['kt'])
	tr_KSaKt = np.trace(KSaKt)
	tr_KStKt = np.trace(np.matmul(np.matmul(covd['kk'], covd['st']), covd['kt']))
	tr_KSnKt = np.trace(np.matmul(np.matmul(covd['kk'], covd['sn']), covd['kt']))
	tr_Sm = np.trace(covd['sm'])
	tr_Sa = np.trace(covd['sa'])
	tr_St = np.trace(covd['st'])
	tr_Sn = np.trace(covd['sn'])
	diagonals_KSaKt = np.array([x[i] for i,x in enumerate(KSaKt)])
	ratio_sum_se1_tr_KSaKt_per_dof = np.sum(covd['se1'])/(tr_KSaKt*covd['se1'].shape[0])
	residual_sum_se1_tr_KSaKt_per_dof = (tr_KSaKt - np.sum(covd['se1']))/covd['se1'].shape[0]
	#print('tr_Sa {} tr_KSaKt {} tr_Sm {}'.format(tr_Sa, tr_KSaKt, tr_Sm))
	#print('tr_St {} tr_KStKt {} tr_Sm {}'.format(tr_St, tr_KStKt, tr_Sm))
	#print('tr_Sn {} tr_KSnKt {} tr_Sm {}'.format(tr_Sn, tr_KSnKt, tr_Sm))
	print('Sum(se1) {} Tr(K Sa Kt) {} ratio_per_dof {} residual_per_dof {}'.format(
			np.sum(covd['se1']), tr_KSaKt,
			ratio_sum_se1_tr_KSaKt_per_dof,
			residual_sum_se1_tr_KSaKt_per_dof))


	ut.plot_defaults()
	f1 = plt.figure()
	a11 = f1.add_subplot(2,2,1)
	a12 = f1.add_subplot(2,2,2)
	a13 = f1.add_subplot(2,2,3)
	a14 = f1.add_subplot(2,2,4)
	
	cmin = min([np.min(x) for x in (covd['sm'], covd['sa'], covd['st'], covd['sn'])])
	cmax = max([np.max(x) for x in (covd['sm'], covd['sa'], covd['st'], covd['sn'])])
	
	a11.imshow(covd['sa'], origin='lower', cmap=cmap, vmin=cmin, vmax=cmax)
	a11.set_title('Sa - a-priori covariance')

	a12.imshow(covd['sm'], origin='lower', cmap=cmap, vmin=cmin, vmax=cmax)
	a12.set_title('Sm - measurement covariance')	

	a13.imshow(covd['st'], origin='lower', cmap=cmap, vmin=cmin, vmax=cmax)
	a13.set_title('St - total covariance')

	a14.imshow(covd['sn'], origin='lower', cmap=cmap, vmin=cmin, vmax=cmax)
	a14.set_title('Sn - smooth covariance')

	#plt.show()


	f2 = plt.figure()
	a21 = f2.add_subplot(2,2,1)
	a22 = f2.add_subplot(2,2,2)

	a21.imshow(KSaKt, origin='lower', cmap=cmap)
	a21.set_title('K Sa Kt')

	a22.imshow(covd['kk'], origin='lower', cmap=cmap)
	a22.set_title('kk - Kernel (Jacobian)')

	a23 = f2.add_subplot(2,2,3)
	a24 = f2.add_subplot(2,2,4)

	a23.imshow(covd['aa'], origin='lower', cmap=cmap)
	a23.set_title('aa - Averaging Kernel')

	a24.imshow(covd['dd'], origin='lower', cmap=cmap)
	a24.set_title('dd - Contribution Function')


	f3 = plt.figure()
	a31 = f3.add_subplot(1,1,1)
	
	a31.plot(covd['se1'], label='Measurement Errors "se1"')
	a31.plot(diagonals_KSaKt, label='Diagonal elements of "K Sa Kt"')
	a31.set_yscale('log')
	a31.legend()
	a31.set_title('Sum(se1)/[Tr(K Sa Kt)*N]={}\n[Tr(K Sa Kt)-Sum(se1)]/N={}'.format(
					ratio_sum_se1_tr_KSaKt_per_dof, residual_sum_se1_tr_KSaKt_per_dof))

	plt.show()



