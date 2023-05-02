from torch.optim.optimizer import Optimizer, required
import copy
import pytorch3d.transforms as t
import torch


class SE3Opt(Optimizer):
	def __init__(self, params,lr=required):
		defaults = dict(lr = lr)
		super(SE3Opt, self).__init__(params, defaults)
	def __setstate__(self, state):
		super(SE3Opt, self).__setstate__(state)
	def step(self):
		for group in self.param_groups:
			for v in group['params']:
				if v.grad is None:
					continue
				J = self.getJacobianRightInv(v)
				#dx = v.grad
				dx = -group['lr'] * v.grad.unsqueeze(2).double()
				dx = torch.bmm(J, dx).squeeze(2)				
				v.data.add_(-group['lr'],dx)
	
	@staticmethod
	def getJacobianRightInv(v):
		eps = 1e-6
		h = eps*torch.eye(6,6, dtype=torch.double )
		N = v.size(0)
		v = v.double()
		res = torch.zeros(N,6,6, dtype=torch.double)
		P = t.se3.se3_exp_map(v, eps = eps)
		for j in range(6):
			h_i  = h[j,:].unsqueeze(0)      
			M = t.se3.se3_exp_map(h_i, eps = eps).expand(N, -1, -1)
			Y = torch.bmm(M, P)
			Y_inv = Y.inverse()
			log = t.se3.se3_log_map(Y_inv,eps = eps)
			dH = -(log + v ) / eps
			res[:,:,j] = dH    
		return res
	
		 
	 
	 
	 

class AccSGD(Optimizer):
	r"""Implements the algorithm proposed in https://arxiv.org/pdf/1704.08227.pdf, which is a provably accelerated method 
	for stochastic optimization. This has been employed in https://openreview.net/forum?id=rJTutzbA- for training several 
	deep learning models of practical interest. This code has been implemented by building on the construction of the SGD 
	optimization module found in pytorch codebase.
	Args:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float): learning rate (required)
		kappa (float, optional): ratio of long to short step (default: 1000)
		xi (float, optional): statistical advantage parameter (default: 10)
		smallConst (float, optional): any value <=1 (default: 0.7)
	Example:
		>>> from AccSGD import *
		>>> optimizer = AccSGD(model.parameters(), lr=0.1, kappa = 1000.0, xi = 10.0)
		>>> optimizer.zero_grad()
		>>> loss_fn(model(input), target).backward()
		>>> optimizer.step()
	"""

	def __init__(self, params, lr=required, kappa = 1000.0, xi = 10.0, smallConst = 0.7, weight_decay=0):
		defaults = dict(lr=lr, kappa=kappa, xi=xi, smallConst=smallConst,
						weight_decay=weight_decay)
		super(AccSGD, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(AccSGD, self).__setstate__(state)

	def step(self, closure=None):
		""" Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			weight_decay = group['weight_decay']
			large_lr = (group['lr']*group['kappa'])/(group['smallConst'])
			Alpha = 1.0 - ((group['smallConst']*group['smallConst']*group['xi'])/group['kappa'])
			Beta = 1.0 - Alpha
			zeta = group['smallConst']/(group['smallConst']+Beta)
			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				if weight_decay != 0:
					d_p.add_(weight_decay, p.data)
				param_state = self.state[p]
				if 'momentum_buffer' not in param_state:
					param_state['momentum_buffer'] = copy.deepcopy(p.data)
				buf = param_state['momentum_buffer']
				buf.mul_((1.0/Beta)-1.0)
				buf.add_(-large_lr,d_p)
				buf.add_(p.data)
				buf.mul_(Beta)

				p.data.add_(-group['lr'],d_p)
				p.data.mul_(zeta)
				p.data.add_(1.0-zeta,buf)

		return loss
