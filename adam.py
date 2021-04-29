import torch
#from . import _functional as F
#from .optimizer import Optimizer
from torch.optim.optimizer import Optimizer
import math

class myAdam(Optimizer):
    r"""Implements Adam, AdamW with and without amsgrad.
    Code is developed using torch.optim.SGD as a template.
     Args:
         paras (iterable): iterable of parametesr to optimizer or dicts defining parameter groups
         lr (float): learning rate (1e-3)
         weight_decay (float, optional): weight decay = 0 corresponds to Adam (default: 0)
         eps (float, optional): stabilizing factor in the denom
         betas (tuple-float, optional): weights of exponential moving averages (default = (0.9, 0.999))
         amsgrd
    Example:
        >>> optimizer = myAdam(model.parameters(), lr = 0.1, weight_decay = 1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
     """
    def __init__(self, params, lr = 1e-3, weight_decay = 0,
                 eps = 1e-8, betas = (0.9, 0.999), 
                 method = 'AdamW',amsgrad = False):
        # check the validity of the parameter values
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eps < 0:
            raise ValueError("Invalid stabilizing factor: {}".format(eps))
        if not 0 <= betas[0] < 1:
            raise ValueError("Invalid weight for exponential moving averages of the gradient: {}".format(betas[0]))
        if not 0<= betas[1] < 1:
            raise ValueError("Invalid weight for exponential moving averages of the squared gradient: {}".format(betas[1]))
        if not (method == 'Adam' or method == 'AdamW'):
            raise ValueError("Invalid method: {}. Please input either Adam or AdamW!".format(method))
        defaults = dict(lr=lr, weight_decay=weight_decay, eps = eps, 
                        betas = betas, method = method, amsgrad = amsgrad)
        # call constructor in the base class
        super(myAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        # why do I need to setdefault?? and which ones shall i set?
        super(myAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            #params_with_grad = []
            #d_p_list = []
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            lr = group['lr']
            method = group['method']

            for p in group['params']:
                if p.grad is None:
                    continue
                    #params_with_grad.append(p)
                    #d_p_list.append(p.grad)
                grad = p.grad.data # not sure - .data or not
                # not sure - why is that self.state[p] refers to the desired par
                state = self.state[p] # reference assignment 
                eps = group['eps']
                # initialize state 
                if len(state) == 0:
                    state['step'] = 0
                    state['mov_avg_grad'] = torch.zeros_like(p.data)
                    state['mov_avg_sq_grad'] = torch.zeros_like(p.data)
                    if amsgrad: # not sure
                        state['max_avg_sq_grad'] = torch.zeros_like(p.data)
                        
                mov_avg, mov_avg_sq = state['mov_avg_grad'], state['mov_avg_sq_grad']
                betas = group['betas']
                
                # increment the step
                state['step'] += 1
                
                # penalize weights based on intended optimizer
                if method == 'Adam':
                    # update gradient by adding that from L2 penalty
                    #grad.add_(group['weight_decay'], p.data)
                    grad.add_(p.data, alpha = group['weight_decay'])
                elif method == 'AdamW':
                    # update parameter based on weight decay
                    p.mul_(1 - lr * weight_decay)
                else: 
                    ValueError("Invalid method: {}. Please input either Adam or AdamW!".format(method))
                
                # compute bias correction factor
                bias_correction = tuple(1 - i ** state['step'] for i in betas)
                # compute the moving average of the gradient
                #mov_avg.mul_(betas[0]).add_(1 - betas[0], grad)
                mov_avg.mul_(betas[0]).add_(grad, alpha = 1 - betas[0])
                # compute the moving average of the squared gradient 
                mov_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value = 1 - betas[1])
                if amsgrad:
                    # compute the sqrt of the maximum of the moving avg. of the squared grad.
                    max_mov_avg_sq = state['max_avg_sq_grad']
                    torch.max(max_mov_avg_sq, mov_avg_sq, out = max_mov_avg_sq)
                    denom = max_mov_avg_sq.sqrt()
                else:
                    # compute the sqrt of the moving avg. of the squared grad.
                    denom = mov_avg_sq.sqrt()
                # compute bias corrected second moment
                denom.div_(math.sqrt(bias_correction[1])).add_(eps)
                # compute bias corrected first moment
                scaled_mov_avg = mov_avg / bias_correction[0]
                # do gradient descent on the weights
                p.addcdiv_(tensor1 = scaled_mov_avg, tensor2 = denom, value = -lr)
        return loss
 