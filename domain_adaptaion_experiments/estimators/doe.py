import torch  
import torch.nn as nn 
import torch.nn.functional as F
from .utils import FF 
import math 

def compute_negative_ln_prob(Y, mu ,ln_var, pdf):
    var = ln_var.exp()
    if pdf == 'gauss':
        negative_ln_prob = 0.5 * (((Y - mu) ** 2 / var).sum(1).mean() + Y.size(1) * math.log(2 * math.pi) + ln_var.sum(1).mean())
    elif pdf == 'logistic':
        whitened = (Y - mu) / var
        adjust = torch.logsumexp(
            torch.stack([torch.zeros(Y.size()).to(Y.device), -whitened]), 0)
        negative_ln_prob = whitened.sum(1).mean + 2 * adjust.sum(1) + ln_var.sum(1).mean()
    else:
        raise ValueError('Unknown PDF: %s' % (pdf))
    return negative_ln_prob
        