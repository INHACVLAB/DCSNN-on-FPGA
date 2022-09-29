import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time 
import global_v as glv

import numpy as np

class SGDBP(torch.autograd.Function): 
    scale = 3  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        ctx.save_for_backward(inputs)
        out = torch.zeros_like(inputs)
        out[torch.where(out.ge(0))] = 1.0
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        inputs = torch.tensor(ctx.saved_tensors)
        grad_input = grad_delta.clone()
        grad = grad_input / (SGDBP.scale * torch.abs(inputs) + 1.0) ** 2
        return grad, None, None