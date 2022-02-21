# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:04:51 2020

@author: 11627
"""

import torch
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()
        self._update_u_v()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            # print(w.view(height,-1).data.shape)
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))  #sigma = (u^T) W v
        setattr(self.module, self.name, w / sigma.expand_as(w))  #update W to W_SN

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)  #conv.weight  ( input_channel , output_channel , kernel_w , kernel_h )

        height = w.data.shape[0]  # height = input_channel
        width = w.view(height, -1).data.shape[1] # width =  output_channel x kernel_w x kernel_h

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False) #initiate left singular vector
                                                                     # shape: (input_channel)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)  #initiate right singular vector
                                                                    # shape: (output_channel x kernel_w x kernel_h)
        # print(u.shape)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]  # will add after update
        # add parameter to module
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()  #更新完W后，再调用原模块的forward()
        return self.module.forward(*args)

if __name__ == '__main__':
    conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
