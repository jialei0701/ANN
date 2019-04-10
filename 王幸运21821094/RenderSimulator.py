#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.autograd
import math
import functools
import hdf5_dataset


import math

import PIL
from PIL import ImageTk
import numpy as np

import random
import matplotlib.pyplot as plt
import config


class SC_log1p3(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_factor = 1

    def forward(self, input):

        num_channels = input.size(1)
        end = min(3, num_channels)



        a = input[:, 0:end, :, :]
        a = a.clamp(min=0, max=1e5)

        a = (a/self.scale_factor).log1p()*self.scale_factor

        if num_channels <= 3:
            return a
        else:

            depth = input[:, end:end+1, :, :]
            depth = depth*150+40
            depth = depth.log1p()/2
            b = input[:, end+1:, :, :]
            return torch.cat([a,depth, b], 1)

    def revert(self, input):
        num_channels = input.size(1)
        assert num_channels <= 3

        input = (torch.exp(input/self.scale_factor) - 1)*self.scale_factor
        return input



class NetRenderMulti3F(Function):

    @staticmethod
    def simple( target_spp, input_grnd, input0):
        target_spp = target_spp.clamp(min=1)

        alpha = torch.sqrt(1/target_spp)

        alpha = alpha.expand_as(input_grnd)

        x = (input_grnd*(1 - alpha) +  input0*alpha)

        return x

    @staticmethod
    def calc_render(guide, input_alpha, alpha, input_bases):
        guide_left = guide.clone()


        result = input_alpha * alpha

        for power in range(len(input_bases)):
            remainder = guide_left.remainder(2)

            result = result + remainder * (input_bases[power] * (2 ** power))
            guide_left -= remainder

            guide_left = guide_left.div(2)

        result = result / (guide + alpha)

        return result

    @staticmethod
    def forward(ctx, spp_map, *inputs):
        assert torch.is_tensor(spp_map)

        spp_map_super_orig = spp_map


        input_alpha = inputs[1]
        inputs_bases = [inputs[0]] + list(inputs[2:-1])
        inputs_ground = inputs[-1]

        # Maximum spp we can support
        max_spp = 2 ** (len(inputs_bases)) - 1.1


        spp_map = (spp_map).clamp(min=1)
        spp_map_original = spp_map.clamp(min=1)
        spp_map = spp_map.clamp(min=1, max=max_spp)
        spp_map_floor = spp_map.floor()
        if False:
            spp_map = spp_map.floor()

        alpha = spp_map - spp_map_floor

        # Combine components to get desired spp
        result = NetRenderMulti3F.calc_render(spp_map_floor, input_alpha, alpha, inputs_bases)
        if True:
            bigger = (spp_map_original > spp_map).float()
            result = (1 - bigger) * result + bigger * NetRenderMulti3F.simple(spp_map_original / max_spp, inputs_ground, result)


        ctx.save_for_backward(inputs_ground, result, spp_map_super_orig)

        return result


    @staticmethod
    def backward(ctx, dL_dout):
        inputs_ground, result, spp_map_super_orig = ctx.saved_tensors

        spp_map_original = torch.round(spp_map_super_orig).clamp(min=1)

        # Calculate derivative
        dS_dn = (inputs_ground - result) / spp_map_original
        dL_din = torch.sum(torch.mul(dL_dout ,Variable(dS_dn)), dim=1, keepdim=True)

        return tuple([dL_din] + [None]*7)



# Calculates renders and it's derivative
# Input: - sampling map
#        - 1 spp
#        - 1 spp
#        - 2 spp
#        - 4 spp
#        - ....
#        - Ground truth image

class NetRenderMulti3(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        return NetRenderMulti3F.apply(*inputs)







