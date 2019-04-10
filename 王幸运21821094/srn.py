#!/usr/bin/env python3

import numpy as np
import torch.nn as nn

import torch.utils.data
import PIL.Image
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets

import argparse
import os
import denoiser

import random
import sys
import special_losses




def load_module(to_net, from_state, flexiable={}, new_params={}):
    flexiable = set(flexiable)
    new_params = set(new_params)
    to_state = to_net.state_dict()



    missing = (set(to_state.keys()) - set(from_state.keys())) - new_params

    missing = [x for x in missing if not any(x.startswith(y) for y in new_params)]

    if len(missing) > 0:
        raise Exception('Missing keys: {}"'.format(missing))

    extra = (set(from_state.keys()) - set(to_state.keys()))
    if len(extra) > 0:
        raise Exception('Extra keys: {}"'.format(extra))


    for from_name, from_param in from_state.items():
        if from_name not in to_state:
            raise Exception("Extra key")

        assert not  isinstance(from_param, torch.nn.parameter.Parameter)

        if to_state[from_name].shape == from_param.shape:
            to_state[from_name].copy_(from_param)

        elif from_name in flexiable:
            to_size = np.array(to_state[from_name].shape)
            from_size = np.array(from_param.shape)

            copy_size = np.minimum(to_size, from_size)

            copy_size =  [slice(0,x) for x in copy_size]
            #to_state[from_name][:, copy_size[1], :, :].copy_(from_param)

            print("WARNING!!! Different size. Skipping!!!!!")

            pass
        elif to_state[from_name].shape != from_param.shape:

            #raise Exception("Different Size {}".format(from_name))
            print("WARNING!!! Different size. Skipping!!!!!")
        else:
            raise Exception("Extra key {}".format(from_name))


def load(f):
    sys.modules['__main__'] = sys.modules[__name__]
    x =  torch.load(f)
    return x


