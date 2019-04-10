#!/usr/bin/env python3

from __future__ import print_function

import visualizer # Must go first

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn


from numpy.random import normal
from numpy.linalg import svd
import torch.optim as optim
import config

from buffer_normalizer import BufferNormalizer
from buffer import Buffer
from perm import Perm
from datalayer import DataLayer
import os
from gglue import Gglue
import time
import full_buffer
import socket
import utilities
import matplotlib.pyplot as plt
import torchvision.models
from tensorboardX import SummaryWriter
import RenderSimulator


from collections import OrderedDict
import srn

import torch.cuda
import sys
from srn import *
import hdf5_dataset

import itertools
import math
import special_losses

# hack for pickle
sys.modules['pytorch_simple_nn'] = sys.modules[__name__]

normalizers = {}

standard_normalizers = {
    "texture":  BufferNormalizer(custom_normalize=[.45, .45]),
    "color_in_log": BufferNormalizer(custom_normalize=[0,1]),


    "shadow": BufferNormalizer(custom_normalize=[.05, .37]),
    "norm1":  BufferNormalizer(custom_normalize=[0, .45]),
    "depth":  BufferNormalizer(custom_normalize=[40,150]),

    "color_in_var": BufferNormalizer(custom_normalize=[0, 1])
}


class DualIter:
    def __init__(self, objs):

        self.objs = [(key, val) for key,val in objs.items()]


    def __iter__(self):
        self.objs_iter = [iter(x[1]) for x in self.objs]
        self.next_obj = -1
        return self

    def __next__(self):
        for idx in range(len(self.objs_iter)):
            self.next_obj = (self.next_obj + 1) % len(self.objs_iter)

            try:
                return self.objs_iter[self.next_obj].next(), self.objs[self.next_obj][0], self.next_obj+1 == len(self.objs_iter)
            except StopIteration:
                pass
        raise StopIteration()

    def __len__(self):
        return sum([len(x[1]) for x in self.objs])



if True:
    for key in standard_normalizers.keys():

        normalizers[key] = standard_normalizers[key]

permutator_group = {
    "color": [
        {"COLOR": "_feature_COLOR_1", "COLOR_VAR": "_var_COLOR_1", "TEXTURE": "_feature_TEXTURE_1_X", "TEXTURE_VAR": "_var_TEXTURE_1_X", "REF_COLOR": "_feature_COLOR_1", "IRRAD": "_feature_IRRAD_1", "COLOR_CBF": "_cbf_COLOR_1", "COLOR_SMALL": "_COLOR_small_1"},
        {"COLOR": "_feature_COLOR_2", "COLOR_VAR": "_var_COLOR_2", "TEXTURE": "_feature_TEXTURE_1_Y", "TEXTURE_VAR": "_var_TEXTURE_1_Y", "REF_COLOR": "_feature_COLOR_2", "IRRAD": "_feature_IRRAD_2", "COLOR_CBF": "_cbf_COLOR_2", "COLOR_SMALL": "_COLOR_small_2"},
        {"COLOR": "_feature_COLOR_3", "COLOR_VAR": "_var_COLOR_3", "TEXTURE": "_feature_TEXTURE_1_Z", "TEXTURE_VAR": "_var_TEXTURE_1_Z", "REF_COLOR": "_feature_COLOR_3", "IRRAD": "_feature_IRRAD_3", "COLOR_CBF": "_cbf_COLOR_3", "COLOR_SMALL": "_COLOR_small_2"},
    ],

}




def roundup_tensor(tensor, size_up):
    size = tensor.shape

    new_size = [min(s + (-s) % su, 19200) for s, su in zip(size, size_up)]


    new_tensor = torch.zeros(new_size).type_as(tensor)

    if len(size) == 4:
        new_tensor[0:size[0], 0:size[1], 0:size[2], 0:size[3]] = tensor[0:size[0], 0:size[1], 0:size[2], 0:size[3]]
    else:
        new_tensor[0:size[0], 0:size[1], 0:size[2]] = tensor[0:size[0], 0:size[1], 0:size[2]]

    return new_tensor


def convert_state(net_state):
    new_state_dict = OrderedDict()

    for k, v in net_state.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        else:
            name = k

        new_state_dict[name] = v

    return new_state_dict


def add_text_patch_group(group, num, id):

    for i in range(num):
        group["TEXTURE_PATCH_" + str(i)] = "_feature_TEXTURE_1_"+str(id)+"_part"+str(i)


for i in range(3):
    add_text_patch_group(permutator_group["color"][i], 16, i)



class FPdata:
    def __init__(self, axillary, inputs, grounds, outputs, loss, error, unique_id, vgg_loss=None, adaptive_map_net=None, adaptive_map_ground=None, pass_spp=None, pass_denpoise=None):
        self.axillary = axillary
        self.inputs = inputs
        self.grounds = grounds
        self.outputs = outputs
        self.loss = loss
        self.error = error
        self.unique_id = unique_id

        self.adaptive_map_net = adaptive_map_net
        self.adaptive_map_ground = adaptive_map_ground

        self.vgg_loss = vgg_loss

        self.pass_spp = pass_spp
        self.pass_denpoise = pass_denpoise


class Denoising:
    class NetDaptor:
        def __init__(self, denoiser, net, buffers_in, buffers_out, buffer_auxiliary=None):
            self.denoiser = denoiser
            self.net = net
            self.buffers_in = buffers_in
            self.buffers_in_proxies = [Gglue.SimpleChannel(x) for x in self.buffers_in]
            self.buffers_out = buffers_out
            self.buffer_auxiliary = buffer_auxiliary

    class NetMapBound(nn.Module):
        def __init__(self, scale=1, bias=1):
            super(Denoising.NetMapBound, self).__init__()
            self.scale = scale
            self.bias = bias

        def forward(self, x):

            x = torch.exp(x)

            x_not_norm = x

            orig_shape = x.size()
            x = x.view(orig_shape[0], -1)

            mult  = self.scale / torch.mean(x, 1, keepdim=True)

            for _ in range(3):
                mult = (self.scale) / torch.clamp(torch.round(x * mult.expand_as(x.data)), min=1).mean(1, keepdim=True) * mult

            x = (x * mult.expand_as(x.data))
            x = x.view(orig_shape)


            if False:
                np_mean_x = mult.data.cpu().numpy()

                print("mean_x", np_mean_x)

                if np.isnan(np.sum(np_mean_x)):
                    print("NaN!!!!")
                    exit(-1)

            self.bias = 0


            x = x.clamp(min=1)

            return x, x_not_norm, 0

    class NetCrossBilateral(nn.Module):
        pass

    class NetBig(nn.Module):
        def pool_skip(self, x):
            return self.pool(x), x

        def merge(self, x, x_skip):
            return torch.cat([x, x_skip], 1)

        def  __init__(self, buffers_in_size, buffers_out_size, passspp=False, second_pass=False):
            super().__init__()

            self.passspp = passspp

            self.buffers_in_size = buffers_in_size
            self.buffers_out_size = buffers_out_size

            self.second_pass = second_pass

            if self.passspp:
                self.buffers_in_size += 1

            self.pool = nn.MaxPool2d(2, 2)
            self.unpool = nn.MaxUnpool2d(2, 2)
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

            kern1_size = 3
            kern1_padding = 1

            def conv_unit(in_layers, out_layers):
                return nn.Sequential(
                    nn.Conv2d(in_layers, out_layers, kern1_size, padding=kern1_padding),
                    nn.LeakyReLU(0.1))

            def convs(layers):
                units = []

                for idx in range(len(layers)-1):
                    units.append(conv_unit(layers[idx], layers[idx+1]))

                return nn.Sequential(*units)

            self.encod0 = convs([self.buffers_in_size, 32, 32])
            self.encod1 = convs([32, 43, 43])
            self.encod2 = convs([43 + (3 if self.second_pass else 0), 57, 57])
            self.encod3 = convs([57, 76, 76])
            self.encod4 = convs([76, 101, 101])

            self.middle = convs([101, 101, 101])

            self.decod4 = convs([101*2, 76, 76])
            self.decod3 = convs([76*2, 57, 57])
            self.decod2 = convs([57*2, 43, 43])
            self.decod1 = convs([43*2, 32, 32])
            self.decod0 = convs([32*2, 128, 64])
            self.final = nn.Conv2d(64, self.buffers_out_size, kern1_size, padding=kern1_padding)

        def forward(self, x, x_small=None):
            x = self.encod0(x)
            x, x_skip0 = self.pool_skip(x)
            x = self.encod1(x)
            x, x_skip1 = self.pool_skip(x)

            if self.second_pass:
                x = torch.cat([x, x_small], 1)

            x = self.encod2(x)
            x, x_skip2 = self.pool_skip(x)
            x = self.encod3(x)
            x, x_skip3 = self.pool_skip(x)
            x = self.encod4(x)
            x, x_skip4 = self.pool_skip(x)

            x = self.middle(x)

            x = self.decod4(self.merge(self.upsample(x), x_skip4))
            x = self.decod3(self.merge(self.upsample(x), x_skip3))
            x = self.decod2(self.merge(self.upsample(x), x_skip2))
            x = self.decod1(self.merge(self.upsample(x), x_skip1))
            x = self.decod0(self.merge(self.upsample(x), x_skip0))
            x = self.final(x)

            return x

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def take_type_opt(self, name):
        if name in self.network_type_opt:

            print("has " + name)
            self.network_type_opt.remove(name)
            return True


        return False

    def set_vis(self, value):
        self.vis = value


    def set_save_border(self, save_border):
        full_buffer.safe_border = (save_border, save_border)

    def __init__(self, network_type, spp, renders_dir,  spp_additional=None, mapmode=None, uniform=False, dataset_filter=None, dualmode=False):
        self.is_cuda = True # torch.cuda.is_available()

        self.criterion = self.right_type(nn.L1Loss(size_average=True))
        self.criterion_mse = self.right_type(nn.MSELoss(size_average=True))
        self.criterion_l1 = self.right_type(nn.L1Loss(size_average=True))


        self.loss_relative_L1 = self.right_type(special_losses.RelativeL1())
        self.loss_edge_loss = self.right_type(special_losses.EdgeLoss())
        self.loss_relative_edge_loss = self.right_type(special_losses.RelativeEdgeLoss())


        self.sc_log1p3 = self.right_type(RenderSimulator.SC_log1p3())


        self.mapmode = mapmode

        self.uniform = uniform
        self.dataset_filter = dataset_filter

        self.optimizer = None

        self.out_dir = "."

        self.dualmode = dualmode

        self.writer = SummaryWriter()
        self.writer_count = 0

        self.tensorboard_every = 10000
        self.tensorboard_graph_every = 20

        self.monitor_writer = special_losses.MonitorWriter(self.writer, self.tensorboard_graph_every, self.tensorboard_every, self)



        self.loss_combo = special_losses.LossCombo(self.monitor_writer,
            ["relative_l1", self.loss_relative_L1, 1],
            ["relative_edge_loss", self.loss_relative_edge_loss, 1],
                                                   )

        self.spp_criterion = self.criterion_mse


        self.running_loss = 0.0
        self.running_loss_count = 0.0
        self.print_error = 5
        self.print_weights = 0
        self.batch_size = 1 if config.config.small_gpu else 6
        self.loss_graph = []
        self.epoch = 0
        self.apply_dir = None

        self.spp_additional = float(spp_additional)

        self.trainloader = None
        self.testloader = None
        self.prenormalized = False

        self.gamma = 1.0
        self.mode = None
        self.model_loaded = False


        self.colorchannels = 3


        self.output_target_buffers = [Buffer(Perm("color", 0, "COLOR"), "color_in")]
        self.output_target_buffers2 = {}

        self.show_img = True
        self.vis = True
        self.generate_gt = False

        self.nodes_real = None
        self.no_gt = False

        self.crop_network = True

        self.model_save_path = "cur_model.model"

        network_type = network_type.split(":")
        self.network_type_opt= set(network_type)


        self.spp=spp

        self.renders_dir = renders_dir

        self.opt_passspp = True
        self.opt_passspp_mult = 1 if self.take_type_opt("passspp") else 0

        assert self.opt_passspp_mult  == 0 # we don't do it anymore

        self.opt_mixset = self.take_type_opt("mixset")
        self.opt_sppopt = self.take_type_opt("sppopt")


        self.prenormalized = True

        self.spp_base = 1

        if '' in self.network_type_opt:
            self.network_type_opt.remove('')
        assert len(self.network_type_opt) == 0

        self.network_single_prefilter = False

        normn="color_in_log"

        bc = [Buffer(Perm("color", i, "COLOR"), normalizer_name=normn) for i in range(self.colorchannels)]
        color_out = [Buffer(Perm("color", i, "COLOR_CBF"), "color_in") for i in range(self.colorchannels)]
        for i in range(self.colorchannels):

            self.output_target_buffers2["color" + str(i)] = bc[i]

        data_in = []
        for i in range(3):
            data_in.append(Buffer(Perm("color", i, "TEXTURE"), "texture"))

        data_in.extend([
            Buffer("_feature_NORM_1_X", "norm1"),
            Buffer("_feature_NORM_1_Y", "norm1"),
            Buffer("_feature_NORM_1_Z", "norm1"),

            Buffer("_feature_DEPTH_1", "depth"),

        ])

        data_in.extend([
            Buffer("_feature_SHADOW_1_Y", "shadow"),
        ])

        color_layers = [self.output_target_buffers2[key] for key in sorted(self.output_target_buffers2.keys())]

        data_in = color_layers + data_in

        data_out = []
        data_out.extend(color_layers)


        self.networks = {}

        data_out = color_layers



        self.forward_pass = self.forward_pass_adapt2


        denoise_net = Denoising.NetBig(len(data_in), 3, passspp=self.opt_passspp)

        self.networks["denoise"] = Denoising.NetDaptor(self, self.right_type(denoise_net), data_in, data_out)

        rnet = RenderSimulator.NetRenderMulti3()

        self.networks["selector"] = Denoising.NetDaptor(self, self.right_type(rnet), data_in, data_out)

        spp_map = nn.Sequential(
            Denoising.NetBig(11,1),
            Denoising.NetMapBound(scale=(self.spp_additional + self.spp_base), bias=0.0))


        self.networks["spp_map"] = Denoising.NetDaptor(self, self.right_type(spp_map), data_in,
                                                       data_out)
        self.data_layer = DataLayer(all_fast_buffers2=self.get_all_fast_buffers(),
                                    all_gt_buffers2=self.get_all_gt_buffers(),
                                    permutator_group2=permutator_group,
                                    normalizers2=normalizers,
                                    output_target_buffers=self.output_target_buffers2)

        self.gglue = Gglue(self.networks, {})

        self.super_dataset = None
        self.subset_dataset = None
        self.subset_dataset_loader = None

        self.active_nodes = None

    def set_mode(self, mode):
        assert mode == "train" or mode == "test" or mode == "info"
        self.mode = mode

    def set_active_nodes(self, nodes):
        self.active_nodes = nodes
        self.gglue.set_active_nodes(nodes)
        self.net2 = self.right_type(self.gglue.generate_superNet(self.nodes_real))


        self.optimizer0 = optim.Adam(self.networks['denoise'].net.parameters(), lr=0.0001)
        self.optimizer1 = optim.Adam(self.networks['spp_map'].net.parameters(), lr=0.0001)
        self.optimizer = optim.Adam(list(self.networks['denoise'].net.parameters()) + list(self.networks['spp_map'].net.parameters()), lr=0.0001)



        self.show_img = True and ("standard" in self.active_nodes or "biliteral_filter" in self.active_nodes or "biliteral_NN" in self.active_nodes)

        self.subset_dataset = None
        self.subset_dataset_loader = None

    def init_active_nodes(self):
        if self.active_nodes is None:
            self.active_nodes = set(self.networks.keys())
            self.set_active_nodes(self.active_nodes)

    def right_type(self, net):
        if self.is_cuda:
            net = net.cuda()

        return net

    @staticmethod
    def get_all_buffer_names(buffers):
        buffer_names = []
        for buffer in buffers:
            buffer_names += buffer.get_all_buffer_names()

        return set(buffer_names)

    def get_all_fast_buffers(self):
        buffers = set()
        for name, daptor  in self.networks.items():
            buffers = buffers.union(set(daptor.buffers_in))

        return buffers

    def get_all_gt_buffers(self):
        buffers = set()
        for name, daptor in self.networks.items():
            buffers = buffers.union(set(daptor.buffers_out))

        return buffers


    def forward_pass_adapt(self, data, train=True, check_points=False, hnodes=[], crop=True, save_graph=False, target_spp_draw = None, not_normalize=False):
        if train:
            requires_grad = True
            volatile=False
        else:
            requires_grad = False
            volatile=True

        # zero the parameter gradients
        if train:
            self.optimizer.zero_grad()

        # get the inputs
        axillary, unique_id, gt_img, inputs = data


        #assert issubclass(type(inputs), torch.FloatTensor) or #issubclass(type(inputs), torch.cuda.FloatTensor)
        #assert issubclass(type(gt_img), torch.FloatTensor) or #issubclass(type(gt_img), torch.cuda.FloatTensor)


        gt_img = gt_img[:, 0:3, :, :]

        if self.is_cuda:
            if not self.generate_gt:
                gt_img = gt_img.cuda()

            inputs = inputs.cuda()

        if train :
            inputs = Variable(inputs)
        else:
            inputs = Variable(inputs, requires_grad=False, volatile=True)

        if train :
            gt_img = Variable(gt_img)

        else:
            gt_img = Variable(gt_img, requires_grad=False, volatile=True)

        inputs = self.sc_log1p3(inputs)
        gt_img = self.sc_log1p3(gt_img)

        do_denoising = False


        if "spp_map" in hnodes:



            outputs = self.networks["spp_map"].net(inputs)
            outputs = outputs[0]


        else:
            if self.opt_passspp:
                inputs2 = torch.cat([inputs, (((target_spp_draw)-4)/4)*self.opt_passspp_mult], 1)
            else:
                inputs2 = inputs

            do_denoising = True
            outputs = self.networks["denoise"].net(inputs2)

        safe_slice = [slice(full_buffer.safe_border[d], -full_buffer.safe_border[d] if full_buffer.safe_border[d] != 0 else None) for d in range(2)]

        vgg_loss = None



        if crop:
            gt_img = gt_img[:, :, safe_slice[0], safe_slice[1]]

        if self.crop_network and crop:
            outputs = outputs[:, :, safe_slice[0], safe_slice[1]]

        num_patches = inputs.size()[0]

        return FPdata(axillary, inputs, gt_img, outputs, 0, (num_patches, {}), unique_id, vgg_loss=vgg_loss)

    def set_normalizations(self, cnormalizers):
        global normalizers
        normalizers = cnormalizers

        self.prenormalized = True
        self.data_layer.init_normalizers(cnormalizers)

    def convert_data(self, data):
        def to_torch(e):
            if isinstance(e, np.ndarray):

                e = torch.from_numpy(e)
                return e # fix me
                return roundup_tensor(e, [ 1, 32, 32])
            else:
                return torch.FloatTensor([e])

        return [to_torch(e).unsqueeze(0).cuda() for e in data]




    def gen_input_rand_map2(self,size, ttype,upscale):
        size_orig = list(size)
        size= list(size)
        size[1] = 1
        size[2] //= upscale
        size[3] //= upscale

        size[2] += 2
        size[3] += 2

        map = torch.rand(*size).type(ttype)


        map = Variable(map)

        map = self.networks["upsample"+str(upscale)].net(map)

        rx = np.random.randint(upscale-1)
        ry = np.random.randint(upscale-1)

        map = map[:, :, rx:rx+size_orig[2], ry:ry+size_orig[3]]

        return map.data

    def gen_input_rand_map(self,input, spp_count):

        size = list(input.size())

        map = self.gen_input_rand_map2(size, input.data.type(), 8)

        map = map/map.mean() * spp_count
        map = map.clamp(min=1)
        map = Variable(map)

        return map

    def gen_spp_map_step(self, input):

        size = list(input.size())

        size = list(size)
        size[1] = 1


        map = torch.ones(*size).type(input.data.type())*np.random.randint(4, 9)

        map = Variable(map)


        return map

    def forward_pass_adapt2(self, data, train=True, check_points=False, hnodes=[], crop=True, draw_prefix="", data_type=None, closure=False, pfp=None):

        print("data_type: ", data_type)

        if train:
            for net in self.networks.values():
                net.net.zero_grad()


        axillary = data[0]
        unique_id = data[1]
        gt_img = data[2]
        mixed_inputs = [data[3+idx] for idx in range(len(data)-3)]

        if random.choice([True, False]):
            mixed_inputs[0], mixed_inputs[1] = mixed_inputs[1], mixed_inputs[0]

        reuse_inputs = True

        #for input in mixed_inputs:
            #assert issubclass(type(input), torch.FloatTensor)

        #assert issubclass(type(gt_img), torch.FloatTensor)

        to_cuda = lambda x: x.cuda()

        if train:
            to_var = lambda x: Variable(x)
        else:
            to_var = lambda x: Variable(x, requires_grad=False, volatile=True)

        if self.is_cuda:
            gt_img = gt_img.cuda()

            mixed_inputs = [to_var(inputs_map.cuda()) for inputs_map in mixed_inputs]

        num_channels = mixed_inputs[0].size(1)
        inputs_grnd = to_var(gt_img[:, :num_channels, :, :])
        gt_img_color = to_var(gt_img[:, 0:3, :, :])

        self.monitor_writer.set_prefix(str(data_type) + "_" + draw_prefix)

        additional_loss = []


        inputs_maps = [mixed_inputs[0]]


        network_spp_map = "spp2_map"

        if (self.mapmode is None or data_type!=0) or True:
            inputs_maps2 = [self.sc_log1p3(inputs_maps[0])]

            target_spps = [self.networks["spp_map"].net(inputs_maps2[0]) ]


        if self.mapmode == "random" and data_type!=1:

            spp_selected = self.gen_input_rand_map(mixed_inputs[0], 4)

        elif self.mapmode == "step":
            spp_selected = self.gen_spp_map_step(mixed_inputs[0])

        elif self.uniform:
            spp_selected = self.gen_input_rand_map(mixed_inputs[0], (self.spp_additional + self.spp_base))*0+(self.spp_additional + self.spp_base)
        else:
            spp_selected = target_spps[0][0]


        do_denoising = not ((self.uniform) and data_type==1)

        if do_denoising:
            input_adapts = []

            inputs_0s = mixed_inputs[0:]

            input_adapts.append(self.networks["selector"].net(spp_selected, *(inputs_0s+[inputs_grnd])))

            input_adapts = [self.sc_log1p3(ina) for ina in input_adapts]

            if self.opt_passspp:
                for idx in range(len(input_adapts)):

                    input_adapts[idx] = torch.cat([input_adapts[idx], ((spp_selected.clamp(min=1)-4)/4)*self.opt_passspp_mult], 1)


            network_name_denoise = "denoise"
            outputss = [self.networks[network_name_denoise].net(input_adapt) for input_adapt in input_adapts]
            gt_img_var = to_var(gt_img[:, 0:3, :, :])
            gt_img_var = self.sc_log1p3(gt_img_var)
            gt_imgs = [gt_img_var] * len(input_adapts)

            gt_img = None
        else:
            gt_imgs = None
            outputs = None
            outputss = None



        safe_slice = [slice(full_buffer.safe_border[d], -full_buffer.safe_border[d] if full_buffer.safe_border[d] != 0 else None) for d in range(2)]

        vgg_loss = None

        if train or self.use_vgg:

            if do_denoising:
                    total_loss = self.loss_combo(outputss[0], gt_imgs[0], additional_loss)

            else:
                if self.uniform:
                    total_loss = None

            if total_loss is not None:
                total_loss.backward()

            if do_denoising:


                self.monitor_writer.add_image('Inputs', visualizer.draw(
                    [input_adapts[0].data[:, 0:3, :, :], inputs_maps[0].data[:, 0:3, :, :]]
                ))

            self.monitor_writer.add_image('Map', visualizer.draw(
                [(spp_selected.data) / 3 / self.spp_additional for idx in range(len([spp_selected]))]
            ))


        if closure:
            return sum_loss


        if crop and gt_imgs is not None:
            gt_img = gt_imgs[0][:, :, safe_slice[0], safe_slice[1]]

        if self.crop_network and crop and outputss is not None:
            outputs = outputss[0][:, :, safe_slice[0], safe_slice[1]]

        if do_denoising:
            num_patches = input_adapts[0].size()[0]
            input_adapts_write = input_adapts[0]
        else:
            num_patches = 0
            input_adapts_write = None


        return FPdata(axillary, input_adapts_write, gt_img, outputs, 0, (num_patches, {}), unique_id, vgg_loss=vgg_loss)


    def set_normalizations(self, cnormalizers):
        global normalizers
        normalizers = cnormalizers

        self.prenormalized = True
        self.data_layer.init_normalizers(cnormalizers)

    def get_subset_dataset(self, required_in_buffers=None, required_out_buffers=None,train=False, apply_renders=None, gglue=None, use_patches=True, modulo=None, workers=None, needed_spps=[], num_group=1, group=0, permute_spp=False):
        if train:
            shuffle=True
        else:
            shuffle=False

        self.subset_dataset = hdf5_dataset.hdf5_dataset(self.renders_dir, needed_spps=needed_spps, num_group=num_group, group=group, permute_spp=permute_spp, dataset_filter=self.dataset_filter)


        workers = 3
        subset_dataset_loader = torch.utils.data.DataLoader(self.subset_dataset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=workers if workers is not None else config.config.num_workers,
            pin_memory=self.is_cuda)


        return subset_dataset_loader

    def train(self, n_epoch):

        self.init_active_nodes()
        self.multi_gpu()

        needed_spps = [0, 1] + hdf5_dataset.get_info(self.renders_dir)

        if self.dualmode:

            if  self.dualmode[0] == 2:
                set_loader = [self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, True, gglue=self.gglue, use_patches=False, needed_spps=needed_spps,  num_group=2, group=0, permute_spp=True),
                              self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, True,
                                                      gglue=self.gglue, use_patches=False, needed_spps=needed_spps, num_group=2, group=1, permute_spp=True)
                              ]

                set_loader_dual = DualIter({0: set_loader[0],
                                            1: set_loader[1]})
            elif self.dualmode[0] == 0:
                set_loader = self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, True,
                                        gglue=self.gglue, use_patches=False, needed_spps=needed_spps, num_group=2,
                                        group=0, permute_spp=True)


                set_loader_dual = DualIter({0: set_loader})

            elif self.dualmode[0] == 1:
                set_loader = self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, True,
                                        gglue=self.gglue, use_patches=False, needed_spps=needed_spps, num_group=2,
                                        group=1, permute_spp=True)


                set_loader_dual = DualIter({1: set_loader})
        else:
            set_loader_dual = self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, True, gglue=self.gglue, use_patches=False, needed_spps=needed_spps,  num_group=1, group=0, permute_spp=True)


        self.net2 = self.net2.train()


        self.data_layer.normalizers_print()

        self.writer_count = 0

        load_round_counter = 0

        save_every = max(500//self.batch_size, len(set_loader_dual) // 10)
        save_every = max(save_every, 1)

        self.output_lines = 6
        needed_output_lines = self.output_lines
        needed_fp = []

        run_count = 0


        while self.epoch < n_epoch:  # loop over the dataset multiple times
            self.epoch += 1
            epoch_start = time.time()

            print('[', end='')

            for i, data in enumerate(set_loader_dual, 0):
                if self.dualmode:
                    (data, data_type, last_item) = data
                else:
                    data_type = -1
                    last_item = True

                load_round_counter += 1
                run_count += 1


                if self.opt_mixset:
                    data_type = np.random.randint(0,2)
                if self.opt_sppopt:
                    data_type = 1

                prefix = "denoise_" if data_type!=1 else "spp_map_"

                self.do_write = True

                if data_type == -1:

                    self.forward_pass(data,
                                                                   draw_prefix=prefix,
                                                                   data_type=data_type)

                    self.optimizer.step()
                elif data_type == 0:
                    fp = self.forward_pass(data,
                                           draw_prefix=prefix,
                                           data_type=data_type)

                    self.optimizer0.step()


                elif data_type == 1:
                    self.forward_pass(data,
                                      draw_prefix=prefix,
                                      data_type=data_type)
                    self.optimizer1.step()

                else:
                    assert False

                if len(set_loader_dual) < 40 or i % int(len(set_loader_dual)/40) == 1:
                    print('=', end='', flush=True)

                if i % save_every == save_every - 1:
                    self.save_state()
                    print('|', end='')


                if last_item:
                    self.writer_count += 1
                    self.monitor_writer.next_step()

            print(']')

            epoch_end = time.time()


            print("[{}]  Time {:.2f}".format(self.epoch, epoch_end - epoch_start))


            if self.epoch % 10 == 0:
                self.save_state()
                print("Model saved")


        print('Finished Training')

    def save_state(self):
        filename = self.model_save_path
        filename_tmp = filename + "~"

        with open(filename_tmp, 'wb') as f:

            networks_states = {name: daptor.net for name, daptor in self.networks.items()}

            torch.save({"networks_states_2": networks_states, "normalizers": self.data_layer.normalizers2,
                        }, f)

            f.flush()
            os.fsync(f.fileno())
            f.close()

            os.rename(filename_tmp, filename)

    def multi_gpu(self):

        for net_name in self.networks.keys():
            self.networks[net_name].net = nn.DataParallel(self.networks[net_name].net)


    def norm_guide(self, guide, mode_uniform):
        guide = guide * (self.spp_additional + self.spp_base) / guide.mean()

        if mode_uniform:
            guide = guide * 0 + (self.spp_additional + self.spp_base)

        guide = guide + (torch.rand(*guide.shape).type_as(guide) - .5)

        guide = torch.round(guide)
        guide = torch.clamp(guide, self.spp_base, self.max_spp_available_minus_base + self.spp_base)

        guide = guide - self.spp_base

        good_elem = guide > 0
        good_elem = good_elem.type(torch.FloatTensor)
        treshold = (guide.sum() - self.spp_additional * guide.numel()) / guide.numel()
        if treshold > 0:
            treshold = (guide.sum() - self.spp_additional * guide.numel()) / good_elem.sum()
            guide = guide - (torch.rand(*guide.shape).type_as(guide) < treshold).type_as(guide)
        else:
            treshold *= -1
            guide = guide + (torch.rand(*guide.shape).type_as(guide) < treshold).type_as(guide)

        guide = torch.clamp(guide, 0, self.max_spp_available_minus_base)

        return guide

    def norm_guide2(self, guide, mode_uniform):
        guide = guide * (self.spp_additional + self.spp_base) / guide.mean()

        if mode_uniform:
            guide = guide * 0 + (self.spp_additional + self.spp_base)

        mult = 1


        for _ in range(5):
            guide2 = torch.clamp(guide*mult, min=self.spp_base)
            guide2 = torch.round(guide2)
            guide2 = torch.clamp(guide2, 0, self.max_spp_available_minus_base)

            mult =  (self.spp_additional + self.spp_base)/guide2.mean() * mult

        guide = torch.round(guide*mult)
        guide = guide - self.spp_base

        guide = torch.clamp(guide, 0, self.max_spp_available_minus_base)


        return guide, mult

    def calc_render(self, guide, input_base, inputs):
        guide = torch.round(guide)
        guide_left = guide.clone()

        result = input_base * self.spp_base

        for power in range(len(inputs)):
            remainder = guide_left.remainder(2)


            result = result + remainder * (inputs[power] * (2 ** power))
            guide_left -= remainder
            guide_left = guide_left.div(2)


        result = result / (guide + self.spp_base)
        return result

    def apply(self):


        train_mode = self.mode == "train"

        self.init_active_nodes()

        mode_uniform = False

        self.net2 = self.net2.train(False)

        self.batch_size = 1

        needed_spps = [0, 1] + hdf5_dataset.get_info(self.renders_dir)
        self.max_spp_available_minus_base = sum(needed_spps[1:]) - 1


        set_loader = self.get_subset_dataset(self.gglue.needed_buffers_in, self.gglue.needed_buffers_out, train_mode,
                                             workers=0, needed_spps =needed_spps)


        if train_mode:
            prefix_name = "test_"
        else:
            prefix_name = "test_"

        for i, data in enumerate(set_loader, 0):

            axillary = data[0]
            unique_id = data[1]
            gt_img = data[2]
            input_base = data[3]

            apply_dir = self.subset_dataset.get_name(int(unique_id.cpu().numpy()))
            print(apply_dir)
            if "Bedroom" in apply_dir:
                continue

            image_size = None

            inputs = [None for _ in range(len(data)-4)]

            for idx in range(len(inputs)):
                inputs[idx] =data[idx+4]

            if image_size is None:
                image_size = list(gt_img.shape[2:])

            print("image_size", image_size)


            gt_img = roundup_tensor(gt_img, [1,1,32,32])
            input_base = roundup_tensor(input_base, [1,1,32,32])

            for idx in range(len(inputs)):
                inputs[idx] = roundup_tensor(inputs[idx], [1,1,32,32])


            hnodes = ["spp_map"]

            fp = self.forward_pass_adapt((axillary, unique_id, gt_img, input_base), train=False, hnodes=hnodes, crop=False)
            guide_orig = fp.outputs.data
            del fp

            # For uniform sampling, set to constant value
            if self.uniform:
                guide_orig = guide_orig*0+(self.spp_additional + self.spp_base)


            guide = guide_orig
            guide = guide

            guide2 = guide-1

            def micro_size(x):
                if len(x.shape) == 4:
                    return x[:, :, :image_size[0], :image_size[1]]
                else:

                    return x[ :, :image_size[0], :image_size[1]]

            if not mode_uniform:
                visualizer.save_as_img(micro_size(guide2.cpu().numpy()[0, :, :, :])/8,
                                       os.path.join(self.out_dir, prefix_name + apply_dir + "_c_map2.png"), self.gamma)



            guide, mult = self.norm_guide2(micro_size(guide), mode_uniform)

            guide = guide.cpu()
            visualizer.save_as_img((guide.numpy()[0,:,:,:] + self.spp_base)/64, os.path.join(self.out_dir, prefix_name + apply_dir + "_c_map.png"), self.gamma)

            guide = roundup_tensor(guide,  [ 1, 1, 32, 32])

            result = self.calc_render(guide, input_base, inputs)


            gt_img_gpu = gt_img.cuda()
            result_gpu = result.cuda()
            guide_gpu =Variable(guide.cuda(), volatile=True, requires_grad=False)


            fp = self.forward_pass_adapt((axillary, unique_id, gt_img_gpu, result_gpu), train=False, hnodes=["denoise"],
                                         crop=False, target_spp_draw=guide_gpu)
            output = fp.outputs.data

            def conv_to_linear(x):
                return self.sc_log1p3.revert(Variable(x, volatile=True, requires_grad=False)).data


            output = conv_to_linear(output)
            output = micro_size(output.cpu().numpy())
            ground = micro_size(gt_img.cpu().numpy())
            input = micro_size(input_base.cpu().numpy())
            input2= micro_size(result.cpu().numpy())


            for idx in range(3):
                output[0,idx,:,:] = normalizers["color_in_log"].reverse_np(output[0,idx,:,:])
                ground[0,idx,:,:] = normalizers["color_in_log"].reverse_np(ground[0,idx,:,:])
                input[0,idx,:,:] = normalizers["color_in_log"].reverse_np(input[0,idx,:,:])
                input2[0,idx,:,:] = normalizers["color_in_log"].reverse_np(input2[0,idx,:,:])

            output = output[0, :, :, :]
            ground = ground[0, 0:3, :, :]
            input = input[0, 0:3, :, :]
            input2 = input2[0, 0:3, :, :]


            visualizer.score_image(output, ground, input, apply_dir, self.out_dir, 0)

            brightness = 1

            if apply_dir.find("Lib") != -1:

                brightness = 2.5
            if apply_dir.find("example3_MB") != -1:

                brightness = 6

            visualizer.save_as_img(output*brightness,
                                   os.path.join(self.out_dir, prefix_name + apply_dir + "_c_network.png"), self.gamma)
            visualizer.save_as_img(ground*brightness, os.path.join(self.out_dir, prefix_name + apply_dir + "_c_grd.png"),
                                   self.gamma)
            visualizer.save_as_img(input*brightness, os.path.join(self.out_dir, prefix_name + apply_dir + "_c_input.png"),
                                   self.gamma)

            visualizer.save_as_img(input2*brightness, os.path.join(self.out_dir, prefix_name + apply_dir + "_c_input2.png"),
                                   self.gamma)



    def load_model(self, f):
        loaded_model = torch.load(f)
        if isinstance(loaded_model, dict):
            if "networks_states_2" in loaded_model:
                for name, net in loaded_model["networks_states_2"].items():
                    if net is not None and name in self.networks :
                        if self.networks[name].net is None:
                            self.networks[name].net = net
                        else:

                            if name != "varb"  :
                            #if True:
                                srn.load_module(self.networks[name].net, convert_state(net.state_dict()), new_params={"superres.", "chooser.", '0.decod4.1.0.bias', '0.decod4.0.0.weight', '0.decod4.0.0.bias', '0.decod4.1.0.weight', '0.encod4.1.0.bias', '0.encod4.1.0.weight', '0.encod4.0.0.weight', '0.encod4.0.0.bias', "estvar."}, flexiable={"encod0.0.weight", "0.middle.0.0.weight"})


                    else:
                        print("Skipping {} sub networks".format(name))

            else:
                for name, net in loaded_model["networks_states"].items():
                    if name in self.networks:
                        self.networks[name].net.load_state_dict(net)
                    else:
                        print("Skipping {} sub networks".format(name))

            self.set_normalizations(loaded_model["normalizers"])

        else:
            self.net.load_state_dict(loaded_model)
        self.model_loaded = True


    def set_save_model_path(self, path):
        self.model_save_path = path
