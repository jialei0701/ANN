import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import full_buffer
from PIL import Image
import numpy as np
import os

import torch
from torch.autograd import Variable
import torch
import torchvision
import denoiser


def calculate_rmse(img, gt):
    diff = img-gt
    assert diff.shape[2] == 3 or  diff.shape[2] == 1
    return (diff*diff/(gt*gt+0.01)).sum()/diff.size



def calc_color(irrad, albedo):
    return irrad * (albedo + 0.005)

def get_color_img(f,  data, target_buffers2, albedo_img=None):
    irrad_index =  f([target_buffers2["irrad"]])[0]
    irrad_img = data.data.cpu()[0, irrad_index, :, :].numpy()
    irrad_img = target_buffers2["irrad"].reverse_np(irrad_img)


    if albedo_img is None:
        albedo_index =  f([target_buffers2["albedo"]])[0]
        albedo_img = data.data.cpu()[0, albedo_index, :, :].numpy()
        albedo_img = target_buffers2["albedo"].reverse_np(albedo_img)

    return calc_color(irrad_img, albedo_img), albedo_img

def togrey(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return plotable_image(img)

def get_color_img_simple(f,  data, output_image_layers, img_num=0):
    color_indices = f(output_image_layers)

    color_img = [None] * len(color_indices)

    if img_num >= data.data.shape[0]:
        img_num = data.data.shape[0] - 1

    for i in range(len(color_indices)):
        buffer = data.data.cpu()[img_num, color_indices[i][0], :, :].numpy()
        color_img[i] = color_indices[i][1].reverse_np(buffer)

    color_img = np.dstack(color_img)

    return color_img





def clip_imp(img):
    return np.clip(img, 0, 1)

def plotable_image(img):
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img

def to_srgb(img):
    return np.power(img, 1/2.2)


def make_img_grid(imgs, width, height):
    one_size = imgs[0].shape[0:2]
    num_channels = imgs[0].shape[2]

    full_size = [one_size[0]*width, one_size[1]*height]


    img = np.zeros(shape=(full_size[0], full_size[1], num_channels), dtype=np.float32)

    num = 0

    for p_x in range(width):
        for p_y in range(height):
            x = p_x*one_size[0]
            y = p_y*one_size[1]

            img[ x:x+one_size[0], y:y+one_size[1], :] = imgs[num]
            num += 1


    return img



def rgb2gray(img):
    if img.shape[2] == 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return img[:, :, 0]




def calcualte_errors(img, ground):
    rmse = calculate_rmse(img, ground)

    return rmse

def join_number_list(list):
    return ",".join(str(x) for x in list)


def convert_img_finale_1(img):
    return np.rollaxis(img, 0, 3).clip(0, 1)

def score_image(filtered,  ground, input, name, cdir, other_loss):
    filtered = convert_img_finale_1(filtered)
    ground = convert_img_finale_1(ground)


    error = calcualte_errors(filtered, ground)


    with open(os.path.join(cdir, "errors.csv"), "a") as file:
        file.write("{},{}\n".format(name, error))

    return error

def save_as_img(img, name, gamma, strideformat=False):
    directory = os.path.dirname(os.path.realpath(name))
    if not os.path.exists(directory):
        os.makedirs(directory)

    if gamma != 1.:
        img = np.power(img, gamma)

    if not strideformat:
        img = np.rollaxis(img, 0, 3)

    img = (img.clip(0, 1)*255).astype(np.uint8)

    if img.shape[2] == 1:
        img = img[:, :, 0]

    Image.fromarray(img).save(name)


def draw(imgs):
    num_inp = imgs[0].size(0)

    im = torch.cat(imgs)
    im = torchvision.utils.make_grid(im, nrow=num_inp)
    im = torch.pow(im.clamp(0, 1), 1 / 2.2)
    im = im.clamp(0, 1)

    return im

