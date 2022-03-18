#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/18
# @Author  : haoli1
# 传统方法收集数据集的使用，获得与目标图片相似的图片
# ssim 结构性相似方法 ，方法的实现是从网上找的
# 但是只能针对两张图像构图完全差不多的，如果是截图的类似，就不太可 唉：）
from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os

import cv2

def ssim(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5      # default
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    _1D_window = (gauss/gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0]**2;
    C2 = K[1]**2;

    mu1 = F.conv2d(image1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(image2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(image1*image1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(image2*image2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(image1*image2, window, padding = window_size//2, groups = channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def process_pic(pic_name):

    I1 = cv2.imread(pic_name)
    I1 = cv2.resize(I1.astype(np.float32),(256,256))
    # tensors
    I1 = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0)/255.0
    I1 = Variable(I1, requires_grad = True)
    return I1




if __name__=="__main__":
    #计算两个文件夹的相似度
    path_1 = '/workspace/sim_pic/candidate/'
    path_2 = '/workspace/sim_pic/non-handraw/'

    K = [0.01, 0.03]
    L = 255;
    window_size = 11

    img_list1 =  [ os.path.join(path_1,i) for i in os.listdir(path_1)]# pic list for candidate pictures,less
    img_list2 =  [ os.path.join(path_2,i) for i in os.listdir(path_2)]# pic list for all pictures ,more
    sim_path=[]

    all_list = [ process_pic(path1) for path1 in img_list1]

    for pic_path2 in img_list2:
        I1 = process_pic(pic_path2)
        ssim_value=[ssim(I1, I2, K, window_size, L).data for I2 in all_list]
        if len([*filter(lambda x: x >= 0.3333, ssim_value)]) > 0:

            sim_path.append({'picture_path:',pic_path2,'value:',ssim_value})

    print(sim_path)
