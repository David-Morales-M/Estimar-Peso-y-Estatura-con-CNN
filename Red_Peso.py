# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:24:02 2021

@author: David Morales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super(UpBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x_up, x_down): 
        x_up = self.conv1(x_up)
        x_up = torch.cat([x_up, x_down], dim=1)
        x_up = self.conv2(x_up)

        return x_up
    
class FinalBlock(nn.Module):
    
    def __init__(self, in_ch, pool_size, h_channel):
        super(FinalBlock, self).__init__()
                
        self.mask_out = nn.Conv2d(in_ch, 2, 1)
        self.joint_out = nn.Conv2d(in_ch, 19, 1)
        self.pool_size = pool_size
        
        self.height_1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1),
            nn.ReLU()
        )
        
        self.height_2 = nn.Sequential(
            nn.Linear(32*32*32, 1024), # 1024
            nn.ReLU(),
            nn.Dropout(0.15), # 0.15
            nn.Linear(1024, 1) # 1024
        )
        
        self.weight_1 = nn.Sequential(
            nn.Conv2d(in_ch, h_channel, 3, padding=1), # ksize=1 for the earliest weight model
            nn.ReLU()
        )
        
        self.weight_2 = nn.Sequential(
            nn.Linear(pool_size*pool_size*h_channel, 1024), # 2048 or 1024 check !
            nn.Dropout(0.15), # 0.15
            nn.ReLU(),
            nn.Linear(1024, 1), # 2048 or 1024 check !
        )
        
    def forward(self, x):
        
        mask = torch.nn.Softmax(1)(self.mask_out(x))
        joint = self.joint_out(x)
        
        adapt = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        adapt_h = F.adaptive_avg_pool2d(x, (32, 32))
        
        height = self.height_1(adapt_h)
        height = height.view(height.size(0), -1)
        height = self.height_2(height)
        
        weight = self.weight_1(adapt)
        weight = weight.view(weight.size(0), -1)
        weight = self.weight_2(weight)
        
        return mask, joint, height, weight

    
class DownBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
                
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

    
class UNet(nn.Module):

    def __init__(self, min_neuron, pool_size=32, h_ch=32):
        super(UNet, self).__init__()
        
        self.conv_down1 = DownBlock(3, min_neuron)
        self.conv_down2 = DownBlock(min_neuron, 2*min_neuron)
        self.conv_down3 = DownBlock(2*min_neuron, 4*min_neuron)
        self.conv_down4 = DownBlock(4*min_neuron, 8*min_neuron)
        self.conv_down5 = DownBlock(8*min_neuron, 16*min_neuron)
        
        self.conv_upsample1 = UpBlock(16*min_neuron, 8*min_neuron)
        self.conv_upsample2 = UpBlock(8*min_neuron, 4*min_neuron)
        self.conv_upsample3 = UpBlock(4*min_neuron, 2*min_neuron)
        self.conv_upsample4 = UpBlock(2*min_neuron, min_neuron)

        self.conv_out = FinalBlock(min_neuron, pool_size, h_ch)
        
    def forward(self, x):
        
        conv1 = self.conv_down1(x)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)(conv1)

        conv2 = self.conv_down2(pool1)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv2)
        
        conv3 = self.conv_down3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)(conv3)
        
        conv4 = self.conv_down4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)(conv4)
        
        conv5 = self.conv_down5(pool4)

        up6 = self.conv_upsample1(conv5, conv4)
        up7 = self.conv_upsample2(up6, conv3)
        up8 = self.conv_upsample3(up7, conv2)
        up9 = self.conv_upsample4(up8, conv1)
        
        return self.conv_out(up9)