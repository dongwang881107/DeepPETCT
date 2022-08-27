#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
from ..tools import pytorchwt
import torch
import torch.nn as nn
import math

class Framelets(nn.Module):
    def __init__(self, in_channels=1, num_features=64, stride=1, padding=1):
        super(Framelets, self).__init__()
        # stage 0
        self.stage_0_0 = self.conv_bn_relu(num_features_in=in_channels, num_features_out=num_features)
        self.stage_0_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        self.stage_0_wt = pytorchwt(feature_num=num_features)
        self.stage_0_HH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features*2)
        self.stage_0_HH_1 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features)
        self.stage_0_HL_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features*2)
        self.stage_0_HL_1 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features)
        self.stage_0_LH_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features*2)
        self.stage_0_LH_1 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features)

        # stage 1
        self.stage_1_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features*2)
        self.stage_1_1 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*2)
        self.stage_1_wt = pytorchwt(feature_num=num_features*2)
        self.stage_1_HH_0 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*4)
        self.stage_1_HH_1 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*2)
        self.stage_1_HL_0 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*4)
        self.stage_1_HL_1 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*2)
        self.stage_1_LH_0 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*4)
        self.stage_1_LH_1 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*2)

        # stage 2
        self.stage_2_0 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*4)
        self.stage_2_1 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*4)
        self.stage_2_wt = pytorchwt(feature_num=num_features*4)
        self.stage_2_HH_0 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*8)
        self.stage_2_HH_1 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*4)
        self.stage_2_HL_0 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*8)
        self.stage_2_HL_1 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*4)
        self.stage_2_LH_0 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*8)
        self.stage_2_LH_1 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*4)

        # stage 3
        self.stage_3_0 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*8)
        self.stage_3_1 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*8)
        self.stage_3_wt = pytorchwt(feature_num=num_features*8)
        self.stage_3_HH_0 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*16)
        self.stage_3_HH_1 = self.conv_bn_relu(num_features_in=num_features*16, num_features_out=num_features*8)
        self.stage_3_HL_0 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*16)
        self.stage_3_HL_1 = self.conv_bn_relu(num_features_in=num_features*16, num_features_out=num_features*8)
        self.stage_3_LH_0 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*16)
        self.stage_3_LH_1 = self.conv_bn_relu(num_features_in=num_features*16, num_features_out=num_features*8)
        self.stage_3_LL_0 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*16)
        self.stage_3_LL_1 = self.conv_bn_relu(num_features_in=num_features*16, num_features_out=num_features*8)

        # reconstruction
        self.stage_3_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*8)
        self.stage_3_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features*8, num_features_out=num_features*4)

        self.stage_2_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*4)
        self.stage_2_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features*4, num_features_out=num_features*2)

        self.stage_1_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features*2)
        self.stage_1_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features*2, num_features_out=num_features)

        self.stage_0_reconstruction_0 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)
        self.stage_0_reconstruction_1 = self.conv_bn_relu(num_features_in=num_features, num_features_out=num_features)

        self.reconstruction_output = nn.Conv2d(num_features, in_channels, kernel_size=1, stride=1, padding=0, bias=True)


    def conv_bn_relu(self, num_features_in, num_features_out):
        layers = []
        layers.append(nn.Conv2d(num_features_in, num_features_out, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.BatchNorm2d(num_features_out))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        # stage 0
        stage_0_0 = self.stage_0_0(x)
        stage_0_1 = self.stage_0_1(stage_0_0)
        # stage_0_transform = self.stage_0_wt.wavelet_n_dec(stage_0_1)
        # stage_0_LL = stage_0_transform[0]
        # [stage_0_LH, stage_0_HL, stage_0_HH] = stage_0_transform[-1]
        [stage_0_LL, stage_0_LH, stage_0_HL, stage_0_HH] = self.stage_0_wt.wavelet_dec(stage_0_1)
        stage_0_LH_0 = self.stage_0_LH_0(stage_0_LH)
        stage_0_LH_1 = self.stage_0_LH_1(stage_0_LH_0)
        stage_0_HL_0 = self.stage_0_HL_0(stage_0_HL)
        stage_0_HL_1 = self.stage_0_HL_1(stage_0_HL_0)
        stage_0_HH_0 = self.stage_0_HH_0(stage_0_HH)
        stage_0_HH_1 = self.stage_0_HH_1(stage_0_HH_0)
        
        # stage 1
        stage_1_0 = self.stage_1_0(stage_0_LL)
        stage_1_1 = self.stage_1_1(stage_1_0)
        # stage_1_transform = self.stage_1_wt.wavelet_n_dec(stage_1_1)
        # stage_1_LL = stage_1_transform[0]
        # [stage_1_LH, stage_1_HL, stage_1_HH] = stage_1_transform[-1]
        [stage_1_LL, stage_1_LH, stage_1_HL, stage_1_HH] = self.stage_1_wt.wavelet_dec(stage_1_1)
        stage_1_LH_0 = self.stage_1_LH_0(stage_1_LH)
        stage_1_LH_1 = self.stage_1_LH_1(stage_1_LH_0)
        stage_1_HL_0 = self.stage_1_HL_0(stage_1_HL)
        stage_1_HL_1 = self.stage_1_HL_1(stage_1_HL_0)
        stage_1_HH_0 = self.stage_1_HH_0(stage_1_HH)
        stage_1_HH_1 = self.stage_1_HH_1(stage_1_HH_0)
        
        # stage 2
        stage_2_0 = self.stage_2_0(stage_1_LL)
        stage_2_1 = self.stage_2_1(stage_2_0)
        # stage_2_transform = self.stage_2_wt.wavelet_n_dec(stage_2_1)
        # stage_2_LL = stage_2_transform[0]
        # [stage_2_LH, stage_2_HL, stage_2_HH] = stage_2_transform[-1]
        [stage_2_LL, stage_2_LH, stage_2_HL, stage_2_HH] = self.stage_2_wt.wavelet_dec(stage_2_1)
        stage_2_LH_0 = self.stage_2_LH_0(stage_2_LH)
        stage_2_LH_1 = self.stage_2_LH_1(stage_2_LH_0)
        stage_2_HL_0 = self.stage_2_HL_0(stage_2_HL)
        stage_2_HL_1 = self.stage_2_HL_1(stage_2_HL_0)
        stage_2_HH_0 = self.stage_2_HH_0(stage_2_HH)
        stage_2_HH_1 = self.stage_2_HH_1(stage_2_HH_0)
        
        # stage 3
        stage_3_0 = self.stage_3_0(stage_2_LL)
        stage_3_1 = self.stage_3_1(stage_3_0)
        # stage_3_transform = self.stage_3_wt.wavelet_n_dec(stage_3_1)
        # stage_3_LL = stage_3_transform[0]
        # [stage_3_LH, stage_3_HL, stage_3_HH] = stage_3_transform[-1]
        [stage_3_LL, stage_3_LH, stage_3_HL, stage_3_HH] = self.stage_3_wt.wavelet_dec(stage_3_1)
        stage_3_LH_0 = self.stage_3_LH_0(stage_3_LH)
        stage_3_LH_1 = self.stage_3_LH_1(stage_3_LH_0)
        stage_3_HL_0 = self.stage_3_HL_0(stage_3_HL)
        stage_3_HL_1 = self.stage_3_HL_1(stage_3_HL_0)
        stage_3_HH_0 = self.stage_3_HH_0(stage_3_HH)
        stage_3_HH_1 = self.stage_3_HH_1(stage_3_HH_0)
        stage_3_LL_0 = self.stage_3_LL_0(stage_3_LL)
        stage_3_LL_1 = self.stage_3_LL_1(stage_3_LL_0)
        
        # reconstruction
        stage_3_reconstruction = self.stage_3_wt.wavelet_rec(stage_3_LL_1, stage_3_LH_1, stage_3_HL_1, stage_3_HH_1)
        # stage_3_transform[0] = stage_3_LL_1
        # stage_3_reconstruction = self.stage_3_wt.wavelet_n_rec(stage_3_transform)
        stage_3_LL_reconstruction_0 = self.stage_3_reconstruction_0(stage_3_reconstruction+stage_3_1)
        stage_3_LL_reconstruction_1 = self.stage_3_reconstruction_1(stage_3_LL_reconstruction_0)

        stage_2_reconstruction = self.stage_2_wt.wavelet_rec(stage_3_LL_reconstruction_1, stage_2_LH_1, stage_2_HL_1, stage_2_HH_1)
        # stage_2_transform[0] = stage_2_LL_1
        # stage_2_reconstruction = self.stage_2_wt.wavelet_n_rec(stage_2_transform)
        stage_2_LL_reconstruction_0 = self.stage_2_reconstruction_0(stage_2_reconstruction+stage_2_1)
        stage_2_LL_reconstruction_1 = self.stage_2_reconstruction_1(stage_2_LL_reconstruction_0)

        stage_1_reconstruction = self.stage_1_wt.wavelet_rec(stage_2_LL_reconstruction_1, stage_1_LH_1, stage_1_HL_1, stage_1_HH_1)
        # stage_1_transform[0] = stage_1_LL_1
        # stage_1_reconstruction = self.stage_1_wt.wavelet_n_rec(stage_1_transform)
        stage_1_LL_reconstruction_0 = self.stage_1_reconstruction_0(stage_1_reconstruction+stage_1_1)
        stage_1_LL_reconstruction_1 = self.stage_1_reconstruction_1(stage_1_LL_reconstruction_0)

        stage_0_reconstruction = self.stage_0_wt.wavelet_rec(stage_1_LL_reconstruction_1, stage_0_LH_1, stage_0_HL_1, stage_0_HH_1)
        # stage_0_transform[0] = stage_0_LL_1
        # stage_0_reconstruction = self.stage_0_wt.wavelet_n_rec(stage_0_transform)
        stage_0_LL_reconstruction_0 = self.stage_0_reconstruction_0(stage_0_reconstruction+stage_0_1)
        stage_0_LL_reconstruction_1 = self.stage_0_reconstruction_1(stage_0_LL_reconstruction_0)
        
        out = self.reconstruction_output(stage_0_LL_reconstruction_1) + x

        if target is not None:
            pairs = {'out': (out, target)}
            return pairs, self.exports(x, out, target)
        else:
            return self.exports(x, out, target)

    def exports(self, x, output, target):
        result = {'input': x, 'output': output}
        if target is not None:
            result['target'] = target
        return result

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('Linear') != -1:
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    framelets = Framelets()
    framelets.cuda()
    x = torch.autograd.Variable(torch.rand(1, 3, 256, 256).cuda(0))
    # x = torch.autograd.Variable(torch.rand(1, 3, 256, 256))
    out = framelets(x)
    # print out.size()
    # print out