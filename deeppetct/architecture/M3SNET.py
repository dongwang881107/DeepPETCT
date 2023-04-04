import torch.nn as nn
import torch
from deeppetct.architecture.blocks import *

class M3SNET(nn.Module):
    # Multi-Modality Multi-Branch Multi-Self-Attention Network with Structure-Promoting Loss for Low-Dose PET/CT Reconstruction
    def __init__(self, sa_mode):
        super(M3SNET, self).__init__()
        print('M3S-NET with {} Attention'.format(sa_mode))

        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.out_channel = 96
        self.acti = 'relu'
        self.sa_mode = sa_mode
        self.num_splits = 2

        # attention blocks
        self.sa_ct = atten_block(self.out_channel, self.num_splits, self.sa_mode)
        self.sa_pet = atten_block(self.out_channel, self.num_splits, self.sa_mode)
        self.sa_com1 = atten_block(self.out_channel*2, self.num_splits, self.sa_mode)
        self.sa_com2 = atten_block(self.out_channel, self.num_splits, self.sa_mode)

        # ct branch
        self.layer_ct1 = conv_block('conv', 1, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_ct2 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_ct3 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_ct4 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_ct5 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)

        # pet branch
        self.layer_pet1 = conv_block('conv', 1, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_pet2 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_pet3 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_pet4 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_pet5 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)

        # combine together
        self.layer_com1 = conv_block('trans', 2*self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_com2 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_com3 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_com4 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer_com5 = conv_block('trans', self.out_channel, 1, self.kernel_size, self.stride, self.padding, self.acti)

    def forward(self, pet10, ct):
        # ct branch
        out_ct = self.layer_ct2(self.layer_ct1(ct))
        out_ct = self.sa_ct(out_ct)
        out_ct = self.layer_ct5(self.layer_ct4(self.layer_ct3(out_ct)))
        # pet branch
        out_pet = self.layer_pet2(self.layer_pet1(pet10))
        res1 = out_pet
        out_pet = self.sa_ct(out_pet)
        out_pet = self.layer_pet4(self.layer_pet3(out_pet))
        res2 = out_pet
        out_pet = self.layer_pet5(out_pet)
        # combine together
        out_com = torch.cat((out_pet,out_ct), dim=1)
        out_com = self.sa_com1(out_com)
        out_com = self.layer_com1(out_com)
        out_com = out_com + res2
        out_com = self.layer_com2(out_com)
        out_com = self.sa_com2(out_com)
        out_com = self.layer_com3(out_com)
        out_com = out_com + res1
        out_com = self.layer_com4(out_com)
        out_com = self.layer_com5(out_com)
        return out_com



