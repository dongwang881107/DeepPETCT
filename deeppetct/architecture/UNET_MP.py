import torch.nn as nn
from deeppetct.architecture.blocks import *

class UNET_MP(nn.Module):
    # Ref: Anatomically aided PET image reconstruction using deep neural networks
    # Medical Physics, 2021
    # Conv2d(stride=2) + ConvTranspose2d(stride=2)
    def __init__(self, sa_mode):
        super(UNET_MP, self).__init__()
        print('UNET_MP: UNET in Medical Physics paper, 2021')
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.sa_mode = sa_mode
        self.num_splits = 2

        # attention blocks
        self.atten_layer = atten_block(128, self.num_splits, self.sa_mode)

        # encoder
        self.layer1 = conv_block('conv', 2, 16, self.kernel_size, 1, self.padding, self.acti)
        self.layer2 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=16, out_channels=16, acti=self.acti)
        self.layer3 = conv_block('conv', 16, 32, self.kernel_size, 1, self.padding, self.acti)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=32, out_channels=32, acti=self.acti)
        self.layer5 = conv_block('conv', 32, 64, self.kernel_size, 1, self.padding, self.acti)
        self.layer6 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti)
        # decoder
        self.layer7 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=128, out_channels=64, acti=self.acti)
        self.layer8 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti)
        self.layer9 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti)
        self.layer10 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=64, out_channels=32, acti=self.acti)
        self.layer11 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti)
        self.layer12 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti)
        self.layer13 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=32, out_channels=16, acti=self.acti)    
        self.layer14 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti)
        self.layer15 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti)
        self.layer16 = conv_block('conv', 16, 1, self.kernel_size, 1, self.padding, self.acti)

    def forward(self, x):
        out = self.layer1(x)
        res1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        res2 = out
        out = self.layer4(out)
        out = self.layer5(out)
        res3 = out
        out = self.layer6(out) 
        out = self.atten_layer(out)
        out = self.layer7(out) 
        out = out + res3
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = out + res2
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out + res1
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        return out

    
