import torch.nn as nn
import torch
from deeppetct.architecture.blocks import *
from pytorch_wavelets import DWTForward, DWTInverse


class REDCNN(nn.Module):
    # Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network
    # Multi-Branch Input (CT+PET)
    # Page 2526, Figure 1
    # Conv2d (s=1) + ConvTranspose2d (s=1)
    def __init__(self, bn_flag, sa_flag):
        super(REDCNN, self).__init__()
        print('Multi-Branch Residual Encoder Decoder CNN with Batch Normalization [{}] and Self-Attention [{}]'.format(bn_flag, sa_flag))

        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.out_channel = 96
        self.acti = 'relu'
        self.bn_flag = bn_flag # batch normalization
        self.sa_flag = sa_flag # self attention
        if self.sa_flag == True:
            self.sa = SelfAttenBlock(self.out_channel)

        # ct branch
        self.layer_ct1 = conv_block('conv', 1, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_ct2 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_ct3 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_ct4 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_ct5 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)

        # pet branch
        self.layer_pet1 = conv_block('conv', 1, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_pet2 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_pet3 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_pet4 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_pet5 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)

        # combine together
        self.layer_com1 = conv_block('trans', 2*self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_com2 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_com3 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_com4 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer_com5 = conv_block('trans', self.out_channel, 1, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)

    def forward(self, pet10, ct):
        out_ct = self.layer_ct5(self.layer_ct4(self.layer_ct3(self.layer_ct2(self.layer_ct1(ct)))))
        out_pet = self.layer_pet1(pet10)
        # res1 = out_pet
        out_pet = self.layer_pet2(out_pet)
        res2 = out_pet
        out_pet = self.layer_pet3(out_pet)
        # res3 = out_pet
        out_pet = self.layer_pet4(out_pet)
        res4 = out_pet
        out_pet = self.layer_pet5(out_pet)
        out_com = torch.cat((out_pet,out_ct), dim=1)
        out_com = self.layer_com1(out_com)
        out_com = out_com + res4
        out_com = self.layer_com2(out_com)
        # out_com = out_com + res3
        out_com = self.layer_com3(out_com)
        out_com = out_com + res2
        out_com = self.layer_com4(out_com)
        # out_com = out_com + res1
        out_com = self.layer_com5(out_com)
        return out_com


class UNET_MP(nn.Module):
    # Ref: Anatomically aided PET image reconstruction using deep neural networks
    # Page 5, Figure 2(b)
    # Conv2d(stride=2) + ConvTranspose2d(stride=2)
    # Not working, has overlapping artifacts
    def __init__(self):
        super(UNET_MP, self).__init__()
        print('UNET_MP: UNET in MP paper')
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.bn_flag = True

        # encoder
        self.layer1 = conv_block('conv', 2, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer2 = down_sampling('conv', 16, 16, self.kernel_size, 2, self.padding, self.acti)
        self.layer3 = conv_block('conv', 16, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer4 = down_sampling('conv', 32, 32, self.kernel_size, 2, self.padding, self.acti)
        self.layer5 = conv_block('conv', 32, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer6 = down_sampling('conv', 64, 128, self.kernel_size, 2, self.padding, self.acti)
        # decoder
        self.layer7 = up_sampling('trans', 128, 64, self.kernel_size, 2, self.padding, self.acti)
        self.layer8 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer9 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer10 = up_sampling('trans', 64, 32, self.kernel_size, 2, self.padding, self.acti)
        self.layer11 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer12 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer13 = up_sampling('trans', 32, 16, self.kernel_size, 2, self.padding, self.acti)    
        self.layer14 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer15 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer16 = conv_block('conv', 16, 1, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)

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

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class UNET_MIA(nn.Module):
    # Ref: Towards lower-dose PET using physics-based uncertainty-aware 
    # multimodal learning with robustness to out-of-distribution data
    # Page 102187, Figure 2
    # Maxpooling(stride=2) + Upsampling(factor=2)
    def __init__(self):
        super(UNET_MIA, self).__init__()
        print('UNET_MIA: UNET in MIA paper')
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.bn_flag = True

        # fuse PET and CT with 1x1 kernel
        self.layer1 = conv_block('conv', 2, 1, 1, 1, 0, self.acti, self.bn_flag)
        # encoder
        self.layer2 = conv_block('conv', 1, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer3 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer4 = down_sampling('maxpooling', kernel_size=2, stride=2, padding=0)
        self.layer5 = conv_block('conv', 64, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer6 = conv_block('conv', 128, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer7 = down_sampling('maxpooling', kernel_size=2, stride=2, padding=0)
        self.layer8 = conv_block('conv', 128, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer9 = conv_block('conv', 256, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer10 = down_sampling('maxpooling', kernel_size=2, stride=2, padding=0)
        self.layer11 = conv_block('conv', 256, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer12 = conv_block('conv', 512, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer13 = down_sampling('maxpooling', kernel_size=2, stride=2, padding=0)
        self.layer14 = conv_block('conv', 512, 1024, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        # decoder
        self.layer15 = conv_block('conv', 1024, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer16 = up_sampling(mode='bilinear')
        self.layer17 = conv_block('conv', 1024, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer18 = conv_block('conv', 512, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer19 = up_sampling(mode='bilinear')
        self.layer20 = conv_block('conv', 512, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer21 = conv_block('conv', 256, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer22 = up_sampling(mode='bilinear')
        self.layer23 = conv_block('conv', 256, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer24 = conv_block('conv', 128, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer25 = up_sampling(mode='bilinear')
        self.layer26 = conv_block('conv', 128, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer27 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer28 = conv_block('conv', 64, 1, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        res1 = out
        out = self.layer6(self.layer5(self.layer4(out)))
        res2 = out
        out = self.layer9(self.layer8(self.layer7(out)))
        res3 = out
        out = self.layer12(self.layer11(self.layer10(out)))
        res4 = out
        out = self.layer16(self.layer15(self.layer14(self.layer13(out))))
        out = torch.cat([res4, out], dim=1)
        out = self.layer19(self.layer18(self.layer17(out)))
        out = torch.cat([res3, out], dim=1)
        out = self.layer22(self.layer21(self.layer20(out)))
        out = torch.cat([res2, out], dim=1)
        out = self.layer25(self.layer24(self.layer23(out)))
        out = torch.cat([res1, out], dim=1)
        out = self.layer28(self.layer27(self.layer26(out)))
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class UNET_TMI(nn.Module):
    # 3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis
    # Page 1311, Figure 3
    # Generator in GAN
    # Conv2d(stride=2) + Upsampling(factor=2)
    def __init__(self):
        super(UNET_TMI, self).__init__()
        print('UNET_TMI: UNET in TMI paper')
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.bn_flag = True

        # fuse PET and CT with 1x1 kernel
        self.layer1 = conv_block('conv', 2, 1, 1, 1, 0, self.acti, self.bn_flag)
        # encoder
        self.layer2 = conv_block('conv', 1, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer3 = down_sampling('conv', 64, 128, self.kernel_size, 2, self.padding, self.acti, self.bn_flag)
        self.layer4 = down_sampling('conv', 128, 256, self.kernel_size, 2, self.padding, self.acti, self.bn_flag)
        self.layer5 = down_sampling('conv', 256, 512, self.kernel_size, 2, self.padding, self.acti, self.bn_flag)
        self.layer6 = down_sampling('conv', 512, 512, self.kernel_size, 2, self.padding, self.acti, self.bn_flag)
        # decoder
        self.layer7 = up_sampling(mode='bilinear')
        self.layer8 = conv_block('conv', 512, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer9 = up_sampling(mode='bilinear')
        self.layer10 = conv_block('conv', 1024, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer11 = up_sampling(mode='bilinear')
        self.layer12 = conv_block('conv', 512, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer13 = up_sampling(mode='bilinear')
        self.layer14 = conv_block('conv', 256, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer15 = conv_block('conv', 128, 1, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        
    def forward(self, x):
        out = self.layer2(self.layer1(x))
        res1 = out
        out = self.layer3(out)
        res2 = out
        out = self.layer4(out)
        res3 = out
        out = self.layer5(out)
        res4 = out
        out = self.layer8(self.layer7(self.layer6(out)))
        out = torch.cat([res4, out], dim=1)
        out = self.layer10(self.layer9(out))
        out = torch.cat([res3, out], dim=1)
        out = self.layer12(self.layer11(out))
        out = torch.cat([res2, out], dim=1)
        out = self.layer14(self.layer13(out))
        out = torch.cat([res1, out], dim=1)
        out = self.layer15(out)
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

# Ref: Deep Convolutional Framelets: A General Deep Learning Framework for Inverse Problems
# Page 1025, Figure 10(b)
# Multi-Channel Input
# SIAM on Imaging Sciences
class Framelets(nn.Module):
    def __init__(self):
        super(Framelets, self).__init__()
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.bn_flag = True
        self.num_channels = 64
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.iwt = DWTInverse(wave='haar', mode='zero')

        # downsampling (wavelet transform by 4 times)
        self.scale0_conv1 = conv_block('conv', 2, self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale0_conv2 = conv_block('conv', self.num_channels, self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale1_conv1 = conv_block('conv', self.num_channels, 2*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale1_conv2 = conv_block('conv', 2*self.num_channels, 2*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale2_conv1 = conv_block('conv', 2*self.num_channels, 4*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale2_conv2 = conv_block('conv', 4*self.num_channels, 4*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale3_conv1 = conv_block('conv', 4*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale3_conv2 = conv_block('conv', 8*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        
        # bottleneck layers
        self.scale4_conv11 = conv_block('conv', 8*self.num_channels, 16*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv12 = conv_block('conv', 16*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv21 = conv_block('conv', 8*self.num_channels, 16*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv22 = conv_block('conv', 16*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv31 = conv_block('conv', 8*self.num_channels, 16*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv32 = conv_block('conv', 16*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv41 = conv_block('conv', 8*self.num_channels, 16*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale4_conv42 = conv_block('conv', 16*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        
        # upsampling (inverse wavelet transform by 4 times)
        self.scale3_deconv1 = conv_block('conv', 5*8*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale3_deconv2 = conv_block('conv', 8*self.num_channels, 4*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale2_deconv1 = conv_block('conv', 5*4*self.num_channels, 4*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale2_deconv2 = conv_block('conv', 4*self.num_channels, 2*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale1_deconv1 = conv_block('conv', 5*2*self.num_channels, 2*self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale1_deconv2 = conv_block('conv', 2*self.num_channels, self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale0_deconv1 = conv_block('conv', 5*self.num_channels, self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.scale0_deconv2 = conv_block('conv', self.num_channels, self.num_channels, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)

    def forward(self, x):
        return True



