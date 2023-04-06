import torch
import torch.nn as nn
import torch.nn.functional as F

from deeppetct.architecture.blocks import *

# generator
class LAGANS_generator(nn.Module):
    def __init__(self):
        super(LAGANS_generator, self).__init__()
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'leaky_relu'

        # fuse PET and CT with 1x1 kernel
        self.layer1 = conv_block('conv', 2, 1, 1, 1, 0, self.acti)
        # encoder
        self.layer2 = conv_block('conv', 1, 64, self.kernel_size, 1, self.padding, self.acti)
        self.layer3 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=128, out_channels=256, acti=self.acti)
        self.layer5 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=256, out_channels=512, acti=self.acti)
        self.layer6 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=512, out_channels=512, acti=self.acti)
        # decoder
        self.layer7 = up_sampling('interp', self.kernel_size, 1, self.padding)
        self.layer8 = conv_block('conv', 512, 512, self.kernel_size, 1, self.padding, self.acti)
        self.layer9 = up_sampling('interp', self.kernel_size, 1, self.padding)
        self.layer10 = conv_block('conv', 1024, 256, self.kernel_size, 1, self.padding, self.acti)
        self.layer11 = up_sampling('interp', self.kernel_size, 1, self.padding)
        self.layer12 = conv_block('conv', 512, 128, self.kernel_size, 1, self.padding, self.acti)
        self.layer13 = up_sampling('interp', self.kernel_size, 1, self.padding)
        self.layer14 = conv_block('conv', 256, 64, self.kernel_size, 1, self.padding, self.acti)
        self.layer15 = conv_block('conv', 128, 1, self.kernel_size, 1, self.padding, self.acti)
        
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

# discriminator
class LAGANS_discriminator(nn.Module):
    def __init__(self, patch_size):
        super(LAGANS_discriminator, self).__init__()
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'leaky_relu'
        self.output_size = int(patch_size/2/2/2/2)

        self.layer1 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=1, out_channels=64, acti=self.acti)
        self.layer2 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti)
        self.layer3 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=128, out_channels=256, acti=self.acti)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=256, out_channels=512, acti=self.acti)

        self.fc = nn.Linear(512*self.output_size*self.output_size,1)
    
    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = out.view(-1, 512*self.output_size*self.output_size)
        out = F.sigmoid(self.fc(out))
        return out

# 3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis
# 2019, IEEE Transactions on Medical Imaging
class LAGANS(nn.Module):
    def __init__(self, patch_size, lambda1):
        super(LAGANS, self).__init__()
        self.generator = LAGANS_generator()
        self.discriminator = LAGANS_discriminator(patch_size)
        self.gen_metric = nn.L1Loss()
        self.lambda1 = lambda1

    # discriminator loss
    def discriminator_loss(self, fake, real, d_fake, d_real):
        dis_loss = torch.mean(torch.log(d_real)) + torch.mean(torch.log(1-d_fake))
        return dis_loss

    # generator loss
    def generator_loss(self, fake, real, d_fake):
        l1_loss = self.gen_metric(fake, real)
        gen_loss = -torch.mean(torch.log(d_fake)) + self.lambda1*l1_loss
        return gen_loss

