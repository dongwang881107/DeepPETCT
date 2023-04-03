import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch import autograd
from torchvision.models import vgg19
from deeppetct.architecture.blocks import *

# generator
class WGANVGG_generator(nn.Module):
    def __init__(self):
        super(WGANVGG_generator, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.acti = 'relu'
        self.num_channels = 32
        
        # convolutional layers
        self.layer1 = conv_block('conv',2,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer2 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer3 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer4 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer5 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer6 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer7 = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer8 = conv_block('conv',self.num_channels,1,self.kernel_size,self.stride,self.padding,self.acti)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        return out

# discriminator
class WGANVGG_discriminator(nn.Module):
    def __init__(self, patch_size):
        super(WGANVGG_discriminator, self).__init__()
        self.patch_size = patch_size
        self.kernel_size = 3
        self.padding = 0
        self.bn_flag = False
        self.acti = 'leaky_relu'
        self.acti_func = get_acti(self.acti)
        self.output_size = math.ceil((math.ceil((math.ceil((self.patch_size-2-2)/2)-2-2)/2)-2-2)/2)
        # convolutional layers
        self.layer1 = conv_block('conv',1,64,self.kernel_size,1,0,self.acti,self.bn_flag)
        self.layer2 = conv_block('conv',64,64,self.kernel_size,2,0,self.acti,self.bn_flag)
        self.layer3 = conv_block('conv',64,128,self.kernel_size,1,0,self.acti,self.bn_flag)
        self.layer4 = conv_block('conv',128,128,self.kernel_size,2,0,self.acti,self.bn_flag)
        self.layer5 = conv_block('conv',128,256,self.kernel_size,1,0,self.acti,self.bn_flag)
        self.layer6 = conv_block('conv',256,256,self.kernel_size,2,0,self.acti,self.bn_flag)
        # fully-connected layers
        self.fc1 = nn.Linear(256*self.output_size*self.output_size,1024)
        self.fc2 = nn.Linear(1024,1)

    def forward(self, x):
        out = self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.fc2(self.acti_func(self.fc1(out)))
        return out

# feature extractor
class WGANVGG_extractor(nn.Module):
    def __init__(self):
        super(WGANVGG_extractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, x):
        out = self.feature(x)
        return out

# WGANVGG: GAN with Wasserstein + perceptual loss
# Low-Dose CT Image Denoising Using a Generative Adversarial Network With Wasserstein Distance and Perceptual Loss
# 2018, TMI, Uni-Modality method
class WGANVGG(nn.Module):
    def __init__(self, patch_size, lambda1, lambda2):
        super(WGANVGG, self).__init__()
        self.generator = WGANVGG_generator()
        self.discriminator = WGANVGG_discriminator(patch_size)
        self.extractor = WGANVGG_extractor()
        self.perc_metric = nn.MSELoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # discriminator loss
    def discriminator_loss(self, fake, real, d_fake, d_real):
        grad_loss = self.gradient_loss(fake, real)
        dis_loss = -torch.mean(d_real) + torch.mean(d_fake) + self.lambda2*grad_loss
        return dis_loss

    # generator loss
    def generator_loss(self, fake, real, d_fake):
        perc_loss = self.perceptual_loss(fake, real)
        gen_loss = -torch.mean(d_fake) + self.lambda1*perc_loss
        return gen_loss

    # perceptual loss
    def perceptual_loss(self, fake, real):
        fake = fake.repeat(1,3,1,1)
        real = real.repeat(1,3,1,1)
        fake_feature = self.extractor(fake)
        real_feature = self.extractor(real)
        perc_loss = self.perc_metric(fake_feature, real_feature)
        return perc_loss

    # gradient loss
    def gradient_loss(self, fake, real):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        eta = torch.Tensor(real.size(0),1,1,1).uniform_(0,1).to(device)
        interp = (eta*real+((1-eta)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        gradients = autograd.grad(outputs=d_interp, inputs=interp,\
            grad_outputs=torch.ones(d_interp.size()).requires_grad_(False).to(device),\
            create_graph=True, retain_graph=True)[0]
        grad_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_loss

# generator
class GAN_TMI_generator(nn.Module):
    def __init__(self):
        super(GAN_TMI_generator, self).__init__()
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'leaky_relu'
        self.bn_flag = True

        # fuse PET and CT with 1x1 kernel
        self.layer1 = conv_block('conv', 2, 1, 1, 1, 0, self.acti, self.bn_flag)
        # encoder
        self.layer2 = conv_block('conv', 1, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer3 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti, bn_flag=self.bn_flag)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=128, out_channels=256, acti=self.acti, bn_flag=self.bn_flag)
        self.layer5 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=256, out_channels=512, acti=self.acti, bn_flag=self.bn_flag)
        self.layer6 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=512, out_channels=512, acti=self.acti, bn_flag=self.bn_flag)
        # decoder
        self.layer7 = up_sampling('interp', self.kernel_size, 2, self.padding)
        self.layer8 = conv_block('conv', 512, 512, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer9 = up_sampling('interp', self.kernel_size, 2, self.padding)
        self.layer10 = conv_block('conv', 1024, 256, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer11 = up_sampling('interp', self.kernel_size, 2, self.padding)
        self.layer12 = conv_block('conv', 512, 128, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer13 = up_sampling('interp', self.kernel_size, 2, self.padding)
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

# discriminator
class GAN_TMI_discriminator(nn.Module):
    def __init__(self, patch_size):
        super(GAN_TMI_discriminator, self).__init__()
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'leaky_relu'
        self.bn_flag = True
        self.output_size = int(patch_size/2/2/2/2)

        self.layer1 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=1, out_channels=64, acti=self.acti, bn_flag=self.bn_flag)
        self.layer2 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti, bn_flag=self.bn_flag)
        self.layer3 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=128, out_channels=256, acti=self.acti, bn_flag=self.bn_flag)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=256, out_channels=512, acti=self.acti, bn_flag=self.bn_flag)

        self.fc = nn.Linear(512*self.output_size*self.output_size,1)
    
    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = out.view(-1, 512*self.output_size*self.output_size)
        out = F.sigmoid(self.fc(out))
        return out

# 3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis
# 2019, IEEE Transactions on Medical Imaging
class GAN_TMI(nn.Module):
    def __init__(self, patch_size, lambda1):
        super(GAN_TMI, self).__init__()
        self.generator = GAN_TMI_generator()
        self.discriminator = GAN_TMI_discriminator(patch_size)
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

