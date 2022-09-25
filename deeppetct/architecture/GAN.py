import torch
import math
import torch.nn as nn

from torch import autograd
from torchvision.models import vgg19
from .blocks import *

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
        self.acti = 'leaky_relu'
        self.acti_func = get_acti(self.acti)
        self.output_size = math.ceil((math.ceil((math.ceil((self.patch_size-2-2)/2)-2-2)/2)-2-2)/2)
        # convolutional layers
        self.layer1 = conv_block('conv',1,64,self.kernel_size,1,0,self.acti)
        self.layer2 = conv_block('conv',64,64,self.kernel_size,2,0,self.acti)
        self.layer3 = conv_block('conv',64,128,self.kernel_size,1,0,self.acti)
        self.layer4 = conv_block('conv',128,128,self.kernel_size,2,0,self.acti)
        self.layer5 = conv_block('conv',128,256,self.kernel_size,1,0,self.acti)
        self.layer6 = conv_block('conv',256,256,self.kernel_size,2,0,self.acti)
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
    def __init__(self, patch_size=64):
        super(WGANVGG, self).__init__()
        self.generator = WGANVGG_generator()
        self.discriminator = WGANVGG_discriminator(patch_size)
        self.extractor = WGANVGG_extractor()
        self.perc_metric = nn.MSELoss()

    # discriminator loss
    def discriminator_loss(self, fake, real, d_fake, d_real, lambda2, device):
        grad_loss = self.gradient_loss(fake, real, device)
        dis_loss = -torch.mean(d_real) + torch.mean(d_fake) + lambda2*grad_loss
        return (dis_loss, grad_loss)

    # generator loss
    def generator_loss(self, fake, real, d_fake, lambda1):
        perc_loss = self.perceptual_loss(fake, real)
        gen_loss = -torch.mean(d_fake) + lambda1*perc_loss
        return (gen_loss, perc_loss)

    # perceptual loss
    def perceptual_loss(self, fake, real):
        fake = fake.repeat(1,3,1,1)
        real = real.repeat(1,3,1,1)
        fake_feature = self.extractor(fake)
        real_feature = self.extractor(real)
        perc_loss = self.perc_metric(fake_feature, real_feature)
        return perc_loss

    # gradient loss
    def gradient_loss(self, fake, real, device):
        eta = torch.Tensor(real.size(0),1,1,1).uniform_(0,1).to(device)
        interp = (eta*real+((1-eta)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        gradients = autograd.grad(outputs=d_interp, inputs=interp,\
            grad_outputs=torch.ones(d_interp.size()).requires_grad_(False).to(device),\
            create_graph=True, retain_graph=True)[0]
        grad_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_loss
