import torch
import sys
import math
import torch.nn as nn

from torch import autograd
from torchvision.models import vgg19

# get activation function
def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

# convolutional block: conv-bn-acti
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti, bn_flag=False):
    if mode == 'conv':
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif mode == 'trans':
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        print('[conv] or [trans]')
        sys.exit(0)
    if bn_flag == True:
        bn = nn.BatchNorm2d(out_channels)
    acti = get_acti(acti)
    layers = [conv,bn,acti] if bn_flag == True else [conv,acti]
    return nn.Sequential(*layers)

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
        self.layer_in = conv_block('conv',2,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer_hidden = conv_block('conv',self.num_channels,self.num_channels,self.kernel_size,self.stride,self.padding,self.acti)
        self.layer_out = conv_block('conv',self.num_channels,1,self.kernel_size,self.stride,self.padding,self.acti)

    def forward(self, x):
        out = self.layer_in(x)
        for _ in range(2,8):
            out = self.layer_hidden(out)
        out = self.layer_out(out)
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
class WGANVGG(nn.Module):
    def __init__(self, patch_size=64):
        super(WGANVGG, self).__init__()
        self.generator = WGANVGG_generator()
        self.discriminator = WGANVGG_discriminator(patch_size)
        self.extractor = WGANVGG_extractor()
        self.perc_metric = nn.MSELoss()

    # discriminator loss
    def discriminator_loss(self, fake, real, d_fake, d_real, lambda2):
        grad_loss = self.gradient_loss(fake, real)
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
    def gradient_loss(self, fake, real):
        eta = torch.FloatTensor(real.size(0),1,1,1).uniform_(0,1)
        interp = (eta*real+((1-eta)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        gradients = autograd.grad(outputs=d_interp, inputs=interp,\
            grad_outputs=torch.ones(d_interp.size()).requires_grad_(False),\
            create_graph=True, retain_graph=True)[0]
        grad_loss = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_loss
