import torch.nn as nn
import torch
import sys


# get activation function
def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

# initialize model
def weights_init(m):
    # in PyTorch
    # default init for Conv3d: kaiming_uniform_ (uniform sample in (-bound, bound))
    # default init for Linear: kaiming_uniform_ (uniform sample in (-bound, bound))
    # default init for BatchNorm3d: weight (uniform sample in (-bound, bound))/bias (0)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif classname.find('BatchNorm3d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')

# convolution block: conv-[bn]-acti
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti, bn_flag=True):
    if mode == 'conv':
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    elif mode == 'trans':
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        print('[conv] | [trans]')
        sys.exit(0)
    bn = nn.BatchNorm3d(out_channels)
    acti = get_acti(acti)
    layers = [conv,bn,acti]
    return nn.Sequential(*layers)

# down sampling block
def down_sampling(mode, kernel_size, stride, padding, in_channels=None, out_channels=None, acti=None, bn_flag=True):
    if mode == 'conv':
        down = conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti)
    elif mode =='maxpooling':
        down = nn.MaxPool3d(kernel_size, stride, padding)
    else:
        print('[conv] | [maxpooling]')
        sys.exit(0)
    layers = [down]
    return nn.Sequential(*layers)

# up sampling block
def up_sampling(mode, kernel_size, stride, padding, in_channels=None, out_channels=None, acti=None, bn_flag=True):
    if mode == 'trans':
        up = conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti)
    elif mode == 'interp':
        up = nn.Upsample(scale_factor=2, mode='trilinear')
    else:
        print('[conv] | [interp]')
        sys.exit(0)
    layers = [up]
    return nn.Sequential(*layers)
