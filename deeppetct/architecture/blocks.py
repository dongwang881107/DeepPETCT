import torch.nn as nn
import sys


# get activation function
def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

# convolution block: conv-[bn]-acti
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti, bn_flag=False):
    if mode == 'conv':
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif mode == 'trans':
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        print('[conv] | [trans]')
        sys.exit(0)
    if bn_flag == True:
        bn = nn.BatchNorm2d(out_channels)
    acti = get_acti(acti)
    layers = [conv,bn,acti] if bn_flag == True else [conv,acti]
    return nn.Sequential(*layers)

# down sampling block
def down_sampling(mode, kernel_size, stride, padding, in_channels=None, out_channels=None, acti=None):
    if mode == 'conv':
        down = conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti)
    elif mode =='maxpooling':
        down = nn.MaxPool2d(kernel_size, stride, padding)
    else:
        print('[conv] | [maxpooling]')
        sys.exit(0)
    layers = [down]
    return nn.Sequential(*layers)

# up sampling block
def up_sampling(mode, kernel_size, stride, padding, in_channels=None, out_channels=None, acti=None):
    if mode == 'trans':
        up = conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti)
    elif mode == 'interp_nearest':
        up = nn.Upsample(scale_factor=2, mode='nearest')
    elif mode =='interp_bilinear':
        up = nn.Upsample(scale_factor=2, mode='bilinear')
    elif mode == 'interp_bicubic':
        up = nn.Upsample(scale_factor=2, mode='bicubic')
    else:
        print('[conv] | [interp_nearest] | [interp_bilinear] | [interp_bicubic]')
        sys.exit(0)
    layers = [up]
    return nn.Sequential(*layers)