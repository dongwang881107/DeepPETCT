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
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti, bn_flag=False):
    if mode == 'conv':
        conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    elif mode == 'trans':
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        print('[conv] | [trans]')
        sys.exit(0)
    if bn_flag == True:
        bn = nn.BatchNorm3d(out_channels)
    acti = get_acti(acti)
    layers = [conv,bn,acti] if bn_flag == True else [conv,acti]
    return nn.Sequential(*layers)

# down sampling block
def down_sampling(mode, kernel_size, stride, padding, in_channels=None, out_channels=None, acti=None):
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

# Self-Attention Block
class SelfAttenBlock(nn.Module):
    def __init__(self,in_dim):
        super(SelfAttenBlock, self).__init__()
        
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        batch_size, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(batch_size,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(batch_size,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(batch_size,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(batch_size,C,width,height)
        
        out = self.gamma*out + x
        return out, attention