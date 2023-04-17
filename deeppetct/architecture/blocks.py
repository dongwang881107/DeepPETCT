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
    # default init for Conv2d: kaiming_uniform_ (uniform sample in (-bound, bound))
    # default init for Linear: kaiming_uniform_ (uniform sample in (-bound, bound))
    # default init for BatchNorm2d: weight (uniform sample in (-bound, bound))/bias (0)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')

# convolution block: conv-bn-acti
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti):
    if mode == 'conv':
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    elif mode == 'trans':
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    else:
        print('[conv] | [trans]')
        sys.exit(0)
    bn = nn.BatchNorm2d(out_channels)
    acti = get_acti(acti)
    layers = [conv,bn,acti]
    return nn.Sequential(*layers)

# Self-Attention Block
class SelfAttenBlock(nn.Module):
    def __init__(self,in_dim):
        super(SelfAttenBlock, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channel, width, height = x.size()
        proj_query  = self.query_conv(x).view(batch_size,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(batch_size,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(batch_size,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(batch_size,channel,width,height)
        
        out = self.gamma*out + x
        return out
    
def atten_block(in_dim):
    atten_block = SelfAttenBlock(in_dim)
    return atten_block