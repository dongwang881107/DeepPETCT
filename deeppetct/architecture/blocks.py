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
class SelfAttention3D(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(SelfAttention3D, self).__init__()
        
        # Define the query, key, and value convolutions
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = num_heads , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = num_heads , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = num_heads , kernel_size= 1)

        # Define scaler
        self.gamma = nn.Parameter(torch.zeros(1))

        # Define the softmax operation
        self.softmax  = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Get the spatial dimensions of the input tensor
        batch_size, channels, depth, width, height = x.size()

        # Calculate the queries, keys, and values
        query  = self.query_conv(x).view(batch_size, -1, depth*width*height).permute(0,2,1)
        key =  self.key_conv(x).view(batch_size, -1, depth*width*height)
        value = self.value_conv(x).view(batch_size, -1, depth*width*height)

        # Calculate the attention scores
        attention =  torch.bmm(query, key) #! easy to get out of memory problem
        attention = self.softmax(attention) 

        # Calculate the weighted sum of the values
        out = torch.bmm(value, attention.permute(0,2,1))
        out = out.view(batch_size, channels, depth, width, height)
        
        # Apply scaling factor and add residual
        out = self.gamma * out + x

        return out

class BlockwiseAttention3D(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(BlockwiseAttention3D, self).__init__()
        self.block_size = 32
        self.num_heads = num_heads
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # Compute query, key, and value
        query = self.query_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        key = self.key_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        value = self.value_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        
        # Compute attention scores
        query_blocks = query.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size).unfold(4, self.block_size, self.block_size)
        key_blocks = key.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size).unfold(4, self.block_size, self.block_size)
        scores = torch.einsum('bijklm,bipqrs->bjklmrs', query_blocks, key_blocks)
        scores = scores / (self.block_size ** 3)
        attention = self.softmax(scores)
        
        # Compute output
        value_blocks = value.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size).unfold(4, self.block_size, self.block_size)
        out_blocks = torch.einsum('bjklmrs,bipqrs->bijklm', attention, value_blocks)
        out = out_blocks.view(batch_size, self.num_heads, channels, depth, height, width)
        out = out.sum(dim=1)
        
        return out

class LocalAttention3D(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(LocalAttention3D, self).__init__()
        self.window_size = 32
        self.num_heads = num_heads
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=num_heads, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        
        # Compute query, key, and value
        query = self.query_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        key = self.key_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        value = self.value_conv(x.unsqueeze(1)).view(batch_size * self.num_heads, -1, depth, height, width)
        
        # Compute attention scores
        scores = torch.einsum('bijk,bilm->bjklm', query, key)
        scores = scores / (self.window_size ** 0.5)
        attention = self.softmax(scores)
        
        # Compute output
        out = torch.einsum('bjklm,bilm->bijk', attention, value)
        out = out.view(batch_size, self.num_heads, channels, depth, height, width)
        out = out.sum(dim=1)
        
        return out
    
def atten_block(in_dim, num_heads, mode):
    if mode == 'self':
        attention_block = SelfAttention3D(in_dim, num_heads)
    elif mode == 'blockwise':
        attention_block = BlockwiseAttention3D(in_dim, num_heads)
    elif mode == 'local':
        attention_block = LocalAttention3D(in_dim, num_heads)
    else:
        print("[self | blockwise | local]")
        sys.exit(0)
    return attention_block
