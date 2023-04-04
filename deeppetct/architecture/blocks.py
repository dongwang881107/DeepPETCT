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
def conv_block(mode, in_channels, out_channels, kernel_size, stride, padding, acti):
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
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        
        # Define the query, key, and value convolutions
        self.query_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)

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

class SlicewiseAttention3D(nn.Module):
    def __init__(self, in_dim, num_slices):
        super(SlicewiseAttention3D, self).__init__()
        self.num_slices = num_slices # number of slices per iteration
        
        # Define the query, key, and value convolutions
        self.query_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)

        # Define scaler
        self.gamma = nn.Parameter(torch.zeros(1))

        # Define the softmax operation
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Get the spatial dimensions of the input tensor
        batch_size, channels, depth, width, height = x.size()

        # split into different slices
        x = x.view(batch_size, channels, depth//self.num_slices, self.num_slices, width, height)

        # Slice-wise attention
        out = torch.zeros_like(x)
        for i in range(depth//self.num_slices):
            # Extract slices
            x_slice = x[:,:,i,:,:,:]

            # Calculate the queries, keys, and values
            query  = self.query_conv(x_slice).view(batch_size, -1, self.num_slices*width*height).permute(0,2,1)
            key =  self.key_conv(x_slice).view(batch_size, -1, self.num_slices*width*height)
            value = self.value_conv(x_slice).view(batch_size, -1, self.num_slices*width*height)

            # Calculate the attention scores
            attention =  torch.bmm(query, key)
            attention = self.softmax(attention) 

            # Calculate the weighted sum of the values
            out_slice = torch.bmm(value, attention.permute(0,2,1))
            out_slice = out_slice.view(batch_size, channels, self.num_slices, width, height)

            out[:,:,i,:,:,:] = out_slice  
        
        # Apply scaling factor and add residual
        out = out.view(batch_size, channels, depth, height, width)
        x = x.view(batch_size, channels, depth, height, width)
        out = self.gamma * out + x

        return out

class BlockwiseAttention3D(nn.Module):
    def __init__(self, in_dim, num_blocks):
        super(BlockwiseAttention3D, self).__init__()
        self.num_blocks = num_blocks # number of blocks for each dimension

        # Define the query, key, and value convolutions
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Define scaler
        self.gamma = nn.Parameter(torch.zeros(1))

        # Define the softmax operation
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # Get the spatial dimensions of the input tensor
        batch_size, channels, depth, height, width = x.size()

        # split x into small blocks
        x = x.view(batch_size, channels, self.num_blocks, depth//self.num_blocks, self.num_blocks,\
                    height//self.num_blocks, self.num_blocks, width//self.num_blocks)

        # Block-wise attention
        out = torch.zeros_like(x)
        for idx in range(self.num_blocks**3):
            # extrac one block 
            i = idx//(self.num_blocks**2)
            j = (idx//self.num_blocks)%self.num_blocks
            k = i%self.num_blocks
            x_block = x[:,:,i,:,j,:,k,:]
            
            # Compute query, key, and value
            query = self.query_conv(x_block).view(batch_size, -1, (depth//self.num_blocks)*(height//self.num_blocks)*(width//self.num_blocks)).permute(0,2,1)
            key = self.key_conv(x_block).view(batch_size, -1, (depth//self.num_blocks)*(height//self.num_blocks)*(width//self.num_blocks))
            value = self.value_conv(x_block).view(batch_size, -1, (depth//self.num_blocks)*(height//self.num_blocks)*(width//self.num_blocks))
            
            # Compute attention scores
            attention =  torch.bmm(query, key)
            attention = self.softmax(attention)
        
            # Calculate the weighted sum of the values
            out_block = torch.bmm(value, attention.permute(0,2,1))
            out_block = out_block.view(batch_size, channels, depth//self.num_blocks, width//self.num_blocks, height//self.num_blocks)

            out[:,:,i,:,j,:,k,:] = out_block
        
        # Apply scaling factor and add residual
        out = out.view(batch_size, channels, depth, height, width)
        x = x.view(batch_size, channels, depth, height, width)
        out = self.gamma * out + x 

        return out
    
def atten_block(in_dim, num_splits, mode):
    if mode == 'original':
        attention_block = SelfAttention3D(in_dim)
    if mode == 'slicewise':
        attention_block = SlicewiseAttention3D(in_dim, num_splits)
    elif mode == 'blockwise':
        attention_block = BlockwiseAttention3D(in_dim, num_splits)
    else:
        print("[original | slicewise | blockwise]")
        sys.exit(0)
    return attention_block
