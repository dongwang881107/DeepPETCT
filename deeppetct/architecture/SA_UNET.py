import torch.nn as nn
import torch
import sys

def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

class SA_UNET(nn.Module):
    def __init__(self):
        super(SA_UNET, self).__init__()
        # mode, in_channel, out_channel, kernel_size, stride, padding, acti
        self.cb1 = ConvBlock('conv',2,16,5,1,'same','relu')
        self.cb2 = ConvBlock('conv',16,16,5,2,'valid','relu')
        self.cb3 = ConvBlock('conv',16,32,5,1,'same','relu')
        self.cb4 = ConvBlock('conv',32,32,5,2,'valid','relu')
        self.cb5 = ConvBlock('conv',32,64,5,1,'same','relu')
        self.cb6 = ConvBlock('conv',64,128,5,2,'valid','relu')

        self.ab = Self_Attn(128,'relu')

        self.cb7 = ConvBlock('trans',128,64,5,2,'valid','relu')
        self.cb8 = ConvBlock('conv',64,64,5,1,'same','relu')
        self.cb9 = ConvBlock('conv',64,64,5,1,'same','relu')
        self.cb10 = ConvBlock('trans',64,32,5,2,'valid','relu')
        self.cb11 = ConvBlock('conv',32,32,5,1,'same','relu')
        self.cb12 = ConvBlock('conv',32,32,5,1,'same','relu')      
        self.cb13 = ConvBlock('trans',32,16,5,2,'valid','relu')
        self.cb14 = ConvBlock('conv',16,16,5,1,'same','relu')
        self.cb15 = ConvBlock('conv',16,16,5,1,'same','relu')
        self.cb16 = ConvBlock('conv',16,1,5,1,'same','relu')

    def forward(self, x):
        # encoder
        # residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        # out += residual_1
        out = self.relu(out)
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class ConvBlock(nn.Module):
    def __init__(self, mode, in_channels, out_channels, kernel_size, stride, padding, acti):
        super().__init__()
        if mode == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        elif mode == 'trans':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            print('[conv] or [trans] only!')
            sys.exit(0)

        self.bn = nn.BatchNorm2d(out_channels)
        self.acti = get_acti(acti)

        self.blocks = nn.Sequential(
            self.conv,
            self.bn,
            self.acti
        )

    def forward(self, x):
        x = self.blocks(x)
        return x



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
