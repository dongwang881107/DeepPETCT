import torch.nn as nn
import torch
import sys

def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

class REDCNN(nn.Module):
    def __init__(self, out_ch=96, kernel_size=5, stride=1, padding=0):
        super(REDCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=kernel_size, stride=stride, padding=padding)

        self.relu = nn.ReLU()

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
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
        else:
            print('[conv] or [trans]')
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

class SA(nn.Module):
    # Self attention layer
    def __init__(self,in_dim):
        super(SA,self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
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

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(batch_size,C,width,height)
        
        out = self.gamma*out + x
        return out
