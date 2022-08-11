import torch.nn as nn
import torch
import sys

def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

class REDCNN_TMI(nn.Module):
    # Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network
    # Page 2526, Figure 1
    # Conv2d (s=1) + ConvTranspose2d (s=1)
    def __init__(self, out_ch=96, kernel_size=5, stride=1, padding=0):
        super(REDCNN_TMI, self).__init__()
        print('RENCNN_TMI: REDCNN in TMI paper')
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
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        res2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        res3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += res3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += res2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        # out += res1
        out = self.relu(out)
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class REDCNN_BN(nn.Module):
    # REDCNN_TMI with batch normalization
    def __init__(self):
        super(REDCNN_BN, self).__init__()
        print('REDCNN_BN: RENCNN with batch normalization')
        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.out_channel = 96
        self.acti = 'relu'
        # encoder
        self.layer1 = ConvBlock('conv', 2, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer2 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer3 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer4 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer5 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        # decoder
        self.layer6 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer7 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer8 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer9 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer10 = ConvBlock('trans', self.out_channel, 1, self.kernel_size, self.stride, self.padding, self.acti)

    def forward(self, x):
        # encoder
        out = self.layer2(self.layer1(x))
        res1 = out
        out = self.layer4(self.layer3(out))
        res2 = out
        # decoder
        out = self.layer6(self.layer5(out))
        out = out + res2
        out = self.layer8(self.layer7(out))
        out = out + res1
        out = self.layer10(self.layer9(out))
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class REDCNN_SA(nn.Module):
    # REDCNN_TMI with batch normalization
    def __init__(self):
        super(REDCNN_SA, self).__init__()
        print('RENCNN_SA: RENCNN with batch normalization and self-attention')
        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.out_channel = 96
        self.acti = 'relu'
        # encoder
        self.layer1 = ConvBlock('conv', 2, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer2 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer3 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer4 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer5 = ConvBlock('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        # decoder
        self.layer6 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer7 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer8 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer9 = ConvBlock('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti)
        self.layer10 = ConvBlock('trans', self.out_channel, 1, self.kernel_size, self.stride, self.padding, self.acti)

        self.sa = SelfAttenBlock(self.out_channel)

    def forward(self, x):
        # encoder
        out = self.layer2(self.layer1(x))
        res1 = out
        out = self.layer4(self.layer3(out))
        res2 = out
        # decoder
        out = self.layer6(self.sa(self.layer5(out)))
        out += res2
        out = self.layer8(self.layer7(out))
        out += res1
        out = self.layer10(self.layer9(out))
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
            print('[conv] or [trans]')
            sys.exit(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.acti = get_acti(acti)

    def forward(self, x):
        x = self.acti(self.bn(self.conv(x)))
        return x

class SelfAttenBlock(nn.Module):
    # Self attention layer
    def __init__(self,in_dim):
        super(SelfAttenBlock, self).__init__()
        
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
