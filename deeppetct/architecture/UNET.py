import torch.nn as nn
import torch
import sys

def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

class UNET_MP(nn.Module):
    # Ref: Anatomically aided PET image reconstruction using deep neural networks
    # Page 5, Figure 2(b)
    # Conv2d(stride=2) + ConvTranspose2d(stride=2)
    # Not working, has overlapping artifacts
    def __init__(self):
        super(UNET_MP, self).__init__()
        print('UNET_MP: UNET in MP paper')
        # mode, in_channel, out_channel, kernel_size, stride, padding, acti
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        # encoder
        self.layer1 = ConvBlock('conv',2,16,self.kernel_size,1,self.padding,self.acti)
        self.layer2 = Down('conv',16,16,self.kernel_size,2,self.padding,self.acti)
        self.layer3 = ConvBlock('conv',16,32,self.kernel_size,1,self.padding,self.acti)
        self.layer4 = Down('conv',32,32,self.kernel_size,2,self.padding,self.acti)
        self.layer5 = ConvBlock('conv',32,64,self.kernel_size,1,self.padding,self.acti)
        self.layer6 = Down('conv',64,128,self.kernel_size,2,self.padding,self.acti)
        # decoder
        self.layer7 = Up('trans',128,64,self.kernel_size,2,self.padding,self.acti)
        self.layer8 = ConvBlock('conv',64,64,self.kernel_size,1,self.padding,self.acti)
        self.layer9 = ConvBlock('conv',64,64,self.kernel_size,1,self.padding,self.acti)
        self.layer10 = Up('trans',64,32,self.kernel_size,2,self.padding,self.acti)
        self.layer11 = ConvBlock('conv',32,32,self.kernel_size,1,self.padding,self.acti)
        self.layer12 = ConvBlock('conv',32,32,self.kernel_size,1,self.padding,self.acti)
        self.layer13 = Up('trans',32,16,self.kernel_size,2,self.padding,self.acti)    
        self.layer14 = ConvBlock('conv',16,16,self.kernel_size,1,self.padding,self.acti)
        self.layer15 = ConvBlock('conv',16,16,self.kernel_size,1,self.padding,self.acti)
        self.layer16 = ConvBlock('conv',16,1,self.kernel_size,1,self.padding,self.acti)

    def forward(self, x):
        out = self.layer1(x)
        res1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        res2 = out
        out = self.layer4(out)
        out = self.layer5(out)
        res3 = out
        out = self.layer6(out) 
        out = self.layer7(out) 
        out = out + res3
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = out + res2
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out + res1
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class UNET_MIA(nn.Module):
    # Ref: Towards lower-dose PET using physics-based uncertainty-aware 
    # multimodal learning with robustness to out-of-distribution data
    # Page 102187, Figure 2
    # Maxpooling(stride=2) + Upsampling(factor=2)
    def __init__(self):
        super(UNET_MIA, self).__init__()
        print('UNET_MIA: UNET in MIA paper')
        # mode, in_channel, out_channel, kernel_size, stride, padding, acti
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        # fuse PET and CT with 1x1 kernel
        self.layer1 = ConvBlock('conv',2,1,1,1,0,self.acti)
        # encoder
        self.layer2 = ConvBlock('conv',1,64,self.kernel_size,1,self.padding,self.acti)
        self.layer3 = ConvBlock('conv',64,64,self.kernel_size,1,self.padding,self.acti)
        self.layer4 = Down('maxpool',kernel_size=2,stride=2,padding=0)
        self.layer5 = ConvBlock('conv',64,128,self.kernel_size,1,self.padding,self.acti)
        self.layer6 = ConvBlock('conv',128,128,self.kernel_size,1,self.padding,self.acti)
        self.layer7 = Down('maxpool',kernel_size=2,stride=2,padding=0)
        self.layer8 = ConvBlock('conv',128,256,self.kernel_size,1,self.padding,self.acti)
        self.layer9 = ConvBlock('conv',256,256,self.kernel_size,1,self.padding,self.acti)
        self.layer10 = Down('maxpool',kernel_size=2,stride=2,padding=0)
        self.layer11 = ConvBlock('conv',256,512,self.kernel_size,1,self.padding,self.acti)
        self.layer12 = ConvBlock('conv',512,512,self.kernel_size,1,self.padding,self.acti)
        self.layer13 = Down('maxpool',kernel_size=2,stride=2,padding=0)
        self.layer14 = ConvBlock('conv',512,1024,self.kernel_size,1,self.padding,self.acti)
        # decoder
        self.layer15 = ConvBlock('conv',1024,512,self.kernel_size,1,self.padding,self.acti)
        self.layer16 = Up('bilinear')
        self.layer17 = ConvBlock('conv',1024,512,self.kernel_size,1,self.padding,self.acti)
        self.layer18 = ConvBlock('conv',512,256,self.kernel_size,1,self.padding,self.acti)
        self.layer19 = Up('bilinear')
        self.layer20 = ConvBlock('conv',512,256,self.kernel_size,1,self.padding,self.acti)
        self.layer21 = ConvBlock('conv',256,128,self.kernel_size,1,self.padding,self.acti)
        self.layer22 = Up('bilinear')
        self.layer23 = ConvBlock('conv',256,128,self.kernel_size,1,self.padding,self.acti)
        self.layer24 = ConvBlock('conv',128,64,self.kernel_size,1,self.padding,self.acti)
        self.layer25 = Up('bilinear')
        self.layer26 = ConvBlock('conv',128,64,self.kernel_size,1,self.padding,self.acti)
        self.layer27 = ConvBlock('conv',64,64,self.kernel_size,1,self.padding,self.acti)
        self.layer28 = ConvBlock('conv',64,1,self.kernel_size,1,self.padding,self.acti)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        res1 = out
        out = self.layer6(self.layer5(self.layer4(out)))
        res2 = out
        out = self.layer9(self.layer8(self.layer7(out)))
        res3 = out
        out = self.layer12(self.layer11(self.layer10(out)))
        res4 = out
        out = self.layer16(self.layer15(self.layer14(self.layer13(out))))
        out = torch.cat([res4, out], dim=1)
        out = self.layer19(self.layer18(self.layer17(out)))
        out = torch.cat([res3, out], dim=1)
        out = self.layer22(self.layer21(self.layer20(out)))
        out = torch.cat([res2, out], dim=1)
        out = self.layer25(self.layer24(self.layer23(out)))
        out = torch.cat([res1, out], dim=1)
        out = self.layer28(self.layer27(self.layer26(out)))
        return out

    @ classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')

class UNET_TMI(nn.Module):
    # 3D Auto-Context-Based Locality Adaptive Multi-Modality GANs for PET Synthesis
    # Page 1311, Figure 3
    # Generator in GAN
    # Conv2d(stride=2) + Upsampling(factor=2)
    def __init__(self):
        super(UNET_TMI, self).__init__()
        print('UNET_TMI: UNET in TMI paper')
        # mode, in_channel, out_channel, kernel_size, stride, padding, acti
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        # fuse PET and CT with 1x1 kernel
        self.layer1 = ConvBlock('conv',2,1,1,1,0,self.acti)
        # encoder
        self.layer2 = ConvBlock('conv',1,64,self.kernel_size,1,self.padding,self.acti)
        self.layer3 = Down('conv',64,128,self.kernel_size,2,self.padding,self.acti)
        self.layer4 = Down('conv',128,256,self.kernel_size,2,self.padding,self.acti)
        self.layer5 = Down('conv',256,512,self.kernel_size,2,self.padding,self.acti)
        self.layer6 = Down('conv',512,512,self.kernel_size,2,self.padding,self.acti)
        # decoder
        self.layer7 = Up('bilinear')
        self.layer8 = ConvBlock('conv',512,512,self.kernel_size,1,self.padding,self.acti)
        self.layer9 = Up('bilinear')
        self.layer10 = ConvBlock('conv',1024,256,self.kernel_size,1,self.padding,self.acti)
        self.layer11 = Up('bilinear')
        self.layer12 = ConvBlock('conv',512,128,self.kernel_size,1,self.padding,self.acti)
        self.layer13 = Up('bilinear')
        self.layer14 = ConvBlock('conv',256,64,self.kernel_size,1,self.padding,self.acti)
        self.layer15 = ConvBlock('conv',128,1,self.kernel_size,1,self.padding,self.acti)
        
    def forward(self, x):
        out = self.layer2(self.layer1(x))
        res1 = out
        out = self.layer3(out)
        res2 = out
        out = self.layer4(out)
        res3 = out
        out = self.layer5(out)
        res4 = out
        out = self.layer8(self.layer7(self.layer6(out)))
        out = torch.cat([res4, out], dim=1)
        out = self.layer10(self.layer9(out))
        out = torch.cat([res3, out], dim=1)
        out = self.layer12(self.layer11(out))
        out = torch.cat([res2, out], dim=1)
        out = self.layer14(self.layer13(out))
        out = torch.cat([res1, out], dim=1)
        out = self.layer15(out)
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

class Down(nn.Module):
    def __init__(self, mode, in_channels=None, out_channels=None, kernel_size=None, stride=None, padding=None, acti=None):
        super().__init__()
        if mode == 'conv':
            self.down = ConvBlock('conv', in_channels, out_channels, kernel_size, stride, padding, acti)
        elif mode == 'maxpool':
            self.down = nn.MaxPool2d(kernel_size, stride, padding)
        else:
            print('[conv] or [maxpool]')
            sys.exit(0)

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, mode, in_channels=None, out_channels=None, kernel_size=None, stride=None, padding=None, acti=None):
        super(ConvBlock, self).__init__()
        if mode == 'trans':
            self.up = ConvBlock('trans', in_channels, out_channels, kernel_size, stride, padding, acti)
        elif mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            print('[trans] or [bilinear]')
            sys.exit(0)
    
    def forward(self, x):
        return self.up(x)

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
