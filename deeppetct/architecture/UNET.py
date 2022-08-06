import torch.nn as nn
import sys

def get_acti(acti):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
    ])[acti]

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        # mode, in_channel, out_channel, kernel_size, stride, padding, acti
        self.cb1  = ConvBlock('conv', 2,  16, 3, 1, 1,'relu')
        self.cb2  = ConvBlock('conv', 16, 16, 3, 2, 1,'relu')
        self.cb3  = ConvBlock('conv', 16, 32, 3, 1, 1,'relu')
        self.cb4  = ConvBlock('conv', 32, 32, 3, 2, 1,'relu')
        self.cb5  = ConvBlock('conv', 32, 64, 3, 1, 1,'relu')
        self.cb6  = ConvBlock('conv', 64, 128,3, 2, 1,'relu')

        self.cb7  = ConvBlock('trans',128,64, 3, 2, [1,1],'relu')
        self.cb8  = ConvBlock('conv', 64, 64, 3, 1, 1,'relu')
        self.cb9  = ConvBlock('conv', 64, 64, 3, 1, 1,'relu')
        self.cb10 = ConvBlock('trans',64, 32, 3, 2, [1,1],'relu')
        self.cb11 = ConvBlock('conv', 32, 32, 3, 1, 1,'relu')
        self.cb12 = ConvBlock('conv', 32, 32, 3, 1, 1,'relu')      
        self.cb13 = ConvBlock('trans',32, 16, 3, 2, [1,1],'relu')
        self.cb14 = ConvBlock('conv', 16, 16, 3, 1, 1,'relu')
        self.cb15 = ConvBlock('conv', 16, 16, 3, 1, 1,'relu')
        self.cb16 = ConvBlock('conv', 16, 1,  3, 1, 1,'relu')


    def forward(self, x):
        out = self.cb1(x)
        res1 = out
        out = self.cb2(out)
        out = self.cb3(out)
        res2 = out
        out = self.cb4(out)
        out = self.cb5(out)
        res3 = out
        out = self.cb6(out)
        out = self.cb7(out)
        out = out + res3
        out = self.cb8(out)
        out = self.cb9(out)
        out = self.cb10(out)
        out = out + res2
        out = self.cb11(out)
        out = self.cb12(out)
        out = self.cb13(out)
        out = out + res1
        out = self.cb14(out)
        out = self.cb15(out)
        out = self.cb16(out)
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
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, \
                padding=padding[0], output_padding=padding[1])
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

