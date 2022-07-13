import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        print('generator')


class Discriminator(nn.Module):
    def __init__(self):
        print('discriminator')

# TODO: build CycleGAN network 
# ! fdfd
# ? fdfd
# // fdfd sdf
# * fdfd

class CycleGAN(nn.Module):
    def __init__(self, out_ch=96, kernel_size=5, stride=1, padding=0):
        super(CycleGAN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.bn5 = nn.BatchNorm2d(out_ch)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=kernel_size, stride=stride, padding=padding)

        self.tbn1 = nn.BatchNorm2d(out_ch)
        self.tbn2 = nn.BatchNorm2d(out_ch)
        self.tbn3 = nn.BatchNorm2d(out_ch)
        self.tbn4 = nn.BatchNorm2d(out_ch)
        self.tbn5 = nn.BatchNorm2d(1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        residual_2 = out
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        residual_3 = out
        out = self.relu(self.bn5(self.conv5(out)))
        # decoder
        out = self.tbn1(self.tconv1(out))
        out += residual_3
        out = self.tbn2(self.tconv2(self.relu(out)))
        out = self.tbn3(self.tconv3(self.relu(out)))
        out += residual_2
        out = self.tbn4(self.tconv4(self.relu(out)))
        out = self.tbn5(self.tconv5(self.relu(out)))
        out += residual_1
        out = self.relu(out)
        return out

    def compute_loss(self, x):
        print('loss')

    def set_optim(self):
        print('optimizer')

    @classmethod
    def compute_loss(cls):
        return nn.MSELoss(reduction='sum')
