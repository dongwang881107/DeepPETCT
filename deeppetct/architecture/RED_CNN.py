import torch.nn as nn

class RED_CNN(nn.Module):
    def __init__(self, out_ch=96, kernel_size=5, stride=1, padding=0):
        super(RED_CNN, self).__init__()
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



