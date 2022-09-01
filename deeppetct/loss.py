import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    "SSIMLoss",
    "TVLoss",
    "wTVLoss",
    "dTVLoss",
    "LossCompose",
]

# Possible loss choices
# nn.L1Loss()
# nn.MSELoss()
# SSIMLoss()
# TVLoss()

class LossCompose:
    def __init__(self, losses, params):
        self.losses = losses
        self.params = params

    def __call__(self, x, y):
        assert(len(self.losses) == len(self.params))
        losses = 0
        for i, loss in enumerate(self.losses):
            losses = losses + self.params[i]*loss(x,y)
        return losses

class SSIMLoss(nn.Module):
    # Structure Similarity Index Measure (SSIM) Loss
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, x, y, window_size, channel, size_average):
        window = self._create_window(window_size, channel)
        window = window.type_as(x)
        mu1 = F.conv2d(x, window, padding=window_size//2)
        mu2 = F.conv2d(y, window, padding=window_size//2)
        mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
        sigma1_sq = F.conv2d(x*x, window, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(y*y, window, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(x*y, window, padding=window_size//2) - mu1*mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1*mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, x, y):
        return (1-self._ssim(x, y, self.window_size, self.channel, self.size_average)).pow(2)

class TVLoss(nn.Module):
    # Total Variation (TV) Loss
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3]-1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

class wTVLoss(nn.Module):
    # Total Variation (TV) Loss
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x, y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3]-1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

class dTVLoss(nn.Module):
    # Total Variation (TV) Loss
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x, y):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3]-1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size