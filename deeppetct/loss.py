import torch
import math
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    "MSELoss",
    "L1Loss",
    "SSIMLoss", 
    "TVLoss",
    "iTVLoss",
    "aTVLoss",
    "iSPLoss",
    "aSPLoss",
    "LossCompose",
]

# compose loss functions together
class LossCompose:
    def __init__(self, losses, params, modalities):
        self.losses = losses
        self.params = params
        self.modalities = modalities
        self.print_loss()

    def print_loss(self):
        print('Loss functions are', end=' ')
        for i, loss in enumerate(self.losses):
            print('[{} ({})]'.format(loss.__class__.__name__, self.modalities[i]), end=' ')
        print('\n')

    def __call__(self, x, y, z):
        # x->pet10, y->ct, z->pet60
        assert(len(self.losses) == len(self.params))
        losses = 0
        for i, loss in enumerate(self.losses):
            losses = losses + self.params[i]*loss(x,y,z,self.modalities[i])
        return losses

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y, z, modality):
        loss = nn.MSELoss()
        if modality == 'PET':
            return loss(x,z)
        elif modality == 'CT':
            return loss(x,y)
        else:
            print('Modality [PET] or [CT]')
            sys.exit(0)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y, z, modality):
        loss = nn.L1Loss()
        if modality == 'PET':
            return loss(x,z)
        elif modality == 'CT':
            return loss(x,y)
        else:
            print('Modality [PET] or [CT]')
            sys.exit(0)

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

    def forward(self, x, y, z, modality):
        if modality == 'PET':
            return (1-self._ssim(x, z, self.window_size, self.channel, self.size_average)).pow(2)
        elif modality == 'CT':
            return (1-self._ssim(x, y, self.window_size, self.channel, self.size_average)).pow(2)
        else:
            print('Modality [PET] or [CT]')
            sys.exit(0)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self,x,y,z,modality):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        tv = 2*(h_tv/count_h+w_tv/count_w)/batch_size
        print(tv)
        return tv

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class iTVLoss(nn.Module):
    # isotropic TV Loss
    def __init__(self):
        super(iTVLoss,self).__init__()

    def forward(self, x, y, z, modality):
        h = x.size()[2]
        w = x.size()[3]
        grad_x = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_x = torch.cat((grad_x,torch.zeros(x.size()[0],x.size()[1],1,x.size()[3])),2)
        grad_y = x[:,:,:,1:]-x[:,:,:,:w-1]
        grad_y = torch.cat((grad_y,torch.zeros(x.size()[0],x.size()[1],x.size()[2],1)),3)
        itv = torch.sqrt(torch.pow(grad_x,2)+torch.pow(grad_y,2)).mean()
        print(itv)
        return itv

class aTVLoss(nn.Module):
    # anisotropic TV Loss
    def __init__(self):
        super(aTVLoss,self).__init__()

    def forward(self, x, y, z, modality):
        h = x.size()[2]
        w = x.size()[3]
        grad_x = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_y = x[:,:,:,1:]-x[:,:,:,:w-1]
        atv = torch.abs(grad_x).mean()+torch.abs(grad_y).mean()
        print(atv)
        return atv

class iSPLoss(nn.Module):
    # isotropic Structure-Promoting loss
    def __init__(self):
        super(iSPLoss,self).__init__()

    def _compute_w(self, x, eta=0.1):
        # compute the edge indicator w function for x
        h = x.size()[2]
        w = x.size()[3]
        # compute grad_x
        grad_x1 = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_x1 = torch.cat((grad_x1,torch.zeros(x.size()[0],x.size()[1],1,x.size()[3])),2)
        grad_x2 = x[:,:,:,1:]-x[:,:,:,:w-1]
        grad_x2 = torch.cat((grad_x2,torch.zeros(x.size()[0],x.size()[1],x.size()[2],1)),3)
        edge_incator = torch.sqrt(eta*eta+torch.pow(grad_x1,2)+torch.pow(grad_x2,2))
        edge_incator = torch.div(eta,edge_incator)
        return edge_incator

    def forward(self, x, y, z, modality):
        h = x.size()[2]
        w = x.size()[3]
        # compute grad_x
        grad_x1 = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_x1 = torch.cat((grad_x1,torch.zeros(x.size()[0],x.size()[1],1,x.size()[3])),2)
        grad_x2 = x[:,:,:,1:]-x[:,:,:,:w-1]
        grad_x2 = torch.cat((grad_x2,torch.zeros(x.size()[0],x.size()[1],x.size()[2],1)),3)
        if modality == 'CT':
            isp = torch.sqrt(torch.pow(grad_x1,2)+torch.pow(grad_x2,2))*self._compute_w(y)
        elif modality == 'PET':
            isp = torch.sqrt(torch.pow(grad_x1,2)+torch.pow(grad_x2,2))*self._compute_w(z)
        else:
            print('Modality [PET] or [CT]')
            sys.exit(0)
        print(isp.mean())
        return isp.mean()

class aSPLoss(nn.Module):
    # anisotropic Stucture-Promoting loss
    def __init__(self):
        super(aSPLoss,self).__init__()

    def _compute_D(self, x, gamma=1, eta=0.1):
        # compute the anisotropic weighting D for x
        h = x.size()[2]
        w = x.size()[3]
        # compute grad_x
        grad_x1 = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_x1 = torch.cat((grad_x1,torch.zeros(x.size()[0],x.size()[1],1,x.size()[3])),2)
        grad_x2 = x[:,:,:,1:]-x[:,:,:,:w-1]
        grad_x2 = torch.cat((grad_x2,torch.zeros(x.size()[0],x.size()[1],x.size()[2],1)),3)
        # compute D
        up = gamma*(torch.pow(grad_x1,2)+torch.pow(grad_x2,2))
        down = eta*eta+(torch.pow(grad_x1,2)+torch.pow(grad_x2,2))
        D = 1 - torch.div(up,down)
        return D

    def forward(self, x, y, z, modality):
        h = x.size()[2]
        w = x.size()[3]
        grad_x1 = x[:,:,1:,:]-x[:,:,:h-1,:]
        grad_x1 = torch.cat((grad_x1,torch.zeros(x.size()[0],x.size()[1],1,x.size()[3])),2)
        grad_x2 = x[:,:,:,1:]-x[:,:,:,:w-1]
        grad_x2 = torch.cat((grad_x2,torch.zeros(x.size()[0],x.size()[1],x.size()[2],1)),3)
        if modality == 'CT':
            D = self._compute_D(y)
        elif modality == 'PET':
            D = self._compute_D(z)
        else:
            print('Modality [PET] or [CT]')
            sys.exit(0)
        asp = torch.sqrt(torch.pow(D*grad_x1,2)+torch.pow(D*grad_x2,2)).mean()
        print(asp)
        return asp