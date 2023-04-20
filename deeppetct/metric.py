import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = [
    "CompareMSE",
    "CompareRMSE",
    "ComparePSNR",
    "CompareSSIM",
    "MetricsCompose",
]

# compose metric functions together
class MetricsCompose:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, x, y):
        metrics = {}
        for m in self.metrics:
            metrics = dict(metrics, **m(x,y))
        return metrics

def compare_MSE(x, y):
    return ((x-y)**2).mean()

def compare_RMSE(x, y):
    return torch.sqrt(compare_MSE(x,y)).item()

def compare_PSNR(x, y):
    return 10*torch.log10((1.**2)/compare_MSE(x,y)).item()

def compare_SSIM(x, y, window_size=11, channel=1, size_average=True):
    window = _create_window(window_size, channel)
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
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item() 

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class CompareMSE:
    def __call__(self, x, y):
        return {'MSE':compare_MSE(x,y)}

class CompareRMSE:
    def __call__(self, x, y):
        return {'RMSE':compare_RMSE(x,y)}

class ComparePSNR:
    def __call__(self, x, y):
        return {'PSNR':compare_PSNR(x,y)}

class CompareSSIM:
    def __call__(self, x, y):
        return {'SSIM':compare_SSIM(x,y)}
