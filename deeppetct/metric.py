import sys
import numpy as np
from skimage.metrics import structural_similarity as compute_SSIM
from skimage.metrics import peak_signal_noise_ratio as compute_PSNR
from skimage.metrics import mean_squared_error as compute_MSE

__all__ = [
    "ComputeMSE",
    "ComputeRMSE",
    "ComputePSNR",
    "ComputeSSIM",
    "MetricsCompose",
]

# compute metrics for all batches
def batch_metric(x, y, name, reduction='sum'):
    metric = 0
    for i in range(x.size()[0]):
        x_slice = x[i,0,:,:].numpy()
        y_slice = y[i,0,:,:].numpy()
        if name == 'MSE':
            metric += compute_MSE(x_slice, y_slice)
        elif name == 'RMSE':
            metric += compute_RMSE(x_slice, y_slice)
        elif name == 'PSNR':
            metric += compute_PSNR(x_slice, y_slice, data_range=1.0)
        elif name == 'SSIM':
            metric += compute_SSIM(x_slice, y_slice, multichannel=False, data_range=1.0)
        else:
            print('MSE | RMSE | PSNR | SSIM')
            sys.exit(0)
    return metric if reduction == 'sum' else metric/x.size()[0]

def compute_RMSE(x, y):
    return np.sqrt(compute_MSE(x,y))

class ComputeMSE:
    def __call__(self, x, y, batch=True):
        return {'MSE':batch_metric(x,y,'MSE')} if batch else {'MSE':compute_MSE(x,y)}

class ComputeRMSE:
    def __call__(self, x, y, batch=True):
        return {'RMSE':batch_metric(x,y,'RMSE')} if batch else {'RMSE':compute_RMSE(x,y)}

class ComputePSNR:
    def __call__(self, x, y, batch=True):
        return {'PSNR':batch_metric(x,y,'PSNR')} if batch else {'PSNR':compute_PSNR(x,y)}

class ComputeSSIM:
    def __call__(self, x, y, batch=True):
        return {'SSIM':batch_metric(x,y,'SSIM')} if batch else {'SSIM':compute_SSIM(x,y)}

class MetricsCompose:
    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, x, y, batch=True):
        metrics = {}
        for m in self.metrics:
            metrics = dict(metrics, **m(x,y,batch=batch))
        return metrics
