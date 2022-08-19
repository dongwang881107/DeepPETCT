import sys
import os
import random
import torch
import torch.optim as optim

# print in both console and file
class LoggingPrinter:
    def __init__(self, filename):
        self.out_file = open(filename, "a")
        self.old_stdout = sys.stdout
        sys.stdout = self
    # executed when the user does a `print`
    def write(self, text): 
        self.old_stdout.write(text)
        self.out_file.write(text)
    # executed when `with` block begins
    def __enter__(self): 
        return self
    # executed when `with` block ends
    def __exit__(self): 
    # restore the original stdout object
        sys.stdout = self.old_stdout
    def flush(self):
        pass

# set up log file
def set_logger(path, name):
    LoggingPrinter(os.path.join(path, name+'.txt'))

# set up seed
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

# set up result folder
def set_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        checkpoint_path = os.path.join(path, 'checkpoint')
        fig_path = os.path.join(path, 'fig')
        stat_path = os.path.join(path, 'stat')
        os.makedirs(checkpoint_path)
        os.makedirs(fig_path)
        os.makedirs(stat_path)
        print('Create path : {}'.format(path))
        print('Create path : {}'.format(checkpoint_path))
        print('Create path : {}'.format(fig_path))
        print('Create path : {}'.format(stat_path))

# set up device
def set_device(device_idx):
    if not torch.cuda.is_available():
        device = 'cpu'
        print('Using CPU')
    else:
        device_idx = sorted(device_idx)
        num_gpus = len(device_idx)
        available_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            device = 'cpu'
            print('Using CPU')
        else: 
            assert(device_idx[0]>=0)
            assert(device_idx[-1]<=(available_gpus-1))
            print('Using {} GPU(s)'.format(num_gpus))
            device = 'cuda'
            torch.backends.cudnn.benchmark = True
    return device

# move optim to device
def optim_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

# set up optimizer and scheduler
def set_optim(model, lr, scheduler):
    gamma = 0.8
    decay_iters = 10
    b1 = 0.5
    b2 = 0.99
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=lr, betas=(b1,b2))
    dis_optimizer = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(b1,b2))
    if scheduler == 'none':
        gamma = 1
        decay_iters = 10000000
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=decay_iters, gamma=gamma)
        dis_scheduler = optim.lr_scheduler.StepLR(dis_optimizer, step_size=decay_iters, gamma=gamma)
    elif scheduler == 'step':
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=decay_iters, gamma=gamma)
        dis_scheduler = optim.lr_scheduler.StepLR(dis_optimizer, step_size=decay_iters, gamma=gamma)
    elif scheduler == 'linear':
        gen_scheduler = optim.lr_scheduler.LinearLR(gen_optimizer, start_factor=gamma, total_iters=decay_iters)
        dis_scheduler = optim.lr_scheduler.LinearLR(dis_optimizer, start_factor=gamma, total_iters=decay_iters)
    elif scheduler == 'exp':
        gen_scheduler = optim.lr_scheduler.ExponentialLR(gen_optimizer, gamma=gamma)
        dis_scheduler = optim.lr_scheduler.ExponentialLR(dis_optimizer, gamma=gamma)
    elif scheduler == 'cos':
        gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, decay_iters, eta_min=1e-5)
        dis_scheduler = optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, decay_iters, eta_min=1e-5)
    elif scheduler == 'reduce':
        gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, 'min', patience=1000)
        dis_scheduler = optim.lr_scheduler.ReduceLROnPlateau(dis_optimizer, 'min', patience=1000)
    else:
        print('none | step | linear | exp | cos | reduce')
        sys.exit(0)
    return gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler
