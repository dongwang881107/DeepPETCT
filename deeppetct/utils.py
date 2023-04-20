import sys
import os
import numpy as np
import torch


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

# set up folder
def set_folder(save_path, case_path, mode):
    case_name = case_path.split('/')[-1]
    case_save_path = save_path + '/' + case_name
    if not os.path.exists(case_save_path):
        os.mkdir(case_save_path)
    recon_path = case_save_path + '/' + mode + '_recon'
    if not os.path.exists(recon_path):
        os.mkdir(recon_path)
    return case_save_path

# set up device
def set_device(device_ids):
    if not torch.cuda.is_available():
        device = 'cpu'
        print('Using CPU')
    else:
        device_ids = sorted(device_ids)
        num_gpus = len(device_ids)
        available_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            device = 'cpu'
            print('Using CPU')
        else: 
            assert(device_ids[0]>=0)
            assert(device_ids[-1]<=(available_gpus-1))
            print('Using {} GPU(s)'.format(num_gpus))
            device = 'cuda'
            torch.backends.cudnn.benchmark = True
    return device

# print metrics
def print_metric(metric_x, metric_pred):
    keys = list(metric_x[0].keys())
    avg_metric = np.zeros([len(keys), 2])
    for i in range(len(metric_x)):
        for j in range(len(keys)):
            avg_metric[j,0] += metric_x[i][keys[j]]
            avg_metric[j,1] += metric_pred[i][keys[j]]
    for j in range(len(keys)):
        print('{:>38} {:<4} = {:<8.4f}'.format('Original average', keys[j], avg_metric[j,0]/len(metric_x)), end='| ')
        print('{:<15} {:<4} = {:<8.4f}'.format('Predict average', keys[j], avg_metric[j,1]/len(metric_x)))

# save metric
def save_metric(metric, metric_path):
    np.save(metric_path, metric)
    print('{:>45} => {:<40}'.format('Metric saved in', metric_path))

