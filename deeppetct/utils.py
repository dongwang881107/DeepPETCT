import sys
import os
import glob
import numpy as np
import torch

# check if the case have already been tested
def check_if_done(path, mode):
    test_status = False
    enhanced_path= os.path.join(path, 'short_'+mode+'_enhanced')
    ori_path = os.path.join(path, 'short_'+mode)
    enhanced_files = glob.glob(enhanced_path+'/I*')
    ori_files = glob.glob(ori_path+'/I*')
    if len(enhanced_files) == len(ori_files) > 0:
        test_status = True
    return test_status

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
def set_folder(case_path, mode):
    save_path = os.path.join(case_path, 'short_'+mode+'_enhanced')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path

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

