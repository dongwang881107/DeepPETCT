import torch
import os
import time
import numpy as np
import torch.nn as nn
from torchinfo import summary


'''
PRINTING
'''
# print usage 
def print_usage():
    return '''
    python {train.py, test.py, plot.py} [optional arguments]
    '''

# print model
def print_model(model, input_size):
    summary(model, input_size, dtypes=[torch.float, torch.float], col_names=["kernel_size", "output_size", "num_params"])

# print arguments
def print_args(args):
    print('{:-^118s}'.format('{} parameters!'.format(args.mode.capitalize()+'ing')))
    count = 0
    for arg, value in args.__dict__.items():
        count = count + 1
        value = str(value) if isinstance(value, list) or value is None else value
        if count == 2:
            print('{:>28} = {:<28}'.format(arg, value))
            count = 0
        else:
            print('{:>28} = {:<28}'.format(arg, value), end='| ')
    if count == 1:
        print('\n')

# print statistics
def print_stat(epoch, total_train_loss, total_valid_loss, total_valid_metric, start_time):
    print('epoch = {:<3} | train_loss = {:<6.4f} | val_loss = {:<7.4f}'.format(epoch+1, total_train_loss[-1], total_valid_loss[-1]), end="| ")
    keys = list(total_valid_metric[0].keys())
    for key in keys:
        txt = key+' = {:<8.4f}'
        print(txt.format(total_valid_metric[-1][key]), end="| ")
    print('time = {:<8.2f}'.format(time.time()-start_time))

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


'''
LOADING
'''
# load statistics
def load_stat(start_epoch, save_path, loss_name, metric_name):
    if start_epoch == 0:
        total_train_loss = []
        total_valid_loss = []
        total_valid_metric = []
    else:
        loss_path = os.path.join(save_path, 'stat', loss_name+'.npy')
        metric_path = os.path.join(save_path, 'stat', metric_name+'.npy')
        total_loss = np.load(loss_path)
        total_train_loss = list(total_loss[:,0][0:start_epoch])
        total_valid_loss = list(total_loss[:,1][0:start_epoch])
        total_valid_metric = list(np.load(metric_path, allow_pickle='TRUE'))
    return total_train_loss, total_valid_loss, total_valid_metric


'''
SAVING
'''
# save checkpoint
def save_checkpoint(model, optimizer, scheduler, save_path, num_epochs, epoch=None):
    checkpoint_path = os.path.join(save_path, 'checkpoint')
    state_dict = {'model':model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),\
        'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'epoch':epoch}
    if epoch is not None:
        if epoch < num_epochs: 
            torch.save(state_dict, checkpoint_path+'/checkpoint_{}.pkl'.format(epoch))
            print('{:>45} => {:<40}'.format('Checkpoint saved in', checkpoint_path+'/checkpoint_{}.pkl'.format(epoch))) 
        else:
            torch.save(state_dict, checkpoint_path+'/checkpoint_final.pkl')
            print('{:>45} => {:<40}'.format('Final checkpoint saved in', checkpoint_path+'/checkpoint_final.pkl')) 
    else:
        torch.save(state_dict, checkpoint_path+'/checkpoint_best.pkl')
        print('{:>45} => {:<40}'.format('Best checkpoint saved in', checkpoint_path+'/checkpoint_best.pkl')) 

# save metric
def save_metric(metric, save_path, metric_name):
    metric_path = os.path.join(save_path, 'stat', metric_name+'.npy')
    np.save(metric_path, metric)
    print('{:>45} => {:<40}'.format('Metric saved in', metric_path))

# save statistics
def save_stat(train_loss, valid_loss, metric, save_path, loss_name, metric_name):
    loss_path = os.path.join(save_path, 'stat', loss_name+'.npy')
    total_loss = np.zeros([len(train_loss),2])
    total_loss[:,0] = train_loss
    total_loss[:,1] = valid_loss
    np.save(loss_path, total_loss)
    print('{:>45} => {:<40}'.format('Train/Valid loss saved in', loss_path))
    metric_path = os.path.join(save_path, 'stat', metric_name+'.npy')
    np.save(metric_path, metric)
    print('{:>45} => {:<40}'.format('Valid metric saved in', metric_path))

# save testing predictions
def save_pred(pred, save_path, pred_name):
    pred_path = os.path.join(save_path, 'stat', pred_name+'.npy')
    np.save(pred_path,pred)
    print('{:>45} => {:<40}'.format('Testing predictions saved in', pred_path))
