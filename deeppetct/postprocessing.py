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

# print model
def print_model(model, gen_size, dis_size):
    print('Generator:')
    summary(model.generator, input_size=gen_size, col_names=["kernel_size", "output_size", "num_params"])
    print('Discriminator:')
    summary(model.discriminator, input_size=dis_size, col_names=["kernel_size", "output_size", "num_params"])

# print statistics
def print_stat(epoch, total_train_loss, total_valid_loss, total_valid_metric, start_time):
    print('epoch = {:<3} | train loss   | dis = {:<8.2f} | gen = {:<8.2f} | valid loss | dis = {:<8.2f} | gen = {:<8.2f}'.\
        format(epoch+1, total_train_loss[-1][0], total_train_loss[-1][1], total_valid_loss[-1][0], total_valid_loss[-1][1]), end="| \n")
    print(' '*12, end="| valid metric | ")
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
        total_train_loss = list(total_loss[0][0:start_epoch,:])
        total_valid_loss = list(total_loss[1][0:start_epoch,:])
        total_valid_metric = list(np.load(metric_path, allow_pickle='TRUE'))
    return total_train_loss, total_valid_loss, total_valid_metric


'''
SAVING
'''
# save checkpoints
def save_checkpoint(model, dis_optim, gen_optim, dis_sched, gen_sched, save_path, num_epochs, epoch=None):
    checkpoint_path = os.path.join(save_path, 'checkpoint')
    state_dict = {'generator':model.generator.module.state_dict() if isinstance(model.generator, nn.DataParallel) else model.generator.state_dict(),\
        'discriminator':model.discriminator.module.state_dict() if isinstance(model.discriminator, nn.DataParallel) else model.discriminator.state_dict(),\
        'gen_optim':gen_optim.state_dict(), 'dis_optim':dis_optim.state_dict(), \
        'gen_sched':gen_sched.state_dict(), 'dis_sched':dis_sched.state_dict(), 'epoch':epoch}
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

# save metrics
def save_metric(metric, save_path, metric_name):
    metric_path = os.path.join(save_path, 'stat', metric_name+'.npy')
    np.save(metric_path, metric)
    print('{:>45} => {:<40}'.format('Metric saved in', metric_path))

# save statistics
def save_stat(train_loss, valid_loss, metric, save_path, loss_name, metric_name):
    loss_path = os.path.join(save_path, 'stat', loss_name+'.npy')
    total_loss = [train_loss, valid_loss]
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
