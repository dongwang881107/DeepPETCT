import time
import numpy as np
import torch

from deeppetct.architecture.blocks import *
from deeppetct.preprocessing import *
from deeppetct.postprocessing import *

class Solver(object):
    def __init__(self, dataloader, model, loss_func, metric_func, args):
        super().__init__()

        # load shared parameters
        self.save_path = args.save_path
        self.dataloader = dataloader
        self.mode = args.mode
        self.args = args

        if self.mode == 'train':
            # load training parameters
            self.checkpoint = args.checkpoint
            self.patch_size = args.patch_size
            self.patch_n = args.patch_n
            self.num_epochs = args.num_epochs
            self.scheduler = args.scheduler
            self.lr = args.lr
            self.gamma = args.gamma
            self.device_idx = args.device_idx
            self.decay_iters = args.decay_iters
            self.print_iters = args.print_iters
            self.save_iters = args.save_iters
            self.loss_func = loss_func
            self.metric_func = metric_func
            self.model = model
            self.device = torch.device(set_device(self.device_idx))
            self.loss_name = args.loss_name
            self.metric_name = args.metric_name
        elif self.mode == 'test':
            # load teseting parameters
            self.device_idx = args.device_idx
            self.num_slices = args.num_slices
            self.metric_func = metric_func
            self.metric_name = args.metric_name
            self.pred_name = args.pred_name
            self.atten_name = args.atten_name
            self.checkpoint = args.checkpoint
            self.device = torch.device(set_device(self.device_idx))
            self.model = model
        else:
            # load plotting parameters
            self.case_idx = args.case_idx
            self.atten_idx = args.atten_idx
            self.trans_idx = args.trans_idx
            self.sag_idx = args.sag_idx
            self.coron_idx = args.coron_idx
            self.data_path = args.data_path
            self.loss_name = args.loss_name
            self.valid_metric_name = args.valid_metric_name
            self.test_metric_name = args.test_metric_name
            self.pred_name = args.pred_name
            self.atten_name = args.atten_name   
            self.not_save_plot = args.not_save_plot
            self.not_plot_loss = args.not_plot_loss
            self.not_plot_metric = args.not_plot_metric
            self.not_plot_pred = args.not_plot_pred
            self.not_plot_atten = args.not_plot_atten

    # training mode
    def train(self):
        start_time = time.time()
        print('{:-^118s}'.format('Training start!'))

        # set up optimizer and scheduler
        optim, scheduler = set_optim(self.model, self.scheduler, self.gamma, self.lr, self.decay_iters)
        
        # load checkpoint if exists
        checkpoint_path = os.path.join(self.save_path, 'checkpoint', self.checkpoint+'.pkl')
        if os.path.exists(checkpoint_path):
            print('{: ^118s}'.format('Loading checkpoint ...'))
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state['model'])
            optim.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start_epoch = state['epoch']
            print('{: ^118s}'.format('Successfully load checkpoint! Training from epoch {}'.format(start_epoch)))
        else:
            print('{: ^118s}'.format('No checkpoint found! Training from epoch 0!'))
            # self.model.apply(weights_init)
            start_epoch = 0
        
        # multi-gpu training and move model to device
        if len(self.device_idx)>1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # compute total patch number
        if (self.patch_size!=None) & (self.patch_n!=None):
            total_train_data = len(self.dataloader[0].dataset)*self.patch_n
            total_valid_data = len(self.dataloader[1].dataset)*self.patch_n
        else:
            total_train_data = len(self.dataloader[0].dataset)
            total_valid_data = len(self.dataloader[1].dataset)

        # load statistics
        total_train_loss, total_valid_loss, total_valid_metric = load_stat(start_epoch, self.save_path, self.loss_name, self.metric_name)
        min_valid_loss = np.inf
        for epoch in range(start_epoch, self.num_epochs):
            # training
            train_loss = 0.0
            for (pet10,ct,pet60) in self.dataloader[0]:
                # move data to device
                pet10 = pet10.float().to(self.device)
                ct = ct.float().to(self.device)
                pet60 = pet60.float().to(self.device)
                # patch training/resize to (batch,feature,weight,height)
                if (self.patch_size!=None) & (self.patch_n!=None):
                    pet10 = pet10.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                    ct = ct.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                    pet60 = pet60.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                # zero the gradients
                self.model.train()
                self.model.zero_grad()
                optim.zero_grad()
                # forward propagation
                pred = self.model(pet10, ct)
                # compute loss
                loss = self.loss_func(pred, ct, pet60)
                # backward propagation
                loss.backward()
                # update weights
                optim.step()
                # update statistics
                train_loss += loss.item()
            # update statistics (average over batch)
            total_train_loss.append(train_loss)
            # update scheduler    
            scheduler.step()

            # validation
            valid_loss = 0.0
            valid_metric = {}
            self.model.eval()
            with torch.no_grad():
                for i, (pet10,ct,pet60) in enumerate(self.dataloader[1]):
                    # move data to device
                    pet10 = pet10.float().to(self.device)
                    ct = ct.float().to(self.device)
                    pet60 = pet60.float().to(self.device)
                    # patch training/resize to (batch,feature,weight,height)
                    if (self.patch_size!=None) & (self.patch_n!=None):
                        pet10 = pet10.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                        ct = ct.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                        pet60 = pet60.view(-1, 1, self.patch_size, self.patch_size, self.patch_size) 
                    # forward propagation
                    pred = self.model(pet10, ct)
                    # compute loss
                    loss = self.loss_func(pred, ct, pet60)
                    # compute metric
                    metric = self.metric_func(pred, pet60)
                    valid_loss += loss.item()
                    valid_metric = metric if i == 0 else {key:valid_metric[key]+metric[key] for key in metric.keys()}
            # update statistics (average over batch)
            total_valid_loss.append(valid_loss)
            total_valid_metric.append({key:valid_metric[key]/total_valid_data for key in valid_metric.keys()})
            # save best checkpoint
            if min_valid_loss > valid_loss:
                print('{: ^118s}'.format('Validation loss decreased! Saving the checkpoint!'))
                save_checkpoint(self.model, optim, scheduler, self.save_path, self.num_epochs)
                min_valid_loss = valid_loss

            # print statictics
            if (epoch+1) % self.print_iters == 0:
                print_stat(epoch, total_train_loss, total_valid_loss, total_valid_metric, start_time)
            # save checkpoints and statistics
            if (epoch+1) % self.save_iters == 0:
                save_checkpoint(self.model, optim, scheduler, self.save_path, self.num_epochs, epoch=epoch+1)
                save_stat(total_train_loss, total_valid_loss, total_valid_metric, self.save_path, self.loss_name, self.metric_name)

        # save results
        print('{:-^118s}'.format('Training finished!'))
        print('Total training time is {:.2f} s'.format(time.time()-start_time))
        print('{:-^118s}'.format('Saving results!'))
        # save final checkpoint and statistics
        save_checkpoint(self.model, optim, scheduler, self.save_path, self.num_epochs, epoch=self.num_epochs)
        save_stat(total_train_loss, total_valid_loss, total_valid_metric, self.save_path, self.loss_name, self.metric_name)

        print('{:-^118s}'.format('Done!'))
    
    # testing mode
    def test(self):
        start_time = time.time()
        print('{:-^118s}'.format('Testing start!'))

        # load checkpoint if exists
        checkpoint_path = os.path.join(self.save_path, 'checkpoint', self.checkpoint+'.pkl')
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state['model'])
        else:
            print('Checkpoint not exist!')
            sys.exit(0)

        # multi-gpu testing and move model to device
        if len(self.device_idx)>1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        # testing
        total_metric_pred = []
        total_metric_x = []
        self.model.eval()
        with torch.no_grad():
            for i, (pet10,ct,pet60) in enumerate(self.dataloader):
                # resize to (batch,feature,weight,height)
                _, depth, height, width = pet10.size()
                pet10 = pet10.view(-1, 1, depth, height, width)
                ct = ct.view(-1, 1, depth, height, width)
                pet60 = pet60.view(-1, 1, depth, height, width)
                # move data to device
                pet10 = pet10.float().to(self.device)
                ct = ct.float().to(self.device)
                pet60 = pet60.float().to(self.device)
                # predict
                pred = torch.zeros_like(pet60)
                for i in range(depth // self.num_slices):
                    start_idx = i*self.num_slices
                    end_idx = i*self.num_slices + self.num_slices
                    pred[:,:,start_idx:end_idx,:,:] =\
                          self.model(pet10[:,:,start_idx:end_idx,:,:], ct[:,:,start_idx:end_idx,:,:])
                pred = pred/torch.max(pred)
                metric_x = self.metric_func(pet10, pet60)
                metric_pred = self.metric_func(pred, pet60)
                total_metric_x.append(metric_x)
                total_metric_pred.append(metric_pred)
                # save predictions
                if i == 0:
                    total_pred = pred
                else:
                    total_pred = torch.cat((total_pred,pred),0)

        # print results
        print_metric(total_metric_x, total_metric_pred)
        print('{:-^118s}'.format('Testing finished!'))
        print('Total testing time is {:.2f} s'.format(time.time()-start_time))
        # save results
        print('{:-^118s}'.format('Saving results!'))
        save_pred(total_pred.cpu(), self.save_path, self.pred_name)
        save_metric((total_metric_x, total_metric_pred), self.save_path, self.metric_name)
        print('{:-^118s}'.format('Done!'))
