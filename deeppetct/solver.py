import time
import numpy as np
import torch

from empatches import EMPatches
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
        else:
            # load teseting parameters
            self.patch_size = args.patch_size
            self.stride = args.stride
            self.device_idx = args.device_idx
            self.num_slices = args.num_slices
            self.metric_func = metric_func
            self.metric_name = args.metric_name
            self.pred_name = args.pred_name
            self.checkpoint = args.checkpoint
            self.device = torch.device(set_device(self.device_idx))
            self.model = model

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
            for (x,y) in self.dataloader[0]:
                # move data to device
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                # patch training/resize to (batch,feature,weight,height)
                if (self.patch_size!=None) & (self.patch_n!=None):
                    x = x.view(-1, 2, self.patch_size, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                # zero the gradients
                self.model.train()
                self.model.zero_grad()
                optim.zero_grad()
                # forward propagation
                pred = self.model(x)
                # compute loss
                loss = self.loss_func(pred, y)
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
                for i, (x,y) in enumerate(self.dataloader[1]):
                    # move data to device
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    # patch training/resize to (batch,feature,weight,height)
                    if (self.patch_size!=None) & (self.patch_n!=None):
                        x = x.view(-1, 2, self.patch_size, self.patch_size, self.patch_size)
                        y = y.view(-1, 1, self.patch_size, self.patch_size, self.patch_size) 
                    # forward propagation
                    pred = self.model(x)
                    # compute loss
                    loss = self.loss_func(pred, y)
                    # compute metric
                    metric = self.metric_func(pred, y)
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
        emp = EMPatches()
        with torch.no_grad():
            for i, (x,y) in enumerate(self.dataloader):
                _, _, depth, height, width = x.size()
                x = x.view(2, depth, height, width)
                y = y.view(depth, height, width)
                # split into patches
                x0_patches, indices = emp.extract_patches(x[0,:,:,:], patchsize=self.patch_size, stride=self.stride, vox=True)
                x1_patches, _ = emp.extract_patches(x[1,:,:,:], patchsize=self.patch_size, stride=self.stride, vox=True)
                y_patches, _ = emp.extract_patches(y, patchsize=self.patch_size, stride=self.stride, vox=True)
                pred_patches = []
                # patch-based testing
                for j in range(len(y_patches)):
                    x0_patch = x0_patches[j].view(1,1,self.patch_size,self.patch_size,self.patch_size)
                    x1_patch = x1_patches[j].view(1,1,self.patch_size,self.patch_size,self.patch_size)
                    x_patch = torch.cat((x0_patch,x1_patch),1)
                    x_patch = x_patch.float().to(self.device)
                    # predict
                    pred_patch = self.model(x_patch)
                    pred_patches.append(pred_patch.squeeze().cpu())
                # merge patches together
                pred = torch.tensor(emp.merge_patches(pred_patches, indices, mode='avg'))
                pred = pred/torch.max(pred)
                # compute metrics
                pred = pred.view(1, 1, depth, height, width).float()
                pet10 = x[0,:,:,:].view(1, 1, depth, height, width).float()
                y = y.view(1, 1, depth, height, width).float()
                metric_x = self.metric_func(pet10, y)
                metric_pred = self.metric_func(pred, y)
                total_metric_x.append(metric_x)
                total_metric_pred.append(metric_pred)
                # save predictions
                pred = pred.squeeze()
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