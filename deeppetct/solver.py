import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from deeppetct.preprocessing import *
from deeppetct.postprocessing import *

class Solver(object):
    def __init__(self, dataloader, model, metric_func, args):
        super().__init__()

        # load shared parameters
        self.save_path = args.save_path
        self.mode = args.mode
        self.dataloader = dataloader
        self.args = args

        if self.mode == 'train':
            # load training parameters
            self.save_path = args.save_path
            self.checkpoint = args.checkpoint
            self.patch_size = args.patch_size
            self.patch_n = args.patch_n
            self.num_epochs = args.num_epochs
            self.scheduler = args.scheduler
            self.lr = args.lr
            self.gamma = args.gamma
            self.device_ids = args.device_ids
            self.decay_iters = args.decay_iters
            self.print_iters = args.print_iters
            self.save_iters = args.save_iters
            self.metric_func = metric_func
            self.model = model
            self.device = torch.device(set_device(self.device_ids))
            self.loss_name = args.loss_name
            self.metric_name = args.metric_name
        elif self.mode == 'test':
            # load teseting parameters
            self.device_ids = args.device_ids
            self.metric_func = metric_func
            self.metric_name = args.metric_name
            self.pred_name = args.pred_name
            self.checkpoint = args.checkpoint
            self.device = torch.device(set_device(self.device_ids))
            self.model = model
        else:
            # load plotting parameters
            self.index = args.index
            self.loss_name = args.loss_name
            self.valid_metric_name = args.valid_metric_name
            self.test_metric_name = args.test_metric_name
            self.pred_name = args.pred_name
            self.not_save_plot = args.not_save_plot
            self.not_plot_loss = args.not_plot_loss
            self.not_plot_metric = args.not_plot_metric
            self.not_plot_pred = args.not_plot_pred

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
            start_epoch = 0
        
        # multi-gpu training and move model to device
        if len(self.device_ids)>1:
            self.model = nn.DataParallel(self.model)
            loss_func = self.model.module.compute_loss()
        else:
            loss_func = self.model.compute_loss()
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
                    x = x.view(-1, 2, self.patch_size, self.patch_size)
                    y = y.view(-1, 1, self.patch_size, self.patch_size)
                # zero the gradients
                self.model.train()
                self.model.zero_grad()
                optim.zero_grad()
                # forward propagation
                pred = self.model(x)
                # compute loss
                loss = loss_func(pred, y)/x.size()[0]
                # backward propagation
                loss.backward()
                # update weights
                optim.step()
                # update statistics
                train_loss += loss.item()*x.size()[0]
            # update statistics (average over batch)
            total_train_loss.append(train_loss/total_train_data)
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
                        x = x.view(-1, 2, self.patch_size, self.patch_size)
                        y = y.view(-1, 1, self.patch_size, self.patch_size) 
                    # forward propagation
                    pred = self.model(x)
                    # compute loss
                    loss = loss_func(pred, y)
                    # compute metric
                    metric = self.metric_func(pred, y)
                    valid_loss += loss.item()
                    valid_metric = metric if i == 0 else {key:valid_metric[key]+metric[key] for key in metric.keys()}
            # update statistics (average over batch)
            total_valid_loss.append(valid_loss/total_valid_data)
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
        if len(self.device_ids)>1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        # testing
        total_metric_pred = []
        total_metric_x = []
        self.model.eval()
        with torch.no_grad():
            for i, (x,y) in enumerate(self.dataloader):
                # resize to (batch,feature,weight,height)
                x = x.view(-1, 2, 144, 144)
                y = y.view(-1, 1, 144, 144)
                # move data to device
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                # predict
                pred = self.model(x)
                pred = pred/torch.max(pred)
                metric_x = self.metric_func(x, y)
                metric_pred = self.metric_func(pred, y)
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

    # plotting mode
    def plot(self):
        start_time = time.time()

        # plotting font and color
        print('{:-^118s}'.format('Plotting start!'))
        fs = 18
        lw = 2.0
        cmap = 'gray_r'
        fraction = 0.045

        # plot training loss
        if self.not_plot_loss:
            loss_path = os.path.join(self.save_path, 'stat', self.loss_name+'.npy')
            total_loss = np.load(loss_path)
            fig = plt.figure()
            plt.xlabel('Epoch', fontsize=fs)
            plt.ylabel('Training Loss', fontsize=fs)
            for j in range(total_loss.shape[-1]):
                plt.plot(total_loss[:,j], linewidth=lw)
            plt.legend(['training','validation'])
            self._plot(fig, self.loss_name)

        # plot validation metric
        if self.not_plot_metric:
            valid_metric_path = os.path.join(self.save_path, 'stat', self.valid_metric_name+'.npy')
            valid_metric = np.load(valid_metric_path, allow_pickle='TRUE')
            keys = list(valid_metric[0].keys())
            metrics = np.zeros([len(valid_metric), len(keys)])
            for i in range(len(valid_metric)):
                for j in range(len(keys)):
                    metrics[i,j] = valid_metric[i][keys[j]]
            for j in range(len(keys)):
                fig = plt.figure()
                plt.xlabel('Epoch', fontsize=fs)
                plt.ylabel(keys[j].upper(), fontsize=fs)
                plt.plot(metrics[:,j], linewidth=lw, label=keys[j])
                plt.legend()
                self._plot(fig, 'valid_'+keys[j].lower())

        # plot predictions
        if self.not_plot_pred:
            pred_path = os.path.join(self.save_path, 'stat', self.pred_name+'.npy')
            test_metric_path = os.path.join(self.save_path, 'stat', self.test_metric_name+'.npy')
            pred = np.load(pred_path)
            test_metric = np.load(test_metric_path, allow_pickle='TRUE')
            data_name = self.dataloader.dataset.get_path()
            if self.index == []:
                self.index = range(len(self.dataloader))
            for i, (x,y) in enumerate(self.dataloader):
                if i in self.index:
                    # load data
                    pet10 = x.squeeze()[0,:,:].numpy()
                    pet60 = y.squeeze().numpy()
                    p = pred[i,:,:,:].squeeze()
                    ct = np.load(data_name[1][i])
                    # normalize data
                    ct = ct/np.max(ct)
                    p = p/np.max(p)
                    data = (pet10,ct,p,pet60)
                    title = ['10s', 'CT', 'Proposed', '60s']
                    fig = plt.figure(figsize=(15,4))
                    for j in range(4):
                        ax = fig.add_subplot(1,4,j+1)
                        im = ax.imshow(data[j], cmap='gray' if j==1 else cmap)
                        ax.set_title(title[j], fontsize=fs)
                        ax.axis('off')
                        fig.colorbar(im, ax=ax, fraction=fraction)
                    title_name = 'Low Dose: '
                    for key in test_metric[0][i].keys():
                        title_name += key+':'+"{:.4f}".format(test_metric[0][i][key])+', '
                    title_name += '\n Proposed: '
                    for key in test_metric[1][i].keys():
                        title_name += key+':'+"{:.4f}".format(test_metric[1][i][key])+', '
                    plt.suptitle(title_name, fontsize=10, y=0.15)
                    fig_name = data_name[0][i].split('/')[-3]+ '_' + \
                               data_name[0][i].split('/')[-1].split('.')[0]
                    self._plot(fig, fig_name)
        print('Total plotting time is {:.2f} s'.format(time.time()-start_time))
        print('{:-^118s}'.format('Done!'))

    # helper 
    def _plot(self, fig, plot_name):
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')
        if self.not_save_plot:
            plot_name = os.path.join(self.save_path, 'fig', plot_name+'.png')
            fig.savefig(plot_name, bbox_inches='tight')
