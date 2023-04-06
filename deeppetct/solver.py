import time
import numpy as np
import torch

from empatches import EMPatches
from deeppetct.preprocessing import *
from deeppetct.postprocessing import *
from deeppetct.architecture.blocks import *

class Solver(object):
    def __init__(self, dataloader, model, metric_func, args):
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
            self.num_iters = args.num_iters
            self.scheduler = args.scheduler
            self.lr = args.lr
            self.lambda1 = args.lambda1
            self.lambda2 = args.lambda2
            self.device_idx = args.device_idx
            self.print_iters = args.print_iters
            self.save_iters = args.save_iters
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
        gen_optim, dis_optim, gen_sched, dis_sched = set_optim(self.model, self.lr, self.scheduler)
        
        # load checkpoint if exists
        checkpoint_path = os.path.join(self.save_path, 'checkpoint', self.checkpoint+'.pkl')
        if os.path.exists(checkpoint_path):
            print('{: ^118s}'.format('Loading checkpoint ...'))
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.generator.load_state_dict(state['generator'])
            self.model.discriminator.load_state_dict(state['discriminator'])
            gen_optim.load_state_dict(state['gen_optim'])
            dis_optim.load_state_dict(state['dis_optim'])
            gen_sched.load_state_dict(state['gen_sched'])
            dis_sched.load_state_dict(state['dis_sched'])
            start_epoch = state['epoch']
            print('{: ^118s}'.format('Successfully load checkpoint! Training from epoch {}'.format(start_epoch)))
        else:
            print('{: ^118s}'.format('No checkpoint found! Training from epoch 0!'))
            self.model.generator.apply(weights_init)
            self.model.discriminator.apply(weights_init)
            start_epoch = 0
        
        # multi-gpu training and move model to device
        if len(self.device_idx)>1:
            self.model.generator = nn.DataParallel(self.model.generator)
            self.model.discriminator = nn.DataParallel(self.model.discriminator)
        self.model.generator = self.model.generator.to(self.device)
        self.model.discriminator = self.model.discriminator.to(self.device)

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
            dis_train_loss = 0.0
            gen_train_loss = 0.0
            for i, (x,real) in enumerate(self.dataloader[0]):
                # move data to device
                x = x.float().to(self.device)
                real = real.float().to(self.device)
                # patch training/resize to (batch,feature,weight,height)
                if (self.patch_size!=None) & (self.patch_n!=None):
                    x = x.view(-1, 2, self.patch_size, self.patch_size, self.patch_size)
                    real = real.view(-1, 1, self.patch_size, self.patch_size, self.patch_size)
                # * train discriminator
                self.model.discriminator.train()
                self.model.discriminator.zero_grad()
                dis_optim.zero_grad()
                for _ in range(self.num_iters):
                    # forward propagation
                    fake = self.model.generator(x).detach()
                    d_fake = self.model.discriminator(fake)
                    d_real = self.model.discriminator(real)
                    # compute loss
                    # dis_loss, grad_loss = self.model.discriminator_loss(fake, real, d_fake, d_real, self.lambda2, self.device)
                    dis_loss = self.model.discriminator_loss(fake, real, d_fake, d_real)
                    # backward propagation
                    dis_loss.backward(retain_graph=True)
                    # update weights
                    dis_optim.step()
                    for p in self.model.discriminator.parameters():
                        p.data.clamp_(-0.01,0.01)
                # update statistics
                dis_train_loss += dis_loss.item()
                # * train generator
                self.model.generator.train()
                self.model.generator.zero_grad()
                gen_optim.zero_grad()
                # forward propagation
                fake = self.model.generator(x)
                d_fake = self.model.discriminator(fake)
                # compute loss
                # gen_loss, perc_loss = self.model.generator_loss(fake, real, d_fake, self.lambda1)
                gen_loss = self.model.generator_loss(fake, real, d_fake)
                # backward propagation
                gen_loss.backward()
                # update weights
                gen_optim.step()
                # update statistics
                gen_train_loss += gen_loss.item()
            # update statistics
            total_train_loss.append((dis_train_loss, gen_train_loss))
            # update scheduler    
            dis_sched.step()
            gen_sched.step()
            
            # validation
            dis_valid_loss = 0.0
            gen_valid_loss = 0.0
            valid_metric = {}
            self.model.generator.eval()
            self.model.discriminator.eval()
            with torch.no_grad():
                for i, (x,real) in enumerate(self.dataloader[1]):
                    # move data to device
                    x = x.float().to(self.device)
                    real = real.float().to(self.device)
                    # patch training/resize to (batch,feature,weight,height)
                    if (self.patch_size!=None) & (self.patch_n!=None):
                        x = x.view(-1, 2, self.patch_size, self.patch_size, self.patch_size)
                        real = real.view(-1, 1, self.patch_size, self.patch_size, self.patch_size) 
                    # forward propagation
                    fake = self.model.generator(x)
                    d_fake = self.model.discriminator(fake)
                    d_real = self.model.discriminator(real)
                    # compute loss
                    with torch.enable_grad():
                        dis_loss = self.model.discriminator_loss(fake, real, d_fake, d_real)
                        gen_loss = self.model.generator_loss(fake, real, d_fake)
                    # compute metric
                    # fake = fake/torch.max(fake)
                    metric = self.metric_func(fake, real)
                    dis_valid_loss += dis_loss.item()
                    gen_valid_loss += gen_loss.item()
                    valid_metric = metric if i == 0 else {key:valid_metric[key]+metric[key] for key in metric.keys()}
            # update statistics (average over batch)
            total_valid_loss.append((dis_valid_loss, gen_valid_loss))
            total_valid_metric.append({key:valid_metric[key]/total_valid_data for key in valid_metric.keys()})
            # save best checkpoint
            if min_valid_loss > gen_valid_loss:
                print('{: ^118s}'.format('Validation loss decreased! Saving the checkpoint!'))
                save_checkpoint(self.model, dis_optim, gen_optim, dis_sched, gen_sched, self.save_path, self.num_epochs)
                min_valid_loss = gen_valid_loss

            # print statictics
            if (epoch+1) % self.print_iters == 0:
                print_stat(epoch, total_train_loss, total_valid_loss, total_valid_metric, start_time)
            # save checkpoints and statistics
            if (epoch+1) % self.save_iters == 0:
                save_checkpoint(self.model, dis_optim, gen_optim, dis_sched, gen_sched, self.save_path, self.num_epochs, epoch=epoch+1)
                save_stat(total_train_loss, total_valid_loss, total_valid_metric, self.save_path, self.loss_name, self.metric_name)

        # save results
        print('{:-^118s}'.format('Training finished!'))
        print('Total training time is {:.2f} s'.format(time.time()-start_time))
        print('{:-^118s}'.format('Saving results!'))
        # save final checkpoint and statistics
        save_checkpoint(self.model, dis_optim, gen_optim, dis_sched, gen_sched, self.save_path, self.num_epochs, epoch=self.num_epochs)
        save_stat(total_train_loss, total_valid_loss, total_valid_metric, self.save_path, self.loss_name, self.metric_name)

        print('{:-^118s}'.format('Done!'))
    
    # testing mode
    def test(self):
        start_time = time.time()
        print('{:-^118s}'.format('Testing start!'))

        # load checkpoint if exists
        checkpoint_path = os.path.join(self.save_path, 'checkpoint', self.checkpoint+'.pkl')
        if os.path.exists(checkpoint_path):
            print('{: ^118s}'.format('Loading checkpoint ...'))
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.generator.load_state_dict(state['generator'])
            print('{: ^118s}'.format('Successfully load {}'.format(self.checkpoint)))
        else:
            print('Checkpoint not exist!')
            sys.exit(0)

        # multi-gpu testing and move model to device
        if len(self.device_idx)>1:
            self.model.generator = nn.DataParallel(self.model.generator)
        self.model.generator = self.model.generator.to(self.device)

        # testing
        total_metric_fake = []
        total_metric_x = []
        self.model.generator.eval()
        emp = EMPatches()
        with torch.no_grad():
            for i, (x,real) in enumerate(self.dataloader):
                _, _, depth, height, width = x.size()
                x = x.view(2, depth, height, width)
                real = real.view(depth, height, width)
                # predict
                # split into patches
                x0_patches, indices = emp.extract_patches(x[0,:,:,:], patchsize=self.patch_size, stride=self.stride, vox=True)
                x1_patches, _ = emp.extract_patches(x[1,:,:,:], patchsize=self.patch_size, stride=self.stride, vox=True)
                real_patches, _ = emp.extract_patches(real, patchsize=self.patch_size, stride=self.stride, vox=True)
                pred_patches = []
                # patch-based testing
                for j in range(len(real_patches)):
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
                real = real.view(1, 1, depth, height, width).float()
                metric_x = self.metric_func(pet10, real)
                metric_pred = self.metric_func(fake, real)
                total_metric_x.append(metric_x)
                total_metric_fake.append(metric_pred)
                # save predictions
                fake = fake.squeeze()
                if i == 0:
                    total_fake = fake
                else:
                    total_fake = torch.cat((total_fake,fake),0)

        # print results
        print_metric(total_metric_x, total_metric_fake)
        print('{:-^118s}'.format('Testing finished!'))
        print('Total testing time is {:.2f} s'.format(time.time()-start_time))
        # save results
        print('{:-^118s}'.format('Saving results!'))
        save_pred(total_fake.cpu(), self.save_path, self.pred_name)
        save_metric((total_metric_x, total_metric_fake), self.save_path, self.metric_name)
        print('{:-^118s}'.format('Done!'))