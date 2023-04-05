import argparse
import warnings

from deeppetct.postprocessing import *
from deeppetct.preprocessing import *
from deeppetct.data import *
from deeppetct.solver import *
from deeppetct.metric import *
from deeppetct.loss import *
from deeppetct.transform import *

import deeppetct.architecture as deeparch

def main(args):
    warnings.filterwarnings('ignore')
    if args.mode == 'train':
        set_seed(args.seed)
        set_folder(args.save_path)
        set_logger(args.save_path, args.log_name)
        print_args(args)
    if args.mode == 'test':
        set_logger(args.save_path, args.log_name)

    # determine tranforms
    train_trans = TransCompose([MyNormalize(), ResizeCT(), SegmentCT(), MyTotensor(), MyVflip(), MyHflip(), MyRotate()])
    valid_trans = TransCompose([MyNormalize(), ResizeCT(), SegmentCT(), MyTotensor()])
    # determine dataloader
    dataloader = get_loader(args.mode, args.data_path, train_trans, valid_trans, args.num_workers,
                            batch_size=args.batch_size if args.mode=='train' else None, 
                            patch_n=args.patch_n if args.mode=='train' else None, 
                            patch_size=args.patch_size if args.mode=='train' else None)
    # determine neural networks
    model = deeparch.unet_mp(sa_mode='blockwise')
    if args.mode == 'train':
        print_model(model, (1,2,64,64,64))
    # determine loss functions
    loss_weights = [1]
    loss_func = LossCompose([nn.MSELoss()], loss_weights)
    # determine metric functions
    metric_func = MetricsCompose([CompareRMSE(), ComparePSNR(), CompareSSIM()])
    # build solver
    solver = Solver(dataloader, model, loss_func, metric_func, args)    

    # training/testing/plotting
    eval('solver.{}()'.format(args.mode))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeepPETCT', usage=print_usage())
    subparsers = parser.add_subparsers(dest = 'mode', required=True, help='train | test')
    
    # training paramters
    subparser_train = subparsers.add_parser('train', help='training mode')
    subparser_train.add_argument('--seed', type=int, default=1000, help='random seed')
    subparser_train.add_argument('--device_idx', nargs='+', type=int, default=[], help='gpu numbers')
    subparser_train.add_argument('--save_path', type=str, default='./result_3d', help='saved path of the results')
    subparser_train.add_argument('--num_workers', type=int, default=0, help='number of workers used')
    subparser_train.add_argument('--log_name', type=str, default='log', help='name of the log file')
    subparser_train.add_argument('--data_path', type=str, default='/Users/dong/Documents/Data/LCPET/pet10to60_old/toy/3d')
    subparser_train.add_argument('--batch_size', type=int, default=1, help='batch size per epoch')
    subparser_train.add_argument('--patch_n', type=int, default=10, help='number of patches extract from one image')
    subparser_train.add_argument('--patch_size', type=int, default=64, help='patch size')
    subparser_train.add_argument('--lr', type=float, default=1e-4, help='learning rate of model')
    subparser_train.add_argument('--scheduler', type=str, default='step', help='type of the scheduler')
    subparser_train.add_argument('--gamma', type=float, default=1, help='decay value of the learning rate')
    subparser_train.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    subparser_train.add_argument('--decay_iters', type=int, default=10, help='number of iterations to decay learning rate')
    subparser_train.add_argument('--save_iters', type=int, default=50, help='number of iterations to save models')
    subparser_train.add_argument('--print_iters', type=int, default=1, help='number of iterations to print statistics')
    subparser_train.add_argument('--loss_name', type=str, default='train_loss', help='name of the training loss')
    subparser_train.add_argument('--metric_name', type=str, default='valid_metric', help='name of the validation metric')
    subparser_train.add_argument('--checkpoint', type=str, default='checkpoint', help='name of the checkpoint')
    
    # testing parameters
    subparser_test = subparsers.add_parser('test', help='testing mode')
    subparser_test.add_argument('--patch_size', type=int, default=64, help='patch size')
    subparser_test.add_argument('--stride', type=int, default=16, help='stride')
    subparser_test.add_argument('--save_path', type=str, default='./test', help='saved path of the results')
    subparser_test.add_argument('--device_idx', nargs='+', type=int, default=[], help='gpu numbers')
    subparser_test.add_argument('--data_path', type=str, default='/Users/dong/Documents/Data/petct/toy')
    subparser_test.add_argument('--arch', type=str, default='unet_mp')
    subparser_test.add_argument('--num_workers', type=int, default=4, help='number of workers used')
    subparser_test.add_argument('--num_slices', type=int, default=10, help='number of slices per reconstruction')
    subparser_test.add_argument('--checkpoint', type=str, default='checkpoint_final', help='name of the checkpoint')
    subparser_test.add_argument('--log_name', type=str, default='log', help='name of the log file')
    subparser_test.add_argument('--metric_name', type=str, default='test_metric', help='name of the metric')
    subparser_test.add_argument('--pred_name', type=str, default='test_pred', help='name of testing predictions')

    args = parser.parse_args()

    # run the main function
    main(args)

