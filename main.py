import argparse
import warnings

from deeppetct.utils import *
from deeppetct.data import *
from deeppetct.solver import *
from deeppetct.metric import *
from deeppetct.transform import *

import deeppetct.architecture as deeparch

def main(args):
    warnings.filterwarnings('ignore')
    set_logger(args.data_path, 'log')

    case_paths = sorted(glob.glob(args.data_path + '/P*'))
    # testing case by case
    print('{:-^118s}'.format('Testing start!'))
    for case_path in case_paths:
        # set path
        case_save_path = set_folder(args.save_path, case_path, args.mode)
        # determine tranforms
        trans = TransCompose([MyNormalize(), ResizeCT(), SegmentCT(), MyTotensor()])
        # determine dataloader
        dataloader = get_loader(case_path, args.mode, trans, args.num_workers)
        # determine neural networks
        model = deeparch.m3snet()
        # determine metric functions
        metric_func = MetricsCompose([CompareRMSE(), ComparePSNR(), CompareSSIM()])
        # build solver
        solver = Solver(case_path, case_save_path,  args.checkpoint, dataloader, model, metric_func, args.mode, args.device_idx)    
        # testing
        solver.test()
    print('{:-^118s}'.format('Testing finished!'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeepPETCT')
    
    # parameters
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final', help='path of the checkpoint')
    parser.add_argument('--data_path', type=str, default='/Users/dong/Documents/Data/petct/big')
    parser.add_argument('--save_path', type=str, default='/Users/dong/Documents/Data/petct/big')
    parser.add_argument('--mode', type=str, default='10s', help='5s|10s|20s|30s')
    parser.add_argument('--device_idx', nargs='+', type=int, default=[], help='gpu numbers')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers used')

    args = parser.parse_args()

    # run the main function
    main(args)

