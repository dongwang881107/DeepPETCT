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

    # get all the testing cases
    case_paths = sorted(glob.glob(args.data_path + '/P*'))

    # case by case testing
    print('{:-^118s}'.format('Testing start!'))
    for case_path in case_paths:
        if os.path.exists(case_path+'/short_'+args.mode):
            # set path
            save_path = set_folder(case_path, args.mode)
            # set logger
            set_logger(save_path, 'log')
            # determine tranforms
            trans = TransCompose([MyNormalize(), ResizeCT(), SegmentCT(), MyTotensor()])
            # determine dataloader
            dataloader = get_loader(case_path, args.mode, trans)
            # determine neural networks
            model = deeparch.m3snet()
            # determine metric functions
            metric_func = MetricsCompose([CompareRMSE(), ComparePSNR(), CompareSSIM()])
            # build solver
            solver = Solver(save_path, args.checkpoint, dataloader, model, metric_func, args.device_idx)    
            # testing
            solver.test()
    print('{:-^118s}'.format('Testing finished!'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeepPETCT')
    
    # parameters
    parser.add_argument('--checkpoint', type=str, default='./model/m3snet_version2.pkl', help='path of the checkpoint')
    parser.add_argument('--data_path', type=str, default='/Users/dong/Documents/Data/petct/big')
    parser.add_argument('--mode', type=str, default='10s', help='5s|10s|20s|30s|10s_pre|10s_post')
    parser.add_argument('--device_idx', nargs='+', type=int, default=[], help='gpu numbers')

    args = parser.parse_args()

    # run the main function
    main(args)

