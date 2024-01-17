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

    # set logger
    set_logger(args.data_path, 'log')

    # determine tranform functions
    trans = TransCompose([MyNormalize(), ResizeCT(), SegmentCT(), MyTotensor()])

    # determine metric functions
    metric_func = MetricsCompose([CompareRMSE(), ComparePSNR(), CompareSSIM()])

    # case by case testing
    print('{:-^118s}'.format('TESTING START!'))
    for case_path in case_paths:
        if not check_if_done(case_path, args.mode):
            if os.path.exists(case_path+'/short_'+args.mode):
                # set path
                save_path = set_folder(case_path, args.mode)
                # set dataloader
                dataloader = get_loader(case_path, args.mode, trans)
                # set neural networks
                model = deeparch.m3snet()
                # build solver
                solver = Solver(save_path, args.checkpoint, dataloader, model, metric_func, args.device_idx)    
                # core test function
                solver.test()
            else:
                print('Case {} do not contain {} PET images! Move to the next case!\n'.format(case_path.split('\\')[-1], args.mode))
        else:
            print('Case {} have alredy been tested! Move to the next case!\n'.format(case_path.split('\\')[-1]))
    print('{:-^118s}'.format('ALL CASES TESTING FINISHED!'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeepPETCT')
    
    # parameters
    parser.add_argument('--checkpoint', type=str, default='./model/m3snet_version2.pkl', help='path of the checkpoint')
    parser.add_argument('--data_path', type=str, default=r'H:\toy')
    parser.add_argument('--mode', type=str, default='10s', help='5s|10s|20s|30s|10s_pre|10s_post')
    parser.add_argument('--device_idx', nargs='+', type=int, default=[], help='gpu numbers')

    args = parser.parse_args()

    # run the main function
    main(args)

