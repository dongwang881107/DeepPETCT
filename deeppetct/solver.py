import pydicom
import time

from deeppetct.architecture.blocks import *
from deeppetct.utils import *

class Solver(object):
    def __init__(self, save_path, checkpoint, dataloader, model, metric_func, device_idx):
        super().__init__()

        # load shared parameters
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.dataloader = dataloader
        self.model = model
        self.metric_func = metric_func
        self.device_idx = device_idx
        self.device = torch.device(set_device(self.device_idx))
    
    # testing mode
    def test(self):
         # load checkpoint if exists
        if os.path.exists(self.checkpoint):
            state = torch.load(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(state['model'])
        else:
            print('Checkpoint not exist!')
            sys.exit(0)

        # multi-gpu testing and move model to device
        if len(self.device_idx)>1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        # testing
        print('{:-^118s}'.format(self.save_path.split('\\')[-2] + ' testing start!'))
        start = time.time()
        total_metric_short_pet = []
        total_metric_pred = []
        self.model.eval()
        with torch.no_grad():
            for i, (short_pet,ct,long_pet) in enumerate(self.dataloader):
                # resize to (batch,feature,weight,height)
                _, _, weight, height = short_pet.size()
                short_pet = short_pet.view(-1, 1, weight, height)
                ct = ct.view(-1, 1, weight, height)
                long_pet = long_pet.view(-1, 1, weight, height)
                # move data to device
                short_pet = short_pet.float().to(self.device)
                ct = ct.float().to(self.device)
                long_pet = long_pet.float().to(self.device)
                # predict
                pred = self.model(short_pet, ct)
                pred = pred/torch.max(pred)
                metric_short_pet = self.metric_func(short_pet, long_pet)
                metric_pred = self.metric_func(pred, long_pet)
                total_metric_short_pet.append(metric_short_pet)
                total_metric_pred.append(metric_pred)
                # save predictions into dicom files
                dcm_path = self.dataloader.dataset.get_path(i)
                dcm = pydicom.dcmread(dcm_path)
                dcm.PixelData = (pred.cpu().numpy()*np.max(dcm.pixel_array)).astype(np.int16)
                dcm.SeriesDescription = '[WB_CTAC_ENHANCED] Body'
                pred_name = dcm_path.split('\\')[-1]
                pred_path = os.path.join(self.save_path, pred_name)
                dcm.save_as(pred_path)

        # print and results
        print_metric(total_metric_short_pet, total_metric_pred)
        metric_path = os.path.join(self.save_path, 'metric.npy')
        save_metric((total_metric_short_pet, total_metric_pred), metric_path)
        end = time.time()
        print('Running time is {:.4f} seconds'.format(end-start))
        print('{:-^118s}'.format(self.save_path.split('\\')[-2] + ' testing finished!\n'))