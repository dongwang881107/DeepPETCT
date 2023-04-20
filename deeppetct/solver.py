from deeppetct.architecture.blocks import *
from deeppetct.utils import *

class Solver(object):
    def __init__(self, data_path, save_path, checkpoint, dataloader, model, metric_func, mode, device_idx):
        super().__init__()

        # load shared parameters
        self.data_path = data_path
        self.save_path = save_path
        self.checkpoint = checkpoint
        self.dataloader = dataloader
        self.model = model
        self.metric_func = metric_func
        self.mode = mode
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
        total_metric_pred = []
        total_metric_x = []
        self.model.eval()
        with torch.no_grad():
            for i, (low_dose,ct,high_dose) in enumerate(self.dataloader):
                # resize to (batch,feature,weight,height)
                _, _, weight, height = low_dose.size()
                low_dose = low_dose.view(-1, 1, weight, height)
                ct = ct.view(-1, 1, weight, height)
                high_dose = high_dose.view(-1, 1, weight, height)
                # move data to device
                low_dose = low_dose.float().to(self.device)
                ct = ct.float().to(self.device)
                high_dose = high_dose.float().to(self.device)
                # predict
                pred = self.model(low_dose, ct)
                pred = pred/torch.max(pred)
                metric_x = self.metric_func(low_dose, high_dose)
                metric_pred = self.metric_func(pred, high_dose)
                total_metric_x.append(metric_x)
                total_metric_pred.append(metric_pred)
                # save predictions
                pred_name = self.dataloader.dataset.get_path(i).split('/')[-1].split('.')[0] + '.npy'
                pred_path = self.save_path + '/' + self.mode + '_recon/' + pred_name
                np.save(pred_path, pred.cpu())

        # print and results
        print_metric(total_metric_x, total_metric_pred)
        metric_path = self.save_path + '/' + self.mode + '_metric.npy'
        save_metric((total_metric_x, total_metric_pred), metric_path)
        print(self.data_path.split('/')[-1]+'{:-^118s}'.format(' testing finished!'))
