from .base_model import BaseModel
from .networks import WaveNetModel as WaveNet
import torch.nn.functional as F
from torch import nn
import torch
from models import constants
import numpy as np

class DDCModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.loss_names = ['ce', 'humaneness_reg', 'total']
        self.metric_names = ['accuracy']
        self.module_names = ['']  # changed from 'model_names'
        self.schedulers = []
        self.net = DDCNet(opt)
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_ce = None
        self.humaneness_reg = None

    def name(self):
        return "DDCNet"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--num_classes', type=int, default=20)
        # parser.add_argument('--output_channels', type=int, default=(4*3))
        # parser.add_argument('--kernel_size', type=int, default=2)
        # parser.add_argument('--bias', action='store_false')
        parser.add_argument('--entropy_loss_coeff', type=float, default=0.0)
        parser.add_argument('--humaneness_reg_coeff', type=float, default=0.0)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.0)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_ = data['input']
        target_ = data['target']
        input_shape = input_.shape
        target_shape = target_.shape
        # 0 batch dimension, 1 window dimension, 2 context time dimension, 3 frequency dimension, 4 mel_window_size dimension, 5 time dimension
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4], input_shape[5])).to(self.device)
        self.input = self.input.permute(0,4,1,2,3) # batch/window x time x temporal_context x frequency_features x mel_window_sizes
        #we collapse all the dimensions of target_ because that is the same way the output of the network is being processed for the cross entropy calculation (see self.forward)
        # here, 0 is the batch dimension, 1 is the window index, 2 is the time dimension, 3 is the output channel dimension
        self.target = target_.reshape((target_shape[0]*target_shape[1]*target_shape[2]*target_shape[3])).to(self.device)

    def forward(self):
        self.output = self.net.forward(self.input)
        x = self.output
        [n, l , classes] = x.size()
        x = x.view(n * l, classes)

        # print(x)
        self.loss_ce = F.cross_entropy(x, self.target)
        if self.opt.entropy_loss_coeff > 0:
            S = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            S = -1.0 * S.mean()
            self.loss_ce += self.opt.entropy_loss_coeff * S
        self.metric_accuracy = (torch.argmax(x,1) == self.target).sum().float()/len(self.target)

        #TODO: implement humaneness_reg maybe
        # problem is we don't have past notes available in input, so need to do that differently
        # just use output I guess :P
        # step_size = self.opt.step_size
        # humaneness_delta = constants.HUMAN_DELTA
        # window_size = int(humaneness_delta/step_size)
        #
        # receptive_field = self.net.module.receptive_field
        # notes = (torch.argmax(input[:,-5:,receptive_field//2-(window_size):receptive_field//2],1)==4).float()
        # distance_factor = torch.tensor(np.exp(-2*np.arange(window_size,0,-1)/window_size)).float().cuda()
        # if self.opt.entropy_loss_coeff > 0:
        #     weights = torch.tensordot(notes,distance_factor,dims=1)
        #     humaneness_reg = F.cross_entropy(x,torch.zeros(weights.shape).long().cuda(), reduction='none')
        #     humaneness_reg = torch.dot(humaneness_reg, weights)
        #     self.loss_humaneness_reg = humaneness_reg
        #     # self.loss_humaneness_reg = 0
        #     self.loss_total = self.loss_ce + self.opt.humaneness_reg_coeff * self.loss_humaneness_reg
        # else:
        #     self.loss_humaneness_reg = 0
        #     self.loss_total = self.loss_ce
        self.loss_humaneness_reg = 0
        self.loss_total = self.loss_ce

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_total.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()
        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass

    def generate(self,y):
        # dimensions of y are: features x window_sizes x time
        receptive_field = 1
        input_length = y.shape[-1]
        y = np.concatenate((np.zeros((y.shape[0],y.shape[1],receptive_field+self.opt.time_shifts//2)),y),2)
        # we also pad at the end to allow generation to be of the same length of song, by padding an amount corresponding to time_shifts
        y = np.concatenate((y,np.zeros((y.shape[0],y.shape[1],self.opt.time_shifts//2))),2)
        input_windowss = []
        time_shifts = self.opt.time_shifts - 1
        # loop that gets the input features for each of the windows, shifted by `ii`, and saves them in `input_windowss`
        for ii in range(-time_shifts//2, time_shifts//2+1):
            input_windows = [y[:,:,self.opt.time_shifts//2+ii:self.opt.time_shifts//2+ii+input_length]]
            input_windows = torch.tensor(input_windows)
            input_windows = (input_windows - input_windows.mean())/torch.abs(input_windows).max()
            # input_windows = (input_windows.permute(3,0,1,2) - input_windows.mean(-1)).permute(1,2,3,0)
            input_windowss.append(input_windows.float())
        input = torch.stack(input_windowss,dim=1).float()
        input_shape = input.shape
        input = input.to(self.device)
        input = input.permute(0,4,1,2,3) # batch/window x time x temporal_context x frequency_features x mel_window_sizes
        if self.opt.cuda:
            return F.softmax(self.net.module.forward(input),2)
        else:
            return F.softmax(self.net.forward(input),2)


class DDCNet(nn.Module):
    def __init__(self,opt):
        super(DDCNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, (7,3)) #assumes CHW format
        # self.pool = nn.MaxPool1d(3, 3)
        self.pool = nn.MaxPool2d((1,3), (1,3))
        self.conv2 = nn.Conv2d(20, 20, 3)
        # self.fc1 = nn.Linear(20 * 9, 256)
        # self.fc2 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(input_size=20*7*8, hidden_size=opt.hidden_dim, num_layers=2, batch_first=True)  # Define the LSTM
        self.hidden_to_state = nn.Linear(opt.hidden_dim,
                                         opt.num_classes)

    def forward(self, x):
        # batch/window x time x temporal_context x frequency_features x mel_window_sizes
        # print(x.shape)
        [N,L,deltaT,dim,winsizes] = x.shape
        x = x.view(N*L,deltaT,dim,winsizes)
        x = x.permute(0,3,1,2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(N,L,20*7*8) #  batch x time x CNN_features
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        lstm_out, _ = self.lstm(x)
        logits = self.hidden_to_state(lstm_out)
        # print(logits.shape)
        return logits
