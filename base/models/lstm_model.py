import torch
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel

class LSTMModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.loss_names = ['ce']
        self.metric_names = ['accuracy']
        self.module_names = ['']  # changed from 'model_names'
        self.schedulers = []
        self.net = LSTMNet(opt)
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_ce = None

    def name(self):
        return "LSTMNet"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hidden_dimension', type=int, default=0)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_ = data['input']
        target_ = data['target']
        input_shape = input_.shape
        target_shape = target_.shape
        # 0 batch dimension, 1 window dimension, 2 input channel dimension, 3 time dimension
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).to(self.device)
        #we collapse all the dimensions of target_ because that is the same way the output of the network is being processed for the cross entropy calculation (see self.forward)
        # here, 0 is the batch dimension, 1 is the window index, 2 is the time dimension, 3 is the output channel dimension
        self.target = target_.reshape((target_shape[0]*target_shape[1]*target_shape[2]*target_shape[3])).to(self.device)

    def forward(self):
        self.output = self.net.forward(self.input)
        x = self.output
        [n, channels, classes, l] = x.size()
        x = x.transpose(1, 3).contiguous()
        x = x.view(n * l * channels, classes)

        self.loss_ce = F.cross_entropy(x, self.target)
        S = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        S = -1.0 * S.mean()
        self.loss_ce += self.opt.entropy_loss_coeff * S
        self.metric_accuracy = (torch.argmax(x,1) == self.target).sum().float()/len(self.target)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_ce.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()


class LSTMNet(nn.Module):

    def __init__(self, opt):
        super().__init__()
        '''  hidden_dim: LSTM Output Dimensionality
             embedding_dim: LSTM Input Dimensionality, which 
             is 12 for chromagram and 20 (default) for MFCC
        '''

        self.hidden_dim = opt.hidden_dim
        if opt.embedding_dim is None:
            embedding_dim = 12  # Chroma, can be changed for MFCC
        else:
            embedding_dim = opt.embedding_dim

        self.lstm = nn.LSTM(embedding_dim, opt.hidden_dim)  # Define the LSTM
        self.hidden_to_state = nn.Linear(opt.hidden_dim,
                                         opt.vocab_size)  # vocab_size used so far is 2001 by default (2000 + empty state)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)  # Input is a 3D Tensor: [length, batch_size, dim] !! Feeder Functions Needed
        state_preoutput = self.hidden_to_state(lstm_out)  # lstm_out shape compatibility (Need to transpose?)
        state_output = F.log_softmax(state_output, dim=1)
        return state_output