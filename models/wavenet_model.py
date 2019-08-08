from .base_model import BaseModel
from .networks import WaveNetModel as WaveNet
import torch.nn.functional as F
import torch
import constants
import numpy as np

class WaveNetModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.loss_names = ['ce', 'humaneness_reg', 'total']
        self.metric_names = ['accuracy']
        self.module_names = ['']  # changed from 'model_names'
        self.schedulers = []
        self.net = WaveNet(layers=opt.layers,
                           blocks=opt.blocks,
                           dilation_channels=opt.dilation_channels,
                           residual_channels=opt.residual_channels,
                           skip_channels=opt.skip_channels,
                           end_channels=opt.end_channels,
                           input_channels=opt.input_channels,
                           output_length=opt.output_length,
                           output_channels=opt.output_channels,
                           num_classes=opt.num_classes,
                           kernel_size=opt.kernel_size,
                           dropout_p=opt.dropout,
                           bias=opt.bias)
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.loss_ce = None
        self.humaneness_reg = None

    def name(self):
        return "WaveNet"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--layers', type=int, default=10, help="Number of layers in each block")
        parser.add_argument('--blocks', type=int, default=4, help="Number of residual blocks in network")
        parser.add_argument('--dilation_channels', type=int, default=32, help="Number of channels in dilated convolutions")
        parser.add_argument('--residual_channels', type=int, default=32, help="Number of channels in the residual link")
        parser.add_argument('--skip_channels', type=int, default=256)
        parser.add_argument('--end_channels', type=int, default=256)
        parser.add_argument('--input_channels', type=int, default=(1+20))
        parser.add_argument('--num_classes', type=int, default=20)
        parser.add_argument('--output_channels', type=int, default=(4*3))
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--bias', action='store_false')
        parser.add_argument('--entropy_loss_coeff', type=float, default=0.0)
        parser.add_argument('--humaneness_reg_coeff', type=float, default=0.0)
        parser.add_argument('--dropout', type=float, default=0.0)
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
        if self.opt.entropy_loss_coeff > 0:
            S = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            S = -1.0 * S.mean()
            self.loss_ce += self.opt.entropy_loss_coeff * S
        self.metric_accuracy = (torch.argmax(x,1) == self.target).sum().float()/len(self.target)

        step_size = self.opt.step_size
        humaneness_delta = constants.HUMAN_DELTA
        window_size = int(humaneness_delta/step_size)
        # print(humaneness_reg.shape)
        receptive_field = self.net.module.receptive_field
        # weights = torch.sum(torch.argmax(self.input[:,-5:,receptive_field//2-(window_size-1):receptive_field//2],1)==4,1).float()
        notes = (torch.argmax(self.input[:,-5:,receptive_field//2-(window_size):receptive_field//2],1)==4).float()
        distance_factor = torch.tensor(np.exp(-2*np.arange(window_size,0,-1)/window_size)).float().cuda()
        # print(notes.shape, distance_factor.shape)
        weights = torch.tensordot(notes,distance_factor,dims=1)
        # print()
        # print(self.input[:,-5:,receptive_field//2-(window_size-1):receptive_field//2].shape)
        # self.loss_humaneness_reg = F.relu(humaneness_reg-1).mean()
        # humaneness_reg = -F.cross_entropy(x,torch.ones(weights.shape).long().cuda(), reduction='none')
        humaneness_reg = F.cross_entropy(x,torch.zeros(weights.shape).long().cuda(), reduction='none')
        humaneness_reg = torch.dot(humaneness_reg, weights)
        self.loss_humaneness_reg = humaneness_reg
        self.loss_total = self.loss_ce + self.opt.humaneness_reg_coeff * self.loss_humaneness_reg

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
