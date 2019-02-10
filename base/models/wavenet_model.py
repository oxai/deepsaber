from .base_model import BaseModel
from .networks import WaveNetModel as WaveNet
import torch.nn.functional as F
import torch


class WaveNetModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['ce']
        self.metric_names = []
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
                           kernel_size=opt.kernel_size,
                           bias=opt.bias)
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]

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
        parser.add_argument('--input_channels', type=int, default=1)
        parser.add_argument('--output_length', type=int, default=1)
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--bias', action='store_false')
        return parser

    def forward(self):
        self.output = self.net.forward(self.input)
        self.loss_ce = F.cross_entropy(self.output, self.target)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_ce.backward()
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




