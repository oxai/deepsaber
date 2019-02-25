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
                           output_channels=opt.output_channels,
                           num_classes=opt.num_classes,
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
        parser.add_argument('--input_channels', type=int, default=(1+(9*3+1)*(5*3)))
        parser.add_argument('--output_length', type=int, default=1)
        parser.add_argument('--num_classes', type=int, default=(9*3+1))
        parser.add_argument('--output_channels', type=int, default=(4*3))
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--bias', action='store_false')
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_ = data['input']
        target_ = data['target']
        input_shape = input_.shape
        target_shape = target_.shape
        self.input = input_.reshape((input_shape[0]*input_shape[1], input_shape[2], input_shape[3])).to(self.device)
        self.target = target_.reshape((target_shape[0]*target_shape[1]*target_shape[2]*target_shape[3])).to(self.device)

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
