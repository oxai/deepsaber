from .base_model import BaseModel
from .networks import WaveNetModel as WaveNet
import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvnetModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['ce']
        self.metric_names = []
        self.module_names = ['conv']  # changed from 'model_names'
        self.netconv = ConvNet(opt)
        self.loss_ce = None
        self.optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.netconv.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.netconv.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]

    def name(self):
        return "ConvNet"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--layers', type=int, default=5, help="Number of layers in each block")
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--input_channels', type=int, default=1, help="Number of feature channels")
        parser.add_argument('--output_channels', type=int, default=4)
        return parser

    def set_input(self, data):
        # move multiple samples of the same song to the second dimension and the reshape to batch dimension
        input_ = data['input'].permute(0, 3, 1, 2)  # bring multiple chunks per song dimension close to batch dimension (0)
        shape = input_.shape
        self.input = input_.reshape((shape[0]*shape[1], shape[2], shape[3]))
        target = data['target'].permute(0, 2, 1)
        shape = target.shape
        self.target = target.reshape((shape[0]*shape[1], shape[2]))

    def forward(self):
        # transform into one batch:
        self.output = self.netconv.forward(self.input)
        self.loss_ce = F.cross_entropy(self.output, self.target)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_ce.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.netconv, requires_grad=True)
        self.forward()
        self.backward()
        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass


class ConvNet(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(opt.input_channels, opt.num_filters, opt.kernel_size),
            nn.InstanceNorm1d(opt.num_filters),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(opt.num_filters, opt.num_filters, opt.kernel_size),
            nn.InstanceNorm1d(opt.num_filters),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )

        layers = []
        for i in range(1, opt.layers + 1):
            layers.extend([
                nn.Conv1d(opt.num_filters * i, opt.num_filters * (i + 1), opt.kernel_size),
                nn.InstanceNorm1d(opt.num_filters),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=0.2),
                nn.ReLU()
            ])
        self.features = nn.Sequential(*layers)

        self.penultimate = nn.Sequential(
            nn.Conv1d(opt.num_filters * (i + 1), opt.num_filters * (i + 1), opt.kernel_size),
            nn.InstanceNorm1d(opt.num_filters),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(500, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, opt.output_channels)
        )

    def forward(self, input):
        initial = self.initial(input)
        features = self.features(initial)
        penultimate = self.penultimate(features)
        resized = F.interpolate(penultimate, 500,  mode='linear')
        return self.final(resized)
