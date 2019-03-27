from .base_model import BaseModel
from .networks import WaveNetModel as WaveNet
import torch.nn.functional as F
import torch
from . import networks

class AdvWaveNetModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ['ce','gen','disc']
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
                           bias=opt.bias)

        self.discriminator = WaveNet(layers=opt.layers,
                                    blocks=opt.blocks,
                                    dilation_channels=opt.dilation_channels,
                                    residual_channels=opt.residual_channels,
                                    skip_channels=opt.skip_channels,
                                    end_channels=opt.end_channels,
                                    input_channels=opt.input_channels,
                                    output_length=1,
                                    output_channels=1,
                                    num_classes=1,
                                    dropout_p=opt.dropout_p,
                                    kernel_size=opt.kernel_size,
                                    bias=opt.bias)

        if self.gpu_ids:
            self.discriminator = networks.init_net(self.discriminator, self.opt.init_type, self.opt.init_gain,
                                self.opt.gpu_ids)  # takes care of pushing net to cuda
            assert torch.cuda.is_available()

        self.gen_optimizers = [torch.optim.Adam([
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
            {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
             'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.disc_optimizers = [torch.optim.Adam([
        {'params': [param for name, param in self.discriminator.named_parameters() if name[-4:] == 'bias'],
            'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
        {'params': [param for name, param in self.discriminator.named_parameters() if name[-4:] != 'bias'],
        'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
        ])]
        self.optimizers = self.gen_optimizers + self.disc_optimizers
        self.loss_ce = None
        self.loss_gen = None
        self.loss_disc = None

#    def update_learning_rate(self):
#        for scheduler in self.schedulers:
#            scheduler.step()
#        lr = self.gen_optimizers[0].param_groups[0]['lr']
#        print('learning rate = %.7f' % lr)


    def name(self):
        return "AdvWaveNet"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--layers', type=int, default=10, help="Number of layers in each block")
        parser.add_argument('--blocks', type=int, default=4, help="Number of residual blocks in network")
        parser.add_argument('--dilation_channels', type=int, default=32, help="Number of channels in dilated convolutions")
        parser.add_argument('--residual_channels', type=int, default=32, help="Number of channels in the residual link")
        parser.add_argument('--skip_channels', type=int, default=256)
        parser.add_argument('--end_channels', type=int, default=256)
        parser.add_argument('--input_channels', type=int, default=(1+20))
        parser.add_argument('--output_length', type=int, default=1)
        parser.add_argument('--num_classes', type=int, default=20)
        parser.add_argument('--output_channels', type=int, default=(4*3))
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--bias', action='store_false')
        parser.add_argument('--frequency_gen_updates', type=int, default=5)
        parser.add_argument('--dropout_p', type=float, default=0.5)
        parser.add_argument('--loss_ce_weight', type=float, default=1.0)
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
        [n, channels, classes, l] = x.size() # l is number of time steps of output (which is opt.output_length)
        x = x.transpose(1, 3).contiguous()
        x = x.view(n * l * channels, classes)

        self.loss_ce = F.cross_entropy(x, self.target)
        self.metric_accuracy = (torch.argmax(x,1) == self.target).sum().float()/len(self.target)
        generated_level = F.softmax(self.output[:,:,:,:-1],2).contiguous().view(n,channels*classes,l-1).cuda()
        p_gen = self.discriminator.forward(torch.cat((self.input[:,:-self.opt.output_channels*self.opt.num_classes,-(l-1):],generated_level),1)).squeeze()
        p_gen = torch.sigmoid(p_gen)
        self.loss_gen = torch.log(1-p_gen).mean() # high when discriminator thinks it likely false
        #lr = self.optimizers[0].param_groups[0]['lr']
        self.loss_gen += self.opt.loss_ce_weight * self.loss_ce

        # p_real = self.discriminator.forward(self.input[:,-self.opt.output_channels*self.opt.num_classes:,:l]).squeeze()
        p_real = self.discriminator.forward(self.input[:,:,:l]).squeeze()
        p_real = torch.sigmoid(p_real)
        self.loss_disc = -self.loss_gen - torch.log(p_real).mean()
        self.loss_disc  += self.opt.loss_ce_weight * self.loss_ce


    def gen_backward(self):
        self.gen_optimizers[0].zero_grad()
        self.loss_gen.backward()
        self.gen_optimizers[0].step()

    def disc_backward(self):
        self.disc_optimizers[0].zero_grad()
        self.loss_disc.backward()
        self.disc_optimizers[0].step()

    def optimize_parameters(self,optimize_generator = False):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        if optimize_generator:
            self.gen_backward()
        else:
            self.disc_backward()
        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass
