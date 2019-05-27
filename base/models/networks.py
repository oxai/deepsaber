import os
import os.path
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, Optimizer
from torch.autograd import Variable, Function
from torch.nn import init
import numpy as np
import Constants


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field = blocks * layers * (kernel_size - 1)
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 input_channels=256,
                 output_length=32,
                 output_channels=1,
                 num_classes=2,
                 kernel_size=2,
                 dropout_p=0,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        # self.dropouts = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channels,
                                    out_channels=self.residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                                                        num_channels=residual_channels,
                                                        dilation=new_dilation,
                                                        dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2
                # self.dropouts.append(nn.Dropout(self.dropout_p))

        self.dropout = nn.Dropout(self.dropout_p)
        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_convs_2 = nn.ModuleList()
        for c in range(self.output_channels):
            self.end_convs_2.append(nn.Conv1d(in_channels=end_channels,
                                        out_channels=self.num_classes,
                                        kernel_size=1,
                                        bias=True))

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = self.dropout(x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        xs = []
        for c in range(self.output_channels):
            xs.append(self.end_convs_2[c](x))

        x = torch.stack(xs)
        x = x.permute(1, 0, 2, 3)
        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        [n, channels, classes, l] = x.size()
        l = self.output_length
        x = x[:, :, :, -l:]
        # x = x.transpose(1, 3).contiguous()
        # x = x.view(n * l * channels, classes)
        return x

    def generate(self,
                 num_samples,
                 conditioning_seq, #the song in our case
                 time_shifts, # number of look ahead times
                 first_samples=None,
                 temperature=1.):
        self.eval()
        # if first_samples is None:
        #     # first_samples = torch.zeros((1,12)) #self.dtype(1).zero_()
        #     first_samples = torch.zeros((1,self.output_channels,self.receptive_field//2)) #self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)

        # num_pad = self.receptive_field - generated.size(2)
        # num_pad = self.receptive_field//2 - generated.size(2)
        # if num_pad > 0:
        #     generated = constant_pad_1d(generated.permute(2,1,0), self.receptive_field//2, pad_start=True).permute(2,1,0)
        #     generated = generated.unsqueeze(0)
        #     print("pad zero")

        #size of this is (1,n_mfcc,timesteps)
        conditioning_seq = constant_pad_1d(conditioning_seq.permute(2,1,0),conditioning_seq.size(2)+self.receptive_field//2,pad_start=True).permute(2,1,0)
        conditioning_seq = constant_pad_1d(conditioning_seq.permute(2,1,0),conditioning_seq.size(2)+self.receptive_field,pad_start=False).permute(2,1,0)

        for i in range(num_samples):
            input = Variable(torch.FloatTensor(1, self.output_channels, self.num_classes, self.receptive_field//2).zero_())
            # input = Variable(torch.FloatTensor(1, self.num_classes, self.receptive_field).zero_())
            # input = input.scatter_(1, generated[:,:,-self.receptive_field:].long(), 1.)
            # input = input.scatter_(2, (generated[:,:,-self.receptive_field:].view(1,12,-1,self.receptive_field).long()-1)%28, 1.)
            input = input.scatter_(2, generated[:,:,-self.receptive_field//2:].unsqueeze(2).long(), 1.)
            shape = input.shape
            input = input.view(shape[0],shape[1]*shape[2],shape[3])
            blocks_reduced_windows_pad = torch.zeros((shape[0],shape[1]*shape[2],self.receptive_field//2 + self.receptive_field%2))
            blocks_reduced_windows_pad[:,Constants.PAD_STATE,:] = 1.0
            input = torch.cat([input,blocks_reduced_windows_pad],2)

            # #this is because "nothing" is a class, but I haven't put a one on the many_hot vector in its position
            # #fixed that now
            # for j in range(self.output_channels):
            #     input[:,self.num_classes*j,:]=0

            featuress = []
            # for ii in range(time_shifts):
            features = conditioning_seq[:,:,i:i+self.receptive_field].float()
            if torch.abs(features).max() > 0: features = (features - features.mean())/torch.abs(features).max()
            featuress.append(features.float())

            # print(features.shape, input.shape)
            input = torch.cat(featuress+[input.float()],1).cuda() #eeh need to have it work without cuda too

            x = self.wavenet(input,dilation_func=self.wavenet_dilate)[0,:,:,0]
            # x = x.transpose(1, 3).contiguous() # need to undertand why transposing here makes a difference.. This line is necessary.. what I mean is that I think it broke when using permute, but perhaps I was being stupid :P
            # x = x.view(self.output_channels ,self.num_classes)

            if temperature > 0:
                x /= temperature
                xs = []
                for i in range(self.output_channels):
                    x1 = x[i,:]
                    prob = F.softmax(x1, dim=0)
                    prob = prob.cpu()
                    np_prob = prob.data.numpy()
                    x1 = np.random.choice(self.num_classes, p=np_prob)
                    xs.append(x1)
                x = Variable(torch.LongTensor([xs]))#np.array([x])
            else:
                x = torch.max(x, 1)[1].float()

            x = x.unsqueeze(2)
            generated = torch.cat((generated, x.float()), 2)

        # generated = (generated / self.input_channels) * 2. - 1
        # mu_gen = mu_law_expansion(generated, self.input_channels)

        self.train()
        return generated[:,:,self.receptive_field//2:]

    #TODO: this should be parallelized.... but I'm not doing it yet :PP
    def generate_no_autoregressive(self,
                 num_samples,
                 conditioning_seq, #the song in our case
                 time_shifts, # number of look ahead times
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            # first_samples = torch.zeros((1,12)) #self.dtype(1).zero_()
            first_samples = torch.zeros((1,self.output_channels,self.receptive_field)) #self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)
        #
        num_pad = self.receptive_field - generated.size(2)
        if num_pad > 0:
            generated = constant_pad_1d(generated.permute(2,1,0), self.receptive_field, pad_start=True).permute(2,1,0)
            generated = generated.unsqueeze(0)
            print("pad zero")

        #size of this is (1,n_mfcc,timesteps)
        conditioning_seq = constant_pad_1d(conditioning_seq.permute(2,1,0),conditioning_seq.size(2)+self.receptive_field,pad_start=True).permute(2,1,0)

        for i in range(num_samples):
            # input = Variable(torch.FloatTensor(1, self.output_channels, self.num_classes, self.receptive_field).zero_())
            # # input = input.scatter_(1, generated[:,:,-self.receptive_field:].long(), 1.)
            # # input = input.scatter_(2, (generated[:,:,-self.receptive_field:].view(1,12,-1,self.receptive_field).long()-1)%28, 1.)
            # input = input.scatter_(2, generated[:,:,-self.receptive_field:].view(1,self.output_channels,-1,self.receptive_field).long(), 1.)
            #
            # shape = input.shape
            # input = input.view(shape[0],shape[1]*shape[2],shape[3])
            # #this is because "nothing" is a class, but I haven't put a one on the many_hot vector in its position
            # #fixed that now
            # for j in range(self.output_channels):
            #     input[:,self.num_classes*j,:]=0

            mfcc_featuress = []
            for ii in range(time_shifts):
                mfcc_features = conditioning_seq[:,:,i+ii:i+ii+self.receptive_field].float()
                if torch.abs(mfcc_features).max() > 0: mfcc_features = (mfcc_features - mfcc_features.mean())/torch.abs(mfcc_features).max()
                mfcc_featuress.append(mfcc_features.float())

            input = torch.cat(mfcc_featuress,1).cuda() #eeh need to have it work without cuda too

            x = self.wavenet(input,dilation_func=self.wavenet_dilate)[0:1, :, :, -1:]
            x = x.transpose(1, 3).contiguous() # need to undertand why transposing here makes a difference.. This line is necessary.. what I mean is that I think it broke when using permute, but perhaps I was being stupid :P
            x = x.view(self.output_channels ,self.num_classes)

            if temperature > 0:
                x /= temperature
                xs = []
                for i in range(self.output_channels):
                    x1 = x[i,:]
                    prob = F.softmax(x1, dim=0)
                    prob = prob.cpu()
                    np_prob = prob.data.numpy()
                    x1 = np.random.choice(self.num_classes, p=np_prob)
                    xs.append(x1)
                x = Variable(torch.LongTensor([xs]))#np.array([x])
            else:
                x = torch.max(x, 1)[1].float()

            x = x.unsqueeze(2)
            generated = torch.cat((generated, x.float()), 2)

        self.train()
        return generated[:,:,self.receptive_field:]

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.input_channels // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = Variable(torch.FloatTensor(1, self.input_channels, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.input_channels, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(Variable(torch.arange(self.input_channels)) - self.input_channels / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.input_channels, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.input_channels) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.input_channels, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.input_channels)
        return mu_gen


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        for q in self.dilated_queues:
            q.dtype = self.dtype
        super().cpu()

### Components ###

def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    def __init__(self, target_size, dimension=0, value=0, pad_start=False):
        super(ConstantPad1d, self).__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

    def forward(self, input):
        self.num_pad = self.target_size - input.size(self.dimension)
        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = input.size()

        size = list(input.size())
        size[self.dimension] = self.target_size
        output = input.new(*tuple(size)).fill_(self.value)
        c_output = output

        # crop output
        if self.pad_start:
            c_output = c_output.narrow(self.dimension, self.num_pad, c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension, 0, c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(input)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension, self.num_pad, cg_output.size(self.dimension) - self.num_pad)
        else:
            cg_output = cg_output.narrow(self.dimension, 0, cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)
        return grad_input


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d(target_size, dimension, value, pad_start)(input)


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def list_all_audio_files(location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", "aiff"))]:
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + location)
    return audio_files


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s

# Helper functions

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=tuple()):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(device=gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.nepoch) / float(opt.nepoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nepoch, eta_min=0)
    elif opt.lr_policy == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=opt.learning_rate / 10, max_lr=opt.learning_rate,
                             step_size=opt.nepoch_decay, mode='triangular2')
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# learning rate schedules


class CyclicLR(object):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `batch_step` should be called after a batch has been used for training.
    To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for eachparam groups.
            Default: 0.001
        max_lr (float or list): Upper boundaries in the cycle for
            each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function. Default: 0.006
        step_size (int): Number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch. Default: 2000
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        last_batch_iteration (int): The index of the last batch. Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         scheduler.batch_step()
        >>>         train_batch(...)

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
