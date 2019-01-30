from __future__ import print_function
import os
import sys
import errno
import time
import socket
import re
from collections import OrderedDict, defaultdict
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentTypeError
import torch
import numpy as np
from PIL import Image


# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382%28v=vs.85%29.aspx
    Official listing of all such codes.
'''


def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


def str_is_int(s):
    r"""
    Check if string is convertable to an integer
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def str2bool(v):
    r"""
    Use with argparse to convert string to bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def print_current_losses_metrics(epoch, i, losses, metrics, t, t_data):
    if i:  # iter is not given in validation (confusing?)
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    else:
        message = '(epoch: %d, validation) ' % (epoch)
    for k, v in (OrderedDict(losses, **metrics)).items():  # not displayed in correct order in python <3.6
        if not i:
            k = '_'.join(k.split('_')[0:-1])
        message += '%s: %.3f ' % (k, v)
    print(message)


def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string


def on_cluster():
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)


def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = dict()
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '{:.2f}{}B'.format(value, s)
    return "{}B".format(n)


def check_mkdir(dir_name):
    try:
        os.mkdir(str(dir_name))
    except FileExistsError:
        pass

def get_flags(filepath):
    with open(filepath, 'r') as argsfile:
        args = eval(argsfile.readline())
    return args


#### torch

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def evaluate_multilabel(predictions, gts):
    r""" Computes metrics, works for single channel too
        Input is numpy in torch tensor form: NxCxHxW
    """
    acc_cls, dice_cls = [],[]
    if predictions.shape[1] > 1:
        predictions = np.exp(predictions) / np.repeat(
                        np.exp(predictions).sum(axis=1)[:,np.newaxis,...], predictions.shape[1], axis=1)
    else:
        predictions = 1/(1 + np.exp(-predictions))
    for c in range(predictions.shape[1]):
        pred = predictions[:,c,...].flatten()
        gt = gts[:,c,...].flatten()
        acc_cls.append(round(float(np.mean(np.array(pred.round() == gt))),2))
        dice_cls.append(round(float(dice_coeff(pred, gt)), 2))
    acc = float(np.mean(acc_cls))
    dice = float(np.mean(dice_cls))
    return acc, acc_cls, dice, dice_cls


def dice_coeff(pred, target):
    r"""This definition generalize to real valued pred and target vector.
    Exact - for numpy arrays
    """
    smooth = 0.0001
    iflat = pred.flatten().round()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) / (tflat.sum() + iflat.sum() + smooth))


def dice_loss(output, target, weights):
    """
    output : NxCxHxW Variable
    target :  NxCxHxW FloatTensor
    weights : C FloatTensor
    """
    eps = 0.0001

    intersection = output * target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    loss_per_channel = weights * (1 - (numerator / denominator))
    return loss_per_channel.sum() / output.size(1)


class MultiLabelSoftDiceLoss(nn.Module):
    def __init__(self, weights=None, num_class=3):
        super(MultiLabelSoftDiceLoss, self).__init__()
        if num_class>1:
            self.sm = nn.Softmax2d()
        else:
            self.sm = nn.Sigmoid()
        self.weights = nn.Parameter(torch.from_numpy(np.array(weights) or np.array([1 for i in range(num_class)])).type(torch.FloatTensor),
                        requires_grad=False)

    def forward(self, outputs, targets):
        return dice_loss(self.sm(outputs), targets, self.weights)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(gt):
    if gt.shape[0] > 1:
        if gt.min() < 0 or gt.max() > 1: #if logit
            if np.any(gt > 500): gt /= gt.max()
            gt = np.exp(gt) / np.repeat(np.exp(gt).sum(axis=0)[np.newaxis,...], gt.shape[0], axis=0)
        gt = np.round(gt)

        if gt.shape[0] == 3:
            r = np.floor((gt[0,...] == 1) * 255)
            g = np.floor((gt[1,...] == 1) * 255)
            b = np.floor((gt[2,...] == 1) * 255)
        elif gt.shape[0] == 2:
            r = np.floor((gt[1,...] == 1) * 255)
            g = np.floor((gt[0,...] == 1) * 255)
            b = np.zeros(gt[0,...].shape)
        elif gt.shape[0] == 1:
            r,g,b = gt*255, gt*255, gt*255
        else:
            raise NotImplementedError
    else:
        if gt.min() < 0 or gt.max() > 1:
            gt = 1/(1+np.exp(-gt))
        gt = np.round(gt)[0,...]
        r,g,b = gt*255, gt*255, gt*255
    gt_colorimg = np.stack([r,g,b], axis=2).astype(np.uint8)
    return gt_colorimg


#### pix2pix/cyclegan utils ####




# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, segmap=False, imtype=np.uint8):
    """
    Takes first image in batch and converts it to tensor for visualisation purposes
    :param input_image:
    :param segmap:
    :param imtype:
    :return: image
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()  # taking the first image only
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    if segmap:
        image_numpy = segmap2img(image_numpy)
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # for segmentation maps with four classes
    return image_numpy.astype(imtype)


def segmap2img(segmap):
    """
    color coding segmap into an image
    """
    if len(segmap.shape) > 2:
        segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)
        if segmap.shape[2] == 4:
            image[image == 1] = 160
            image[image == 2] = 200
            image[image == 3] = 250
        elif segmap.shape[2] == 3:
            image[image == 1] = 200
            image[image == 2] = 250
        elif segmap.shape[2] == 2:
            image[image == 1] = 250
        else:
            raise ValueError("Conversion of map to image not supported for shape {}".format(segmap.shape))
    else:
        image = segmap * 250
    image = image[:, :, np.newaxis].repeat(3, axis=2)
    return image


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



#### NOT USED ###

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index =-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate_singlelabel(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes)) #confusion matrix
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) #intersection over union?
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class PolyLR(object):
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
        self.max_iter = float(max_iter)
        self.init_lr_groups = []
        for p in optimizer.param_groups:
            self.init_lr_groups.append(p['lr'])
        self.param_groups = optimizer.param_groups
        self.curr_iter = curr_iter
        self.lr_decay = lr_decay

    def step(self):
        for idx, p in enumerate(self.param_groups):
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x


def summary(model, input_size, device, batch_size=-1):
    """
    Prints out a detailed summary of the pytorch model.
    From: https://github.com/sksq96/pytorch-summary
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :return:
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
