import os
import torch
from contextlib import contextmanager
from base.utils import utils
from collections import OrderedDict
from . import networks

# Benefits of having one skeleton, e.g. for train - is that you can keep all the incremental changes in
# one single code, making it your streamlined and updated script -- no need to keep separate logs on how
# to implement stuff


class BaseModel:
    """
    Philosophy: a model is different from a pytorch module, as a model may contain
    multiple networks that have a forward method that is not sequential
    (thing about VAE-GAN)

    call method is forward() -- as in Module

    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
        self.loss_names = []
        self.metric_names = []
        self.module_names = []  # changed from 'model_names'
        self.visual_names = []
        self.visual_types = []  # added to specify how to plot (e.g. in case output is a segmentation map)
        self.image_paths = []
        self.optimizers = []
        self.schedulers = []
        self.input = None
        self.target = None
        self.output = None

    def name(self):
        return 'BaseModel'

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        ABSTRACT METHOD
        :param parser:
        :param is_train:
        :return:
        """
        return parser

    def set_input(self, data):
        self.input = data['input']
        self.target = data['target']
        if self.opt.gpu_ids and (self.input.device.type == 'cpu' or self.target.device.type == 'cpu'):
            self.input = self.input.cuda()
            self.target = self.target.cuda()

    def forward(self):
        pass

    # load and print networks; create schedulers;
    def setup(self):
        """
        This method shouldn't be overwritten.
        1. Initialises networks and pushes nets and losses to cuda,
        2. Sets up schedulers
        3. Loads and prints networks;
        :param opt:
        :param parser:
        :return:
        """
        if self.gpu_ids:
            self.net = networks.init_net(self.net, self.opt.init_type, self.opt.init_gain,
                                self.opt.gpu_ids)  # takes care of pushing net to cuda
            assert torch.cuda.is_available()
            ### what is this thing for?
            # for loss_name in self.loss_names:
            #     loss = getattr(self, loss_name).cuda()
            #     loss = loss.to(self.device)
            #     setattr(self, loss_name, loss)
        if self.is_train:
            self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        if not self.is_train or self.opt.continue_train:
            load_suffix = 'iter_%d' % self.opt.load_iter if self.opt.load_iter > 0 else self.opt.load_epoch
            self.load_networks(load_suffix)
        for module_name in self.module_names:
            net = getattr(self, "net" + module_name)
            net.train()
        self.print_networks(self.opt.verbose)

    # make models eval mode during test time
    def eval(self):
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    @contextmanager
    def start_validation(self):
        """
        Context manager for setting up meter that average the validation metrics over validation data-set,
        and then set the val attributes of the model.
        Use the yielded function 'update_validation_meters' to compute running average of validation metrics
        """
        # __enter__ #
        loss_meters = {loss_name: utils.AverageMeter() for loss_name in self.loss_names}
        metric_meters = {metric_name: utils.AverageMeter() for metric_name in self.metric_names}
        model = self

        def update_validation_meters():
            # update meters (which remain hidden from main)
            for loss_name in model.loss_names:
                loss = getattr(model, 'loss_' + loss_name)
                loss_meters[loss_name].update(loss, model.opt.batch_size)
            for metric_name in model.metric_names:
                metric = getattr(model, 'metric_' + metric_name)
                metric_meters[metric_name].update(metric, self.opt.batch_size)

        # as #
        yield update_validation_meters
        # __exit__ #
        # Copy values to validation fields
        for loss_name in self.loss_names:
            loss_val_name = 'loss_' + loss_name + '_val'
            loss = loss_meters[loss_name].avg
            setattr(self, loss_val_name, loss)
        for metric_name in self.metric_names:
            metric_val_name = 'metric_' + metric_name + '_val'
            metric = metric_meters[metric_name].avg
            setattr(self, metric_val_name, metric)
        for visual_name in self.visual_names:
            visual_val = getattr(self, visual_name)
            setattr(self, visual_name + "_val", visual_val)

    def optimize_parameters(self):
        pass

    def evaluate_parameters(self):
        """
        Abstract method that I added -- pix2pix code did not compute evaluation metrics,
        but for many tasks they can be quite useful
        Updates metrics values (metric must start with 'metric_')
        """
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_losses(self, is_val=False):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                name = name if not is_val else name + "_val"
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def get_current_metrics(self, is_val=False):
        metric_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                name = name if not is_val else name + "_val"
                metric_ret[name] = float(getattr(self, 'metric_' + name))
        return metric_ret

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self, is_val=False):
        visual_ret = OrderedDict()
        for name, kind in zip(self.visual_names, self.visual_types):
            if isinstance(name, str):
                name = name if not is_val else name + "_val"
                visual_ret[name + "_" + kind] = getattr(self, name)
        return visual_ret

    # save models to the disk
    def save_networks(self, epoch):
        for name in self.module_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    # load models from the disk
    def load_networks(self, epoch):
        for name in self.module_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # if not self.opt.gpu_ids:
                #    state_dict = {key[6:]: value for key, value in
                #                    state_dict.items()}  # remove data_parallel's "module."
                net.load_state_dict(state_dict)

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.module_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    utils.summary(net, (3, self.opt.fine_size, self.opt.fine_size), self.device.type)
                    # print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # shares modules memory, so that models can be used in multiprocessing
    def share_memory(self):
        # TODO is this necessary for cuda weights?
        for module_name in self.module_names:
            net = getattr(self, 'net' + module_name)
            if net is not None:
                net.share_memory()
