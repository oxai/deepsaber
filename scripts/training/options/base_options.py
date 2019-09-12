import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
TRAINING_DIR = os.path.join(SCRIPTS_DIR, 'training')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)
import argparse
import multiprocessing as mp
import torch
import importlib
import pkgutil
import models
import scripts.training.data as data
import scripts.misc.utils.utils as utils
import json

class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         add_help=False)  # TODO - check that help is still displayed
        parser.add_argument('--task', type=str, default='scripts.training', help="Module from which dataset and model are loaded")
        parser.add_argument('-d', '--data_dir', type=str, default='/Users/andreachatrian/Documents/Repositories/oxai/beatsaber/DataE')
        parser.add_argument('--dataset_name', type=str, default="song")
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--num_windows', default=16, type=int)
        parser.add_argument('--pad_batches', action='store_true', help='whether to pad batches sequences to the length of the longest in the minibatch')
        parser.add_argument('--augment', type=int, default=0)
        parser.add_argument('--model', type=str, default="wavenet", help="The network model used for beatsaberification")
        parser.add_argument('--init_type', type=str, default="normal")
        parser.add_argument('--init_gain', default=0.02, type=float)
        parser.add_argument('--eval', action='store_true', help='use eval mode during validation / test time.')
        parser.add_argument('--do_validation', action='store_true', help='use eval mode during validation / test time.')
        parser.add_argument('-nf', '--num_filters', type=int, default=15, help='mcd number of filters for unet conv layers')
        parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
        parser.add_argument('--learning_rate_patience', default=50, type=int)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--loss_weight', default=None)
        parser.add_argument('--gpu_ids', default='-1', type=str, help='gpu ids (comma separated numbers - e.g. 1,2,3), =-1 for cpu use')
        parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')
        parser.add_argument('--experiment_name', default="experiment_name", type=str)
        parser.add_argument('--checkpoints_dir', default=TRAINING_DIR, type=str, help='checkpoint folder')
        parser.add_argument('--load', action='store_true', help='whether to load model or not.')
        parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, whether load models by iteration')
        parser.add_argument('-ad', '--augment_dir', type=str, default='')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--fork_processes', action='store_true', help="Set method to create dataloader child processes to fork instead of spawn (could take up more memory)")

        parser.add_argument('--using_bpm_time_division', action='store_true', help="Whether to use the time divisions which are divisors of the beats")

        parser.add_argument('--beat_subdivision', default=16, type=int)
        parser.add_argument('--step_size', default=0.01, type=float)

        self.parser = parser
        self.is_train = None
        self.opt = None

    def gather_options(self):
        # get the basic options
        opt, _ = self.parser.parse_known_args()

        # load task module and task-specific options
        task_name = opt.task
        task_options = importlib.import_module("{}.options.task_options".format(task_name))  # must be defined in each task folder
        self.parser = argparse.ArgumentParser(parents=[self.parser, task_options.TaskOptions().parser])
        opt, _ = self.parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name, task_name)
        parser = model_option_setter(self.parser, self.is_train)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options fsldkn
        dataset_name = opt.dataset_name
        print(dataset_name, task_name)
        dataset_option_setter = data.get_option_setter(dataset_name, task_name)
        parser = dataset_option_setter(parser, self.is_train)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        file_name_json = os.path.join(expr_dir, 'opt.json')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        with open(file_name_json, 'wt') as opt_file:
            opt_file.write(json.dumps(vars(opt)))

    def parse(self):

        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # check options:
        if opt.loss_weight:
            opt.loss_weight = [float(w) for w in opt.loss_weight.split(',')]
            if len(opt.loss_weight) != opt.num_class:
                raise ValueError("Given {} weights, when {} classes are expected".format(
                    len(opt.loss_weight), opt.num_class))
            else:
                opt.loss_weight = torch.tensor(opt.loss_weight)

        self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # set multiprocessing
        if opt.workers > 0 and not opt.fork_processes:
            mp.set_start_method('spawn', force=True)

        self.opt = opt
        return self.opt
