import importlib
from torch.utils.data import DataLoader
from .base_dataset import BaseDataset


def find_dataset_using_name(dataset_name, task_name):
    # Given the option --dataset_name [datasetname],
    # the file "data/datasetname_dataset.py"
    # will be imported.
    try:
        task_module = importlib.import_module(task_name)
        dataset_filename = task_name + ".data." + dataset_name + "_dataset"
        datasetlib = importlib.import_module(dataset_filename, package=task_module)
    except (ModuleNotFoundError, ImportError):
        # if module not found, attempt to load from base
        task_name = 'base'
        task_module = importlib.import_module(task_name)
        dataset_filename = task_name + ".data." + dataset_name + "_dataset"
        datasetlib = importlib.import_module(dataset_filename, package=task_module)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.

    def is_subclass(subclass, superclass):
        return next(iter(subclass.__bases__)).__module__.endswith(superclass.__module__)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            if is_subclass(cls, BaseDataset) or any(is_subclass(cls_b, BaseDataset) for cls_b in cls.__bases__):
                dataset = cls

    if dataset is None:
        raise NotImplementedError("In {}.py, there should be a subclass of BaseDataset with class name that matches {} in lowercase.".format(
              dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name, task_name):
    dataset_class = find_dataset_using_name(dataset_name, task_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, validation_phase=False):
    dataset = find_dataset_using_name(opt.dataset_name, opt.task)
    if validation_phase:
        opt.phase = "val"
    instance = dataset(opt)
    print('dataset [{}] was created {}'.format(instance.name(), "(val)" if validation_phase else ''))
    return instance


def create_dataloader(dataset):
    is_val = dataset.opt.phase == "val"
    return DataLoader(dataset,
                      batch_size=dataset.opt.batch_size if not is_val else dataset.opt.val_batch_size,
                      shuffle=not is_val, num_workers=dataset.opt.workers)
