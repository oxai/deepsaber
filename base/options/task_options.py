import argparse
import importlib


class TaskOptions:
    """
    Base class to be inherited from task instances when they want to add task-dependent options.
    E.g. segmentation options for images.
    The options from this object are added to the options in BaseOptions
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(add_help=False)

    def add_actions(self, parser):
        self.actions = self.parser._actions
        for action in self.actions:
            for i, ex_action in enumerate(parser._actions):
                if action.option_strings == ex_action.option_strings:
                    parser._actions[i] = action
        return parser


def get_task_options(task_name):

    task_module = importlib.import_module(task_name)
    options_filename = task_name + ".options." + task_name.lower() + "_options"
    optionslib = importlib.import_module(options_filename, package=task_module)
    options = None
    target_options_name = task_name.replace('_', '') + 'options'
    for name, cls in optionslib.__dict__.items():
        if name.lower() == target_options_name.lower() \
           and next(iter(cls.__bases__)).__module__.endswith(TaskOptions.__module__):  # check that base class is BaseModel
            options = cls

    if options is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (options_filename, target_options_name))
    return options()