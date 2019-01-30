import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return 0

    def make_subset(self, indices):
        return data.Subset(self, indices)