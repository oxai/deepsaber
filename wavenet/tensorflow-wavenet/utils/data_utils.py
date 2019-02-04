import itertools
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def merge_and_split(inputs, labels):
    df = inputs.reset_index().merge(labels.reset_index(), on='utterance',
                                    how='inner').set_index('utterance')
    return df.feat, df.label


def list_to_sparse(inputs):
    """Convert list of lists into scipy coo matrix.
    """

    data = list(itertools.chain(*inputs))
    row = list(itertools.chain(
        *[itertools.repeat(i, len(x)) for i, x in enumerate(inputs)]))
    col = list(itertools.chain(*[range(len(x)) for x in inputs]))

    s = coo_matrix((data, (row, col)), shape=(
            len(inputs), np.max([len(x) for x in inputs])))

    return s

# def next_batch():
#     start = 0
#     end = 0
#     batch_size, data_length = 1, 441
#     while end != data_length:
#         end += batch_size
#         end = data_length if end >= data_length else end
#         yield (start, end)
#         start = end
#
# class BatchGenerator(object):
#     def __init__(self, data, batch_size=1):
#         self.inputs, self.labels = data
#         self.batch_size = batch_size
#         self.data_length = len(self.inputs)
#         self.sequence_length = np.array([x.shape[0] for x in self.inputs])
#     def next_batch(self):
#         self._suffle()
#         start = 0
#         end = 0
#         batch_size, data_length = self.batch_size, self.data_length
#         while end != data_length:
#             end += batch_size
#             end = data_length if end >= data_length else end
#             yield (start,end)
#             start = end
#     def _suffle(self):
#         permutation = np.random.permutation(self.data_length)
#         self.inputs = self.inputs[permutation]
#         self.labels = self.labels[permutation]
#         self.sequence_length = self.sequence_length[permutation]
#     def _get(self, start, end):
#         sequence_length = self.sequence_length[start:end]
#         batch_sequence_length = np.max(sequence_length)
#         inputs = np.array([np.pad(x, pad_width=((0, batch_sequence_length - len(x)),
#                                                 (0, 0)), mode='constant') for x in self.inputs[start:end]])
#         labels = list_to_sparse(self.labels[start:end])
#         return inputs, labels, sequence_length

class BatchGenerator(object):
    def __init__(self, data, batch_size=1):
        self.inputs, self.labels = data
        self.batch_size = batch_size
        self.data_length = len(self.inputs)
        self.sequence_length = pd.Series([x.shape[0] for x in self.inputs])
        # self.sequence_length = np.array([x.shape[0] for x in self.inputs])

    def next_batch(self):
        self._suffle()

        start = 0
        end = 0
        batch_size, data_length = self.batch_size, self.data_length
        while end != data_length:
            end += batch_size
            end = data_length if end >= data_length else end

            yield self._get(start, end)
            start = end

    def _suffle(self):
        permutation = np.random.permutation(self.data_length)
        self.inputs = self.inputs[permutation]
        self.labels = self.labels[permutation]
        self.sequence_length = self.sequence_length[permutation]

    def _get(self, start, end):
        sequence_length = self.sequence_length[start:end]
        batch_sequence_length = np.max(sequence_length)
        inputs = np.array([np.pad(x, pad_width=((0, batch_sequence_length - len(x)),
                                                (0, 0)), mode='constant') for x in self.inputs[start:end]])

        labels = np.array([np.pad(x, pad_width=(0, batch_sequence_length - len(x)), mode='constant') for x in self.labels[start:end]])

        sequence_weights = np.array([np.pad(np.ones(x.shape), pad_width=(0, batch_sequence_length - len(x)), mode='constant') for x in self.labels[start:end]])

        # labels = list_to_sparse(self.labels[start:end])
        # labels = list_to_sparse(self.labels[start:end])

        return inputs, labels, sequence_length, sequence_weights
