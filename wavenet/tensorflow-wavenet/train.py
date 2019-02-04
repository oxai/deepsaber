import argparse
import numpy as np
import tensorflow as tf
from utils.data_utils import BatchGenerator
from utils.data_utils import list_to_sparse
from utils.data_utils import merge_and_split
from utils.reader import AudioReader
from utils.reader import KaldiReader
from utils.reader import LabelReader
from utils.timit import Timit
from wavenet.models import WaveNet
from tensorflow.contrib.keras import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='CTC-based automatic speech recognition')
    parser.add_argument(
        '--train_feat', default='./data/processed/train.39.cmvn.scp')
    parser.add_argument(
        '--train_label', default='./data/material/train.lbl')
    parser.add_argument('--residual_channels', type=int, default=16)
    parser.add_argument('--dilation_channels', type=int, default=32)
    parser.add_argument('--skip_channels', type=int, default=16)
    parser.add_argument('--initial_kernel_size', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--num_residual_blocks', type=int, default=3)
    parser.add_argument('--num_dilation_layers', type=int, default=5)
    parser.add_argument('--raw_waveform', action='store_true')
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--valid', type=str, nargs=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=50)

    return parser.parse_args()


def sparse_phone_mapping(st, fn=Timit.convert_61_to_39):
    """Map sparse phone tensor row by row according to fn.
    """

    ret = [[] for _ in range(st.dense_shape[0])]
    start, end, mark = 0, 0, 0

    for i, v in zip(st.indices, st.values):
        ret[i[0]].append(v)

    ret = [fn(seq) for seq in ret]

    ret = list_to_sparse(ret)

    return tf.SparseTensorValue(np.array([ret.row, ret.col]).T,
                                np.array(ret.data),
                                ret.shape)


def edit_distance(_hypothesis, _truth):
    """Calculate the edit distance on Timit dataset,
       where phones should be converted from 61 phones
       set to 39 one first.
    """

    graph = tf.Graph()
    with graph.as_default():
        hypothesis = tf.sparse_placeholder(tf.int32)
        truth = tf.sparse_placeholder(tf.int32)
        ed = tf.reduce_mean(tf.edit_distance(hypothesis, truth))

    with tf.Session(graph=graph) as sess:
        ret = sess.run(ed, feed_dict={
            hypothesis: sparse_phone_mapping(_hypothesis),
            truth: sparse_phone_mapping(_truth)
        })

    return ret


def eval(sess, model, x, y, batch_size=32):
    gen = BatchGenerator((x, y), batch_size=batch_size)
    ler, loss = 0, 0
    for inputs, labels, sequence_length in gen.next_batch():
        indices = np.array([labels.row, labels.col]).T
        values = np.array(labels.data)
        dense_shape = labels.shape
        _labels = tf.SparseTensorValue(indices, values, dense_shape)

        decoded, _loss = sess.run([model.decoded, model.loss], feed_dict={
            model.inputs: inputs,
            model.labels: _labels,
            model.sequence_length: sequence_length
        })

        _ler = edit_distance(decoded[0], _labels)

        ler += _ler * len(inputs)
        loss += _loss * len(inputs)

    return ler / len(x), loss / len(x)


def train(model, x, y, epochs, batch_size=1, validation_data=None):
    train_op = model.optimizer.minimize(model.loss)
    train_gen = BatchGenerator((x, y), batch_size=batch_size)

    steps_per_epoch = np.ceil(
        train_gen.data_length / batch_size).astype(np.int32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            prog_bar = utils.Progbar(steps_per_epoch)
            print('Epoch {}/{}'.format(epoch, epochs))
            for step, (inputs, labels, sequence_length) in enumerate(train_gen.next_batch(), 1):

                indices = np.array([labels.row, labels.col]).T
                values = np.array(labels.data)
                dense_shape = labels.shape

                _, loss = sess.run([train_op, model.loss], feed_dict={
                    model.inputs: inputs,
                    model.labels: tf.SparseTensorValue(indices, values, dense_shape),
                    model.sequence_length: sequence_length
                })

                if steps_per_epoch == step:
                    update_values = [('loss', loss)]
                    if validation_data is not None:
                        val_ler, val_loss = eval(
                            sess, model, *validation_data)
                        update_values += [('val_loss', val_loss),
                                          ('val_ler', val_ler)]

                    prog_bar.update(step, values=update_values, force=True)
                else:
                    prog_bar.update(step, values=[('loss', loss)])


def main(args):
    if args.raw_waveform:
        # input is raw waveform
        feat, num_features = AudioReader.read(args.train_feat)
    else:
        # input is kaldi format
        feat, num_features = KaldiReader.read(args.train_feat)

    label, num_labels, label_map = LabelReader.read(
        args.train_label, label_map=None)

    x_train, y_train = merge_and_split(feat, label)

    validation_data = None
    if args.valid is not None:
        if args.raw_waveform:
            valid_feat, num_valid_features = AudioReader.read(args.valid[0])
        else:
            valid_feat, num_valid_features = KaldiReader.read(args.valid[0])

        assert num_features == num_valid_features

        valid_label, _, _ = LabelReader.read(
            args.valid[1], label_map=label_map)
        validation_data = merge_and_split(valid_feat, valid_label)

    num_classes = num_labels + 1
    model = WaveNet(num_features,
                    num_classes,
                    args.residual_channels,
                    args.dilation_channels,
                    args.skip_channels,
                    initial_kernel_size=args.initial_kernel_size,
                    kernel_size=args.kernel_size,
                    num_residual_blocks=args.num_residual_blocks,
                    num_dilation_layers=args.num_dilation_layers,
                    downsampling=args.raw_waveform,
                    causal=args.causal)

    train(model, x_train, y_train, args.epoch,
          batch_size=args.batch_size, validation_data=validation_data)


if __name__ == '__main__':
    args = parse_args()
    main(args)
