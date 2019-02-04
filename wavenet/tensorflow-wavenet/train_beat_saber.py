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
    for inputs, labels, sequence_length, sequence_weights in gen.next_batch():
        # indices = np.array([labels.row, labels.col]).T
        # values = np.array(labels.data)
        # dense_shape = labels.shape
        # _labels = tf.SparseTensorValue(indices, values, dense_shape)

        sess.run([model.acc_op, model.accuracy], feed_dict={
            model.inputs: inputs,
            model.labels: labels,
            model.sequence_length: np.array(sequence_length),
            model.sequence_weights: np.array(sequence_weights)
        })

        decoded, _loss, _ler = sess.run([model.decoded, model.loss, model.accuracy], feed_dict={
            model.inputs: inputs,
            model.labels: labels,
            model.sequence_length: np.array(sequence_length),
            model.sequence_weights: np.array(sequence_weights)
        })

        #_ler = edit_distance(decoded[0], _labels)

        ler += _ler * len(inputs)
        loss += _loss * len(inputs)

    return ler / len(x), loss / len(x)


def train(model, x, y, epochs, batch_size=1, validation_data=None):
    train_op = model.optimizer.minimize(model.loss)
    train_gen = BatchGenerator((x, y), batch_size=batch_size)

    steps_per_epoch = np.ceil(
        train_gen.data_length / batch_size).astype(np.int32)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(1, epochs + 1):
            train_gen.__init__((x,y),batch_size)
            prog_bar = utils.Progbar(steps_per_epoch)
            print('Epoch {}/{}'.format(epoch, epochs))
            for step, (inputs, labels, sequence_length, sequence_weights) in enumerate(train_gen.next_batch(), 1):

                # indices = np.array([labels.row, labels.col]).T
                # values = np.array([labels.data])
                # dense_shape = labels.shape

                _, loss = sess.run([train_op, model.loss], feed_dict={
                    model.inputs: inputs,
                    model.labels: labels,
                    # model.labels: tf.SparseTensorValue(indices, values, dense_shape),
                    model.sequence_length: np.array(sequence_length),
                    model.sequence_weights: np.array(sequence_weights)
                })

                if steps_per_epoch == step:
                    update_values = [('loss', loss)]
                    if validation_data is not None:
                        val_ler, val_loss = eval(
                            sess, model, *validation_data)
                        update_values += [('val_loss', val_loss),
                                          ('val_accuracy', val_ler)]

                    prog_bar.update(step, values=update_values)#, force=True)
                else:
                    prog_bar.update(step, values=[('loss', loss)])

        saver.save(sess, './my_test_model',global_step=1000)





import pickle
features_list = pickle.load(open("../speech-to-text-wavenet/features_list.p","rb"), encoding='latin1')
levels_list = pickle.load(open("../speech-to-text-wavenet/levels_list.p","rb"), encoding='latin1')

max([np.max(level[:,0]) for level in levels_list])
max([np.max(level[:,1]) for level in levels_list])
max([np.max(level[:,2]) for level in levels_list])
max([np.max(level[:,3]) for level in levels_list])

levels_list = [((level[:,0]*4+level[:,1])*3+level[:,2])*4+level[:,3] for level in levels_list]

num_classes = 9*4*3*4

num_features = 20

# features_list[0].shape

residual_channels=16
dilation_channels=32
skip_channels=16
initial_kernel_size=2
kernel_size=2
num_residual_blocks=3
num_dilation_layers=5
raw_waveform=False
causal=True
valid=2
batch_size=16
epoch=250

features_list = [features.T for features in features_list]

# foo=np.random.permutation(len(features_list))
# features_list[foo]
import pandas as pd
# features_list = pd.Series(features_list)
# levels_list = pd.Series(levels_list)

# np.array([x.shape[0] for x in features_list])[0:1][0]

# bar[foo]

model = WaveNet(num_features,
                num_classes,
                residual_channels,
                dilation_channels,
                skip_channels,
                initial_kernel_size=initial_kernel_size,
                kernel_size=kernel_size,
                num_residual_blocks=num_residual_blocks,
                batch_size=batch_size,
                num_dilation_layers=num_dilation_layers,
                downsampling=raw_waveform,
                causal=causal)

x_train = pd.Series(features_list[:-100])
y_train = pd.Series(levels_list[:-100])
validation_data = (pd.Series(features_list[-100:]), pd.Series(levels_list[-100:]))
train(model, x_train, y_train, epoch,
      batch_size=batch_size, validation_data=validation_data)


import pickle
test_features_list = pickle.load(open("../speech-to-text-wavenet/test_features_list.p","rb"), encoding='latin1')
test_features_list = [features.T for features in test_features_list]
with tf.Session() as sess:
    # sess.run(tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    output = graph.get_tensor_by_name("output:0")
    inputs = graph.get_tensor_by_name("placeholder/inputs:0")
    test_output= sess.run(output, feed_dict={
        inputs: test_features_list[0:],
    })

import pickle
# pickle.dump(test_output[0],open("test_output.p","wb"))
test_output = pickle.load(open("test_output.p","rb"))

len(test_output)

sum(test_output==0)

notes = [{"_time":float(i/16.0), "_cutDirection":int(x//(4*3*4)-1), "_lineIndex":int(x%(4*3*4)), "_lineLayer":int(x%(3*4)), "_type":int(x%(4))} for i,x in enumerate(test_output) if x != 0]

len(notes)

song_json = {u'_beatsPerBar': 16,
 u'_beatsPerMinute': 125,
 u'_events': [],
 u'_noteJumpSpeed': 10,
 u'_notes': notes,
 u'_obstacles': [],
 u'_shuffle': 0,
 u'_shufflePeriod': 0.25,
 u'_version': u'1.5.0'}

import json

with open("Easy.json", "w") as f:
    f.write(json.dumps(song_json))
