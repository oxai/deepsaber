import tensorflow as tf


def temporal_padding(inputs, paddings):
    paddings = [[0, 0], [paddings[0], paddings[1]], [0, 0]]

    return tf.pad(inputs, paddings)


def conv1d(inputs,
           filters,
           kernel_size,
           strides=1,
           padding='same',
           data_format='channels_last',
           dilation_rate=1,
           activation=None,
           use_bias=False,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           bias_initializer=tf.zeros_initializer(),
           name=None):
    """A general wrapper of tf.layers.conv1d() supporting 
       1. 'causal' padding method used for WaveNet.
       2. batch normalization when use_bias is False for accuracy.
    """

    if padding == 'causal':
        left_pad = dilation_rate * (kernel_size - 1)
        inputs = temporal_padding(inputs, (left_pad, 0))
        padding = 'valid'

    outputs = tf.layers.conv1d(inputs,
                               filters,
                               kernel_size,
                               strides=strides,
                               padding=padding,
                               data_format=data_format,
                               dilation_rate=dilation_rate,
                               activation=activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer)

    if not use_bias:
        axis = -1 if data_format == 'channels_last' else 1
        outputs = tf.layers.batch_normalization(outputs,
                                                axis=axis,
                                                name='{}-batch_normalization'.format(name))

    return outputs


def dilation_layer(inputs,
                   residual_channels,
                   dilation_channels,
                   skip_channels,
                   kernel_size,
                   dilation_rate,
                   data_format='channels_last',
                   causal=True,
                   use_bias=False,
                   name=None):
    """Implementation of dilation layer used for WaveNet.
    """

    conv_filter = conv1d(inputs,
                         dilation_channels,
                         kernel_size,
                         data_format=data_format,
                         dilation_rate=dilation_rate,
                         activation=tf.nn.tanh,
                         padding='causal' if causal else 'same',
                         use_bias=use_bias,
                         name='{}-filter'.format(name))

    conv_gate = conv1d(inputs,
                       dilation_channels,
                       kernel_size,
                       data_format=data_format,
                       dilation_rate=dilation_rate,
                       activation=tf.nn.sigmoid,
                       padding='causal' if causal else 'same',
                       use_bias=use_bias,
                       name='{}-gate'.format(name))

    outputs = conv_filter * conv_gate
    skip_connections = conv1d(outputs,
                              skip_channels,
                              1,
                              padding='same',
                              use_bias=use_bias,
                              name='{}-1x1-conv-skip'.format(name))

    transformed_outputs = conv1d(outputs,
                                 residual_channels,
                                 1,
                                 padding='same',
                                 use_bias=use_bias,
                                 name='{}-1x1-conv-transform'.format(name))

    return transformed_outputs + inputs, skip_connections
