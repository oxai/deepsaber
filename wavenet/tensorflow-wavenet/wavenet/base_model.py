import tensorflow as tf


class BaseModel(object):
    """Base model.
    """

    def __init__(self, num_features, num_classes,batch_size):
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size


class CTCModel(BaseModel):
    """Tensorflow model with connectionist temporal classification (CTC) loss.
    """

    def __init__(self, num_features, num_classes,batch_size):
        super(CTCModel, self).__init__(num_features, num_classes,batch_size)
        self._build_graph()

    def _build_graph(self):
        """Build the computational gragh.
        """

        self._creat_placeholders()
        logits = self._create_logits()
        self._create_ctc_loss(logits)
        self._create_optimizer()
        self._create_eval(logits)

    def _creat_placeholders(self):
        """Create placeholders including inputs, labels and sequence length.
        """

        with tf.variable_scope('placeholder'):
            self._inputs = tf.placeholder(
                tf.float32, shape=[None, None, self.num_features], name='inputs')

            # self._labels = tf.sparse_placeholder(tf.int32, name='labels')
            self._labels = tf.placeholder(tf.int32, shape=[None,None], name='labels')

            self._sequence_length = tf.placeholder(
                tf.int32, shape=[None, ], name='sequence_length')
            self._sequence_weights = tf.placeholder(
                tf.float32, shape=[None, None], name='sequence_weights')

    def _create_logits(self):
        """Create logits according to inputs.
        """

        raise NotImplementedError(
            'This is the abstract method. Subclasses should implement this.')

    def _create_ctc_loss(self, logits):
        """Create CTC loss between labels and logits.
        """

        # with tf.variable_scope('loss'):
        #     self._loss = tf.reduce_mean(tf.nn.ctc_loss(
        #         self.labels, logits, self.sequence_length, ctc_merge_repeated=False))
        with tf.variable_scope('loss'):
            self._loss = tf.contrib.seq2seq.sequence_loss(
                targets=self.labels, logits=logits, weights=self.sequence_weights)

    def _create_optimizer(self):
        """Create adam optimizer.
        """

        with tf.variable_scope('optimizer'):
            self._optimizer = tf.train.AdamOptimizer(name='optimizer')

    def _create_eval(self, logits):
        """Create evaluation terms including decoded outputs and
           edit distance between decoded outputs and labels.
        """

        self.decoded = tf.transpose(tf.argmax(logits,2), name="output")
        with tf.variable_scope('eval'):
            # self.decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            #     logits, self.sequence_length)
            # self.decoded, neg_sum_logits = logits, logits
            self._accuracy, self._acc_op = tf.metrics.accuracy(labels=self.labels, predictions=tf.transpose(tf.argmax(logits,2)), weights=self.sequence_weights)
            # self.edit_distance = tf.reduce_mean(tf.edit_distance(
            #     tf.cast(self.decoded[0], tf.int32), self.labels))

    @property
    def inputs(self):
        return self._inputs

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def acc_op(self):
        return self._acc_op

    @property
    def labels(self):
        return self._labels

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def sequence_weights(self):
        return self._sequence_weights

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer
