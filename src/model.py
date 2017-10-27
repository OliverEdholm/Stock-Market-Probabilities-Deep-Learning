# imports
import logging
from six.moves import xrange

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


# setup
logging.basicConfig(level=logging.DEBUG)


# classes
class Model:
    def __init__(self,
                 learning_rate,
                 input_n_datapoints,
                 rnn_layer_size,
                 n_rnn_layers,
                 fcn_layer_size,
                 n_fcn_layers,
                 learning_rate_decay_factor,
                 use_lstm=True):
        self.learning_rate = learning_rate
        self.input_n_datapoints = input_n_datapoints
        self.rnn_layer_size = rnn_layer_size
        self.n_rnn_layers = n_rnn_layers
        self.fcn_layer_size = fcn_layer_size
        self.n_fcn_layers = n_fcn_layers
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.use_lstm = use_lstm

        logging.debug('building learning rate op')
        self.learning_rate = tf.Variable(float(self.learning_rate),
                                         trainable=False)
        self.learning_rate_decay_operation = self.learning_rate.assign(
            self.learning_rate * self.learning_rate_decay_factor)

        logging.debug('building placeholders')
        self.keep_prob = tf.placeholder('float32')

        self.X = tf.placeholder('float32', [None, self.input_n_datapoints, 1])
        self.X_ = tf.unstack(tf.layers.batch_normalization(self.X), axis=1)

        self.y = tf.placeholder('float32', [None, 3])

        self.y_ = self._get_model()

        logging.debug('building accuracy vars')
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.y_, 1),
                                    tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        logging.debug('building loss')
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                        logits=self.y_)
            )
            tf.summary.scalar('cross_entropy', self.loss)

        logging.debug('building optimizer')
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.training_operation = optimizer.minimize(self.loss)

        logging.debug('building merged summaries')
        self.merged = tf.summary.merge_all()

    def _get_model(self):
        logging.debug('building model')
        logging.debug('building rnn part')
        if self.use_lstm:
            rnn_cell = rnn.BasicLSTMCell
        else:
            rnn_cell = rnn.GRUCell

        if self.n_rnn_layers > 1:
            rnn_cell = rnn.MultiRNNCell(
                [rnn_cell(self.rnn_layer_size)
                 for _ in xrange(self.n_rnn_layers)]
            )

        rnn_outputs, _ = rnn.static_rnn(rnn_cell, self.X_,
                                        dtype=tf.float32)

        logging.debug('building fcn part')
        fcn_layers = [tf.nn.dropout(layers.fully_connected(rnn_outputs[-1],
                                    self.fcn_layer_size), self.keep_prob)]
        for idx in xrange(self.n_fcn_layers):
            if idx == self.n_fcn_layers - 1:
                fcn_layer = layers.fully_connected(fcn_layers[-1],
                                                   3)
            else:
                fcn_layer = layers.fully_connected(fcn_layers[-1],
                                                   self.fcn_layer_size)

            fcn_layers.append(fcn_layer)

        for idx, fcn_layer in enumerate(fcn_layers):
            tf.summary.histogram('fcn_{}'.format(idx + 1), fcn_layer)

        output = fcn_layers[-1]

        return output

    def step(self, session, X_batch, y_batch, dropout=1.0,
             get_outputs=False, get_summary=False):
        input_feed = [self.training_operation, self.loss]

        if get_outputs:
            input_feed.append(self.y_)

        if get_summary:
            input_feed.append(self.merged)

        X_inputs = {self.X: X_batch}
        y_inputs = {self.y: y_batch}
        output_feed = session.run(input_feed,
                                  feed_dict={**X_inputs, **y_inputs, 
                                             self.keep_prob: dropout})

        return output_feed[1:]
