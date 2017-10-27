# imports
from config import DEFAULT_BATCH_SIZE
from config import DEFAULT_DATA_SET_PATH
from config import DEFAULT_LEARNING_RATE
from config import DEFAULT_LEARNING_RATE_DECAY_FACTOR
from config import DEFAULT_RNN_LAYER_SIZE
from config import DEFAULT_N_RNN_LAYERS
from config import DEFAULT_FCN_LAYER_SIZE
from config import DEFAULT_N_FCN_LAYERS
from config import DEFAULT_INPUT_N_DATAPOINTS
from config import DEFAULT_BREAK_STEP_INTERVAL
from config import DEFAULT_N_BATCHES_TO_ACCURACY_EVAL
from config import DEFAULT_TENSORBOARD_PATH
from config import DEFAULT_CHECKPOINTS_PATH
from config import DEFAULT_SAVE_STEP_INTERVAL
from config import DEFAULT_DROPOUT

import sys
sys.path.append('src')
from model import Model
from data import DataSet

import os
import logging
from shutil import rmtree
from six.moves import xrange

import numpy as np
import tensorflow as tf

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_float('learning_rate', DEFAULT_LEARNING_RATE,
                   'Learning rate when training with AdamOptimizer.')
flags.DEFINE_float('dropout', DEFAULT_DROPOUT,
                   'Dropout when training.')
flags.DEFINE_float('learning_rate_decay_factor',
                   DEFAULT_LEARNING_RATE_DECAY_FACTOR,
                   'What factor learning rate should be multiplied with when '
                   'loss isn\'t decreasing.')
flags.DEFINE_integer('rnn_layer_size', DEFAULT_RNN_LAYER_SIZE,
                     'Size of RNN layers.')
flags.DEFINE_integer('n_rnn_layers', DEFAULT_N_RNN_LAYERS,
                     'Number of RNN layers in the network.')
flags.DEFINE_integer('fcn_layer_size', DEFAULT_FCN_LAYER_SIZE,
                     'Size of normal fully connected layers.')
flags.DEFINE_integer('n_fcn_layers', DEFAULT_N_FCN_LAYERS,
                     'Number of normal fully connected layers.')
flags.DEFINE_integer('input_n_datapoints', DEFAULT_INPUT_N_DATAPOINTS,
                     'Number of datapoints to have as input.')
flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE,
                     'Batch size to use in model.')
flags.DEFINE_integer('break_step_interval', DEFAULT_BREAK_STEP_INTERVAL,
                     'At what step interval to update learning rate and '
                     'tensorboard.')
flags.DEFINE_integer('save_step_interval', DEFAULT_SAVE_STEP_INTERVAL,
                     'At what step interval to save the model.')
flags.DEFINE_integer('n_batches_to_accuracy_eval',
                     DEFAULT_N_BATCHES_TO_ACCURACY_EVAL,
                     'How many batches should be used for accuracy eval.')
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to dataset.')
flags.DEFINE_string('tensorboard_path', DEFAULT_TENSORBOARD_PATH,
                    'Path to tensorboard.')
flags.DEFINE_string('checkpoints_path', DEFAULT_CHECKPOINTS_PATH,
                    'Path to checkpoints.')
FLAGS = flags.FLAGS


# functions
def main():
    if os.path.exists(FLAGS.checkpoints_path):
        ans = input('A folder at {} already exists, do you wish to remove it?'
                    ' y / n: '.format(FLAGS.checkpoints_path)).lower()[0]
        if ans == 'y':
            logging.info('removing folder at {}'.format(
                FLAGS.checkpoints_path))
            rmtree(FLAGS.checkpoints_path)

    os.makedirs(FLAGS.checkpoints_path)

    if os.path.exists(FLAGS.tensorboard_path):
        logging.info('removing folder at {}'.format(FLAGS.tensorboard_path))
        rmtree(FLAGS.tensorboard_path)

    data_set = DataSet.load(FLAGS.data_set_path)

    model = Model(FLAGS.learning_rate,
                  FLAGS.input_n_datapoints,
                  FLAGS.rnn_layer_size,
                  FLAGS.n_rnn_layers,
                  FLAGS.fcn_layer_size,
                  FLAGS.n_fcn_layers,
                  FLAGS.learning_rate_decay_factor,
                  use_lstm=True)

    logging.debug('initializing')
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        train_writer = tf.summary.FileWriter(FLAGS.tensorboard_path,
                                             sess.graph)

        n_params = np.sum([np.prod(v.get_shape().as_list())
                           for v in tf.trainable_variables()])
        logging.debug('training {} parameters'.format(n_params))
        step = 1
        avg_losses = []
        losses = []
        while True:
            X_batch, y_batch = data_set.get_random_batch(
                FLAGS.batch_size)

            loss, summary = model.step(sess, X_batch, y_batch, 
                                       dropout=FLAGS.dropout,
                                       get_summary=True)
            losses.append(loss)
            step += 1

            if step % FLAGS.break_step_interval == 0:
                avg_loss = np.mean(losses)

                if len(avg_losses) > 2 and avg_loss > max(avg_losses[-3:]):
                    sess.run(model.learning_rate_decay_operation)
                    logging.info('decreased learning rate, it is now:'
                                 ' {}'.format(sess.run(model.learning_rate)))

                avg_losses.append(avg_loss)

                accuracies = []
                for _ in xrange(FLAGS.n_batches_to_accuracy_eval):
                    X_batch, y_batch = data_set.get_random_batch(
                        FLAGS.batch_size, is_test_data=True)

                    X_inputs = {model.X: X_batch}
                    y_inputs = {model.y: y_batch}
                    accuracy = sess.run(model.accuracy, feed_dict={**X_inputs,
                                                                   **y_inputs,
                                                                   model.keep_prob: 1.0})
                    accuracies.append(accuracy)

                avg_accuracy = np.mean(accuracies)

                logging.info('step: {}, loss: {}, accuracy: {}'.format(
                    step, avg_loss, avg_accuracy))
                losses = []
                train_writer.add_summary(summary, step)

            if step % FLAGS.save_step_interval == 0:
                logging.info('saving checkpoint of model')
                saver.save(sess, '{}/model'.format(FLAGS.checkpoints_path))


if __name__ == '__main__':
    main()
