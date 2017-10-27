# imports
from config import DEFAULT_DATA_SET_PATH
from config import DEFAULT_DATA_PATH
from config import DEFAULT_N_DAYS_PER_DATAPOINT
from config import DEFAULT_INPUT_N_DATAPOINTS
from config import DEFAULT_TEST_DATA_PERCENTAGE
from config import DEFAULT_OVER_PERCENTAGE
from config import DEFAULT_UNDER_PERCENTAGE

import sys
sys.path.append('src')
from data import DataSet

import os
import logging

import tensorflow as tf

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
flags = tf.app.flags
flags.DEFINE_string('data_set_path', DEFAULT_DATA_SET_PATH,
                    'Path to save dataset on.')
flags.DEFINE_string('data_path', DEFAULT_DATA_PATH,
                    'Path to data csv file.')
flags.DEFINE_integer('n_days_per_datapoint', DEFAULT_N_DAYS_PER_DATAPOINT,
                     'Number of days between each datapoint.')
flags.DEFINE_integer('input_n_datapoints', DEFAULT_INPUT_N_DATAPOINTS,
                     'Number of datapoints that the model gets inputed with.')
flags.DEFINE_float('test_data_percentage', DEFAULT_TEST_DATA_PERCENTAGE,
                   'Percentage of data to put aside for evaluation.')
flags.DEFINE_float('over_percentage', DEFAULT_OVER_PERCENTAGE,
                   'Percentage that price must exceed to be the output of '
                   '"over".')
flags.DEFINE_float('under_percentage', DEFAULT_UNDER_PERCENTAGE,
                   'Percentage that price must be lower than to be the '
                   'output if "under".')
FLAGS = flags.FLAGS


# functions
def main():
    if os.path.exists(FLAGS.data_set_path):
        ans = input('A file at {} already exists, do you wish to remove it?'
                    ' y / n: '.format(FLAGS.data_set_path)).lower()[0]
        if ans == 'y':
            logging.info('removing file at {}'.format(FLAGS.data_set_path))
            os.remove(FLAGS.data_set_path)

    data = DataSet(FLAGS.data_path,
                   FLAGS.n_days_per_datapoint,
                   FLAGS.input_n_datapoints,
                   FLAGS.test_data_percentage,
                   FLAGS.over_percentage,
                   FLAGS.under_percentage)

    logging.info('amount of training data: {}'.format(
        len(data.train_X)))
    logging.info('amount of test data: {}'.format(len(data.test_X)))

    n_over = 0
    n_under = 0
    n_same = 0
    for v in data.train_y:
        if v[0] == 1:
            n_over += 1
        elif v[1] == 1:
            n_under += 1
        else:
            n_same += 1

    s = n_over + n_under + n_same
    print('percentile of over: {}'.format(n_over / s))
    print('percentile of under: {}'.format(n_under / s))
    print('percentile of same: {}'.format(n_same / s))

    data.save(FLAGS.data_set_path)


if __name__ == '__main__':
    main()
