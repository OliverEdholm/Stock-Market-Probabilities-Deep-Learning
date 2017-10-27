# imports
import logging
from datetime import datetime
from datetime import timedelta
from six.moves import xrange
from six.moves import cPickle
from copy import deepcopy
from random import randint

import numpy as np
import pandas as pd


# classes
class Date:
    def __init__(self, date_string):
        self.date = self._date_string_to_datetime(date_string)

    def _date_string_to_datetime(self, date_string):
        year, month, day = [int(n) for n in date_string.split('-')]

        return datetime(year=year, month=month, day=day)

    def _get_date_num(self, num):
        num_str = str(num)

        if len(num_str) == 1:
            num_str = '0' + num_str

        return num_str

    def add_n_days(self, n_days):
        self.date = self.date + timedelta(days=n_days)

    def is_older(self, date2):
        date1 = self.date
        date2 = date2.date

        if date1.year > date2.year:
            return True
        elif date1.month > date2.month:
            return True
        elif date1.day > date2.day:
            return True

        return False

    def get_string_date(self):
        year_str = self._get_date_num(self.date.year)
        month_str = self._get_date_num(self.date.month)
        day_str = self._get_date_num(self.date.day)

        return '-'.join([year_str, month_str, day_str])


class DataSet:
    def __init__(self, path, n_days_per_datapoint, input_n_datapoints,
                 test_data_percentage, over_percentage, under_percentage):
        self.n_days_per_datapoint = n_days_per_datapoint
        self.input_n_datapoints = input_n_datapoints
        self.test_data_percentage = test_data_percentage
        self.over_percentage = over_percentage 
        self.under_percentage = under_percentage 

        self.data = self._get_data(path)

        prices = self._get_prices(self.data)

        X, y = self._get_model_inputs(prices)

        out = self._distribute_data(X, y)

        self.train_X = out[0]
        self.train_y = out[1]
        self.test_X = out[2]
        self.test_y = out[3]

    def _get_data(self, path):
        logging.debug('getting data from {}'.format(path))
        df = pd.read_csv(path)

        return df

    def _get_prices(self, data):
        date_strings = list(data['Date'])

        start_date = Date(date_strings[0])
        end_date = Date(date_strings[-1])

        prices = []
        cur_date = deepcopy(start_date)
        while end_date.is_older(cur_date):
            price = data.loc[data['Date'] ==
                             cur_date.get_string_date()]['Close'].real

            if len(price):
                prices.append(price[0])
            else:
                prices.append(None)

            cur_date.add_n_days(self.n_days_per_datapoint)

        return prices

    def _get_model_inputs(self, prices):
        X = []
        y = []

        for idx in xrange(len(prices) - self.input_n_datapoints):
            out = prices[idx + self.input_n_datapoints]

            if out is not None:
                inp = prices[idx: idx + self.input_n_datapoints]
                inp_prices = [-1 if price is None else price
                              for price in inp]

                latest = inp_prices[-1]
                if latest == -1:
                    continue

                X.append(np.array(inp_prices))

                if out > latest * self.over_percentage:
                    out_vector = [1, 0, 0]
                elif out < latest * self.under_percentage:
                    out_vector = [0, 1, 0]
                else:
                    out_vector = [0, 0, 1]
                out_vector = np.array(out_vector)

                y.append(out_vector)

        return X, y

    def _indices_to_values(self, indices, encoder_inputs, decoder_inputs):
        new_encoder_inputs = []
        new_decoder_inputs = []

        for idx in indices:
            new_encoder_inputs.append(encoder_inputs[idx])
            new_decoder_inputs.append(decoder_inputs[idx])

        return new_encoder_inputs, new_decoder_inputs 

    def _distribute_data(self, inputs, outputs, random=False):
        test_size = round(len(inputs) * self.test_data_percentage)
        if random:
            train_indices = [idx
                             for idx in xrange(len(inputs))]

            test_size = round(len(inputs) * self.test_data_percentage)

            test_indices = []
            for _ in xrange(test_size):
                idx = randint(0, len(train_indices) - 1)

                test_indices.append(train_indices[idx])
                train_indices.pop(idx)

            train_inputs, train_outputs = self._indices_to_values(
                train_indices, inputs, outputs)
            test_inputs, test_outputs = self._indices_to_values(
                test_indices, inputs, outputs)
        else:
            train_inputs = inputs[:-test_size]            
            train_outputs = outputs[:-test_size]            

            test_inputs = inputs[-test_size:]
            test_outputs = outputs[-test_size:]

        return train_inputs, train_outputs, \
            test_inputs, test_outputs

    def save(self, file_path):
        logging.info('saving dataset at {}'.format(file_path))
        with open(file_path, 'wb') as pkl_file:
            pkl_file.flush()

            cPickle.dump(self, pkl_file)

    @staticmethod
    def load(file_path):
        logging.info('loading dataset at {}'.format(file_path))
        with open(file_path, 'rb') as pkl_file:
            return cPickle.load(pkl_file)

    def get_random_batch(self, batch_size, is_test_data=False):
        if not is_test_data:
            X = self.train_X
            y = self.train_y
        else:
            X = self.test_X
            y = self.test_y

        batch_X = []
        batch_y = []
        for _ in xrange(batch_size):
            data_idx = randint(0, len(X) - 1)

            inp = X[data_idx]
            inp = [[n] for n in inp]
            batch_X.append(inp)

            batch_y.append(y[data_idx])

        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)

        return batch_X, batch_y
