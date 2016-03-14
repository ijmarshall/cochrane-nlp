# coding: utf-8

# # Multitask Learning
# 
# Use a single shared representation to predict targets of interest

import os
from collections import OrderedDict
import plac
import numpy as np
from sklearn.cross_validation import KFold

import keras
from keras.models import model_from_json
from keras.utils.layer_utils import model_summary

from support import produce_labels, ValidationCallback
from data_loader import DataLoader

import logging
from logging import info as info

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ModelTrainer(DataLoader):
    def load_model(self, model_fname, weights_fname=None):
        """Load a keras model from disk

        Parameters
        ----------
        model_fname : the name of the model file on disk to load
        weights_fname : the name of the weights file on disk to load

        """
        assert os.path.isfile(model_fname)

        info('Loading model {}...'.format(model_fname))

        with open(model_fname, 'r') as f:
            self.model = model_from_json(f.read())

        model_summary(self.model)

        if not weights_fname:
            return

        assert os.path.isfile(weights_fname)

        self.model.load_weights(weights_fname)

        info('Done!')

    def do_train_val_split(self, n_folds=5):
        """Split the data up into train and validation"""

        info('Doing {}-fold train-validation split...'.format(n_folds))

        # Train Test Split
        fold = KFold(len(self.abstracts_padded), n_folds=n_folds, shuffle=True)
        p = iter(fold)
        train_idxs, val_idxs = next(p)
        self.num_train, self.num_val = len(train_idxs), len(val_idxs)

        # Extract training data to pass to keras fit()
        self.train_data = OrderedDict(produce_labels(self.label_names,
                                                     self.class_sizes,
                                                     self.ys[:, train_idxs]))
        self.train_data.update({'input': self.abstracts_padded[train_idxs]})

        # Extract validation data to validate over
        self.val_data = OrderedDict(produce_labels(self.label_names,
                                                   self.class_sizes,
                                                   self.ys[:, val_idxs]))
        self.val_data.update({'input': self.abstracts_padded[val_idxs]})

        info('Done!')

    def train(self, nb_epoch, batch_size, val_every):
        """Train the model for a fixed number of epochs

        Save the weights after every epoch

        """
        info('Begin training for {} epochs...'.format(nb_epoch))

        val_callback = ValidationCallback(self.val_data, batch_size, self.num_train, val_every)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights.hd5', verbose=2)
        
        history = self.model.fit(self.train_data, batch_size=batch_size,
                                 nb_epoch=nb_epoch, verbose=2,
                                 callbacks=[checkpointer, val_callback])

        info('Done!')

@plac.annotations(
        model=('keras model file', 'option', None, str),
        weights=('weights file', 'option', None, str),
        labels=('labels to predict', 'option'),
        nb_epoch=('number of epochs', 'option', None, int),
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int)
)
def main(model, weights, labels='gender,phase_1', nb_epoch=5, batch_size=128, val_every=1):
    labels = labels.split(',')

    mt = ModelTrainer()

    mt.load_embeddings()
    mt.load_labels(labels)
    mt.load_model(model, weights)

    mt.do_train_val_split()
    mt.train(nb_epoch, batch_size, val_every)

if __name__ == '__main__':
    plac.call(main)
