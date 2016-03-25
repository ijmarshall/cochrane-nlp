# coding: utf-8

# # Multitask Learning
# 
# Use a single shared representation to predict gender and phase 2

# ### Load Embeddings and Abstracts

# In[1]:

import os
import sys

from collections import OrderedDict

import plac
import pickle

import numpy as np

from sklearn.cross_validation import KFold

import keras

from keras.models import Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.layer_utils import model_summary
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from support import classinfo_generator, produce_labels, ValidationCallback

class Model:
    def load_embeddings(self):
        """Load word embeddings and abstracts
        
        embeddings_info dict
        --------------------
        abstracts: full-text abstracts
        abstracts_padded: abstracts indexed and padded
        embeddings: embedding matrix
        word_dim: dimension word embeddings
        word2idx: dictionary from word to embedding index
        idx2word: dictionary from embedding index to word
        maxlen: size of each padded abstract
        vocab_size: number of words in the vocabulary
            
        """
        embeddings_info = pickle.load(open('pickle/embeddings_info.p', 'rb'))

        self.abstracts = embeddings_info['abstracts']
        self.abstracts_padded = embeddings_info['abstracts_padded']
        self.embeddings = embeddings_info['embeddings']
        self.word_dim = embeddings_info['word_dim']
        self.word2idx, idx2word = embeddings_info['word2idx'], embeddings_info['idx2word']
        self.maxlen = embeddings_info['maxlen']
        self.vocab_size = embeddings_info['vocab_size']

    def load_labels(self, label_names, composite_labels):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        # Use composite labels or factored labels?
        if composite_labels:
            symbols_file, labels_file = 'composite_labels.p', 'composite_binarized.p'
        else:
            symbols_file, labels_file = 'pruned_dataset.p', 'binarized_dataset.p'

        pruned_dataset = pickle.load(open('pickle/{}'.format(symbols_file), 'rb'))
        binarized_dataset = pickle.load(open('pickle/{}'.format(labels_file), 'rb'))

        # Only consider subset of labels passed in
        pruned_dataset = pruned_dataset[label_names]
        binarized_dataset = binarized_dataset[label_names]

        self.ys = np.array(binarized_dataset).T # turn labels into numpy array

        # Get class names and sizes
        class_info = list(classinfo_generator(pruned_dataset))
        class_names, self.class_sizes = zip(*class_info)
        class_names = {label: classes for label, classes in zip(label_names, class_names)}

        self.label_names = label_names

    def do_train_val_split(self):
        """Split data up into separate train and validation sets

        Use sklearn's function.

        """
        fold = KFold(len(self.abstracts_padded),
                     n_folds=5,
                     shuffle=True,
                     random_state=0) # for reproducibility!
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

    def build_model(self, nb_filter, filter_len, hidden_dim,
            dropout_prob, dropout_emb, task_specific, reg, backprop_emb, word2vec_init):
        """Build keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
        dropouts = {}

        ### BEGIN LAYER NAMES #################################################
                                                                              #
        input = 'input'
        embedding = 'embedding'
        dropouts[embedding] = embedding + '_'
        conv = 'conv'
        pool = 'pool'
        flat = 'flat'
        shared = 'shared'
        dropouts[shared] = shared + '_'

        if task_specific:
            task_specifics = {label: '{}_rep'.format(label) for label in self.label_names}

            # Add dropout
            for label in self.label_names:
                specific_rep = task_specifics[label]
                dropouts[specific_rep] = specific_rep + '_'

        probs = {label: '{}_probs'.format(label) for label in self.label_names}
        outputs = self.label_names
                                                                              #
        ### END LAYER NAMES ###################################################
        

        ### BEGIN GRAPH CONSTRUCTION ##########################################
                                                                              #
        model = Graph()

        model.add_input(name=input,
                        input_shape=[self.maxlen],
                        dtype='int') # dtype='int' is 100% necessary for some reason!

        init_embeddings = [self.embeddings] if word2vec_init else None
        model.add_node(Embedding(input_dim=self.vocab_size, output_dim=self.word_dim,
                                 weights=init_embeddings,
                                 input_length=self.maxlen,
                                 trainable=backprop_emb),
                       name=embedding,
                       input=input)

        model.add_node(Dropout(dropout_emb), name=dropouts[embedding], input=embedding)
        model.add_node(Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_len,
                                     activation='relu',
                                     W_regularizer=l2(reg)),
                       name=conv,
                       input=dropouts[embedding])

        model.add_node(MaxPooling1D(pool_length=self.maxlen-(filter_len-1)), name=pool, input=conv)
        model.add_node(Flatten(), name=flat, input=pool)
        model.add_node(Dense(hidden_dim, activation='relu', W_regularizer=l2(reg)),
                       name=shared,
                       input=flat)
        model.add_node(Dropout(dropout_prob), name=dropouts[shared], input=shared)

        for label, num_classes in zip(self.label_names, self.class_sizes):
            # Fork the graph and predict probabilities for each target from shared representation

            if task_specific: 
                # Final dense hidden layer for task-specific representation

                specific_rep = task_specifics[label]

                model.add_node(Dense(hidden_dim, activation='relu', W_regularizer=l2(reg)),
                               name=specific_rep,
                               input=dropouts[shared])

                model.add_node(Dropout(dropout_prob),
                               name=dropouts[specific_rep],
                               input=specific_rep)

                model.add_node(Dense(output_dim=num_classes, activation='softmax', W_regularizer=l2(reg)),
                               name=probs[label],
                               input=dropouts[specific_rep])
            else:
                # Straight from shared representation to softmax

                model.add_node(Dense(output_dim=num_classes, activation='softmax', W_regularizer=l2(reg)),
                               name=probs[label],
                               input=dropouts[shared])

        for label in self.label_names:
            model.add_output(name=label, input=probs[label]) # separate output for each label

        model.compile(optimizer='rmsprop',
                    loss={label: 'categorical_crossentropy' for label in self.label_names}) # CE for all the targets

                                                                              #
        ### END GRAPH CONSTRUCTION ############################################

        model_summary(model)

        self.model = model

    def train(self, nb_epoch, batch_size, val_every, weights_loc, class_weight, composite_labels):
        """Train the model for a fixed number of epochs

        Set up callbacks first.

        """
        val_callback = ValidationCallback(self.val_data, batch_size, self.num_train, val_every)
        checkpointer = ModelCheckpoint(filepath=weights_loc, verbose=2)

        if class_weight:
            if composite_labels:
                class_weights_fname = 'composite_weights.p'
            else:
                class_weights_fname = 'class_weights.p'

            # Load class weights and filter down to only labels we are considering
            all_class_weights = pickle.load(open('pickle/{}'.format(class_weights_fname, 'rb')))
            class_weights = {label: weights for label, weights in all_class_weights.items() if label in self.label_names}
        else:
            class_weights = {} # no weighting

        history = self.model.fit(self.train_data, batch_size=batch_size,
                                 nb_epoch=nb_epoch, verbose=2,
                                 callbacks=[checkpointer, val_callback],
                                 class_weight=class_weights)


@plac.annotations(
        nb_epoch=('number of epochs', 'option', None, int),
        labels=('labels to predict', 'option'),
        task_specific=('whether to include an addition task-specific hidden layer', 'option', None, str),
        nb_filter=('number of filters', 'option', None, int),
        filter_len=('length of filter', 'option', None, int),
        hidden_dim=('size of hidden state', 'option', None, int),
        dropout_prob=('dropout probability', 'option', None, float),
        dropout_emb=('perform dropout after the embedding layer', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        backprop_emb=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int),
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        class_weight=('enfore class balance through loss scaling', 'option', None, str),
        word2vec_init=('initialize embeddings with word2vec', 'option', None, str),
        composite_labels=('use composite labels as opposed to factored ones', 'option', None, str)
)
def main(nb_epoch=5, labels='gender,phase_1', task_specific='False',
        nb_filter=128, filter_len=2, hidden_dim=128, dropout_prob=.5, dropout_emb='True',
        reg=0, backprop_emb='False', batch_size=128, val_every=1, exp_group='',
        class_weight='False', word2vec_init='True',
        composite_labels='False'):
    """Training process

    1. Load embeddings and labels
    2. Build the keras model and load weights files
    3. Train

    """
    labels = labels.split(',')

    # Convert boolean strings to actual booleans
    task_specific = True if task_specific == 'True' else False
    backprop_emb = True if backprop_emb == 'True' else False
    class_weight = True if class_weight == 'True' else False
    dropout_emb = dropout_prob if dropout_emb == 'True' else 1e-100
    word2vec_init = True if word2vec_init == 'True' else False
    composite_labels = True if composite_labels == 'True' else False

    weights_fname = '+'.join(arg.lstrip('-') for arg in sys.argv[1:])

    m = Model()
    m.load_embeddings()
    m.load_labels(labels, composite_labels)
    m.do_train_val_split()
    m.build_model(nb_filter, filter_len, hidden_dim, dropout_prob, dropout_emb,
                  task_specific, reg, backprop_emb, word2vec_init)

    weights_loc = 'weights/{}/{}'.format(exp_group, weights_fname)
    if os.path.isfile(weights_loc):
        m.model.load_weights(weights_loc)
    else:
        print >> sys.stderr, 'weights file {} not found!'.format(weights_loc)

    m.train(nb_epoch, batch_size, val_every, weights_loc, class_weight, composite_labels)


if __name__ == '__main__':
    plac.call(main)
