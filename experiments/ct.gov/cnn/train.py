# coding: utf-8

# # Multitask Learning
# 
# Use a single shared representation to predict gender and phase 2

# ### Load Embeddings and Abstracts

# In[1]:

import pickle

import numpy as np

import keras
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from support import classinfo_generator, produce_labels, AccuracyCallback

class Model:
    def load_embeddings(self):
        """Load word embeddings and training examples"""

        embeddings_info = pickle.load(open('embeddings_info.p', 'rb'))

        self.abstracts = embeddings_info['abstracts']
        self.abstracts_padded = embeddings_info['abstracts_padded']
        self.embeddings = embeddings_info['embeddings']
        self.word_dim = embeddings_info['word_dim']
        self.word2idx, idx2word = embeddings_info['word2idx'], embeddings_info['idx2word']
        self.maxlen = embeddings_info['maxlen']
        self.vocab_size = embeddings_info['vocab_size']

    def load_labels(self):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        # Dataframes of labels
        pruned_dataset = pickle.load(open('pruned_dataset.p', 'rb'))
        binarized_dataset = pickle.load(open('binarized_dataset.p', 'rb'))

        ys = np.array(binarized_dataset).T # turn labels into numpy array
        self.labels = pruned_dataset.columns.tolist() # extract label names

        # Get class names and sizes
        class_info = list(classinfo_generator(pruned_dataset))
        class_names, self.class_sizes = zip(*class_info)
        class_names = {label: classes for label, classes in zip(self.labels, class_names)}

        # Extract training data to pass to keras fit()
        self.data = dict(produce_labels(self.labels, self.class_sizes, ys))
        self.data.update({'input': self.abstracts_padded})

        # Extract validation data
        self.val_dict = {label: y_row for label, y_row in zip(self.labels, ys)}
        self.val_dict.update({'input': self.abstracts_padded})

    def build_model(self):
        """Build keras model
        
        Current architecture is:
            embedding -> conv -> pool -> fc -> fork
            
        """
        nb_filter = 20
        filter_length = 2
        hidden_dims = 32
        nb_epoch = 35
        batch_size = 10

        model = Graph()
        model.add_input(name='input', input_shape=[self.maxlen], dtype='int') # dtype='int' is 100% necessary for some reason!
        model.add_node(Embedding(input_dim=self.vocab_size, output_dim=self.word_dim,
            weights=[self.embeddings],
            input_length=self.maxlen,
            trainable=False),
                    name='embedding', input='input')

        model.add_node(Dropout(0.25), name='dropout1', input='embedding')

        model.add_node(Convolution1D(nb_filter=nb_filter,
                                    filter_length=filter_length,
                                    activation='relu'),
                    name='conv',
                    input='dropout1')
        model.add_node(MaxPooling1D(pool_length=self.maxlen-1), name='pool', input='conv') # non-maximum suppression
        model.add_node(Flatten(), name='flat', input='pool')
        model.add_node(Dense(hidden_dims), name='z', input='flat')
        model.add_node(Activation('relu'), name='shared', input='z')
        model.add_node(Dropout(0.25), name='dropout2', input='shared')

        # Fork the graph and predict probabilities for each target from shared representation
        for label, num_classes in zip(self.labels, self.class_sizes):
            model.add_node(Dense(output_dim=num_classes, activation='softmax'), name='{}_probs'.format(label), input='dropout2')
            model.add_output(name=label, input='{}_probs'.format(label))

        model.compile(optimizer='rmsprop',
                    loss={label: 'categorical_crossentropy' for label in self.labels}) # CE for all the targets

        self.model = model

    def train(self):
        """Train the model for a fixed number of epochs

        Save the weights after every epoch

        """
        batch_size = 6
        num_epochs = 10

        callback = AccuracyCallback(self.data, self.val_dict, batch_size, num_train=len(self.abstracts_padded), val_every=2)
        checkpointer = keras.callbacks.ModelCheckpoint(filepath="weights.hdf5", verbose=1)

        self.model.fit(self.data, validation_data=self.data,
                batch_size=batch_size, callbacks=[callback, checkpointer], nb_epoch=10)

if __name__ == '__main__':
    model = Model()

    model.load_embeddings()
    model.load_labels()
    model.build_model()
    model.train()
