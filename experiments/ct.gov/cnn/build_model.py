# coding: utf-8

# # Multitask Learning
# 
# Use a single shared representation to predict gender and phase 2

# ### Load Embeddings and Abstracts

# In[1]:

import plac

import numpy as np

import keras

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.layer_utils import model_summary

from support import classinfo_generator, produce_labels, ValidationCallback
from data_loader import DataLoader

class ModelBuilder(DataLoader):
    """Class that builds a model
    
    This class will build a keras model and save it to disk, which a
    ModelTrainer can then load and train.
    
    """
    def build_model(self, nb_filter, filter_len, hidden_dim):
        """Build keras model

        Check to see if one already exists on disk. If so, use that one instead.
        
        Current architecture is embedding -> conv -> pool -> fc -> fork.
            
        """
        model = Graph()

        # Input Layer
        model.add_input(name='input',
                        input_shape=[self.maxlen],
                        dtype='int') # dtype='int' is 100% necessary for some reason!

        # Embedding Layer with dropout
        model.add_node(Embedding(input_dim=self.vocab_size, output_dim=self.word_dim,
                                 weights=[self.embeddings],
                                 input_length=self.maxlen,
                                 trainable=False),
                       name='embedding', input='input')
        model.add_node(Dropout(0.25), name='dropout1', input='embedding')

        # Convolution layer
        model.add_node(Convolution1D(nb_filter=nb_filter,
                                     filter_length=filter_len,
                                     activation='relu'),
                       name='conv',
                       input='dropout1')

        # Non-maximum Supression
        model.add_node(MaxPooling1D(pool_length=self.maxlen-1), name='pool', input='conv')

        # Flatten Layer
        model.add_node(Flatten(), name='flat', input='pool')

        # Dense Layer
        model.add_node(Dense(hidden_dim), name='z', input='flat')
        model.add_node(Activation('relu'), name='shared', input='z')
        model.add_node(Dropout(0.25), name='dropout2', input='shared')

        # Fork the graph and predict probabilities for each target from shared representation
        for label_name, num_classes in zip(self.label_names, self.class_sizes):
            model.add_node(Dense(output_dim=num_classes, activation='softmax'),
                           name='{}_probs'.format(label_name),
                           input='dropout2')
            model.add_output(name=label_name, input='{}_probs'.format(label_name)) # separate output for each label

        model.compile(optimizer='rmsprop', # CE for all the targets
                    loss={label_name: 'categorical_crossentropy' for label_name in self.label_names})

        model_summary(model)

        self.model = model

    def save_model(self):
        """Save model to disk"""

        json_string = self.model.to_json()
        fname = '{}.json'.format('+'.join(self.label_names))
        open(fname, 'w').write(json_string)

        print 'Wrote {} to disk!'.format(fname)

@plac.annotations(
        nb_filter=('number of filters', 'option', None, int),
        filter_len=('length of filter', 'option', None, int),
        hidden_dim=('size of hidden state', 'option', None, int),
        labels=('labels to predict', 'option'),
)
def main(nb_filter=128, filter_len=2, hidden_dim=128, labels='gender,phase_1'):

    labels = labels.split(',')

    mb = ModelBuilder()

    mb.load_embeddings()
    mb.load_labels(labels)

    mb.build_model(nb_filter, filter_len, hidden_dim)
    mb.save_model()

if __name__ == '__main__':
    plac.call(main)
