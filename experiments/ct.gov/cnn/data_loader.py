# coding: utf-8

# ### Load Embeddings and Abstracts

# In[1]:

import pickle

import numpy as np

from support import classinfo_generator

import logging
from logging import info as info

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DataLoader:
    """Class to load in abstracts and labels
    
    The motivation for this class is that both the training code and the model
    building code need access to both information about the abstracts and the
    labels. So each of those components subclass this class to get that
    functionality.
    
    """
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
        info('Loading embeddings...')

        embeddings_info = pickle.load(open('embeddings_info.p', 'rb'))

        self.abstracts = embeddings_info['abstracts']
        self.abstracts_padded = embeddings_info['abstracts_padded']
        self.embeddings = embeddings_info['embeddings']
        self.word_dim = embeddings_info['word_dim']
        self.word2idx, idx2word = embeddings_info['word2idx'], embeddings_info['idx2word']
        self.maxlen = embeddings_info['maxlen']
        self.vocab_size = embeddings_info['vocab_size']

        info('Done!')

    def load_labels(self, label_names):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        info('Loading labels {}...'.format(label_names))

        # Dataframes of labels
        pruned_dataset = pickle.load(open('pruned_dataset.p', 'rb'))
        binarized_dataset = pickle.load(open('binarized_dataset.p', 'rb'))

        # Only consider subset of labels passed in
        pruned_dataset = pruned_dataset[label_names]
        binarized_dataset = binarized_dataset[label_names]

        self.ys = np.array(binarized_dataset).T # turn labels into numpy array

        # Get class names and sizes
        class_info = list(classinfo_generator(pruned_dataset))
        class_names, self.class_sizes = zip(*class_info)
        class_names = {label: classes for label, classes in zip(label_names, class_names)}

        self.label_names = label_names

        info('Done!')
