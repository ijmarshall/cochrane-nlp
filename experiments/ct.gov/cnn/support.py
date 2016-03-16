import sys
import operator

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import keras
from keras.utils.np_utils import to_categorical

import sklearn


class TestCallback(keras.callbacks.Callback):
    def __init__(self):
        super(TestCallback, self).__init__()

    def on_batch_end(self, batch, logs={}):
        print len(self.model.validation_data)
        print self.model.output_order

class ValidationCallback(keras.callbacks.Callback):
    """Callback to compute accuracy during training"""

    def __init__(self, val_data, batch_size, num_train, val_every):
        """Callback to compute accuracy during training
        
        Parameters
        ----------
        val_data : dict containing input and labels
        batch_size : number of examples per batch
        num_train : number of examples in training set
        val_every : number of times to to validation during an epoch

        Also save the validation accuracies when you compute them.
        
        """
        super(ValidationCallback, self).__init__()

        self.val_data = val_data
        self.num_batches_since_val = 0
        num_minis_per_epoch = (num_train/batch_size) # number of minibatches per epoch
        self.K = num_minis_per_epoch / val_every # number of batches to go before doing validation
        
    def on_batch_end(self, batch, logs={}):
        """Do validation if it's been a while
        
        Concretely print out fscores for each class val_every number of times
        per epoch.
        
        """
        return # don't do anything here right now!

        # Hasn't been long enough since your last validation run?
        if self.num_batches_since_val < self.K-1:
            self.num_batches_since_val += 1
            return
            
        loss = self.model.evaluate(self.val_data)
        print 'val loss:', loss

        predictions = self.model.predict(self.val_data)
        
        for label, ys_pred in predictions.items():
            # f1 score
            ys_val = self.val_data[label].argmax(axis=1) # convert categorical to label-based
            f1 = sklearn.metrics.f1_score(ys_val,
                                          ys_pred.argmax(axis=1),
                                          average='macro')

            print '{} f1:'.format(label), f1

        self.num_batches_since_val = 0

    def on_epoch_end(self, epoch, logs={}):
        """Evaluate validation loss and f1
        
        Compute macro f1 score (unweighted average across all classes)
        
        """
        # loss
        loss = self.model.evaluate(self.val_data)
        print 'val loss:', loss

        # f1
        predictions = self.model.predict(self.val_data)
        for label, ys_pred in predictions.items():
            ys_val = self.val_data[label].argmax(axis=1) # convert categorical to indexes
            f1 = sklearn.metrics.f1_score(ys_val,
                                          ys_pred.argmax(axis=1),
                                          average='macro')

            print '{} f1:'.format(label), f1

        sys.stdout.flush() # try and flush stdout so condor prints it!

def plot_confusion_matrix(confusion_matrix, columns):
    df = pd.DataFrame(confusion_matrix, columns=columns, index=columns)
    axes = plt.gca()
    axes.imshow(df, interpolation='nearest')
    tick_marks = np.arange(len(columns))
    plt.xticks(tick_marks, df.index, rotation=90)
    plt.yticks(tick_marks, df.index)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    width = height = len(columns)

    for x in xrange(width):
        for y in xrange(height):
            axes.annotate(str(confusion_matrix[x][y]) if confusion_matrix[x][y] else '', xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

def classinfo_generator(df):
    """Generates tuples of class sizes and class names
    
    Parameters
    ----------
    df : dataframe which contains the training labels
    
    """
    for column in df.columns:
        categories = df[column].cat.categories
        yield categories, len(categories)

def produce_labels(label_names, class_sizes, ys):
    """Generates dict of label_names for a minibatch for each objective
    
    Parameters
    ----------
    label_names : list of label names (order must correspond to label_names in ys)
    class_sizes : list of class sizes
    ys : labels
    
    Will produce a dict like:
    
    {gender: 2darray, phase: 2darray, ..., masking: 2darray}
    
    where 2darray has one-hot label_names for every row.
    
    """
    num_objectives, num_train = ys.shape
    
    for label, num_classes, y_row in zip(label_names, class_sizes, ys):        
        ys_block = to_categorical(y_row)

        # Take into account missing label_names!
        missing_data = np.argwhere(y_row == -1).flatten()
        ys_block[missing_data] = 0
        
        yield (label, ys_block)

def examples_generator(dataset, target='gender', num_examples=None):
    """Generate indexes into dataset to pull out examples of classes
    
    Generate n examples for each class where n is the number of examples for the class
    we have the fewest examples for.
    
    """
    labels = dataset[target].unique()
    
    if not num_examples:
        num_class_examples = dataset.groupby('gender').size()
        num_examples = min(num_class_examples) # only get a number of examples such that we have perfect class balance
    
    for label in labels:
        for idx, entry in dataset[dataset[target] == label][:num_examples].iterrows():
            yield idx

    dataset = dataset.loc[list(examples_generator(dataset, num_examples=50))]

    dataset.groupby('gender').size()
