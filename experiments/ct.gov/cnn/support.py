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

    def __init__(self, val_data, batch_size, num_train, val_every, val_weights,
            f1_weights, save_weights):
        """Callback to compute f1 during training
        
        Parameters
        ----------
        val_data : dict containing input and labels
        batch_size : number of examples per batch
        num_train : number of examples in training set
        val_every : number of times to to validation during an epoch
        f1_weights : location to save model weights

        Also save model weights whenever a new best f1 is reached.
        
        """
        super(ValidationCallback, self).__init__()

        self.val_data = val_data
        self.num_batches_since_val = 0
        num_minis_per_epoch = (num_train/batch_size) # number of minibatches per epoch
        self.K = num_minis_per_epoch / val_every # number of batches to go before doing validation
        self.best_f1 = 0
        self.f1_weights = f1_weights
        self.val_weights = val_weights
        self.save_weights = save_weights
        
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
            # f1 score
            ys_val = self.val_data[label]

            # Rows that have *no* label have all zeros. Get rid of them!
            valid_idxs = ys_val.any(axis=1)
            if not np.any(valid_idxs):
                continue # masked out label - go onto the next

            f1 = sklearn.metrics.f1_score(ys_val[valid_idxs].argmax(axis=1),
                                          ys_pred[valid_idxs].argmax(axis=1),
                                          average=None)

            print '{} f1: {}'.format(label, list(f1))
            sys.stdout.flush() # try and flush stdout so condor prints it!

            if not self.save_weights:
                continue

            macro_f1 = np.mean(f1)
            if macro_f1 > self.best_f1:
                self.best_f1 = macro_f1 # update new best f1
                self.model.save_weights(self.f1_weights, overwrite=True) # save model weights!

        if not self.save_weights:
            return

        # Save val weights no matter what!
        self.model.save_weights(self.val_weights, overwrite=True)

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

def produce_labels(label_names, ys, class_sizes):
    """Generates dict of label_names for a minibatch for each objective
    
    Parameters
    ----------
    label_names : list of label names (order must correspond to label_names in ys)
    class_sizes : number of classes in each label
    ys : labels
    
    Will produce a dict like:
    
    {gender: 2darray, phase: 2darray, ..., masking: 2darray}
    
    where 2darray has one-hot label_names for every row.
    
    """
    num_objectives, num_train = ys.shape
    
    for label, y_row, class_size in zip(label_names, ys, class_sizes):
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
