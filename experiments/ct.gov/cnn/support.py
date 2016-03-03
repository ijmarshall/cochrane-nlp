import operator

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

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

def produce_labels(labels, class_sizes, ys):
    """Generates dict of labels for a minibatch for each objective
    
    Parameters
    ----------
    labels : list of label names (order must correspond to labels in ys)
    class_sizes : list of class sizes
    ys : labels
    
    Will produce a dict like:
    
    {gender: 2darray, phase: 2darray, ..., masking: 2darray}
    
    where 2darray has one-hot labels for every row.
    
    """
    num_objectives, batch_size = ys.shape
    
    for label, num_classes, y_row in zip(labels, class_sizes, ys):        
        ys_block = np.zeros([batch_size, num_classes])
        ys_block[np.arange(batch_size), y_row] = 1
        
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
