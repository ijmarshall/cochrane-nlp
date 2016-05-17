import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

import wordcloud


def word_cloud(words, axes, title):
    wc = wordcloud.WordCloud().generate(words)
    axes.imshow(wc)
    axes.axis('off')
    plt.title(title)

def sprint(message):
    """Helper function for printing eye catching messages"""

    print '*'*5, message, '*'*5

def duplicate_words(pairs):
    """Helper function which yields a number of duplicated words proportional to
    their corresponding coefficients"""

    for coef, word in pairs:
        for _ in range(int(coef*100)):
            yield word

def unpack_generator(d):
    """Transform the following dict:
    
    {c1: [(clf11, s11), (clf12, s12), ..., (clf1k, s1k)],
     c2: [(clf21, s21), (clf22, s22), ..., (clf2k, s2k)]
     .
     .
     .
     cc: [(clfc1, sc1), (clfc2, sc2), ..., (clf1k, sck)]}
     
    to:
     
    {c1: [s11, s12, ..., s1k],
     c2: [s21, s22, ..., s2k]
     .
     .
     .
     cc: [sc1, sc2, ..., sck]}
     
    to make putting into a dataframe easier. We're just purging the clfs.
    
    """
    for target, pairs in d.items():
        yield target, [f1 for _, f1 in pairs]

def plot_and_pickle(evaluations):
    """Plot f1 scores in a bar chart and show std for each target

    evaluations : dict of the form

        {c1: [(clf11, s11), (clf12, s12), ..., (clf1k, s1k)],
         c2: [(clf21, s21), (clf22, s22), ..., (clf2k, s2k)]
         .
         .
         .
         cc: [(clfc1, sc1), (clfc2, sc2), ..., (clf1k, sck)]}

    where ck is the k'th target and sk is the k'th score on the test set from
    cross-validation.

    """
    # clfs, f1s = zip(*pairs)

    # Plot
    evals = dict(unpack_generator(evaluations))
    df = pd.DataFrame(evals)
    df.mean().plot(kind='bar', yerr=df.std())

    for target, pairs in evaluations.items():
        # Make directory for this target
        folder = '{}/{}'.format('models', target)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Pickle the f1 scores with the models
        with open('{}/{}_clf.p'.format(folder, target), 'wb') as f:
            pickle.dump(dict(pairs), f)

        # for i, clf in enumerate(clfs):
        #     with open('{}/{}_clf{}.p'.format(folder, target, i), 'wb') as f:
        #         pickle.dump(clf, f)
