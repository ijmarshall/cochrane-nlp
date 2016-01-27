import os
import operator
import re
from collections import namedtuple
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from support import word_cloud, sprint, duplicate_words

import sklearn
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

# Verbosity levels
VERBOSITY_MAXIMAL = 3


def extract_target(abstracts_targets, target):
    """Filter away all columns not relevant to predicting `target`

    abstracts_targets : dataframe consisting of abstracts, pmid, and targets for
    prediction
    target : ct.gov field of interest for prediction

    Additionally filter away abstracts without a label for `target`

    """
    df = abstracts_targets.ix[:, ['abstract', 'pmid', target]]
    df = df[df[target].notnull()] # filter away abstracts with no label
    
    return df

def filter_sparse_classes(df, target, verbose, threshold=30):
    """Filters away classes which we have less than `threshold` number of
    examples for.
    
    Parameters
    ----------
    df : dataframe returned from extract_targets()
    target : ct.gov field of interest for prediction
    verbose : print distribution of classes if True
    
    """
    sizes = df.groupby(target).size()

    filtered_df = df[df[target].isin(sizes[sizes >= threshold].index)]

    if verbose:
        sprint('Class Breakdown')
        print filtered_df.groupby(target).size()

    return filtered_df

def view_class_examples(df, target):
    """Prints an example abstract for each class of `target`

    df : dataframe returned from filter_sparse_classes()
    target : ct.gov field of interest for prediction

    """
    labels = df[target].unique()

    indexes = [df[df[target] == label].iloc[0].name for label in labels]

    for index in indexes:
        pm_url = 'https://www.google.com/search?q=pmid+' + df.iloc[index].pmid + '&btnI=I' # I'm Feeling Lucky

        print '*'*5, df.iloc[index][target], '*'*5
        print df.iloc[index].abstract

def word_cloud_classes(df, target):
    """Dispaly a word cloud for each class in `target`

    df : dataframe returned from filter_sparse_classes()
    target : ct.gov field of interest for prediction

    """
    labels = df[target].unique()

    fig = plt.figure(figsize=(12, 2*len(labels)))
    plt.clf()

    for i, label in enumerate(np.sort(labels), start=1):
        axes = fig.add_subplot(int(np.ceil(len(labels)/2.)), 2, i)
        words = ' '.join(df[df[target] == label].abstract)

        word_cloud(words, axes, label)

    fig.suptitle('Most Common Words per Class')
    plt.axis('off')
    plt.show()

def train_test_split(df, target):
    """Returns a train/test split for abstracts when predicting `target`

    df : dataframe returned from filter_sparse_classes()
    target : ct.gov field of interest for prediction

    """
    return sklearn.cross_validation.train_test_split(df.abstract, df[target])

def vectorize(abstracts_train, vect_type):
    """Vectorizes abstracts

    abstracts_train : abstracts for training returned from train_test_split()

    A TfidfVectorizer is used with use_idf=False in order to make the change to
    using a HashingVectorizer in the future simple

    """
    assert vect_type == 'hashing' or vect_type == 'tfidf'

    if vect_type == 'hashing':
        vectorizer = HashingVectorizer(ngram_range=(1, 2), stop_words='english', binary=True)
    elif vect_type == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', use_idf=False, binary=True)

    vectorizer.fit(abstracts_train)

    return vectorizer.transform(abstracts_train), vectorizer

def get_vocabulary(vectorizer):
    """Extract and order vocabulary from `vectorizer`

    vectorizer : vectorizer returned from vectorize()

    This function only needs to be called when we are interested in doing model
    introspection (e.g. plotting word clouds and finding the most important
    words

    """
    return [word for word, index in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]

def do_grid_search(X_train, ys_train, verbosity=False, k=5, num_alphas=10):
    """Do a grid search over regularization term

    X_train : training set examples
    ys_train : training set labels
    k : number of folds to use in cross-validation
    num_alphas : number of alphas to search over
    verbosity : plot f1 and all scores for each hyperparameter setting if maximal

    Macro f1 scores for each setting of the regularization term are also
    plotted.

    """
    verbosity = 3 if verbosity else 0

    M, N = X_train.shape

    sprint('Grid Search')
    clf = SGDClassifier(class_weight='auto', n_iter=int(np.ceil(10**6/(M-M/k)))) # http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use

    parameters = {
        'alpha': np.logspace(-1, -4, num_alphas)
    }

    grid_search = GridSearchCV(clf, parameters, verbose=verbosity, scoring='f1_macro', cv=k)
    grid_search.fit(X_train, ys_train)

    # Get scores for different hyperparam settings into dataframe
    df = pd.DataFrame(grid_search.grid_scores_, columns=grid_search.grid_scores_[0]._fields)

    # Explode cv scores for each hyperparam setting
    scores = df.cv_validation_scores.apply(pd.Series)
    scores = scores.rename(columns=lambda x: 's{}'.format(x))
    score_columns = scores.columns

    scores['f1'], scores['err'] = scores.mean(axis=1), scores.std(axis=1) # mean f1 and stddev for cv scores

    alphas = df.parameters.apply(lambda x: pd.Series(x))

    df = pd.concat([alphas, scores], axis=1).fillna(0) # concatenate the two back together

    # Plot f1 and all the scores for each hyperparam setting?
    if verbosity == VERBOSITY_MAXIMAL:
        axes = df['f1'].plot(yerr=df.err, linewidth=.5)
        for s in score_columns:
            axes = df[s].plot(ax=axes, style='.', c='black')

        # Fix axes
        tick_marks = np.arange(len(alphas))
        plt.xticks(tick_marks, df.alpha.round(4), rotation=90)
        axes.set_xlabel('alpha')
        axes.set_ylabel('macro f1')

        plt.show()
    
    return grid_search

def predict(clf, df, target, vectorizer, verbosity):
    """Predict test labels

    clf : classifier used in prediction
    df : dataframe with test examples
    target : ct.gov field of interest for prediction
    vectorizer : vectorizer used to vectorize train abstract examples
    verbosity : print performance numbers if True

    """
    abstracts_test, ys_test = df.abstract, df[target]
    X_test = vectorizer.transform(abstracts_test)
    predictions = clf.predict(X_test)
    
    # Compute f1s for all classes
    lb = sklearn.preprocessing.LabelBinarizer()
    f1s = sklearn.metrics.f1_score(lb.fit_transform(ys_test), lb.fit_transform(predictions), average=None)
    f1 = np.mean(f1s)

    if verbosity == VERBOSITY_MAXIMAL:
        # Display f1s
        Classes = namedtuple('Classes', [re.sub('[^0-9a-zA-Z]+', '_', class_) for class_ in clf.classes_])

        sprint('Performance')
        print 'f1s: {}'.format({label: f1 for label, f1 in zip(clf.classes_, f1s)})
        print

    if verbosity:
        print 'Test f1: {}'.format(f1)
    
    return predictions, f1

def print_confusion_matrix(ys_test, predictions, clf):
    """Print confusion matrix

    ys_test : test abstract labels
    predictions : test abstract label predictions
    clf : classifer used to make predictions

    """
    confusion_matrix = sklearn.metrics.confusion_matrix(ys_test, predictions)

    fig = plt.figure()
    plt.clf()

    labels = clf.classes_

    plt.imshow(confusion_matrix, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    width = height = len(labels)

    for x in xrange(width):
        for y in xrange(height):
            plt.annotate(str(confusion_matrix[x][y]) if confusion_matrix[x][y] else '', xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

def most_important_features(clf, vocabulary, verbose=False):
    """Display word clouds of most important features

    clf : trained classifier
    vocabulary : list of words in same order as clf features
    verbose : return if False

    """
    if not verbose:
        return

    fig = plt.figure(figsize=(20, 20))
    plt.clf()

    labels = clf.classes_
    for i, (weights, title) in enumerate(zip(clf.coef_, labels), start=1):
        pairs = sorted(zip(weights, vocabulary), reverse=True)[:10]
        pairs = [(pair[0], re.sub('\s+', '_', pair[1])) for pair in pairs]

        duped_words = list(duplicate_words(pairs))

        axes = fig.add_subplot(int(np.ceil(len(labels)/2.)), 2, i)
        word_cloud(' '.join(duped_words), axes, title)

    fig.suptitle('Most Important Words per Class')

    plt.axis('off')
    plt.show()

def best_clf_cv(df, target, show_wc, vect_type, verbosity):
    """Yield the best classifier found via cross-fold validation
    
    df : dataframe with train examples
    target : ct.gov field of interest for prediction
    vect_type : use Hashing vectorizer if 'hashing' (else use TfidfVectorizer)
    show_wc : display word clouds if True (vect_type must be tfidf)
    verbosity : 0: print nothing; 1: print only average f1; 5+: display alpha search

    1. View class examples (optional)
    2. Word cloud classes (optional)
    3. Train/test split
    4. Vectorize training set
    5. Extract ordered vocabulary (for model introspection - optional)
    6. Grid search over regularization terms
    7. Return best clf found

    Also yields the vectorizer for the training data.

    """
    print
    if show_wc: word_cloud_classes(df, target)
    abstracts_train, ys_train = df.abstract, df[target]
    X_train, vectorizer = vectorize(abstracts_train, vect_type)
    if vect_type == 'tfidf': vocabulary = get_vocabulary(vectorizer)
    grid_search = do_grid_search(X_train, ys_train, verbosity, k=3, num_alphas=5)

    return grid_search.best_estimator_, vectorizer

def do_pipeline(abstracts_targets, target,
        vect_type='hashing', show_wc=False, verbosity=0, do_pickle=False):
    """Execute ct.gov fixed-class prediction pipeline

    abstracts_targets : dataframe consisting of abstracts, pmid, and targets for
    prediction
    target : ct.gov field of interest for prediction
    vect_type : use Hashing vectorizer if 'hashing' (else use TfidfVectorizer)
    show_wc : display word clouds if True (vect_type must be tfidf)
    verbosity : 0: print nothing; 1: print only average f1; 5+: display alpha search
    do_pickle : pickle vectorizer and model if True

    1. Extract targets
    2. Filter away sparse classes
        - This box is necessary because sklearn.metrics.f1_score() complains
          when there's a class in the prediction set that's not in the true_y
          set.  Additionally having a small number of classes hurts overall
          performance due to the fact macro_f1 scoring is currently being used.
    
    For each fold (train, test) split of the original data:
        - Get best clf from cross-fold validation on the training data
        - Make predictions on test set and evaluate performance
        - Print confusion matrix (optional)
        - Display most important features (optional)
        - Pickle vectorizer and model (optional)

    Yields the best clf found during cross-validation and its performance on the
    test set. This is done for each fold in the dataset.

    """
    assert not (vect_type == 'hashing' and show_wc)

    df = extract_target(abstracts_targets, target)
    df = filter_sparse_classes(df, target, verbosity)
    # view_class_examples(df, 'intervention_model')

    for train, test in KFold(n=len(df), n_folds=3):
        best_clf, vectorizer = best_clf_cv(df.iloc[train], target, show_wc, vect_type, verbosity) # do cross validation on training set
        predictions, test_f1 = predict(best_clf, df.iloc[test], target, vectorizer, verbosity) # predict on test

        if verbosity == VERBOSITY_MAXIMAL: print_confusion_matrix(ys_test, predictions, best_clf)
        if show_wc: most_important_features(best_clf, vocabulary)
        # if do_pickle: pickle_model(best_clf, target)

        yield best_clf, test_f1
