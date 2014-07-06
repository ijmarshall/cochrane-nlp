#
#   low memory modular vectorizer
#


#   does not require to build up a list of dicts,
#   which is prohibitively expensive in memory for
#   a large number of PDFs

#   also uses hashvectorizer, so does not maintain
#   a mapping

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from itertools import izip
import numpy as np
import scipy


class ModularVectorizer(object):

    def __init__(self, *args, **kwargs):
        self.vec = InteractionHashingVectorizer(*args, **kwargs)

    def builder_clear(self):
        self.X = None

    def _combine_matrices(self, X_part, weighting=1):
        X_part.data.fill(weighting)

        if self.X is None:
            self.X = X_part
        else:
            self.X = self.X + X_part
            # assuming we have no collisions, the interaction terms shouldn't be identical
            # if there are collisions, this is ok since they should form a tiny proportion
            # of the data (they will have values > weighting)

    def builder_add_docs(self, X_docs, weighting=1, interactions=None, prefix=None, low=None):
        X_part = self.vec.transform(X_docs, i_vec=interactions, i_term=prefix, low=low)
        self._combine_matrices(X_part, weighting=weighting)

    builder_add_interaction_features = builder_add_docs # identical fn here; but for compatability

    def builder_transform(self):
        return self.X

    builder_fit_transform = builder_transform # identical fn here; but for compatability


class InteractionHashingVectorizer(HashingVectorizer):
    """
    Same as HashingVectorizer,
    but with an option to add interaction prefixes to the
    tokenized words, and option to take a binary mask vector
    indicating which documents to add interactions for
    """
    def __init__(self, *args, **kwargs):

        # this subclass requires certain parameters - check these

        assert kwargs.get("analyzer", "word") == "word" # only word tokenization (default)
        assert kwargs.get("norm") is None # don't normalise words (i.e. counts only)
        assert kwargs.get("binary") == True 
        assert kwargs.get("non_negative") == True

        super(InteractionHashingVectorizer, self).__init__(*args, **kwargs)

    def build_analyzer(self, i_term=None):
        """Return a callable that handles preprocessing and tokenization"""

        preprocess = self.build_preprocessor()

        # only does word level analysis
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()

        return lambda doc: self._word_ngrams(
            tokenize(preprocess(self.decode(doc))), stop_words, i_term=i_term)


    def _word_ngrams(self, tokens, stop_words=None, i_term=None):
        """
        calls super of _word_ngrams, then adds interaction prefix onto each token
        """
        tokens = super(InteractionHashingVectorizer, self)._word_ngrams(tokens, stop_words)

        # import pdb; pdb.set_trace()

        if i_term:
            return [i_term + token for token in tokens]
        else:
            return tokens

    def _iter_interact_docs(self, X, i_vec):
        for doc, interacts in izip(X, i_vec):
            if interacts:
                yield doc
            else:
                yield ""


    def _limit_features(self, csr_matrix, low=2, high=None, limit=None):
        """
        Lower bound on features, so that > n docs much contain the feature
        """
        
        assert isinstance(csr_matrix, scipy.sparse.csr_matrix) # won't work with other sparse matrices
        # (most can be converted with .tocsr() method)

        indices_to_remove = np.where(np.asarray(csr_matrix.sum(axis=0) < low)[0])[0]
        # csr_matrix.sum(axis=0) < low: returns Boolean matrix where total features nums < low
        # np.asarray: converts np.matrix to np.array
        # [0]: since the array of interest is the first (and only) item in an outer array
        # np.where: to go from True/False to indices of Trues

        
        data_filter = np.in1d(csr_matrix.indices, indices_to_remove)
        # gets boolean array, where the columns of any non-zero values are to be removed
        # (i.e. their index is in the indices_to_remove array)

        # following three lines for info/debugging purposes
        # to show how many unique features are being removed
        num_total_features = len(np.unique(csr_matrix.indices)) 
        num_features_to_remove = np.sum(np.in1d(indices_to_remove, np.unique(csr_matrix.indices)))
        print "%d/%d features will be removed" % (num_features_to_remove, num_total_features)

        csr_matrix.data[data_filter] = 0
        # set the values to be removed to 0 to start with

        csr_matrix.eliminate_zeros()
        # then run the np optimised routine to delete those 0's (and free a little memory)
        # NB zeros are superfluous since a sparse matrix

        return csr_matrix

    def transform(self, X_s, y=None, i_vec=None, i_term=None, high=None, low=None, limit=None):
        """
        same as HashingVectorizer transform, except allows for interaction list
        which is an iterable the same length as X filled with True/False
        this method adds an empty row to docs labelled as False
        """
        analyzer = self.build_analyzer(i_term=i_term)

        if i_vec is None:
            X = self._get_hasher().transform(analyzer(doc) for doc in X_s)
        else:
            X = self._get_hasher().transform(analyzer(doc) for doc in self._iter_interact_docs(X_s, i_vec))
        
        X.data.fill(1)

        # import pdb; pdb.set_trace()

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        if low:
            X = self._limit_features(X, low=low)
        return X

    # Alias transform to fit_transform for convenience
    # (repeated here or else is alised to the parent class transform)
    fit_transform = transform




