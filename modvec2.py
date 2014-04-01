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



class ModularVectorizer(object):

    def builder_clear(self):
        self.X = None

    def _combine_matrices(self, X_part):
        if self.X is None:
            self.X = X_part
        else:
            self.X = self.X + X_part

    def builder_add_docs(self, X_docs, prefix = None, weighting=1):
        vec = InteractionHashingVectorizer(interaction_prefix=prefix)
        X_part = vec.transform(X_docs)
        self._combine_matrices(X_part)

    def builder_add_interaction_features(self, X_docs, interactions, prefix=None):
        vec = InteractionHashingVectorizer(interaction_prefix=prefix)
        X_part = vec.transform(X_docs, i=interactions)
        self._combine_matrices(X_part)

    def builder_transform(self):
        return self.X

    builder_fit_transform = builder_transform


class InteractionHashingVectorizer(HashingVectorizer):
    """
    Same as HashingVectorizer,
    but with an option to add interaction prefixes to the
    tokenized words, and option to take a binary mask vector
    indicating which documents to add interactions for
    """
    def __init__(self, *args, **kwargs):

        self.interaction_prefix = kwargs.pop("interaction_prefix", None)
        # remove from the kwargs before calling super.__init__

        assert kwargs.get("analyzer", "word") == "word"
        # this subclass only does word tokenization

        super(InteractionHashingVectorizer, self).__init__(*args, **kwargs)


    def _word_ngrams(self, tokens, stop_words=None):
        """
        calls super of _word_ngrams, then adds interaction prefix onto each token
        """
        tokens = super(InteractionHashingVectorizer, self)._word_ngrams(tokens, stop_words)

        if self.interaction_prefix:
            return [self.interaction_prefix + token for token in tokens]
        else:
            return tokens

    def _iter_interact_docs(self, X, i):
        for doc, interacts in izip(X, i):
            if interacts:
                yield doc
            else:
                yield ""

    def transform(self, X, y=None, i=None):
        """
        same as HashingVectorizer transform, except allows for interaction list
        which is an iterable the same length as X filled with True/False
        this method adds an empty row to docs labelled as False
        """
        analyzer = self.build_analyzer()
        if i == None:
            X = self._get_hasher().transform(analyzer(doc) for doc in X)
        else:
            X = self._get_hasher().transform(analyzer(doc) for doc in self._iter_interact_docs(X, i))
        if self.binary:
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    # Alias transform to fit_transform for convenience
    # (repeated here or else is alised to the parent class transform)
    fit_transform = transform




