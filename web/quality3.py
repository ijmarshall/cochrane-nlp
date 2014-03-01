#
# quality3.py - ModularCountVectorizer
#


#
# TODO - change the name to something better! (can't do it without re pickling the models which will take 2-3h)
#


import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer

SIMPLE_WORD_TOKENIZER = re.compile("[a-zA-Z]{2,}") # regex of the rule used by sklearn CountVectorizer

class ModularCountVectorizer():
    """
    Similar to CountVectorizer from sklearn, but allows building up
    of feature matrix gradually, and adding prefixes to feature names
    (to identify interaction terms)
    """

    def __init__(self, *args, **kwargs):
        self.data = []
        self.vectorizer = DictVectorizer(*args, **kwargs)

    def _transform_X_to_dict(self, X, prefix=None):
        """
        makes a list of dicts from a document
        1. word tokenizes
        2. creates {word1:1, word2:1...} dicts
        (note all set to '1' since the DictVectorizer we use assumes all missing are 0)
        """
        return [self._dict_from_word_list(self._word_tokenize(document, prefix=prefix)) for document in X]

    def _word_tokenize(self, text, prefix=None):
        """
        simple word tokenizer using the same rule as sklearn
        punctuation is ignored, all 2 or more letter characters are a word
        """

        # print "text:"
        # print text
        # print "tokenized words"
        # print SIMPLE_WORD_TOKENIZER.findall(text)

        if prefix:
            return [prefix + word.lower() for word in SIMPLE_WORD_TOKENIZER.findall(text)]
        else:
            return [word.lower() for word in SIMPLE_WORD_TOKENIZER.findall(text)]


    def _dict_from_word_list(self, word_list):
        return {word: 1 for word in word_list}

    def _dictzip(self, dictlist1, dictlist2):
        """
        zips together two lists of dicts of the same length
        """
        # checks lists must be the same length
        if len(dictlist1) != len(dictlist2):
            raise IndexError("Unable to combine featuresets with different number of examples")

        output = []



        for dict1, dict2 in zip(dictlist1, dictlist2):
            output.append(dict(dict1.items() + dict2.items()))
            # note that this *overwrites* any duplicate keys with the key/value from dictlist2!!

        return output

    def transform(self, X, prefix=None):
        # X is a list of document strings
        # word tokenizes each one, then passes to a dict vectorizer
        dict_list = self._transform_X_to_dict(X, prefix=prefix)
        return self.vectorizer.transform(dict_list)

    def fit_transform(self, X, prefix=None):
        # X is a list of document strings
        # word tokenizes each one, then passes to a dict vectorizer
        dict_list = self._transform_X_to_dict(X, prefix=prefix)
        return self.vectorizer.fit_transform(dict_list)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


    def builder_clear(self):
        self.builder = []
        self.builder_len = 0

    def builder_add_docs(self, X, prefix = None):
        if not self.builder:
            self.builder_len = len(X)
            self.builder = self._transform_X_to_dict(X)
        else:
            X_dicts = self._transform_X_to_dict(X, prefix=prefix)
            self.builder = self._dictzip(self.builder, X_dicts)

    def builder_add_interaction_features(self, X, interactions, prefix=None):
        if prefix is None:
            raise TypeError('Prefix is required when adding interaction features')

        doc_list = [(sent if interacting else "") for sent, interacting in zip(X, interactions)]
        self.builder_add_docs(doc_list, prefix)

    def builder_fit_transform(self):
        return self.vectorizer.fit_transform(self.builder)

    def builder_transform(self):
        return self.vectorizer.transform(self.builder)   



