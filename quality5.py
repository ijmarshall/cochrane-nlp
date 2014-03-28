#
#   experiments
#

from tokenizer import sent_tokenizer, word_tokenizer
import sklearn
import numpy as np
import progressbar
import re
import biviewer
import codecs
import yaml
from unidecode import unidecode
from nltk.corpus import stopwords

from sklearn.externals import six

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier
import sklearn.metrics

import pprint


import csv

import difflib


############################################################
#   
#   data reader
#
############################################################

class RoBData:

    CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                    "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting", "Other"]

    REGEX_QUOTE_PRESENT = re.compile("Quote\:")
    REGEX_QUOTE = re.compile("\"(.*?)\"") # retrive blocks of text in quotes
    REGEX_ELLIPSIS = re.compile("\s*[\[\(]?\s?\.\.+\s?[\]\)]?\s*") # to catch various permetations of "..." and "[...]"


    def __init__(self, test_mode=False, show_progress=True):

        self.domain_map = self._load_domain_map()
        self.pdfviewer = biviewer.PDFBiViewer()

        self.max_studies = 200 if test_mode else len(self.pdfviewer)

        self.show_progress= show_progress
        


    def generate_data(self, doc_level_only=False):
        """
        simultaneously generate document and sentence data
        (though for simple models may not need sentence data)
        """
        if self.show_progress:
                p = progressbar.ProgressBar(self.max_studies, timer=True)

        self.data = []

        matcher = PDFMatcher() # for matching Cochrane quotes with PDF sentences


        for study_id, study in enumerate(self.pdfviewer):

            if study_id > self.max_studies:
                break

            if self.show_progress:
                p.tap()

            pdf_text = self._preprocess_pdf(study.studypdf)

            if not doc_level_only:
                matcher.load_pdftext(pdf_text) # load the PDF once

            ###
            #   simplification here - where a domain is repeated just use the first judgement
            #   (where the further judegments are probably for different outcomes)
            ###

            # start off with core domains being unknown, populate known ones later
            doc_y = {domain: 0 for domain in self.CORE_DOMAINS} # 0 = unknown

            sent_y = {domain: [] for domain in self.CORE_DOMAINS}

            quality_data = study.cochrane["QUALITY"]
            for domain in quality_data:
                domain["QUOTES"] = self._preprocess_cochrane(domain["DESCRIPTION"])                
                
                mapped_domain = self._map_to_core_domain(domain["DOMAIN"])
                if doc_y[mapped_domain] == 0:
                    # only add the results for first instance per study

                    doc_y[mapped_domain] = 1 if domain["RATING"] == "YES" else -1
                    # simplifying to LOW risk of bias = 1 *v* HIGH/UNKNOWN risk = -1

                    if domain["QUOTES"] and not doc_level_only:
                        matcher.load_quotes(domain["QUOTES"])
                        sent_y[mapped_domain] = matcher.generate_y()

            if not doc_level_only:
                study_data = {"doc-text": pdf_text, "doc-y": doc_y,
                          "sent-spans": matcher.sent_indices, "sent-y": sent_y}
            else:
                study_data = {"doc-text": pdf_text, "doc-y": doc_y}

            self.data.append(study_data)

    def _preprocess_pdf(self, pdftext):
        pdftext = unidecode(pdftext)
        pdftext = re.sub("\n", " ", pdftext) # preprocessing rule 1
        return pdftext

    def _preprocess_cochrane(self, rawtext):

        # regex clean up of cochrane strings
        processedtext = unidecode(rawtext)
        processedtext = re.sub(" +", " ", processedtext)

        # extract all parts in quotes
        quotes = self.REGEX_QUOTE.findall(processedtext)

        # then split at any ellipses
        quote_parts = []
        for quote in quotes:
            quote_parts.extend(self.REGEX_ELLIPSIS.split(quote))
        return quote_parts

    def _map_to_core_domain(self, domain_free_text):
        
        mapped_domain = self.domain_map.get(domain_free_text)

        # note the map includes non-core categories
        return mapped_domain if mapped_domain in self.CORE_DOMAINS else "Other"

        
    def _load_domain_map(self, filename="data/domain_names.txt"):

        with codecs.open(filename, 'rb', 'utf-8') as f:
            raw_data = yaml.load(f)

        mapping = {}
        for key, value in raw_data.iteritems():
            for synonym in value:
                mapping[synonym] = key

        return mapping


############################################################
#   
#   match quotes from Cochrane to PDF
#
############################################################



class PDFMatcher():
    """
    matches and generates sent tokens from pdf text
    """
    def __init__(self, quotes=None, pdftext=None):
        # load a sequence matcher; turn autojunk off (since buggy for long strings)
        # http://stackoverflow.com/questions/20875795/python-passing-sequencematcher-in-difflib-an-autojunk-false-flag-yields-error
        self.sequencematcher = difflib.SequenceMatcher(None, autojunk=False)

        # if quotes:
        #     self.quotes = self.load_quotes(quotes)
        # if pdftext:
        #     self.pdftext = self.load_pdftext(pdftext)


    def load_quotes(self, quotes):
        self.quotes = quotes

    def load_pdftext(self, pdftext):
        self.pdftext = pdftext
        self.lenpdf = len(pdftext)
        self.sequencematcher.set_seq2(self.pdftext)
        self.sent_indices =  sent_tokenizer.span_tokenize(self.pdftext)


    def _overlap(self, t1, t2):
        """
        finds out whether two tuples overlap
        """
        return t2[0] < t1[1] and t1[0] < t2[1]


    def generate_X(self):
        X = []
        # go through sentence indices
        # make X (list of sentences)
        for (start_i, end_i) in self.sent_indices:
            X.append(self.pdftext[start_i: end_i])
        return X


    def generate_y(self, min_char_match=20):
        """
        returns indices of any matching sentences
        """
        good_match = False # this will be set to True if sufficent matching characters in
                           # at least one of the parts of the quotes

        match_indices = []

        # go through quotes, match using difflib
        # and keep any matches which are long enough so likely true matches
        for quote in self.quotes:

            self.sequencematcher.set_seq1(quote)

            best_match = self.sequencematcher.find_longest_match(0, len(quote), 0, self.lenpdf)

            # only interested in good quality matches
            if best_match.size > min_char_match:
                good_match = True
                match_indices.append((best_match.b, best_match.b + best_match.size)) # add (start_i, end_i) tuples (of PDF indices)

        
        y = []
        if not good_match:
            # if quality criteria not met, leave here
            # (i.e. return empty lists [])
            return y

        # go through sentence indices
        # and calculate y, if there is *any* overlap with matched quoted text then
        # append the sentence index
        for sent_i, (start_i, end_i) in enumerate(self.sent_indices):
            # if any overlaps with quotes, then y = True, else False
            if any((self._overlap((start_i, end_i), match_tuple) for match_tuple in match_indices)):
                y.append(sent_i)
        return y



############################################################
#   
#   data filters
#
############################################################

class DataFilter(object):

    def __init__(self, data_instance):
        self.data_instance = data_instance
        self.available_ids = self._get_available_ids()

    def _get_available_ids(self):
        """
        subclass this to obtain the subset of ids available
        """

        # in the base class return *all* ids
        return [i for i, doc in enumerate(self.data_instance.data)]


    def Xy(self, doc_indices):
        pass



class DomainFilter(DataFilter):
    def __init__(self, *args, **kwargs):
        """
        params: domain = one of the CORE_DOMAINS to filter data by
        """
        self.domain = kwargs.pop("domain") # remove from the kwargs before calling super.__init__
        super(DomainFilter, self).__init__(*args, **kwargs)


class DocFilter(DomainFilter):

    def _get_available_ids(self):
        return [i for i, doc_i in enumerate(self.data_instance.data) if doc_i["doc-y"][self.domain] != 0]

    def Xy(self, doc_indices):
        X = []
        y = []
        for i in doc_indices:
            doc_i = self.data_instance.data[i]
            X.append(doc_i["doc-text"])
            y.append(doc_i["doc-y"][self.domain])
        return X, y


class SentFilter(DomainFilter):

    def _get_available_ids(self):
        return [i for i, doc_i in enumerate(self.data_instance.data) if doc_i["sent-y"][self.domain]]

    def Xy(self, doc_indices):
        X = []
        y = []
        for i in doc_indices:
            doc_i = self.data_instance.data[i]
            X.append(doc_i["doc-text"])
            y.append(doc_i["doc-y"][self.domain])
        return X, y

class MultiTaskDocFilter(DataFilter):
    """
    for training a multi-task model (i.e. one model for all domains)
    """

    # subclasses _get_available_ids() default behaviour
    # i.e. return all documents

    def Xy(self, doc_indices):
        raise NotImplemented("Xy not used in MultiTaskDocFilter - you probably want Xyi() for interaction terms")

    def Xyi(self, doc_indices):
        X = []
        y = []
        interactions = []

        for i in doc_indices:
            doc_i = self.data_instance.data[i]
            for domain, judgement in doc_i["doc-y"].iteritems():                
                if judgement == 0: # skip the blanks
                    continue 
                X.append(doc_i["doc-text"])
                y.append(judgement)
                interactions.append(domain)
        return X, y, interactions




############################################################
#   
#   metrics
#
############################################################




class BinaryMetricsRecorder(object):
    """
    records results of folds, and outputs them in various formats

    """

    METRIC_NAMES = ["n", "f1", "precision", "recall", "accuracy"]
    METRIC_FUNCTIONS = [lambda preds, test: len(preds),
                        sklearn.metrics.f1_score, sklearn.metrics.precision_score,
                        sklearn.metrics.recall_score, sklearn.metrics.accuracy_score]


    def __init__(self, title="Untitled", domains=["default"]):
        self.title = title
        self.domains = domains
        self.fold_metrics = {k: [] for k in domains}
        

    def add_preds_test(self, preds, test, domain="default"):
        """
        add a fold of data
        """
        fold_metric = {metric_name: metric_fn(test, preds) for metric_name, metric_fn
                    in zip (self.METRIC_NAMES, self.METRIC_FUNCTIONS)}

        fold_metric["domain"] = domain
        fold_metric["fold"] = len(self.fold_metrics[domain])
        
        self.fold_metrics[domain].append(fold_metric)

    def _means(self, domain):

        summaries = {k: [] for k in self.METRIC_NAMES}

        for row in self.fold_metrics[domain]:
            for metric_name in self.METRIC_NAMES:
                summaries[metric_name].append(row[metric_name])

        means = {metric_name: np.mean(summaries[metric_name]) for metric_name in self.METRIC_NAMES}

        means["domain"] = domain
        means["fold"] = "mean"
        means["n"] = np.sum(summaries["n"]) # overwrite with the sum

        return means


    def save_csv(self, filename):

        output = []
        for domain in self.domains:
            output.extend(self.fold_metrics[domain])
            output.append(self._means(domain))
            output.append({}) # blank line



        with open(filename, 'wb') as f:
            w = csv.DictWriter(f, ["domain", "fold"] + self.METRIC_NAMES)
            w.writeheader()
            w.writerows(output)

        

############################################################
#   
#   Vectorizer
#
############################################################



class ModularCountVectorizer():
    """
    Similar to CountVectorizer from sklearn, but allows building up
    of feature matrix gradually, and adding prefixes to feature names
    (to identify interaction terms)
    """
    STOP_WORDS = set(stopwords.words('english'))
    SIMPLE_WORD_TOKENIZER = re.compile("[a-zA-Z]{2,}") # regex of the rule used by sklearn CountVectorizer


    def __init__(self, *args, **kwargs):
        self.data = []
        self.vectorizer = DictVectorizer(*args, **kwargs)

    def _transform_X_to_dict(self, X, prefix=None, weighting=1):
        """
        makes a list of dicts from a document
        1. word tokenizes
        2. creates {word1:1, word2:1...} dicts
        (note all set to '1' since the DictVectorizer we use assumes all missing are 0)
        """
        return [self._dict_from_word_list(
            self._word_tokenize(document, prefix=prefix), weighting=1) for document in X]

    def _word_tokenize(self, text, prefix=None, stopword=True):
        """
        simple word tokenizer using the same rule as sklearn
        punctuation is ignored, all 2 or more letter characters are a word
        """


        stop_word_list = self.STOP_WORDS if stopword else set()

        if prefix:
            return [prefix + word.lower() for word in self.SIMPLE_WORD_TOKENIZER.findall(text) 
                        if not word.lower() in stop_word_list]
        else:
            return [word.lower() for word in self.SIMPLE_WORD_TOKENIZER.findall(text) 
                        if not word.lower() in stop_word_list]


    def _dict_from_word_list(self, word_list, weighting=1):
        return {word: weighting for word in word_list}

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

    def fit_transform(self, X, prefix=None, max_features=None, low=2):
        # X is a list of document strings
        # word tokenizes each one, then passes to a dict vectorizer
        dict_list = self._transform_X_to_dict(X, prefix=prefix)
        X = self.vectorizer.fit_transform(dict_list)

        if max_features is not None or low is not None:
            X, removed = self._limit_features(X.tocsc(), 
                        self.vectorizer.vocabulary_, low=low, limit=max_features)
            print "pruned %s features!" % len(removed)
            X = X.tocsc()

        return self.vectorizer.fit_transform(dict_list)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names()


    def builder_clear(self):
        self.builder = []
        self.builder_len = 0

    def builder_add_docs(self, X, prefix = None, weighting=1):
        #pdb.set_trace()
        if not self.builder:
            self.builder_len = len(X)
            self.builder = self._transform_X_to_dict(X, prefix=prefix, weighting=weighting)
        else:
            X_dicts = self._transform_X_to_dict(X, prefix=prefix, weighting=weighting)
            self.builder = self._dictzip(self.builder, X_dicts)

    def builder_add_interaction_features(self, X, interactions, prefix=None):
        if prefix is None:
            raise TypeError('Prefix is required when adding interaction features')

        doc_list = [(sent if interacting else "") for sent, interacting in zip(X, interactions)]
        self.builder_add_docs(doc_list, prefix)

        
    def builder_fit_transform(self, max_features=None, low=2):
        X = self.vectorizer.fit_transform(self.builder)
        if max_features is not None or low is not None:
            X, removed = self._limit_features(X.tocsc(), 
                        self.vectorizer.vocabulary_, low=low, limit=max_features)
            print "pruned %s features!" % len(removed)
            X = X.tocsc()

        return X #self.vectorizer.fit_transform(self.builder)

    def builder_transform(self):
        return self.vectorizer.transform(self.builder)   


    def _limit_features(self, cscmatrix, vocabulary, high=None, low=None,
                        limit=None):
        """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
        if high is None and low is None and limit is None:
            return cscmatrix, set()

        # Calculate a mask based on document frequencies
        dfs = self._document_frequency(cscmatrix)
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            # backward compatibility requires us to keep lower indices in ties!
            # (and hence to reverse the sort by negating dfs)
            mask_inds = (-dfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(six.iteritems(vocabulary)):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]

        return cscmatrix[:, kept_indices], removed_terms

    def _document_frequency(self, X):
        """Count the number of non-zero values for each feature in csc_matrix X."""
        return np.diff(X.indptr)



############################################################
#   
#   experiments
#
############################################################


class ExperimentBase(object):

    def __init__(self, dat):
        self.metrics = BinaryMetricsRecorder(dat.CORE_DOMAINS)

    def run(self):
        pass

    def _process_fold(self):
        pass


class SimpleModel(ExperimentBase):
    """
    Models each domain separately
    (produces 6 independent models)
    """
    pass


class MultitaskModel(ExperimentBase):
    """
    Models all domains together, and uses interaction
    features to predict with domain specific information
    """
    pass






def simple_model_test(data_filter=DocFilter):

    dat = RoBData(test_mode=False)
    dat.generate_data(doc_level_only=True)


    metrics = BinaryMetricsRecorder(domains=dat.CORE_DOMAINS)

    stupid_metrics = BinaryMetricsRecorder(domains=dat.CORE_DOMAINS)

    for domain in dat.CORE_DOMAINS:

        docs = data_filter(dat, domain=domain)
        uids = np.array(docs.available_ids)
        print "%d docs obtained for domain: %s" % (len(uids), domain)


        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

        no_studies = len(uids)

        kf = KFold(no_studies, n_folds=5, shuffle=False)

        for train, test in kf:

            X_train_d, y_train = docs.Xy(uids[train])
            X_test_d, y_test = docs.Xy(uids[test])

            vec = CountVectorizer()

            X_train = vec.fit_transform(X_train_d)
            X_test = vec.transform(X_test_d)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            metrics.add_preds_test(y_preds, y_test, domain=domain)

            stupid_metrics.add_preds_test([1] * len(y_test), y_test, domain=domain)

    metrics.save_csv('test_output_full.csv')

    stupid_metrics.save_csv('stupid_output.csv')
            
    


def multitask_test():

    dat = RoBData(test_mode=False)
    dat.generate_data(doc_level_only=True)


    metrics = BinaryMetricsRecorder(domains=dat.CORE_DOMAINS)

    train_docs = MultiTaskDocFilter(dat)
    train_uids = np.array(train_docs.available_ids)

    tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
    clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

    no_studies = len(train_uids)

    kf = KFold(no_studies, n_folds=5, shuffle=False)

    for train, test in kf:

        print "new fold!"

        X_train_d, y_train, i_train = train_docs.Xyi(train_uids[train])

        # build up test data
        interactions = {domain:[] for domain in dat.CORE_DOMAINS}
        for doc_text, doc_domain in zip(X_train_d, i_train):
            for domain in dat.CORE_DOMAINS:
                if domain == doc_domain:
                    interactions[domain].append(doc_text)
                else:
                    interactions[domain].append("")

        # add to vec
        vec = ModularCountVectorizer()
        vec.builder_clear()

        vec.builder_add_docs(X_train_d) # add base features

        for domain in dat.CORE_DOMAINS:
            vec.builder_add_docs(interactions[domain], prefix=domain+"-i-") # then add interactions

        X_train = vec.builder_fit_transform()
        

        clf.fit(X_train, y_train)


        for domain in dat.CORE_DOMAINS:

            test_docs = DocFilter(dat, domain=domain) # test on regular doc model
            domain_uids = np.array(test_docs.available_ids)

            test_uids = np.intersect1d(train_uids[test], domain_uids)

            X_test_d, y_test = test_docs.Xy(test_uids)

            # build up test vector

            vec.builder_clear()
            vec.builder_add_docs(X_test_d) # add base features
            vec.builder_add_docs(X_test_d, prefix=domain+'-i-') # add interactions

            X_test = vec.builder_transform()

            y_preds = clf.predict(X_test)

            metrics.add_preds_test(y_preds, y_test, domain=domain)


    metrics.save_csv('multitask.csv')


            





    





if __name__ == '__main__':
    # simple_model_test(data_filter=DocFilter)
    multitask_test()
    






