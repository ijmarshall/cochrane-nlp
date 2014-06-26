#
#   experiments
#

import pickle
import re
import pdb
import csv
import os
import sys



import numpy as np
import sklearn
import codecs
import yaml
from unidecode import unidecode
from nltk.corpus import stopwords
from sklearn.externals import six


from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier
import sklearn.metrics


import cochranenlp
from ..output.progressbar import ProgressBar
from ..output.metrics import BinaryMetricsRecorder
from ..readers import biviewer
from ..ml.modhashvec import ModularVectorizer, InteractionHashingVectorizer
from ..textprocessing.tokenizer import sent_tokenizer, word_tokenizer

import pprint

import collections
import csv

import difflib

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)




############################################################
#   
#   data reader
#
############################################################


CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting", "Other"]


class RoBData:


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
            p = ProgressBar(self.max_studies, timer=True)

        self.data = collections.defaultdict(list) # indexed by PMID, list of data entries

        matcher = PDFMatcher() # for matching Cochrane quotes with PDF sentences

        pmids_encountered = collections.Counter()

        for study_id, study in enumerate(self.pdfviewer):

            if study_id > self.max_studies:
                break

            if self.show_progress:
                p.tap()


            

            pdf_text = self._preprocess_pdf(study.studypdf["text"])

            if not doc_level_only:
                matcher.load_pdftext(pdf_text) # load the PDF once

            ###
            #   simplification here - where a domain is repeated just use the first judgement
            #   (where the further judegments are probably for different outcomes)
            ###

            # start off with core domains being unknown, populate known ones later
            doc_y = {domain: 0 for domain in CORE_DOMAINS} # 0 = unknown

            sent_y = {domain: [] for domain in CORE_DOMAINS}

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

            study_data = {"doc-text": pdf_text, "doc-y": doc_y}

            if not doc_level_only:
                study_data.update({"sent-spans": matcher.sent_indices, "sent-y": sent_y})

            self.data[study.studypdf["pmid"]].append(study_data)

    def _preprocess_pdf(self, pdftext):
        pdftext = unidecode(pdftext)
        pdftext = re.sub("\n", " ", pdftext) # preprocessing rule 1
        return pdftext

    def _preprocess_cochrane(self, rawtext):

        if rawtext is None: # look here if some problem with None type appearing later on in sentence processing (may need to change to empty string)
            return None

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
        return mapped_domain if mapped_domain in CORE_DOMAINS else "Other"

        
    def _load_domain_map(self, filename=os.path.join(cochranenlp.PATH, "data", "domain_names.txt")):

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

    def get_ids(self, filter_domain=None, filter_sents=False, pmid_instance=0):
        if filter_sents and not filter_domain:
            raise ValueError("filter_sents set to true; therefore a specific domain needs to be passed in filter_domain")

        domain_filter = (lambda v: v[pmid_instance]["doc-y"][filter_domain] != 0) if filter_domain else lambda v: True
        sent_filter = lambda v: v[pmid_instance]["sent-y"][filter_domain] if filter_sents else lambda v: True
        instance_filter = lambda v: len(v) > pmid_instance
 
        filter_pipe = [instance_filter, domain_filter, sent_filter]
        return np.array([k for k, v in self.data_instance.data.iteritems() if all((f(v) for f in filter_pipe))])

    def Xy(self, doc_indices):
        pass



# class DomainFilter(DataFilter):
#     def __init__(self, *args, **kwargs):
#         """
#         params: domain = one of the CORE_DOMAINS to filter data by
#         """
#         self.domain = kwargs.pop("domain") # remove from the kwargs before calling super.__init__
#         super(DomainFilter, self).__init__(*args, **kwargs)


class DocFilter(DataFilter):

    def Xy(self, doc_indices, pmid_instance=0, domain=None):
        if domain is None:
            raise ValueError("DocFilter requires specific domain to retrieve data for")

        X = []
        y = []
        for i in doc_indices:
            doc_i = self.data_instance.data[i]
            if pmid_instance >= len(doc_i) or doc_i[pmid_instance]["doc-y"][domain]==0:
                continue
                # skip if no information in this instance
                # i.e. if there either isn't a n'th record for the study, or the domain isn't recorded in it
                # (fails silently)
            X.append(doc_i[pmid_instance]["doc-text"])
            y.append(doc_i[pmid_instance]["doc-y"][domain])
        return X, y

class SentFilter(DataFilter):

    def Xy(self, doc_indices, pmid_instance=0, domain=None):
        if domain is None:
            raise ValueError("SentFilter requires specific domain to retrieve data for")

        X = []
        y = []
        for i in doc_indices:

            doc_i = self.data_instance.data[i][pmid_instance]

            # make sentence list
            sents = [doc_i["doc-text"][start:end] for start, end in doc_i["sent-spans"]]
            X.extend(sents)

            # and make boolean array
            sent_y = -np.ones(len(sents), dtype=np.int8) # most are -1s
            sent_y[doc_i["sent-y"][domain]] = 1 # except these ones
            y = np.append(y, sent_y)

        return X, y


class MultiTaskDocFilter(DataFilter):
    """
    for training a multi-task model (i.e. one model for all domains)
    """

    def Xy(self, doc_indices):
        raise NotImplemented("Xy not used in MultiTaskDocFilter - you probably want Xyi() for (X, y, interaction term) tuples")

    def Xyi(self, doc_indices, pmid_instance=0):
        X = []
        y = []
        interactions = []

        for i in doc_indices:
            doc_i = self.data_instance.data[i][pmid_instance]
            for domain, judgement in doc_i["doc-y"].iteritems():                
                if judgement == 0: # skip the blanks
                    continue 
                X.append(doc_i["doc-text"])
                y.append(judgement)
                interactions.append(domain)            
        return X, y, interactions



ModularCountVectorizer = ModularVectorizer
############################################################
#   
#   experiments
#
############################################################
class ExperimentBase(object):

    def __init__(self):
        self.metrics = BinaryMetricsRecorder(domains=CORE_DOMAINS)

    def run(self, doc_filter, train_ids, test_ids, train_instance=0, test_instance=0):

        for domain in CORE_DOMAINS:
            logging.info('domain %s' % domain)

            # allow ids to be domain specific
            if isinstance(train_ids, dict):
                train_ids_domain = train_ids[domain]
            else:
                train_ids_domain = train_ids

            if isinstance(test_ids, dict):
                test_ids_domain = test_ids[domain]
            else:
                test_ids_domain = test_ids


            X_train_d, y_train = doc_filter.Xy(uids[train_ids_domain], domain=domain)
            X_test_d, y_test = doc_filter.Xy(uids[test_ids_domain], domain=domain)

            y_preds = self._get_y_preds_from_fold(X_train_d, y_train, X_test_d)
            
            self.metrics.add_preds_test(y_preds, y_test, domain=domain)

    def save(self, filename):
        logging.info('saving data')
        self.metrics.save_csv(filename)

    def _get_y_preds_from_fold(self, X_train_d, y_train, X_test_d):
        logging.info('modelling fold')
        clf = self._get_model()
        vec = self._get_vec()

        X_train = vec.transform(X_train_d, low=2)
        X_test = vec.transform(X_test_d)

        clf.fit(X_train, y_train)
        return clf.predict(X_test)

    
    def _get_vec(self):
        return InteractionHashingVectorizer(norm=None, non_negative=True, binary=True)
    
    def _get_model(self):
        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        return GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='accuracy')





class SimpleModel(ExperimentBase):
    """
    Models each domain separately
    (produces 6 independent models)
    """
    def __init__(self):
        pass
    def _get_data(self):
        return RoBData(test_mode=False)

    def _get_filter(self, dat):
        return DocFilter(dat)


class MultitaskModel(ExperimentBase):
    """
    Models all domains together, and uses interaction
    features to predict with domain specific information
    """
    def run(self):

        uids = self._get_uids()
        kf = KFold(len(uids), n_folds=folds, shuffle=False)

        for train, test in kf:

            X_train_d, y_train, i_train = self.filter.Xyi(uids[train])
            X_test_d, y_test, i_test = self.filter.Xyi(uids[test])



    
    def _get_y_preds_from_fold(self, X_train_d, y_train, i_train, X_test_d, i_test):
        logging.info('building up training data')
        interactions = {domain:[] for domain in CORE_DOMAINS}
        for doc_text, doc_domain in zip(X_train_d, i_train):
            for domain in CORE_DOMAINS:
                if domain == doc_domain:
                    interactions[domain].append(True)
                else:
                    interactions[domain].append(False)

        logging.info('adding test data to vectorizer')
        vec = ModularCountVectorizer()
        vec.builder_clear()

        logging.info('adding base features')
        vec.builder_add_docs(X_train_d, low=10) # add base features

        for domain in CORE_DOMAINS:
            logging.info('adding interactions for domain %s' % (domain,))
            print np.sum(interactions[domain]), "/", len(interactions[domain]), "added for", domain
            vec.builder_add_interaction_features(X_train_d, interactions=interactions[domain], prefix=domain+"-i-", low=2) # then add interactions

        logging.info('fitting vectorizer')
        X_train = vec.builder_fit_transform()
        
        logging.info('fitting model')
        clf.fit(X_train, y_train)




##################################################
#
# helper methods to setup/run experiments.
#
##################################################
def simple_model_test(data_filter=DocFilter):

    dat = RoBData(test_mode=False)
    dat.generate_data(doc_level_only=True)


    metrics = BinaryMetricsRecorder(domains=CORE_DOMAINS)

    stupid_metrics = BinaryMetricsRecorder(domains=CORE_DOMAINS)


    multitask_docs = MultiTaskDocFilter(dat) # use the same ids as the multitask model
    multitask_uids = np.array(multitask_docs.available_ids)
    no_studies = len(multitask_uids)
    kf = KFold(no_studies, n_folds=5, shuffle=False)

    for domain in CORE_DOMAINS:

        docs = data_filter(dat, domain=domain)
        uids = np.array(docs.available_ids)
        print "%d docs obtained for domain: %s" % (len(uids), domain)


        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

        no_studies = len(uids)

        

        for train, test in kf:

            X_train_d, y_train = docs.Xy(np.intersect1d(uids, multitask_uids[train]))
            X_test_d, y_test = docs.Xy(np.intersect1d(uids, multitask_uids[test]))

            # vec = CountVectorizer(min_df=2)
            vec = InteractionHashingVectorizer(norm=None, non_negative=True, binary=True)

            X_train = vec.fit_transform(X_train_d, low=2)
            X_test = vec.transform(X_test_d)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            metrics.add_preds_test(y_preds, y_test, domain=domain)

            stupid_metrics.add_preds_test([1] * len(y_test), y_test, domain=domain)

    metrics.save_csv('simple_acc.csv')
    stupid_metrics.save_csv('stupid_output.csv')
            




def multitask_test(fold=None, n_folds_total=5, pickle_metrics=False, 
                                metrics_out_dir=None):
    """run multitask experiment.

    if fold a fold is specified, run only that fold. 
    """

    logging.info('loading data into memory')
    dat = RoBData(test_mode=False)
    dat.generate_data(doc_level_only=True)


    logging.info('loading metric recorder')
    metrics = BinaryMetricsRecorder(domains=CORE_DOMAINS)


    logging.info('generating training documents')
    train_docs = MultiTaskDocFilter(dat)
    logging.info('generating training ids')
    train_uids = np.array(train_docs.available_ids)

    logging.info('setting model parameters')
    tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
    clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

    no_studies = len(train_uids)
    logging.info('calculating folds')
    kf = KFold(no_studies, n_folds=n_folds_total, shuffle=False)
    if fold is not None:
        kf = [list(kf)[fold]]
        metrics_out_path = os.path.join(
                metrics_out_dir, "metrics_%s.pickle" % fold)

    for train, test in kf:
        logging.info('new fold starting!')

        X_train_d, y_train, i_train = train_docs.Xyi(train_uids[train])

        logging.info('building up test data')
        interactions = {domain:[] for domain in CORE_DOMAINS}
        for doc_text, doc_domain in zip(X_train_d, i_train):
            for domain in CORE_DOMAINS:
                if domain == doc_domain:
                    interactions[domain].append(True)
                else:
                    interactions[domain].append(False)

        logging.info('adding test data to vectorizer')
        vec = ModularCountVectorizer()
        vec.builder_clear()

        logging.info('adding base features')
        vec.builder_add_docs(X_train_d, low=10) # add base features

        for domain in CORE_DOMAINS:
            logging.info('adding interactions for domain %s' % (domain,))
            print np.sum(interactions[domain]), "/", len(interactions[domain]), "added for", domain
            vec.builder_add_interaction_features(X_train_d, interactions=interactions[domain], prefix=domain+"-i-", low=2) # then add interactions

        logging.info('fitting vectorizer')
        X_train = vec.builder_fit_transform()
        
        logging.info('fitting model')
        clf.fit(X_train, y_train)


        for domain in CORE_DOMAINS:

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

            if pickle_metrics:
                with open(metrics_out_path, 'wb') as out_f:
                    pickle.dump(metrics, out_f)


    if fold is None:
        metrics.save_csv('multitask_acc.csv')
    else:
        metrics.save_csv(os.path.join(metrics_out_path, 'multitask.csv'))



if __name__ == '__main__':
    e = ExperimentBase()
    e.run()
    # simple_model_test()

    






