from tokenizer import sent_tokenizer, word_tokenizer
import biviewer
import pdb
import re
import progressbar
import collections
import string
from unidecode import unidecode
import codecs

import yaml
from pprint import pprint

import numpy as np
import math

import difflib

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV

from sklearn.feature_extraction import DictVectorizer

from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support
import random
import operator 

from sklearn.cross_validation import KFold

from journalreaders import PdfReader

import cPickle as pickle

from sklearn.metrics import f1_score, make_scorer, fbeta_score
import nltk
from nltk.corpus import stopwords

REGEX_QUOTE_PRESENT = re.compile("Quote\:")
REGEX_QUOTE = re.compile("\"(.*?)\"") # retrive blocks of text in quotes
REGEX_ELLIPSIS = re.compile("\s*[\[\(]?\s?\.\.+\s?[\]\)]?\s*") # to catch various permetations of "..." and "[...]"

SIMPLE_WORD_TOKENIZER = re.compile("[a-zA-Z]{2,}") # regex of the rule used by sklearn CountVectorizer

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]
                # "OTHER" is generated in code, not in the mapping file
                # see data/domain_names.txt for various other criteria
                # all of these are available via QualityQuoteReader

ALL_DOMAINS = CORE_DOMAINS[:] # will be added to later

RoB_CLASSES = ["YES", "NO", "UNKNOWN"]

# @TODO move me
domain_str = lambda d: d.lower().replace(" ", "_")


def show_most_informative_features(vectorizer, clf, n=1000):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest). 
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))

    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    out_str = []
    for (c1, f1), (c2, f2) in top:
        out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
    feature_str = "\n".join(out_str)
    return feature_str


def show_most_informative_features_ynu(vectorizer, clf, n=10):
    ###
    # note that in the multi-class case, clf.coef_ will
    # have k weight vectors, which I believe are one per
    # each class (i.e., each is a classifier discriminating
    # one class versus the rest). 

    combinations =  ["NO vs (YES + UNKNOWN)", "UNKNOWN vs (YES + NO)", "YES vs (NO + UNKNOWN)"]

    out_str = []

    for i, combination in enumerate(combinations):

        out_str.append(combination)
        out_str.append("*" * 20)

        c_f = sorted(zip(clf.coef_[i], vectorizer.get_feature_names()))

        if n == 0:
            n = len(c_f)/2

        top = zip(c_f[:n], c_f[:-(n+1):-1])
    
        for (c1, f1), (c2, f2) in top:
            out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2))
    feature_str = "\n".join(out_str)
    return feature_str

def load_domain_map(filename="data/domain_names.txt"):

    with codecs.open(filename, 'rb', 'utf-8') as f:
        raw_data = yaml.load(f)

    mapping = {}

    for key, value in raw_data.iteritems():
        for synonym in value:
            mapping[synonym] = key

    return mapping



class QualityQuoteReader2():
    """
    iterates through Cochrane Risk of Bias information
    v2 maintains unique ids for all source studies + does not filter by whether a quote is present
    returns list of quotes where they are available, else none
    """

    def __init__(self, sent=False, test_mode=False):
        self.BiviewerView = collections.namedtuple('BiViewer_View', ['uid', 'cochrane', 'studypdf'])
        self.pdfviewer = biviewer.PDFBiViewer()
        self.domain_map = load_domain_map()
        if test_mode:
            self.test_mode_break_point = 500
        else:
            self.test_mode_break_point = None

        

    def __iter__(self):
        """
        run through PDF/Cochrane data
        preprocesses PDF text
        and maps domain title to one of the core Risk of Bias domains if possible
        """

        progress_bar_limit = len(self.pdfviewer) if self.test_mode_break_point is None else self.test_mode_break_point
        p = progressbar.ProgressBar(progress_bar_limit, timer=True)

        for uid, study in enumerate(self.pdfviewer):

            if self.test_mode_break_point and (uid >= self.test_mode_break_point):
                break

            p.tap()
            quality_data = study.cochrane["QUALITY"]
            for domain in quality_data:
                
                domain["QUOTES"] = self.preprocess_cochrane(domain["DESCRIPTION"])
                try:
                    domain["DOMAIN"] = self.domain_map[domain["DOMAIN"]] # map domain titles to our core categories
                except:
                    domain["DOMAIN"] = "OTHER"
                    
            yield self.BiviewerView(uid=uid, cochrane={"QUALITY": quality_data}, studypdf=self.preprocess_pdf(study.studypdf))


    def __len__(self):
        return len(self.pdfviewer) if self.test_mode_break_point is None else self.test_mode_break_point


    def preprocess_pdf(self, pdftext):
        pdftext = unidecode(pdftext)
        pdftext = re.sub("\n", " ", pdftext) # preprocessing rule 1
        return pdftext

    def preprocess_cochrane(self, rawtext):

        # regex clean up of cochrane strings
        processedtext = unidecode(rawtext)
        processedtext = re.sub(" +", " ", processedtext)

        # extract all parts in quotes
        quotes = REGEX_QUOTE.findall(processedtext)

        # then split at any ellipses
        quote_parts = []
        for quote in quotes:
            quote_parts.extend(REGEX_ELLIPSIS.split(quote))
        return quote_parts
        

class PDFMatcher():
    """
    matches and generates sent tokens from pdf text
    """
    def __init__(self, quotes=None, pdftext=None):
        # load a sequence matcher; turn autojunk off (since buggy for long strings)
        self.sequencematcher = difflib.SequenceMatcher(None, autojunk=False)

        if quotes:
            self.quotes = self.load_quotes(quotes)
        if pdftext:
            self.pdftext = self.load_pdftext(pdftext)



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
        t1s, t1e = t1
        t2s, t2e = t2

        # true if either start of t1 is inside t2, or start of t2 is inside t1
        return (t2s <= t1s <= t2e) or (t1s <= t2s <= t1e) 


    def generate_X(self):
        X = []
        # go through sentence indices
        # make X (list of sentences)
        for (start_i, end_i) in self.sent_indices:
            X.append(self.pdftext[start_i: end_i])
        return X


    def generate_y(self, min_char_match=20):
        """
        returns X: list of sentence strings
                y: numpy vector of 1, -1 (for positive/negative examples)
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
            # (i.e. return empty lists [], [])
            return y

        # otherwise continue and generate feature and answer vectors

        # get indices of sentences (rather than split)
        sent_indices = sent_tokenizer.span_tokenize(self.pdftext)

        # go through sentence indices
        # make X (list of sentences)
        # and calculate y, if there is *any* overlap with matched quoted text then
        # y = True
        for (start_i, end_i) in sent_indices:



            # if any overlaps with quotes, then y = True, else False
            if any((self._overlap((start_i, end_i), match_tuple) for match_tuple in match_indices)):
                y.append(1)
            else:
                y.append(-1)
        return y



class SentenceModel():
    """
    predicts whether sentences contain risk of bias information
    - uses data from Cochrane quotes only
    """

    def __init__(self, test_mode=False):
        self.quotereader = QualityQuoteReader2(test_mode=test_mode) # this now runs through all studies
        

    def generate_data(self, uid_filter=None):
        """
        tokenizes and processes the raw text from pdfs and cochrane
        saves in self.X_list and self.y_list (both dicts)

        """
        
        test_domains = CORE_DOMAINS # may change later to access various "other" domains

        # one feature matrix X across all domains
        self.X_list = [] # key will be unique id, value will be text
        # but multiple y vectors; one for each test domain
        self.y_list = {domain: [] for domain in test_domains}

        self.y_judgements = {domain: [] for domain in test_domains} # used in subclasses to make hybrid models


        self.X_uids = []
        self.y_uids = {domain: [] for domain in test_domains}



        
        for uid, cochrane_data, pdf_data in self.quotereader:

            if uid_filter is not None and uid not in uid_filter:
                continue        


            matcher = PDFMatcher()
            matcher.load_pdftext(pdf_data)
                    
            X_study = matcher.generate_X()

            self.X_list.extend(X_study)
            self.X_uids.extend([uid] * len(X_study))

            domains_done_already = [] # for this particular study
            # (we're ignoring multiple quotes per domain at the moment and getting the first...)

            for domain in cochrane_data["QUALITY"]:

                if domain["DOMAIN"] not in test_domains or domain["DOMAIN"] in domains_done_already:
                    
                    continue # skip if a domain is repeated in a study (though note that this is likely due to different RoB per *outcome* which is ignored here)

                if domain["QUOTES"]:
                    matcher.load_quotes(domain["QUOTES"])
                    y_study = matcher.generate_y()



                    self.y_list[domain["DOMAIN"]].extend(y_study)
                    self.y_uids[domain["DOMAIN"]].extend([uid] * len(y_study))
                    self.y_judgements[domain["DOMAIN"]].extend([domain["RATING"]] * len(y_study))

                    domains_done_already.append(domain["DOMAIN"])
                    


        self.y = {domain: np.array(self.y_list[domain]) for domain in test_domains}

        self.X_uids = np.array(self.X_uids)
        self.y_uids = {domain: np.array(self.y_uids[domain]) for domain in test_domains}
        self.y_judgements = {domain: np.array(self.y_judgements[domain]) for domain in test_domains}

        # self.vectorize()

    def vectorize(self):

        self.vectorizer = ModularCountVectorizer()

        self.X = self.vectorizer.fit_transform(self.X_list)




    def load_text(self, filename):
        """
        loads the original text of all PDFs, to debugging and looking at predicted text from corpus
        NB this is a large file
        """
        with open(filename, 'rb') as f:
            self.X_list = pickle.load(f)


    def __len__(self):
        """
        returns the total number of studies (not features)
        """
        return len(self.quotereader)

    def len_domain(self, domain):
        return len(np.unique(self.y_uids[domain]))
            
        
    def domain_X_filter(self, domain):
        """
        returns X_filter for a domain
        """
        y_study_uids = np.unique(self.y_uids[domain])
        X_filter = np.nonzero([(X_uid in y_study_uids) for X_uid in self.X_uids])[0]
        return X_filter

    def domain_uids(self, domain):
        unique_study_ids = np.unique(self.y_uids[domain])
        return unique_study_ids

    def X_y_uid_filtered(self, uids, domain):
        X_all = self.X_domain_all(domain=domain)
        y_all = self.y_domain_all(domain=domain)

        filter_ids = np.nonzero([(y_study_id in uids) for y_study_id in self.y_uids[domain]])[0]
        X_filtered = X_all[filter_ids]
        y_filtered = y_all[filter_ids]

        return X_filtered, y_filtered



    def get_all_domains(self):
        return self.y.keys()


    def X_get_sentence(self, select_sent_id, domain):

        y_study_ids = np.unique(self.y[domain].study_ids)
        X_filter = np.nonzero([X_study_id in y_study_ids for X_study_id in self.X.study_ids])[0]
        return self.X_list.data[X_filter[select_sent_id]]


    def X_domain_all(self, domain):
        """
        retrieve X data for a domain
        """

        X_filter = self.domain_X_filter(domain)
        return self.X[X_filter]

    def y_domain_all(self, domain):
        return self.y[domain]


    # def X_y_filtered(self, filter_ids, domain):

    #     X_all = self.X_domain_all(domain=domain)
    #     y_all = self.y_domain_all(domain=domain)

    #     # np.unique always returns ordered ids
    #     unique_study_ids = np.unique(self.y_uids[domain])

    #     mapped_ids = [unique_study_ids[filter_id] for filter_id in filter_ids]

    #     filter_ids = np.nonzero([(y_study_id in mapped_ids) for y_study_id in self.y_uids[domain]])[0]
        

    #     X_filtered = X_all[filter_ids]
    #     y_filtered = y_all[filter_ids]

    #     return X_filtered, y_filtered
    

class DocumentLevelModel(SentenceModel):
    """
    for predicting the risk of bias
    as "HIGH", "LOW", or "UNKNOWN" for a document
    using a binary bag-of-words as features for each document
    """


    def generate_data(self, uid_filter=None, binarize=False):
        """
        tokenizes and processes the raw text from pdfs and cochrane
        saves in self.X_list and self.y_list (both dicts)

        """
        
        test_domains = CORE_DOMAINS # may change later to access various "other" domains

        # one feature matrix X across all domains
        self.X_list = [] # key will be unique id, value will be text
        # but multiple y vectors; one for each test domain
        self.y_list = {domain: [] for domain in test_domains}


        self.X_uids = []
        self.y_uids = {domain: [] for domain in test_domains}

        for uid, cochrane_data, pdf_data in self.quotereader:

            if uid_filter is not None and uid not in uid_filter:
                continue        

            X_study = [pdf_data] # this time the X is the whole pdf data

            self.X_list.extend(X_study)
            self.X_uids.extend([uid] * len(X_study))

            domains_done_already = [] # for this particular study
            # (we're ignoring multiple quotes per domain at the moment and getting the first...)

            for domain in cochrane_data["QUALITY"]:

                if domain["DOMAIN"] not in test_domains or domain["DOMAIN"] in domains_done_already:
                    continue # skip if a domain is repeated in a study (though note that this is likely due to different RoB per *outcome* which is ignored here)

                if binarize:
                    y_study = 1 if domain["RATING"]=="YES" else -1 # binarize
                else:
                    y_study = domain["RATING"]
                
                self.y_list[domain["DOMAIN"]].append(y_study)
                self.y_uids[domain["DOMAIN"]].append(uid)

                domains_done_already.append(domain["DOMAIN"])

  
        self.y = {domain: np.array(self.y_list[domain]) for domain in test_domains}

        self.X_uids = np.array(self.X_uids)
        self.y_uids = {domain: np.array(self.y_uids[domain]) for domain in test_domains}



class MultiTaskDocumentModel(DocumentLevelModel):
    '''
    The idea here is to train a single model across all domains. Basically:

        y_ij = sign{(w0 + w_j) * x_i}

    for document x_i, where a w_j is learned for each domain and w0 is a shared
    weight vector (across domains).
    '''

    def vectorize(self):

        self.vectorizer = ModularCountVectorizer()
        self.vectorizer.builder_clear()

        self.X_mt_labels = [] # which rows correspond to which doc/interactions? 
        self.y_mt = []
        self.uids_to_row_indices = {}
        self.row_indices_to_uids, self.row_indices_to_domains = [], []
        # number of rows in the 'multi-task' matrix, which
        # will vary depending on how many docs have labels
        # for how many domains
        n_rows = 0 # (equal to len(self.X_mt_labels)

        '''
        the vectorizer wants all the documents at once,
        so here we are going to build them up. we're only
        going to add interaction copies for a given document
        for those domains that we have an associated label.
        '''
        docs = []
        # which indices in docs correspond to copies for 
        # which domains?
        domains_to_interaction_doc_indices = defaultdict(list)
        for i, doc in enumerate(self.X_list):
            # `intercept' document
            uid = self.X_uids[i]

            # add |CORE_DOMAINS| copies for each instance.
            for domain in CORE_DOMAINS:
                d_str = domain_str(domain)
                if uid in self.domain_uids(domain):
                    # get the label (note that we match up the
                    # uid to do this)
                    uids_to_lbls = dict(zip(self.y_uids[domain], 
                                        self.y_domain_all(domain=domain)))
                    #y_index = self.y_uids(domain).index(uid)
                    #domain_ys = self.y_domain_all(domain=domain)
                    #self.y_mt.append(domain_ys[y_index])
                    self.y_mt.append(uids_to_lbls[uid])

                    # append interaction copy of doc
                    docs.append(doc)
                    self.row_indices_to_uids.append(uid)
                    self.row_indices_to_domains.append(domain)
                    self.X_mt_labels.append("%s-%s" % (i, d_str))
                    domains_to_interaction_doc_indices[d_str].append(n_rows)

                    n_rows += 1

        '''
        now actually ad documents and interaction copies to 
        the vectorizer. 
        '''
        #for i, doc in enumerate(self.X_list):
        # `intercept' document
        self.vectorizer.builder_add_docs(docs)

        for domain in CORE_DOMAINS:
            d_str = domain_str(domain)
            interaction_list = []
            for i in xrange(len(docs)):
                if i in domains_to_interaction_doc_indices[d_str]:
                    interaction_list.append(docs[i])
                else:
                    interaction_list.append("")
            self.vectorizer.builder_add_docs(interaction_list, prefix=d_str+"-")

        # BCW -- attempting to upper bound features!
        # note that this will keep the <max_features> most common
        # features, regardless of whether or not they are 'interaction'
        # features
        self.X = self.vectorizer.builder_fit_transform(max_features=50000)
   

    def X_y_uid_filtered(self, uids, domain=None):
        X_indices, y = [], []
        for i in xrange(self.X.shape[0]):
            if domain is None and self.row_indices_to_uids[i] in uids:
                # if domain is None, return big multi-task design matrix
                # -- you'll want to do this, e.g., for training
                X_indices.append(i)
                y.append(self.y_mt[i])
            elif domain == self.row_indices_to_domains[i] and self.row_indices_to_uids[i] in uids:
                # otherwise (when domain is not None), return 
                # instances for only the target domain
                # (e.g., for testing)
                X_indices.append(i)
                y.append(self.y_mt[i])

        return self.X[X_indices], y
      

class MultiTaskHybridDocumentModel(MultiTaskDocumentModel):
    '''
    same as the MultiTaskDocumentModel, except takes in sentence
    level modelling too into the mix
    '''

    def vectorize(self):

        self.vectorizer = ModularCountVectorizer()
        self.vectorizer.builder_clear()

        self.X_mt_labels = [] # which rows correspond to which doc/interactions? 
        self.y_mt = []
        self.uids_to_row_indices = {}
        self.row_indices_to_uids, self.row_indices_to_domains = [], []
        # number of rows in the 'multi-task' matrix, which
        # will vary depending on how many docs have labels
        # for how many domains
        n_rows = 0 # (equal to len(self.X_mt_labels)

        '''
        the vectorizer wants all the documents at once,
        so here we are going to build them up. we're only
        going to add interaction copies for a given document
        for those domains that we have an associated label.
        '''
        docs = []
        high_prob_sents = defaultdict(list)
        # which indices in docs correspond to copies for 
        # which domains?
        domains_to_interaction_doc_indices = defaultdict(list)
        for i, doc in enumerate(self.X_list):
            # `intercept' document
            uid = self.X_uids[i]

            # add |CORE_DOMAINS| copies for each instance.
            for domain in CORE_DOMAINS:
                d_str = domain_str(domain)
                if uid in self.domain_uids(domain):
                    # get the label (note that we match up the
                    # uid to do this)
                    uids_to_lbls = dict(zip(self.y_uids[domain], 
                                        self.y_domain_all(domain=domain)))
                    #y_index = self.y_uids(domain).index(uid)
                    #domain_ys = self.y_domain_all(domain=domain)
                    #self.y_mt.append(domain_ys[y_index])
                    self.y_mt.append(uids_to_lbls[uid])

                    # append interaction copy of doc
                    docs.append(doc)


                    high_prob_sents[domain].append(self.get_sent_predictions_for_doc(doc, domain))

                    for high_prob_domain in CORE_DOMAINS:
                        if high_prob_domain != domain:
                            high_prob_sents[high_prob_domain].append("")




                    self.row_indices_to_uids.append(uid)
                    self.row_indices_to_domains.append(domain)
                    self.X_mt_labels.append("%s-%s" % (i, d_str))
                    domains_to_interaction_doc_indices[d_str].append(n_rows)

                    n_rows += 1

        '''
        now actually add documents and interaction copies to 
        the vectorizer. 
        '''
        #for i, doc in enumerate(self.X_list):
        # `intercept' document
        self.vectorizer.builder_add_docs(docs)

        for domain in CORE_DOMAINS:
            d_str = domain_str(domain)
            interaction_list, sent_interaction_list = [], []
            for i in xrange(len(docs)):
                if i in domains_to_interaction_doc_indices[d_str]:
                    interaction_list.append(docs[i])
                    sent_interaction_list.append(high_prob_sents[domain][i])
                else:
                    interaction_list.append("")
                    sent_interaction_list.append("")

            self.vectorizer.builder_add_docs(interaction_list, prefix=d_str+"-doc-")
            self.vectorizer.builder_add_docs(sent_interaction_list, prefix=d_str+"-sent-")

        self.X = self.vectorizer.builder_fit_transform()

    def get_sent_predictions_for_doc(self, doc, domain):

        # tokenize into sentences
        sents = sent_tokenizer.tokenize(doc)

        # vectorize the sentences
        X_sents = self.sent_vectorizer.transform(sents)

        # get predicted 1 / -1 for the sentences
        pred_class = self.sent_clfs[domain].predict(X_sents)

        # get the sentences which are predicted 1
        positive_sents = [sent for sent, pred in zip(sents, pred_class) if pred==1]

        # make a single string per doc
        rob_sents = " ".join(positive_sents)

        return rob_sents

    def set_sent_model(self, sent_clfs, sent_vectorizer):
        """
        set a model which takes in a list of sentences;
        and returns -1 or 1
        """
        self.sent_clfs = sent_clfs
        self.sent_vectorizer = sent_vectorizer



class HybridDocModel(DocumentLevelModel):
    """
    for predicting the risk of bias
    as "HIGH", "LOW", or "UNKNOWN" for a document
    using a binary bag-of-words as features for each document
    """


    def vectorize(self, domain=None):

        if domain is None:
            raise TypeError("this class requires domain specific vectorization")

        self.vectorizer = ModularCountVectorizer()
        self.vectorizer.builder_clear()

        X_filter = self.domain_X_filter(domain)
        predictions = self.get_sent_predictions_for_domain(domain)

        self.vectorizer.builder_add_docs([self.X_list[i] for i in X_filter])
        self.vectorizer.builder_add_docs(predictions, prefix="high-prob-sent-", weighting=10)

        self.X = self.vectorizer.builder_fit_transform()

    def get_sent_predictions_for_domain(self, domain):

        uids = self.domain_uids(domain)

        predictions = []

        for uid in uids:

            # get the index of the study with specified uid
            study_index = np.nonzero(self.X_uids==uid)[0][0]

            # tokenize into sentences
            sents = sent_tokenizer.tokenize(self.X_list[study_index])

            # vectorize the sentences
            X_sents = self.sent_vectorizer.transform(sents)

            # get predicted 1 / -1 for the sentences
            pred_class = self.sent_clf.predict(X_sents)

            # get the sentences which are predicted 1
            positive_sents = [sent for sent, pred in zip(sents, pred_class) if pred==1]

            # make a single string per doc
            doc = " ".join(positive_sents)
            
            predictions.append(doc)
        
        return predictions

    def set_sent_model(self, doc_clf, doc_vectorizer):
        """
        set a model which takes in a list of sentences;
        and returns -1 or 1
        """
        self.sent_clf = doc_clf
        self.sent_vectorizer = doc_vectorizer

    def X_y_uid_filtered(self, uids, domain):
        X_all = self.X
        y_all = self.y_domain_all(domain=domain)

        filter_ids = np.nonzero([(y_study_id in uids) for y_study_id in self.y_uids[domain]])[0]
        X_filtered = X_all[filter_ids]
        y_filtered = y_all[filter_ids]

        return X_filtered, y_filtered





# class HybridDocModel2(HybridDocModel):
#     """
#     for predicting the risk of bias
#     as "HIGH", "LOW", or "UNKNOWN" for a document
#     using a binary bag-of-words as features for each document
#     """


#     def vectorize(self, domain=None):

#         if domain is None:
#             raise TypeError("this class requires domain specific vectorization")

#         self.vectorizer = ModularCountVectorizer()
#         self.vectorizer.builder_clear()
#         X_filter = self.domain_X_filter(domain)

#         predictions = self.get_sent_predictions_for_domain(domain)

#         # self.vectorizer.builder_add_docs([self.X_list[i] for i in X_filter])
        
#         self.vectorizer.builder_add_docs(predictions, prefix="high-prob-sent-")

#         self.X = self.vectorizer.builder_fit_transform()






class HybridModel(SentenceModel):
    """
    predicts whether sentences contain risk of bias information
    - uses real RoB judgements
    """


    def vectorize(self, domain=None, interaction_classes=["YES", "NO"]):

        if domain is None:
            raise TypeError("this class requires domain specific vectorization")

        self.vectorizer = ModularCountVectorizer()
        
        self.vectorizer.builder_clear()

        X_filter = self.domain_X_filter(domain)

        sents = [self.X_list[i] for i in X_filter]

        self.vectorizer.builder_add_docs(sents)

        for interaction_class in interaction_classes:

            self.vectorizer.builder_add_interaction_features(sents, self.y_judgements[domain]==interaction_class, prefix="rob-" + interaction_class + "-")

        self.X = self.vectorizer.builder_fit_transform()


    def X_y_uid_filtered(self, uids, domain):
        X_all = self.X
        y_all = self.y_domain_all(domain=domain)

        filter_ids = np.nonzero([(y_study_id in uids) for y_study_id in self.y_uids[domain]])[0]
        X_filtered = X_all[filter_ids]
        y_filtered = y_all[filter_ids]

        return X_filtered, y_filtered



class HybridModelProbablistic(HybridModel):
    """
    predicts whether sentences contain risk of bias information
    - requires a model to be passed in which can predice RoB judgements from
    full text document
    """

    def vectorize(self, domain=None, interaction_classes=["YES", "NO"], use_vectorizer=None):

        if domain is None:
            raise TypeError("this class requires domain specific vectorization")

        if use_vectorizer is None:
            self.vectorizer = ModularCountVectorizer()
        else:
            self.vectorizer = use_vectorizer

        self.vectorizer.builder_clear()
        X_filter = self.domain_X_filter(domain)

        predictions = self.get_doc_predictions_for_domain(domain)



        sents = [self.X_list[i] for i in X_filter]
        self.vectorizer.builder_add_docs(sents)

        for interaction_class in interaction_classes:
            self.vectorizer.builder_add_interaction_features(sents, predictions==interaction_class, prefix="rob-" + interaction_class + "-")


        if use_vectorizer is None:
            self.X = self.vectorizer.builder_fit_transform()
        else:
            self.X = self.vectorizer.builder_transform()




    def get_doc_predictions_for_domain(self, domain):

        uids = self.domain_uids(domain)

        predictions = []

        for uid in uids:

            # get the indices of all sentences in the study with specified uid
            X_filter = np.nonzero(self.X_uids==uid)[0]

            # make a single string per doc
            doc = " ".join([self.X_list[i] for i in X_filter])
            # vectorize the docs, then predict using the model
            X_doc = self.doc_vectorizer.transform(doc)
            prediction = self.doc_clf.predict(X_doc)

            # add the same prediction for each sentence
            predictions.extend([prediction[0]] * len(X_filter))
        
        return np.array(predictions)

    def set_doc_model(self, doc_clf, doc_vectorizer):
        """
        set a model which takes in a full text doc;
        outputs a doc class "YES", "NO", or "UNKNOWN"
        """
        self.doc_clf = doc_clf
        self.doc_vectorizer = doc_vectorizer




class ModularCountVectorizer():
    """
    Similar to CountVectorizer from sklearn, but allows building up
    of feature matrix gradually, and adding prefixes to feature names
    (to identify interaction terms)
    """

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

        # print "text:"
        # print text
        # print "tokenized words"
        # print SIMPLE_WORD_TOKENIZER.findall(text)
        stop_word_list = stopwords.words('english') if stopword else []

        if prefix:
            return [prefix + word.lower() for word in SIMPLE_WORD_TOKENIZER.findall(text) 
                        if not word.lower() in stop_word_list]
        else:
            return [word.lower() for word in SIMPLE_WORD_TOKENIZER.findall(text) 
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


    def _cap_features(self, n=50000):
        # count up each token.
        token_counts_d = defaultdict(int)
        for x_d in self.builder:
            for t in x_d:
                token_counts_d[t] += 1

        # sort by frequency
        sorted_d = sorted(
            token_counts_d.iteritems(), key=operator.itemgetter(1), reverse=True)
        
        keep_these = [t[0] for t in sorted_d[:n]]

        builder_filtered = []
        for x_d in self.builder:
            filtered_d = {}
            for token in keep_these:
                if token in x_d:
                    filtered_d[token] = x_d[token]
            builder_filtered.append(filtered_d)
        #pdb.set_trace()
        self.builder = builder_filtered

        

    def builder_fit_transform(self, max_features=None):
        if max_features is not None:
            self._cap_features(n=max_features)

        pdb.set_trace()
        return self.vectorizer.fit_transform(self.builder)

    def builder_transform(self):
        return self.vectorizer.transform(self.builder)   





def sentence_prediction_test(class_weight={1: 5, -1:1}, model=SentenceModel(test_mode=True)):
    print
    print
    print

    print "Sentence level prediction"
    print "=" * 40
    print

    s = model


    print "Model name:\t" + s.__class__.__name__
    print s.__doc__

    print "class_weight=%s" % (str(class_weight),)
    
    
    s.generate_data()
    s.vectorize()
    
    for test_domain in CORE_DOMAINS:
        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        


        domain_uids = s.domain_uids(test_domain)
        no_studies = len(domain_uids)




        kf = KFold(no_studies, n_folds=5, shuffle=False, indices=True)

        # # tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        # tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 1, 5)]}]
        # clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
        

        print "making scorer"
        ftwo_scorer = make_scorer(fbeta_score, beta=2)

        tuned_parameters = [{"alpha": np.logspace(-4, -1, 10)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 1, 10)]}]
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring=ftwo_scorer)


        metrics = []

        for fold_i, (train, test) in enumerate(kf):



            X_train, y_train = s.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = s.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

            metrics.append(fold_metric) # get the scores for positive instances

            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            

            # if not sample and list_features:
            #     # not an obvious way to get best features for ensemble
            #     print show_most_informative_features(s.vectorizer, clf)
            

        # summary score

        summary_metrics = np.mean(metrics, axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])


        # then train all for most informative features
        clf = SGDClassifier(loss="hinge", penalty="L2", alpha=0.01, class_weight={1: 5, -1: 1})
        X_all = s.X_domain_all(test_domain)
        y_all = s.y_domain_all(test_domain)

        clf.fit(X_all, y_all)

        print show_most_informative_features(s.vectorizer, clf)





def binary_doc_prediction_test(model=DocumentLevelModel(test_mode=False)):
    print
    print
    print

    print "Binary doc prediction"
    print "=" * 40
    print

    s = model


    
    s.generate_data(binarize=True)
    s.vectorize()
    
    for test_domain in CORE_DOMAINS:
        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        


        domain_uids = s.domain_uids(test_domain)
        no_studies = len(domain_uids)




        kf = KFold(no_studies, n_folds=5, shuffle=False, indices=True)

        # # tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        # tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 1, 5)]}]
        # clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
        

        # print "making scorer"
        # ftwo_scorer = make_scorer(fbeta_score, beta=2)
        tuned_parameters = {"alpha": np.logspace(-4, -1, 10), "class_weight": [{1: i, -1: 1} for i in np.logspace(-1, 1, 10)]}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring="f1")


        metrics = []

        for fold_i, (train, test) in enumerate(kf):



            X_train, y_train = s.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = s.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

            metrics.append(fold_metric) # get the scores for positive instances

            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            

            # if not sample and list_features:
            #     # not an obvious way to get best features for ensemble
            #     print show_most_informative_features(s.vectorizer, clf)
            

        # summary score

        summary_metrics = np.mean(metrics, axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])


        # then train all for most informative features
        clf = SGDClassifier(loss="hinge", penalty="L2", alpha=0.01, class_weight={1: 5, -1: 1})
        X_all = s.X_domain_all(test_domain)
        y_all = s.y_domain_all(test_domain)

        clf.fit(X_all, y_all)

        print show_most_informative_features(s.vectorizer, clf)

def multitask_document_prediction_test(model=MultiTaskDocumentModel(test_mode=False), 
                                        test_domain=CORE_DOMAINS[0]):
    print "multitask!"
    d = model
    d.generate_data(binarize=True) # some variations use the quote data internally 
                      # for sentence prediction (for additional features)

    # d.X_uids contains all the UIds.
    # d.y_uids contains a dictionary mapping domains to the UIds for
    # which we have labels (in said domain)
    #pdb.set_trace()
    all_uids = d.X_uids
    d.vectorize()

    ####
    # the major change here is we don't loop through the domains!
    tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
    clf = GridSearchCV(SGDClassifier(loss="log", penalty="L2"), 
                                tuned_parameters, scoring='f1')

    kf = KFold(len(all_uids), n_folds=5, shuffle=False) ### TODO 250 is totally random
    metrics = defaultdict(list)




    for fold_i, (train, test) in enumerate(kf):

        print "Training on fold", fold_i,
        # note that we do *not* pass in a domain here, because
        # we use *all* domain training data
        X_train, y_train = d.X_y_uid_filtered(all_uids[train])
        print "done!"
        

        clf.fit(X_train, y_train)

        print "Testing on fold", fold_i,
        for domain in CORE_DOMAINS:
            # multitask uses same trained model for all domains, but test on separate test data
            X_test, y_test = d.X_y_uid_filtered(all_uids[test], domain)

            y_preds = clf.predict(X_test)
                
            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]
            metrics[domain].append(fold_metric) # get the scores for positive instances (save them up since all in the wrong order here!)
        print "done!"

    # then recreate in the right order for printout
    for domain in CORE_DOMAINS:

        print
        print domain
        print "*" * 60
        print

        for fold_i, fold_metric in enumerate(metrics[domain]):
            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
        # summary score

        summary_metrics = np.mean(metrics[domain], axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])



def multitask_hybrid_document_prediction_test(model=MultiTaskHybridDocumentModel(test_mode=False)):
    
    print "multitask! and hybrid! :)"
    d = model
    d.generate_data(binarize=True) # some variations use the quote data internally 
                      # for sentence prediction (for additional features)

    # d.X_uids contains all the UIds.
    # d.y_uids contains a dictionary mapping domains to the UIds for
    # which we have labels (in said domain)
    #pdb.set_trace()
    all_uids = d.X_uids
    # d.vectorize()

    ####
    # the major change here is we don't loop through the domains!
    tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
    clf = GridSearchCV(SGDClassifier(loss="log", penalty="L2"), 
                                tuned_parameters, scoring='f1')

    kf = KFold(len(all_uids), n_folds=5, shuffle=False) ### TODO 250 is totally random
    metrics = defaultdict(list)



    print "...generating sentence data,,,",
    s = SentenceModel(test_mode=False)
    s.generate_data()
    s.vectorize()
    print "done!"

    sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]


    for fold_i, (train, test) in enumerate(kf):


        


        sent_clfs = defaultdict(list)

        for domain in CORE_DOMAINS:
            sents_X, sents_y = s.X_domain_all(domain=domain), s.y_domain_all(domain=domain)

            sent_clfs[domain] = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
            sent_clfs[domain].fit(sents_X, sents_y)



        print "Training on fold", fold_i,

        d.set_sent_model(sent_clfs, s.vectorizer)
        d.vectorize()

        # note that we do *not* pass in a domain here, because
        # we use *all* domain training data
        X_train, y_train = d.X_y_uid_filtered(all_uids[train])

        clf.fit(X_train, y_train)
        print "done!"

        print "Testing on fold", fold_i,
        for domain in CORE_DOMAINS:
            # multitask uses same trained model for all domains, but test on separate test data
            X_test, y_test = d.X_y_uid_filtered(all_uids[test], domain)

            y_preds = clf.predict(X_test)
                
            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]
            metrics[domain].append(fold_metric) # get the scores for positive instances (save them up since all in the wrong order here!)
        print "done!"

    # then recreate in the right order for printout
    for domain in CORE_DOMAINS:

        print
        print domain
        print "*" * 60
        print

        for fold_i, fold_metric in enumerate(metrics[domain]):
            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
        # summary score

        summary_metrics = np.mean(metrics[domain], axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])





def document_prediction_test(model=DocumentLevelModel(test_mode=False)):

    print "Document level prediction"
    print "=" * 40
    print

    d = model
    d.generate_data() # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)

    d.vectorize()

    for test_domain in CORE_DOMAINS:
        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        # f1_prefer_nos = make_scorer(f1_score, pos_label="NO")

        tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
        clf = GridSearchCV(SGDClassifier(loss="log", penalty="L2"), tuned_parameters, scoring='f1')

       
        # clf = SGDClassifier(loss="hinge", penalty="L2")

        domain_uids = d.domain_uids(test_domain)
        no_studies = len(domain_uids)

        kf = KFold(no_studies, n_folds=5, shuffle=False)

        metrics = []


        for fold_i, (train, test) in enumerate(kf):

            X_train, y_train = d.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = d.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)
            


            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds, labels=RoB_CLASSES))[:3]

            print ('fold %d\t' % (fold_i)) + '\t'.join(RoB_CLASSES)

            # for metric_type, scores in zip(["prec.", "recall", "f1"], fold_metric):
            #     print "%s\t%.2f\t%.2f\t%.2f" % (metric_type, scores[0], scores[1], scores[2])

            # print

            # print clf.best_params_

            #### START CONFUSION

            real_no_indices = (y_test=="NO")
            print "The actual NOs were predicted as..."
            print collections.Counter(y_preds[real_no_indices])

            #### END CONFUSION



            metrics.append(fold_metric) # get the scores for positive instances

            # print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])


        mean_scores = np.mean(metrics, axis=0)

        print "=" * 40
        print 'means \t' + '\t'.join(RoB_CLASSES)

        for metric_type, scores in zip(["prec.", "recall", "f1"], mean_scores):
            print "%s\t%.2f\t%.2f\t%.2f" % (metric_type, scores[0], scores[1], scores[2])
        print


        # then train all for most informative features
        clf = SGDClassifier(loss="hinge", penalty="L2", alpha=0.01)
        X_all = d.X_domain_all(test_domain)
        y_all = d.y_domain_all(test_domain)

        clf.fit(X_all, y_all)

        print show_most_informative_features_ynu(d.vectorizer, clf)




def simple_hybrid_prediction_test(model=HybridModel(test_mode=True)):

    print "Hybrid prediction"
    print "=" * 40
    print


    s = model
    s.generate_data() # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)



    for test_domain in CORE_DOMAINS:

        s.vectorize(test_domain)

        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        domain_uids = s.domain_uids(test_domain)
        no_studies = len(domain_uids)

        kf = KFold(no_studies, n_folds=5, shuffle=False)


        # tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight":  [{1: i, -1: 1} for i in np.logspace(0, 1, 5)]}]
        # clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

        print "making scorer"
        ftwo_scorer = make_scorer(fbeta_score, beta=2)

        tuned_parameters = [{"alpha": np.logspace(-4, -1, 10)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 1, 10)]}]
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring=ftwo_scorer)





        metrics = []

        for fold_i, (train, test) in enumerate(kf):



            X_train, y_train = s.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = s.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

            metrics.append(fold_metric) # get the scores for positive instances

            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            



            metrics.append(fold_metric) # get the scores for positive instances



        # summary score

        summary_metrics = np.mean(metrics, axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])





# def simple_hybrid_prediction_test(model=HybridModel(test_mode=True)):

#     print "Hybrid prediction"
#     print "=" * 40
#     print


#     s = model
#     s.generate_data() # some variations use the quote data internally 
#                                                 # for sentence prediction (for additional features)



#     for test_domain in CORE_DOMAINS:

#         s.vectorize(test_domain)

#         print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
#         domain_uids = s.domain_uids(test_domain)
#         no_studies = len(domain_uids)

#         kf = KFold(no_studies, n_folds=5, shuffle=False)


#         tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight":  [{1: i, -1: 1} for i in np.logspace(0, 1, 5)]}]
#         clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')

        



#         metrics = []

#         for fold_i, (train, test) in enumerate(kf):



#             X_train, y_train = s.X_y_uid_filtered(domain_uids[train], test_domain)
#             X_test, y_test = s.X_y_uid_filtered(domain_uids[test], test_domain)

#             clf.fit(X_train, y_train)

#             y_preds = clf.predict(X_test)

#             fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

#             metrics.append(fold_metric) # get the scores for positive instances

#             print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            



#             metrics.append(fold_metric) # get the scores for positive instances



#         # summary score

#         summary_metrics = np.mean(metrics, axis=0)
#         print "=" * 40
#         print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])



def true_hybrid_prediction_test(model, test_mode=False):

    print "True Hybrid prediction"
    print "=" * 40
    print


    s = model
    s.generate_data() # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)

    s_cheat = HybridModel(test_mode=False)
    s_cheat.generate_data()



    for test_domain in CORE_DOMAINS:

        

        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        domain_uids = s.domain_uids(test_domain)
        no_studies = len(domain_uids)
        kf = KFold(no_studies, n_folds=5, shuffle=False)

        print "making scorer"
        ftwo_scorer = make_scorer(fbeta_score, beta=2)

        tuned_parameters = [{"alpha": np.logspace(-4, -1, 10)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 1, 10)]}]
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring=ftwo_scorer)


        metrics = []

        for fold_i, (train, test) in enumerate(kf):

            print "training doc level model with test data, please wait..."

            d = DocumentLevelModel(test_mode=False)
            d.generate_data(uid_filter=domain_uids[train])
            d.vectorize()
            doc_X, doc_y = d.X_domain_all(domain=test_domain), d.y_domain_all(domain=test_domain)


            doc_tuned_parameters = {"alpha": np.logspace(-4, -1, 10)}
            doc_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), doc_tuned_parameters, scoring='f1')

            doc_clf.fit(doc_X, doc_y)

            s.set_doc_model(doc_clf, d.vectorizer)


            s_cheat.vectorize(test_domain)
            s.vectorize(test_domain, use_vectorizer=s_cheat.vectorizer)


            X_train, y_train = s_cheat.X_y_uid_filtered(domain_uids[train], test_domain)
            # train on the *true* labels

            X_test, y_test = s.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

            metrics.append(fold_metric) # get the scores for positive instances

            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            



            metrics.append(fold_metric) # get the scores for positive instances



        # summary score

        summary_metrics = np.mean(metrics, axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])



def hybrid_doc_prediction_test(model=HybridDocModel(test_mode=True)):

    print "Hybrid doc level prediction"
    print "=" * 40
    print


    d = model
    d.generate_data() # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)


    for test_domain in CORE_DOMAINS:
        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        domain_uids = d.domain_uids(test_domain)
        no_studies = len(domain_uids)
        kf = KFold(no_studies, n_folds=5, shuffle=False)
        tuned_parameters = {"alpha": np.logspace(-4, -1, 5)}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')


        metrics = []

        for fold_i, (train, test) in enumerate(kf):

            

            s = SentenceModel(test_mode=False)
            s.generate_data(uid_filter=domain_uids[train])
            s.vectorize()
            sents_X, sents_y = s.X_domain_all(domain=test_domain), s.y_domain_all(domain=test_domain)



            sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
            sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
            sent_clf.fit(sents_X, sents_y)
            d.set_sent_model(sent_clf, s.vectorizer)
            d.vectorize(test_domain)

            X_train, y_train = d.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = d.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds, labels=RoB_CLASSES))[:3]

            print ('fold %d\t' % (fold_i)) + '\t'.join(RoB_CLASSES)

            for metric_type, scores in zip(["prec.", "recall", "f1"], fold_metric):
                print "%s\t%.2f\t%.2f\t%.2f" % (metric_type, scores[0], scores[1], scores[2])

            print





            metrics.append(fold_metric) # get the scores for positive instances

            # print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])


        mean_scores = np.mean(metrics, axis=0)

        print "=" * 40
        print 'means \t' + '\t'.join(RoB_CLASSES)

        for metric_type, scores in zip(["prec.", "recall", "f1"], mean_scores):
            print "%s\t%.2f\t%.2f\t%.2f" % (metric_type, scores[0], scores[1], scores[2])
        print



def binary_hybrid_doc_prediction_test(model=HybridDocModel(test_mode=True)):

    print "Binary hybrid doc level prediction"
    print "=" * 40
    print


    d = model
    d.generate_data(binarize=True) # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)


    for test_domain in CORE_DOMAINS:

        

        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        domain_uids = d.domain_uids(test_domain)
        no_studies = len(domain_uids)
        kf = KFold(no_studies, n_folds=5, shuffle=False)
        tuned_parameters = {"alpha": np.logspace(-4, -1, 10), "class_weight": [{1: i, -1: 1} for i in np.logspace(-1, 1, 10)]}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')


        metrics = []

        for fold_i, (train, test) in enumerate(kf):

            

            s = SentenceModel(test_mode=True)
            s.generate_data(uid_filter=domain_uids[train])
            s.vectorize()
            sents_X, sents_y = s.X_domain_all(domain=test_domain), s.y_domain_all(domain=test_domain)



            sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
            sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
            sent_clf.fit(sents_X, sents_y)
            d.set_sent_model(sent_clf, s.vectorizer)
            d.vectorize(test_domain)


            X_train, y_train = d.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = d.X_y_uid_filtered(domain_uids[test], test_domain)

            clf.fit(X_train, y_train)

            y_preds = clf.predict(X_test)

            fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

            metrics.append(fold_metric) # get the scores for positive instances

            print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            

            metrics.append(fold_metric) # get the scores for positive instances




            # print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])

        summary_metrics = np.mean(metrics, axis=0)
        print "=" * 40
        print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])






        # then train all for most informative features

        s = SentenceModel(test_mode=True)
        s.generate_data(uid_filter=domain_uids)
        s.vectorize()
        sents_X, sents_y = s.X_domain_all(domain=test_domain), s.y_domain_all(domain=test_domain)

        sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
        sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
        sent_clf.fit(sents_X, sents_y)
        d.set_sent_model(sent_clf, s.vectorizer)
        d.vectorize(test_domain)

        clf = SGDClassifier(loss="hinge", penalty="L2", alpha=0.1, class_weight={1: 1, -1: 1})
        X_all, y_all = d.X_y_uid_filtered(domain_uids, test_domain)

        clf.fit(X_all, y_all)

        print show_most_informative_features(s.vectorizer, clf)




def binary_hybrid_doc_prediction_test2(model=HybridDocModel(test_mode=True)):

    print "Binary hybrid doc level prediction version 2 (maybe quicker!!)"
    print "=" * 40
    print


    d = model
    d.generate_data(binarize=True) # some variations use the quote data internally 
                                                # for sentence prediction (for additional features)


    for test_domain in CORE_DOMAINS:

        

        print ("*"*40) + "\n\n" + test_domain + "\n\n" + ("*" * 40)
        
        domain_uids = d.domain_uids(test_domain)
        no_studies = len(domain_uids)
        kf = KFold(no_studies, n_folds=5, shuffle=False)
        tuned_parameters = {"alpha": np.logspace(-4, -1, 10), "class_weight": [{1: i, -1: 1} for i in np.logspace(-1, 1, 10)]}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='f1')
        

        metrics = []

        s = SentenceModel(test_mode=True)
        s.generate_data(uid_filter=domain_uids)
        s.vectorize()

        

        for fold_i, (train, test) in enumerate(kf):

            

            sents_X, sents_y = s.X_y_uid_filtered(domain_uids[test], test_domain)
            # sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
            # sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')


            sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}]
            sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2", class_weight={1:5, -1:1}), sent_tuned_parameters, scoring='recall')


            sent_clf.fit(sents_X, sents_y)
            d.set_sent_model(sent_clf, s.vectorizer)
            d.vectorize(test_domain)


            X_train, y_train = d.X_y_uid_filtered(domain_uids[train], test_domain)
            X_test, y_test = d.X_y_uid_filtered(domain_uids[test], test_domain)



            clf.fit(X_train, y_train)

        
            # print show_most_informative_features(s.vectorizer, clf.best_estimator_)
            print show_most_informative_features(s.vectorizer, clf)

        #     y_preds = clf.predict(X_test)

        #     fold_metric = np.array(sklearn.metrics.precision_recall_fscore_support(y_test, y_preds))[:,1]

        #     metrics.append(fold_metric) # get the scores for positive instances

        #     print "fold %d:\tprecision %.2f, recall %.2f, f-score %.2f" % (fold_i, fold_metric[0], fold_metric[1], fold_metric[2])
            

        #     metrics.append(fold_metric) # get the scores for positive instances






        # summary_metrics = np.mean(metrics, axis=0)
        # print "=" * 40
        # print "mean score:\tprecision %.2f, recall %.2f, f-score %.2f" % (summary_metrics[0], summary_metrics[1], summary_metrics[2])






        # # then train all for most informative features

        # sents_X, sents_y = s.X_domain_all(domain=test_domain), s.y_domain_all(domain=test_domain)

        # sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
        # sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
        # sent_clf.fit(sents_X, sents_y)


        # d.set_sent_model(sent_clf, s.vectorizer)
        # d.vectorize(test_domain)

        # clf = SGDClassifier(loss="hinge", penalty="L2", alpha=0.5, class_weight={1: 1, -1: 1})
        # X_all, y_all = d.X_y_uid_filtered(domain_uids, test_domain)
        # clf.fit(X_all, y_all)

        



def main():
    # true_hybrid_prediction_test(model=HybridModelProbablistic(test_mode=False))
    # sentence_prediction_test(model=SentenceModel(test_mode=False))
    # simple_hybrid_prediction_test(model=HybridModel(test_mode=False))
    # binary_doc_prediction_test()
    #print "Try weighting sentences better"
    #binary_hybrid_doc_prediction_test2()
    # binary_hybrid_doc_prediction_test()

    # hybrid_doc_prediction_test(model=HybridDocModel2(test_mode=False))
    # document_prediction_test(model=DocumentLevelModel(test_mode=False))

    # multitask_document_prediction_test(model=MultiTaskDocumentModel(test_mode=True))
    multitask_hybrid_document_prediction_test(model=MultiTaskHybridDocumentModel(test_mode=False))

if __name__ == '__main__':
    main()

