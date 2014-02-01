#####################################################
#                                                   #
#   Predicting risk of bias from full text papers   #
#                                                   #
#####################################################
import pdb 

from tokenizer import sent_tokenizer, word_tokenizer
import biviewer
import re
import progressbar
import collections
import string
from unidecode import unidecode
import codecs

import yaml
import numpy as np
import math

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import SGDClassifier

from sklearn.cross_validation import KFold



from collections import Counter

QUALITY_QUOTE_REGEX = re.compile("Quote\:\s*[\'\"](.*?)[\'\"]")

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]
                # there is a seventh domain "Other", but not listed here since covers multiple areas
                # see data/domain_names.txt for various other criteria
                # all of these are available via QualityQuoteReader



def word_sent_tokenize(raw_text):
    return [(word_tokenizer.tokenize(sent)) for sent in sent_tokenizer.tokenize(raw_text)]


def describe_data():

    perdomain_output = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]
    perdomain_quotes = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]

    overall_output = Counter()
    overall_quotes = Counter()

    q = QualityQuoteReader()

    for i, study in enumerate(q):
            
        for domain in study.cochrane["QUALITY"]:

            domain_text = domain["DOMAIN"].replace("\xc2\xa0", " ")

            if domain_text in CORE_DOMAINS:
                domain_index = CORE_DOMAINS.index(domain_text)
            else:
                domain_index = 6 # other

            perdomain_output[domain_index][domain["RATING"]] += 1
            overall_output[domain["RATING"]] += 1

            if QUALITY_QUOTE_REGEX.match(domain['DESCRIPTION']):
                perdomain_quotes[domain_index][domain["RATING"]] += 1
                overall_quotes[domain["RATING"]] += 1

    print
    print "ALL"
    for domain, counts in zip(CORE_DOMAINS + ["OTHER"], perdomain_output):
        print
        print domain
        print
        print counts

    print
    print "OVERALL"
    print
    print overall_output



def flatten_list(l):
    return [item for sublist in l for item in sublist]


def sublist(l, indices):
    if isinstance(indices, tuple):
        indices = [indices]

    output = [l[start: end] for (start, end) in indices]
    return flatten_list(output)

def np_indices(indices):
    if isinstance(indices, tuple):
        indices = [indices]

    output = [np.arange(start, end) for (start, end) in indices]
    return np.hstack(output)





def show_most_informative_features(vectorizer, clf, n=50):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))

    if n == 0:
        n = len(c_f)/2

    top = zip(c_f[:n], c_f[:-(n+1):-1])
    print
    print "%d most informative features:" % (n, )
    print
    for (c1, f1), (c2, f2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2)


def load_domain_map(filename="data/domain_names.txt"):

    with codecs.open(filename, 'rb', 'utf-8') as f:
        raw_data = yaml.load(f)

    mapping = {}

    for key, value in raw_data.iteritems():
        for synonym in value:
            mapping[synonym] = key

    return mapping

class QualityQuoteReader():
    """
    iterates through Cochrane Risk of Bias information for domains where there is a quote only
    """

    def __init__(self, quotes_only=True):
        self.BiviewerView = collections.namedtuple('BiViewer_View', ['cochrane', 'studypdf'])
        self.pdfviewer = biviewer.PDFBiViewer()
        self.domain_map = load_domain_map()
        self.quotes_only = quotes_only


    def __iter__(self):
        """
        run through PDF/Cochrane data, and return filtered data of interest
        preprocesses PDF text
        and maps domain title to one of the core Risk of Bias domains if possible
        """

        used_pmids = set()

        p = progressbar.ProgressBar(len(self.pdfviewer), timer=True)

        for study in self.pdfviewer:

            p.tap()

            quality_quotes = []
            quality_data = study.cochrane["QUALITY"]


            for domain in quality_data:
                domain['DESCRIPTION'] = self.preprocess_cochrane(domain['DESCRIPTION'])
                if QUALITY_QUOTE_REGEX.match(domain['DESCRIPTION']) or not self.quotes_only:
                    domain_text = domain["DOMAIN"].replace("\xc2\xa0", " ")

                    try:
                        domain["DOMAIN"] = self.domain_map[domain_text] # map domain titles to our core categories
                    except:
                        domain["DOMAIN"] = "UNMAPPED"
                    quality_quotes.append(domain)

            if quality_quotes:
                yield self.BiviewerView(cochrane={"QUALITY": quality_quotes}, studypdf=self.preprocess_pdf(study.studypdf))
                # returns only the quality data with quotes in it for ease of use; preprocesses pdf text


    def preprocess_pdf(self, pdftext):
        pdftext = unidecode(pdftext)
        pdftext = re.sub("\n", " ", pdftext) # preprocessing rule 1
        return pdftext

    def preprocess_cochrane(self, cochranetext):
        cochranetext = unidecode(cochranetext)
        return cochranetext

    def domains(self):
        domain_headers = set((value for key, value in self.domain_map.iteritems()))
        return list(domain_headers)






def _get_domains_from_study(study):
    return [domain["DOMAIN"] for domain in study.cochrane["QUALITY"]]

def _simple_BoW(study):
    return [s for s in word_tokenizer.tokenize(study.studypdf) 
                if not s in string.punctuation]

def _get_study_level_X_y(test_domain=CORE_DOMAINS[4]):
    '''
    return X, y for the specified test domain. here
    X will be of dimensionality equal to the number of 
    studies for which we have the test_domain data. 
    '''
    X, y = [], []
    #study_counter = 0
    q = QualityQuoteReader(quotes_only=False)


    map_lbl = lambda lbl: 1 if lbl=="YES" else -1

    for i, study in enumerate(q):
        domain_in_study = False
        pdf_tokens = study.studypdf
            

        for domain in study.cochrane["QUALITY"]:

            quality_rating = domain["RATING"]
            #### for now skip unknowns, test YES v NO
            if quality_rating == "UNKNOWN":
                quality_rating = "NO"
                # break

            # note that the 2nd clause deals with odd cases 
            # in which a domain is *repeated* for a study,
            if domain["DOMAIN"] == test_domain and not domain_in_study:

                domain_in_study = True
                #study_counter += 1
                #pdf_tokens = word_sent_tokenize(study.studypdf)

                X.append(pdf_tokens)
                y.append(map_lbl(quality_rating))

        
                
        if not domain_in_study:
            #y.append("MISSING")
            pass
            

        #if len(y) != len(X):
        #    pdb.set_trace()
        
    #pdb.set_trace()
    vectorizer = CountVectorizer(max_features=10000)
    Xvec = vectorizer.fit_transform(X)            

    return Xvec, y, vectorizer

def predict_domains_for_documents():
    X, y, vec = _get_study_level_X_y()

    # note that asarray call below, which seems necessary for 
    # reasons that escape me (see here 
    # https://github.com/scikit-learn/scikit-learn/issues/2508)
    clf = SGDClassifier(loss="hinge", penalty="l2", class_weight={1:10, 0:1})






    cv_res = cross_validation.cross_val_score(
                clf, X, np.asarray(y), 
                score_func=sklearn.metrics.f1_score, cv=5)

    print cv_res

    ### train on all
    model = clf.fit(X, y)
    print show_most_informative_features(vec, model, n=50)
    

def predict_sentences_reporting_bias():
    X, y, X_sents, vec, study_sent_indices = _get_sentence_level_X_y()
    
    

    clf = SGDClassifier(loss="hinge", penalty="l2")


    kf = KFold(len(study_sent_indices), n_folds=5, shuffle=True)

    for fold_i, (train, test) in enumerate(kf):

        print "making test sentences"
        
        test_indices = [study_sent_indices[i] for i in test]
        train_indices = [study_sent_indices[i] for i in train]

        X_sents_test = sublist(X_sents, test_indices)
        # [X_sents[i] for i in test]
        
        print "done!"

        print "generating split"
        X_train = X[np_indices(train_indices)]
        y_train = y[np_indices(train_indices)]
        X_test = X[np_indices(test_indices)]
        y_test = y[np_indices(test_indices)]
        print "done!"

        print "fitting model..."
        clf.fit(X_train, y_train)
        print "done!"

        print "making predictions"
        y_preds = clf.predict(X_test)
        print "done!"
        

        f1 = sklearn.metrics.f1_score(y_test, y_preds)
        recall = sklearn.metrics.recall_score(y_test, y_preds)
        precision = sklearn.metrics.precision_score(y_test, y_preds)

        print "fold %d:\tf1: %.2f\trecall: %.2f\tprecision: %.2f" % (fold_i, f1, recall, precision)

        for start, end in test_indices:

            study_X = X[np_indices((start, end))]
            study_y = y[np_indices((start, end))]

            pred_probs = clf.decision_function(study_X)

            max_index = np.argmax(pred_probs) + start

            real_index = np.where(study_y==1)[0][0] + start

            print "Max distance +ve %.2f:\n%s\n" % (np.max(pred_probs), X_sents[max_index])

            print "Actual answer:\n%s\n\n" % (X_sents[real_index])


            # min_index = np.argmin(pred_probs) + start

            # print "Max distance -ve %.2f:\n%s\n\n" % (np.min(pred_probs), X_sents[min_index])





        # for i, (y_pred, sent) in enumerate(zip(y_preds, X_sents_test)):

        #     if y_pred > 0:
        #         print
        #         print "%d: %s" % (i, sent)







    # cv_res = cross_validation.cross_val_score(
    #             clf, X, np.asarray(y), 
    #             score_func=sklearn.metrics.recall_score, cv=5)

    # print cv_res

    model = clf.fit(X, y)
    print show_most_informative_features(vec, model, n=50)


    


def _get_sentence_level_X_y(test_domain=CORE_DOMAINS[0]):
    q = QualityQuoteReader()
    y = []
    X_words = []
    
    study_sent_indices = [] # list of (start, end) indices corresponding to each study
    sent_index_counter = 0


    domains = q.domains()
    counter = 0

    for i, study in enumerate(q):

        # fast forward to the matching domain
        for domain in study.cochrane["QUALITY"]:
            if domain["DOMAIN"] == test_domain:
                break
        else:
            # if no matching domain continue to the next study
            continue


        try:
            quote = QUALITY_QUOTE_REGEX.search(domain["DESCRIPTION"]).group(1)
        except:
            print "Unable to extract quote:"
            print domain["DESCRIPTION"]
            raise

        quote_words = word_tokenizer.tokenize(quote)
        pdf_sents = sent_tokenizer.tokenize(study.studypdf)

 
        quote_sent_bow = set((word.lower() for word in quote_words))

        rankings = []

        for pdf_i, pdf_sent in enumerate(pdf_sents):

            pdf_words = word_tokenizer.tokenize(pdf_sent)
        
            pdf_sent_bow = set((word.lower() for word in pdf_words))

            if not pdf_sent_bow or not quote_sent_bow:
                prop_quote_in_sent = 0
            else:
                prop_quote_in_sent = 100* (1 - (float(len(quote_sent_bow-pdf_sent_bow))/float(len(quote_sent_bow))))

            # print "%.0f" % (prop_quote_in_sent,)

            rankings.append((prop_quote_in_sent, pdf_i))

        rankings.sort(key=lambda x: x[0], reverse=True)
        best_match_index = rankings[0][1]
        # print quote
        # print pdf_tokens[best_match_index]

        y_study = np.zeros(len(pdf_sents))
        y_study[best_match_index] = 1

        y.extend(y_study)
        X_words.extend(pdf_sents)
        sent_end_index = sent_index_counter + len(pdf_sents)
        study_sent_indices.append((sent_index_counter, sent_end_index))
        sent_index_counter = sent_end_index



                    
                    
                


    print len(X_words)
    print X_words[0]

    print "fitting vectorizer"
    vectorizer = CountVectorizer(max_features=10000)
    X = vectorizer.fit_transform(X_words)            
    print "done!"
    y = np.array(y)

    return X, y, X_words, vectorizer, study_sent_indices


    

    print "Finished! %d studies included domain %s" % (counter, test_domain)




def test_pdf_cache():

    pdfviewer = biviewer.PDFBiViewer()
    pdfviewer.cache_pdfs()




if __name__ == '__main__':
    # predict_domains_for_documents()
    # test_pdf_cache()
    predict_sentences_reporting_bias()
    # getmapgaps()
