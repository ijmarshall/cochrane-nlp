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

import yaml
import numpy as np

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import SGDClassifier

QUALITY_QUOTE_REGEX = re.compile("Quote\:\s*[\'\"](.*?)[\'\"]")

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]
                # there is a seventh domain "Other", but not listed here since covers multiple areas
                # see data/domain_names.txt for various other criteria
                # all of these are available via QualityQuoteReader



def word_sent_tokenize(raw_text):
    return [(word_tokenizer.tokenize(sent)) for sent in sent_tokenizer.tokenize(raw_text)]


def load_domain_map(filename="data/domain_names.txt"):

    with open(filename, 'rb') as f:
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

    def __init__(self):
        self.BiviewerView = collections.namedtuple('BiViewer_View', ['cochrane', 'studypdf'])
        self.pdfviewer = biviewer.PDFBiViewer()
        self.domain_map = load_domain_map()


    def __iter__(self):
        """
        run through PDF/Cochrane data, and return filtered data of interest
        preprocesses PDF text
        and maps domain title to one of the core Risk of Bias domains if possible
        """

        for study in self.pdfviewer:

            quality_quotes = []
            quality_data = study.cochrane["QUALITY"]

            for domain in quality_data:
                domain['DESCRIPTION'] = self.preprocess_cochrane(domain['DESCRIPTION'])
                if QUALITY_QUOTE_REGEX.match(domain['DESCRIPTION']):
                    domain["DOMAIN"] = self.domain_map[domain["DOMAIN"]] # map domain titles to our core categories
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

def _get_study_level_X_y(test_domain=CORE_DOMAINS[0]):
    '''
    return X, y for the specified test domain. here
    X will be of dimensionality equal to the number of 
    studies for which we have the test_domain data. 
    '''
    X, y = [], []
    #study_counter = 0
    q = QualityQuoteReader()

    map_lbl = lambda lbl: 1 if lbl=="YES" else -1

    for i, study in enumerate(q):
        domain_in_study = False
        pdf_tokens = study.studypdf#_simple_BoW(study)
            

        for domain in study.cochrane["QUALITY"]:
            # note that the 2nd clause deals with odd cases 
            # in which a domain is *repeated* for a study,
            if domain["DOMAIN"] == test_domain and not domain_in_study:
                domain_in_study = True
                #study_counter += 1
                #pdf_tokens = word_sent_tokenize(study.studypdf)

                X.append(pdf_tokens)

                quality_rating = domain["RATING"]
                #### for now lump 'unknown' together with 
                #### 'no'
                if quality_rating == "UNKNOWN":
                    quality_rating = "NO"


                y.append(map_lbl(quality_rating))

                
        if not domain_in_study:
            #y.append("MISSING")
            pass
            
        #if len(y) != len(X):
        #    pdb.set_trace()
        
    #pdb.set_trace()
    vectorizer = CountVectorizer(max_features=10000)
    Xvec = vectorizer.fit_transform(X)            

    return Xvec, y

def predict_domains_for_documents():
    X, y = _get_study_level_X_y()

    # note that asarray call below, which seems necessary for 
    # reasons that escape me (see here 
    # https://github.com/scikit-learn/scikit-learn/issues/2508)
    clf = SGDClassifier(loss="hinge", penalty="l2")
    cv_res = cross_validation.cross_val_score(
                clf, X, numpy.asarray(y), 
                score_func=sklearn.metrics.f1_score, cv=5)
    
    
    



def main():
    q = QualityQuoteReader()

    y = []
    X_words = []

    domains = q.domains()

    test_domain = CORE_DOMAINS[1]


    counter = 0

    for i, study in enumerate(q):
        for domain in study.cochrane["QUALITY"]:
            if domain["DOMAIN"] == test_domain:
                #pdb.set_trace()
                try:
                    quote = QUALITY_QUOTE_REGEX.search(domain["DESCRIPTION"]).group(1)
                except:
                    print "Unable to extract quote:"
                    print domain["DESCRIPTION"]
                    raise

                quote_tokens = word_sent_tokenize(quote)
                pdf_tokens = word_sent_tokenize(study.studypdf)

                
                for quote_i, quote_sent in enumerate(quote_tokens):

                    quote_sent_bow = set((word.lower() for word in quote_sent))

                    rankings = []

                    for pdf_i, pdf_sent in enumerate(pdf_tokens):
                    
                        pdf_sent_bow = set((word.lower() for word in pdf_sent))

                        if not pdf_sent_bow or not quote_sent_bow:
                            prop_quote_in_sent = 0
                        else:
                            prop_quote_in_sent = 100* (1 - (float(len(quote_sent_bow-pdf_sent_bow))/float(len(quote_sent_bow))))

                        # print "%.0f" % (prop_quote_in_sent,)

                        rankings.append((prop_quote_in_sent, pdf_i))

                    rankings.sort(key=lambda x: x[0], reverse=True)
                    best_match_index = rankings[0][1]
                    print quote
                    print pdf_tokens[best_match_index]

                    y_study = np.zeros(len(pdf_tokens))
                    y_study[best_match_index] = 1

                    y.append(y_study)
                    X_words.extend(pdf_tokens)
                    
                    counter += 1

                    print y_study
                break 

    y = np.array(y).flatten()
    pdb.set_trace()
    print "Finished! %d studies included domain %s" % (counter, test_domain)





def test_pdf_cache():

    pdfviewer = biviewer.PDFBiViewer()
    pdfviewer.cache_pdfs()




if __name__ == '__main__':
    main()
    # test_pdf_cache()