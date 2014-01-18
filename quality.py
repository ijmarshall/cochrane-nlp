#####################################################
#                                                   #
#   Predicting risk of bias from full text papers   #
#                                                   #
#####################################################




from tokenizer import sent_tokenizer, word_tokenizer
import biviewer
import re
import progressbar
import collections

from unidecode import unidecode

import yaml
import numpy as np


QUALITY_QUOTE_REGEX = re.compile("Quote\:\s*[\'\"](.*?)[\'\"]")


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







def main():


    q = QualityQuoteReader()

    y = []
    X_words = []

    domains = q.domains()

    test_domain = "Blinding of participants and personnel"

    print domains


    counter = 0

    for i, study in enumerate(q):

        for domain in study.cochrane["QUALITY"]:
            if domain["DOMAIN"] == test_domain:
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

    print "Finished! %d studies included domain %s" % (counter, test_domain)





def test_pdf_cache():

    pdfviewer = biviewer.PDFBiViewer()
    pdfviewer.cache_pdfs()






if __name__ == '__main__':
    main()
    # test_pdf_cache()