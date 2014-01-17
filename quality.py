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

import yaml


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
                if re.match('Quote:', domain['DESCRIPTION']):
                    domain["DOMAIN"] = self.domain_map[domain["DOMAIN"]] # map domain titles to our core categories
                    quality_quotes.append(domain)

            if quality_quotes:
                yield self.BiviewerView(cochrane={"QUALITY": quality_quotes}, studypdf=self.preprocess_pdf(study.studypdf))
                # returns only the quality data with quotes in it for ease of use; preprocesses pdf text


    def preprocess_pdf(self, pdftext):
        pdftext = re.sub("\n", " ", pdftext) # preprocessing rule 1
        return pdftext




def main():


    q = QualityQuoteReader()

    for i, study in enumerate(q):
        print i
        for domain in study.cochrane["QUALITY"]:

            quote = re.search("Quote\: ?[\'\"](.*?)[\'\"]", domain["DESCRIPTION"]).group(1)

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

                # best_indices = []
                # for ranking in rankings:
                #     if ranking[0] > 50:
                #         best_indices.append(ranking[1])

                # print quote

                # print len(best_indices), "indices matched"
                
                # for ind in best_indices:
                    # print pdf_tokens[ind]

                    





        if i > 5:
            break



def test_pdf_cache():

    pdfviewer = biviewer.PDFBiViewer()
    pdfviewer.cache_pdfs()






if __name__ == '__main__':
    main()
    # test_pdf_cache()