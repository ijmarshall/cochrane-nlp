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


    def __iter__(self):
        " run through PDF/Cochrane data, and return filtered data of interest "

        for study in self.pdfviewer:

            quality_quotes = []
            quality_data = study.cochrane["QUALITY"]

            for domain in quality_data:
                if re.match('Quote:', domain['DESCRIPTION']):
                    quality_quotes.append(domain)

            if quality_quotes:
                yield self.BiviewerView(cochrane={"QUALITY": quality_quotes}, studypdf=study.studypdf)
                # returns only the quality data with quotes in it for ease of use







def main():

    m = load_domain_map()
    q = QualityQuoteReader()

    for i, study in enumerate(q):
        print i
        for domain in study.cochrane["QUALITY"]:
            print m[domain["DOMAIN"]]
            print
            print domain["DESCRIPTION"]
            print
            print
        if i > 5:
            break



def test_pdf_cache():

    pdfviewer = biviewer.PDFBiViewer()
    pdfviewer.cache_pdfs()






if __name__ == '__main__':
    main()
    # test_pdf_cache()