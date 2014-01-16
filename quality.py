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







def main():
    q = QualityQuoteReader()

    for i, study in enumerate(q):
        print i
        print study
        if i == 10:
            break










if __name__ == '__main__':
    main()