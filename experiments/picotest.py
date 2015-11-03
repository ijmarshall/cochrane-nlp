#
#   testing methods of distant supervision from CDSR to PDF
#


from cochranenlp.readers import biviewer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def word_list(text):
    text = text.lower()
    word_set = set(re.split('[\s\.\,\;]+', text))
    # word_set = set(re.split('[^a-z]+', text))
    # word_set = set(re.findall('[^a-z]+', text))
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)

# def num_count(text):
#     text = text.lower()
#     words = set(re.split('[^a-z]+', text))



def main(arg=None):
    # Get an instance of the object that allows you to view cochrane aligned with full-text pdf publications
    p = biviewer.PDFBiViewer()

    # Pick a random publication
    p_max = len(p) - 1
    p_i = random.randint(0, p_max)

    # Get the text of the pdf of that publication
    pdf = p[p_i].studypdf['text']
    pdf = pdf.decode('utf-8')
    pdf_sents = sent_tokenize(pdf)

    for part in ["CHAR_PARTICIPANTS"]:#, "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]:

        print part
        print p[p_i].cochrane['cdsr_filename']
        print "*" * 40

        # Get the population summary from cochrane for this pdf
        t = p[p_i].cochrane["CHARACTERISTICS"][part]
        cdsr_words = word_list(t)

        # print cdsr_words

        # For each sentence in the pdf, count how many words overlap with the cochrane population summary
        intersects = []
        for i, sent in enumerate(pdf_sents):
            sent_words = word_list(sent)
            intersects.append(len(cdsr_words.intersection(sent_words)))
        intersects = np.array(intersects)

        if arg == "plot":
            sns.set_palette("deep", desat=.6)
            sns.set_context(rc={"figure.figsize": (8, 4)})
            sns.distplot(intersects, kde=False)
            # sns.factorplot("frequency", 
                # data=, x_order=range(0, 20))
            plt.show()
        elif arg == "pc":
            intersects_pc = (intersects*100)/len(cdsr_words)
            sns.set_palette("deep", desat=.6)
            sns.set_context(rc={"figure.figsize": (8, 4)})

            sns.distplot(np.array(intersects_pc), kde=False)
            # sns.factorplot("frequency", 
                # data=, x_order=range(0, 20))
            plt.show()


        # Find all the pdf sentences for which the maximum number of words matched
        max_val = max(intersects)
        max_indices = [i for i, j in enumerate(intersects) if j == max_val]

        print "Population summary from CDSR:"
        print t
        print

        print "PDF sentences which have highest overlap:"
        for v in max_indices:
            print pdf_sents[v].replace('\n', ' ')
        print


if __name__ == '__main__':
    arg = None
    try:
        arg = sys.argv[1]
    except:
        pass

    main(arg=arg)
