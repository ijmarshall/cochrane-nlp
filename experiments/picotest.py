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
    word_set = set(re.split('[^a-z]+', text))
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)





def main(arg=None):
    p = biviewer.PDFBiViewer()

    p_max = len(p) - 1
    p_i = random.randint(0, p_max)

    pdf = p[p_i].studypdf['text']
    pdf_sents = sent_tokenize(pdf)

    for part in ["CHAR_PARTICIPANTS"]:#, "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]:

        print part
        print "*" * 40

        t = p[p_i].cochrane["CHARACTERISTICS"][part]
        cdsr_words = word_list(t)

        # print cdsr_words

        
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


        max_val = max(intersects)
        max_indices = [i for i, j in enumerate(intersects) if j == max_val]

        print "Text from CDSR:"
        

        
        print t
        print

        print "Text from PDF:"
        
        
        for v in max_indices:
            print pdf_sents[v]
        print


if __name__ == '__main__':
    arg = None
    try:
        arg = sys.argv[1]
    except:
        pass

    main(arg=arg)