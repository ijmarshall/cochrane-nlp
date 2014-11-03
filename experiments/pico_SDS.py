#
#   testing methods of distant supervision from CDSR to PDF
#

import re
import random
import sys
import csv 
import pdb
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

from cochranenlp.readers import biviewer

def word_list(text):
    text = text.lower()
    word_set = set(re.split('[^a-z]+', text))
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)


def output_data_for_labeling(N=5, output_file_path="for_labeling.csv", cutoff=4):
    ''' '''
    with open(output_file_path, 'wb') as outf:
        csv_writer = csv.writer(outf, delimiter=",")

        csv_writer.writerow(
            ["study id", "PICO field", "CDSR sentence", 
            "candidate sentence", "rating"])

        for i in xrange(N):
            p = biviewer.PDFBiViewer()
            p_max = len(p) - 1
            p_i = random.randint(0, p_max)
            
            pdf = p[p_i].studypdf['text']
            study_id = "%s" % p[p_i][1]['pmid']

            best_sentences = []
            pdf_sents = sent_tokenize(pdf)

            for pico_field in ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]:

                # i.e., number of shared tokens
                sentence_scores = []
                # sentence texts
                sentences = []

                # target sentence
                t = p[p_i].cochrane["CHARACTERISTICS"][pico_field]
                cdsr_words = word_list(t)

                for i, sent in enumerate(pdf_sents):
                    sent_words = word_list(sent)
                    sentences.append(sent)
                    sentence_scores.append(len(cdsr_words.intersection(sent_words)))

                ranked_sentences, scores = [list(x) for x in zip(*sorted(
                        zip(sentences, sentence_scores), 
                        key=itemgetter(1), reverse=True))]

                # don't take more than 10 sentences
                num_to_keep = min(len([score for score in scores if score >= cutoff]), 10)
                ## what to do if this is zero?
                if num_to_keep == 0:
                    print "no sentences passed threshold!"
                else:
                    for candidate in ranked_sentences[:num_to_keep]:
                        csv_writer.writerow([
                            study_id, pico_field.replace("CHAR_", " "), t, candidate, ""])

           


if __name__ == '__main__':
    arg = None
    try:
        arg = sys.argv[1]
    except:
        pass

    main(arg=arg)