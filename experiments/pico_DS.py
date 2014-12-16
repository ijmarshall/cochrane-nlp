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

from readers import biviewer

PICO_DOMAINS = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

def word_list(text):
    text = text.lower()
    word_set = set(re.split('[^a-z]+', text))
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)

def all_PICO_DS(cutoff=4, max_sentences=10):
    ###
    # for now we'll just grab sentences and `labels' according
    # to simple criteria.
    ###
    X_y_dict = {domain: {"sentences":[], "y":[]} for domain in PICO_DOMAINS}
    p = biviewer.PDFBiViewer()
    for n, study in enumerate(p):
        if n % 100 == 0:
            print "on study %s" % n 

        ### TMP TMP TMP
        if n > 300:
            return X_y_dict
            
        pdf = study.studypdf['text']
        study_id = "%s" % study[1]['pmid']
        pdf_sents = sent_tokenize(pdf)

        for pico_field in PICO_DOMAINS:
            ranked_sentences_and_scores = \
                    get_ranked_sentences_for_study_and_field(study, 
                                pico_field, pdf_sents=pdf_sents)
    
                
            # in this case, there was no supervision in the
            # CDSR so we just keep on moving
            if ranked_sentences_and_scores is None:
                pass 
            else:
                #### 
                # then add all sentences that meet our DS
                # criteria as positive examples; all others 
                # are -1. 
                ###

                # the :2 throws away the shared tokens here.
                ranked_sentences, scores = ranked_sentences_and_scores[:2]
                # don't take more than max_sentences sentences
                #num_to_keep = min(len([score for score in scores if score >= cutoff]), max_sentences)

                pos_count = 0 # place an upper-bound on the number of pos. instances.
                for sent, score in zip(ranked_sentences, scores):
                    X_y_dict[pico_field]["sentences"].append(sent)
                    cur_y = -1
                    if pos_count < max_sentences and score >= cutoff:
                        cur_y = 1
                        pos_count += 1  
                    X_y_dict[pico_field]["y"].append(cur_y)

    return X_y_dict

def output_data_for_labeling(N=5, output_file_path="for_labeling.csv", cutoff=4, max_sentences=10):
    ''' generate a CSV file for labeling matches '''

    with open(output_file_path, 'wb') as outf:
        csv_writer = csv.writer(outf, delimiter=",")

        csv_writer.writerow(
            ["study id", "PICO field", "CDSR sentence", 
            "candidate sentence", "rating"])

        for i in xrange(N):
            p = biviewer.PDFBiViewer()
            p_max = len(p) - 1

            # randomly select a study
            p_i = random.randint(0, p_max)

            # store details, sentence-tokenize
            study = p[p_i]
            pdf = study.studypdf['text']
            study_id = "%s" % study[1]['pmid']
            pdf_sents = sent_tokenize(pdf)

            for pico_field in PICO_DOMAINS:
                ranked_sentences_and_scores = get_ranked_sentences_for_study_and_field(study, 
                            pico_field, pdf_sents=pdf_sents)
                
                # in this case, there was no target label in the
                # CDSR so we just keep on moving
                if ranked_sentences_and_scores is None:
                    pass 
                else:
                    # the :2 throws away the shared tokens here.
                    ranked_sentences, scores = ranked_sentences_and_scores[:2]
                    # don't take more than max_sentences sentences
                    num_to_keep = min(len([score for score in scores if score >= cutoff]), max_sentences)

                
                    ## what to do if this is zero?
                    if num_to_keep == 0:
                        print "no sentences passed threshold!"
                    else:
                        target_text = study.cochrane["CHARACTERISTICS"][pico_field]
                        for candidate in ranked_sentences[:num_to_keep]:
                            csv_writer.writerow([
                                study_id, pico_field.replace("CHAR_", " "), 
                                                target_text, candidate, ""])

              
def get_ranked_sentences_for_study_and_field(study, PICO_field, pdf_sents=None):
    '''
    Given a study (readers.biviewer.Biviewer_View object) and 
    a PICO field (one of: "CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", 
    "CHAR_OUTCOMES"), return sentences in the PDF ranked w.r.t. how 
    closely (in terms of number of shared tokens) they match the 
    target sentence/summary in the CDSR.

    To avoid forcing re-tokenization for each field (in cases
    where one is retrieving relevant sentences for multiple 
    PICO fields), one may optionally pass in pdf_sents; i.e., the
    tokenized sentences within the pdf text. If this is None, 
    tokenization will be performed here.
    '''

    # tokenize if necessary 
    if pdf_sents is None:
        pdf = study.studypdf['text']
        pdf_sents = sent_tokenize(pdf)

    # i.e., number of shared tokens
    sentence_scores = []
    
    # sentence texts
    sentences = []

    # record the shared words
    shared_tokens = []

    t = study.cochrane["CHARACTERISTICS"][PICO_field]
    if t is None:
        # no supervision here!
        return None 

    # this is the target sentence
    t = t.decode("utf-8", errors="ignore")
   
    cdsr_words = word_list(t)

    for i, sent in enumerate(pdf_sents):
        sent = sent.decode("utf-8", errors="ignore")
        sent_words = word_list(sent)
        sentences.append(sent)
        shared_tokens_i = cdsr_words.intersection(sent_words)
        shared_tokens.append(shared_tokens_i)
        sentence_scores.append(len(shared_tokens_i))

    ranked_sentences, scores = [list(x) for x in zip(*sorted(
            zip(sentences, sentence_scores), 
            key=itemgetter(1), reverse=True))]

    return ranked_sentences, scores, shared_tokens


if __name__ == '__main__':
    arg = None
    try:
        arg = sys.argv[1]
    except:
        pass

    main(arg=arg)