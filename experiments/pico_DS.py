#
#   Derives distant supervision for PICO.
#
#   @TODO this should almost certainly live in the sds
#   package... or something? On the other hand, this is
#   lacks the "s" in "sds": perhaps we should refactor
#   and create a DS package somewhere?
#


import re
import random
import sys
import csv
import pdb
from operator import itemgetter
from collections import Counter
import cPickle as pickle
import string

import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix, csc_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# cochranenlp stuff
import cochranenlp.ml.pico_vectorizer
from cochranenlp.ml.pico_vectorizer import PICO_vectorizer
from cochranenlp.output import progressbar
from cochranenlp.readers import biviewer
from cochranenlp.textprocessing.indexnumbers import NumberTagger
numberswap = NumberTagger().swap

import cochranenlp
DATA_PATH = cochranenlp.config["Paths"]["base_path"] # to data


import sys
reload(sys)

import os

sys.setdefaultencoding('utf8')


# for generating DS features from sentences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize



PICO_DOMAINS = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

def word_list(text, old_style=True):

    if old_style:
        return word_list_old_style(text)
    text = text.lower()

    text = numberswap(text)
    word_set = set(re.split('[^a-z0-9]+', text)) 
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)

def word_list_old_style(text):
    text = text.lower()
    word_set = set(re.split('[^a-z]+', text)) 
    stop_set = set(stopwords.words('english'))
    return word_set.difference(stop_set)


def all_PICO_DS(cutoff=4, max_sentences=10, add_vectors=True, pickle_DS=True):
    '''
    Generates all available `labeled' PICO training data via naive
    DS strategy of token matching and returns a nested dictionary;
    the top level are PICO domains, and sentences / `labels' are
    nested beneath these.

    If the vectorize flag is True, then we return (sparse)
    feature representations of the sentences.
    @TODO @TODO
    should probably better engineer these features! e.g., should
    at least swap nums in, and probably other tricks...

    WARNING this is pretty slow and there is a lot of data!
    you should set the pickle_DS flag to True and to dump this
    data to disk and then read it in directly rather than
    re-generating it every time.
    '''
    

    ###
    # for now we'll just grab sentences and `labels' according
    # to simple criteria.
    ###
    sentences_y_dict = {
        domain: {"sentences":[], "y":[], "pmids":[], "CDSR_id":[], "positional_features":[]} for
            domain in PICO_DOMAINS}


    

    p = biviewer.PDFBiViewer()


    progress = progressbar.ProgressBar(len(p), timer=True)

    for n, study in enumerate(p):
        progress.tap()


        pdf = study.studypdf['text'].decode("utf8", errors="ignore")
        
    
        study_id = "%s" % study[1]['pmid']

        pdf_sents = sent_tokenize(pdf)

        if len(pdf_sents) < 10: # these are all junk
            continue

        cochrane_id = study.cochrane['cdsr_filename']
        for pico_field in PICO_DOMAINS:
            ranked_sentences_and_scores = get_ranked_sentences_for_study_and_field(study, pico_field, pdf_sents=pdf_sents, get_positional_features=True)

            # in this case, there was no supervision in the
            # CDSR so we just keep on moving
            if ranked_sentences_and_scores is None:
                #
                # IM: note that ranked_sentences_and_scores returns None
                # where empty CDSR strings, but returns empty lists
                # where empty PDF strings (i.e. no sentences)
                # this case is caught later on

                pass
            

            else:
                ####
                # then add all sentences that meet our DS
                # criteria as positive examples; all others
                # are -1.
                ###

                # the :2 throws away the shared tokens here.
                ranked_sentences, scores = ranked_sentences_and_scores[:2]

                positional_features = ranked_sentences_and_scores[3]

                pos_count = 0 # place an upper-bound on the number of positive instances.

                for sent, score, position in zip(ranked_sentences, scores, positional_features):
                    sentences_y_dict[pico_field]["sentences"].append(sent) # IM: why y_dict??
                    sentences_y_dict[pico_field]["positional_features"].append(position)


                    # IM: Note that we're potentially including docs with all
                    # negative sentences (where none pass the threshold)
                    # though this seems to not happen much anecdotally.
                    # may wish to exclude later

                    cur_y = -1
                    if pos_count < max_sentences and score >= cutoff:
                        cur_y = 1
                        pos_count += 1
                    sentences_y_dict[pico_field]["y"].append(cur_y)
                    sentences_y_dict[pico_field]["pmids"].append(study_id)
                    sentences_y_dict[pico_field]["CDSR_id"].append(cochrane_id)

    # add vector representations to the dictionary
    if add_vectors:


        sentences_y_dict, domain_vectorizers = vectorize(sentences_y_dict)


        if pickle_DS:

            print "pickling..."
            with open(os.path.join(DATA_PATH, "sds_sentence_data.pickle"), 'wb') as outf:
                pickle.dump(sentences_y_dict, outf)

            with open(os.path.join(DATA_PATH, "sds_vectorizers.pickle"), 'wb') as outf:
                pickle.dump(domain_vectorizers, outf)
            print "done!"

        return sentences_y_dict, domain_vectorizers

    return sentences_y_dict


def vectorize(sentences_y_dict):
    '''
    Vectorize the sentences in each pico domain and
    mutate the sentences_y_dict parameter by adding
    these representations.
    '''
    domain_vectorizers = {}
    for domain in PICO_DOMAINS:

        all_sentences = sentences_y_dict[domain]["sentences"]

        domain_vectorizers[domain] = PICO_vectorizer()
        sentences_y_dict[domain]["X"] = domain_vectorizers[domain].fit_transform(all_sentences, extra_features=sentences_y_dict[domain]["positional_features"])

    
    return sentences_y_dict, domain_vectorizers


def output_data_for_labeling(N=7, output_file_path="for_labeling-2-24-15_brian.csv", 
                                cutoff=4, max_sentences=10, exclude_list=None):
    ''' generate a CSV file for labeling matches '''

    if exclude_list is None:
        exclude_list = []

    with open(output_file_path, 'wb') as outf:
        csv_writer = csv.writer(outf, delimiter=",")

        csv_writer.writerow(
            ["study id", "PICO field", "CDSR sentence",
            "candidate sentence", "rating"])

        #for i in xrange(N):
        count = 0
        while count < N:
            p = biviewer.PDFBiViewer()
            p_max = len(p) - 1

            # randomly select a study
            p_i = random.randint(0, p_max)

            # store details, sentence-tokenize
            study = p[p_i]
            pdf = study.studypdf['text']

            
            study_id = "%s" % study[1]['pmid']
          
            if int(study_id) not in exclude_list:
                count += 1
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
                        num_to_keep = min(len([score for score in scores if score >= cutoff]),
                                            max_sentences)
                        # IM: max_sentences defaults to 10
                        # cutoff defaults to 4 (chopping off the most uninformative left side bulk of the histogram)
                        # therefore we're keeping up to a max of 10 sents
                        # from those with >= 4 token matches


                        ## what to do if this is zero?
                        if num_to_keep == 0:
                            print "no sentences passed threshold!"
                        else:
                            target_text = study.cochrane["CHARACTERISTICS"][pico_field]
                            for candidate in ranked_sentences[:num_to_keep]:
                                csv_writer.writerow([
                                    study_id, pico_field.replace("CHAR_", " "),
                                                    target_text, candidate, ""])


def get_ranked_sentences_for_study_and_field(study, PICO_field, pdf_sents=None, get_positional_features=False):
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
        pdf = study.studypdf['text'].decode("utf-8", errors="ignore")
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



    if get_positional_features:

        # generate positional features here (quintiles)
        num_sents = len(sentences)
        quintile_cutoff = num_sents / 5 # the integer

        if quintile_cutoff == 0:
            sentence_quintiles = [{"DocTooSmallForQuintiles" : 1} for ii in xrange(num_sents)]
            print "tiny file encountered... len=%d" % num_sents


        else:
            sentence_quintiles = [{"DocumentPositionQuintile%d" % (ii/quintile_cutoff): 1} for ii in xrange(num_sents)]

        ranked_sentences, scores, positions = [list(x) for x in zip(*sorted(zip(sentences, sentence_scores, sentence_quintiles), key=lambda x: x[1], reverse=True))] or [[],[],[]]

        return ranked_sentences, scores, shared_tokens, positions
    else:
        ranked_sentences, scores = [list(x) for x in zip(*sorted(zip(sentences, sentence_scores), key=lambda x: x[1], reverse=True))] or [[],[]]
        return ranked_sentences, scores, shared_tokens

    #  IM to check: ranked_sentences = [] and scores = [] if no sentences in doc??

    

def main(arg):

    all_PICO_DS()



if __name__ == '__main__':
    arg = None
    try:
        arg = sys.argv[1]
    except:
        pass

    main(arg=arg)
