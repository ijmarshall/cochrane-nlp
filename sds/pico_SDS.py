'''
Supervised distant supervision for PICO extraction.

Here we aim to go from the information (direct
distant supervision for PICO task) contained in
the annotations file to feature vectors and labels for the
candidate filtering task. In the writeup nomenclature,
this is to generate \tilde{x} and \tilde{y}.

@TODO Technically to be consistent this module should probably
live in the cochranenlpexperiments lib rather than here
'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pdb
import random
import csv
import sys
import bz2
import cPickle as pickle
import os
import time
import copy
from datetime import datetime
from collections import defaultdict
import itertools 

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC 

from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import pandas as pd 

import hickle 

import cochranenlp
from cochranenlp.readers import biviewer
from cochranenlp.ml.pico_vectorizer import PICO_vectorizer

# this module allows us to grab the ranked
# sentences. this is possibly not the ideal
# location.
from cochranenlp.experiments import pico_DS

DATA_PATH = cochranenlp.config["Paths"]["base_path"] # to data

domains = pico_DS.PICO_DOMAINS



def output_labeled_data(output_dir="sds/results/",
                            y_dict_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_sentence_data10.pickle", 
                            domain_v_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_vectorizers10.pickle"):
    
    sentences_y_dict, domain_vectorizers = _unpickle_PICO_DS(y_dict_pickle, domain_v_pickle)

    output_str = [["row_num", "pmid", "domain", "sentence", "label" ]]

    for domain in domains[1:]:
        print "on domain: %s " % domain
        domain_DS = sentences_y_dict[domain]

        ###
        # and this is where the direct supervision is loaded up
        print "generating DS learning tasks..."

        DS_learning_tasks, z_dict, domains_pmids_targets = get_DS_features_and_labels()
        domain_supervision = DS_learning_tasks[domain]
        
        for i, pmid in enumerate(domain_DS["pmids"]):
            if pmid in domain_supervision["pmids"]:
                cur_sentence = domain_DS["sentences"][i]
                cur_label = _match_direct_to_distant_supervision(
                                domain_supervision, pmid, cur_sentence)
                

                # if this is None, it means the current sentence
                # was not found in the candidate set, implicitly
                # this means it is a -1.
                if cur_label is None:
                    cur_label = "-1 (distant)"
                cur_str = ["%s" % i, "%s" % pmid, domain, cur_sentence, "%s" % cur_label]
                output_str.append(cur_str)

    
    with open(os.path.join(output_dir, "for-ani.csv"), 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(output_str)


def dump_robot(output_dir="sds/results/",                      
                y_dict_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_sentence_data10.pickle", 
                domain_v_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_vectorizers10.pickle"):

    ### 
    # load in DS; we assume this is pickled already.
    print "unpickling DS from %s..." % y_dict_pickle
    sentences_y_dict, domain_vectorizers = _unpickle_PICO_DS(y_dict_pickle,
                                                domain_v_pickle)
    
    ###
    # and this is where the direct supervision is loaded up
    print "generating DS learning tasks..."
    DS_learning_tasks, z_dict, domains_pmids_targets = get_DS_features_and_labels()
    print "done."

    for domain in pico_DS.PICO_DOMAINS:
        print "on domain %s" % domain 
        output_path = os.path.join(output_dir, "%s-labeling.csv" % domain)
        domain_DS = sentences_y_dict[domain]
        domain_supervision = DS_learning_tasks[domain]

    
        ###
        # these were directly labeled.
        labeled_pmids_for_domain = list(
                    set(DS_learning_tasks[domain]["pmids"]))


        ###
        # load in \tilde{X} for SDS
        print "loading X_tilde_dict ... just the once though!"
        X_tilde_dict, target_text_dict = get_DS_features_for_all_data(z_dict)
        print "ok."
        
        train_rows, test_rows = [], []
        directly_supervised_indicators = np.zeros(len(domain_DS["pmids"])).astype("int8")

        unique_pmids = list(set(domain_DS["pmids"]))

        pmids_already_seen = []
        prev_pmid = None
        for i, pmid in enumerate(domain_DS["pmids"]):
            if prev_pmid is None:
                pass 
            elif pmid != prev_pmid:
                pmids_already_seen.append(prev_pmid)

            cur_sentence = domain_DS["sentences"][i]
            
            train_rows.append(i)

            ###
            # here we need to overwrite any labels for which
            # we have explicit supervision!
            if pmid in domain_supervision["pmids"]:

                cur_label = _match_direct_to_distant_supervision(
                    domain_supervision, pmid, cur_sentence)

                # if this is None, it means the current sentence
                # was not found in the candidate set, implicitly
                # this means it is a -1.
                if cur_label is None:
                    cur_label = -1
                else:
                    # we train only on 2's (highly relevant sentences), as we 
                    # are aiming for a high-precision model
                    try:
                        cur_label = _score_to_binary_lbl(cur_label, threshold=2, zero_one=False)
                    except:
                        print "Ah! something wrong with this label."
                        pdb.set_trace()
                domain_DS["y"][i] = cur_label
                directly_supervised_indicators[i] = 1

            # this is to make sure we don't wind up
            # with duplicate instances of documents
            # in the test set
            elif pmid not in pmids_already_seen:
                test_rows.append(i)

            prev_pmid = pmid

        print "huzzah -- data all set up."
        
        ###
        # assemble train and test sets
        X_train_DS = domain_DS["X"][train_rows]
        y_train_DS = np.array(domain_DS["y"])[train_rows]
        pmids_train_DS = [domain_DS["pmids"][train_row] for train_row in train_rows]
        directly_supervised_indicators_train = directly_supervised_indicators[train_rows]

        ###
        # build an SDS classifier
        
        # this transforms the small amount of direct supervision we
        # have for the mapping task from candidate sentences to
        # the best sentences into feature vectors and labels with
        # which to train a model
        #
        # TODO probably X_direct should be renamed to X_tilde_direct
        ###
        X_direct, y_direct, direct_pmids, vectorizer = generate_X_y(
                                                domain_supervision,
                                                return_pmids_too=True) 
        


        # the above returns tuples of numeric and textual
        # features for each candidate. below we vectorize
        # these.
        X_train_tilde = []
        train_sentences_DS = [] 
        train_tilde_pmids = []
        for j in train_rows:
            cur_X = X_tilde_dict[domain]["X"][j]
            X_tilde_v = None
            if cur_X is not None:
                X_tilde_v = _to_vector(cur_X, vectorizer)
            X_train_tilde.append(X_tilde_v)

            train_tilde_pmids.append(X_tilde_dict[domain]['pmids'][j])
            train_sentences_DS.append(domain_DS["sentences"][j])

        # build the SDS model!
        testing_pmids = [] # in this case we use *all* available labels!
        print "building SDS model..."
        sds_clf = build_SDS_model(X_direct, y_direct, direct_pmids,
                                directly_supervised_indicators_train,
                                X_train_tilde, train_tilde_pmids,
                                train_rows, testing_pmids,
                                X_train_DS, y_train_DS, pmids_train_DS,
                                domain, sentences=train_sentences_DS)


        # dump? 
        # see: https://gist.github.com/ijmarshall/e2797042e86326bc307c
        #pdb.set_trace()
        coefs = csr_matrix(sds_clf.coef_)
        model_tuple_out = (sds_clf.intercept_, coefs.data, coefs.indices, coefs.indptr)
        model_out_path = os.path.join(output_dir, '%s.rbt' % domain)
        hickle.dump(model_tuple_out, model_out_path, compression='gzip')
        
        # note that the vectorizers (for all domains) are already 
        # available in the domain_v_pickle object
        vec_out_path = os.path.join(output_dir, "%s_vectorizer.bz2_pickle" % domain)
        vec_out_f = bz2.BZ2File(vec_out_path, 'w') 
        pickle.dump(domain_vectorizers[domain], vec_out_f)
        vec_out_f.close()
        print "\n\n --- woo! %s pickled and available at: %s; vectorizer in %s --- \n\n" % (domain, model_out_path, vec_out_path)


def use_model_to_generate_labeling_file(N, k=3, output_dir="sds/results/",
                            y_dict_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_sentence_data10.pickle", 
                            domain_v_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_vectorizers10.pickle"):
    ''' generate file for annotation using models; this will select N studies at random '''    

    ###
    # setup output path.
    output_path = os.path.join(output_dir, "%s-labeling.csv" % int(time.time()))
    print "will write labeling file out to %s" % output_path

 
    domain_index = 2
    domain = "CHAR_OUTCOMES"

    ### 
    # load in DS; we assume this is pickled already.
    print "unpickling DS from %s..." % y_dict_pickle
    sentences_y_dict, domain_vectorizers = _unpickle_PICO_DS(y_dict_pickle,
                                                domain_v_pickle)
    
    print "ok!"    
    domain_DS = sentences_y_dict[domain]

    ###
    # and this is where the direct supervision is loaded up
    print "generating DS learning tasks..."
    DS_learning_tasks, z_dict, domains_pmids_targets = get_DS_features_and_labels()
    domain_supervision = DS_learning_tasks[domain]
    print "done."

    #pdb.set_trace()
    
    ###
    # these were directly labeled.
    labeled_pmids_for_domain = list(
                set(DS_learning_tasks[domain]["pmids"]))
    

    # now pick N new studies to have the ugrads label
    print "selecting study PMIDs to label..."
    pmids_to_label = pico_DS.pick_N_pmids(N, labeled_pmids_for_domain)
    # these are the IDs to labeled (will basically be treated as 'test IDs')
    testing_pmids = pmids_to_label
    print "done. here are the PMIDs I selected: %s" % pmids_to_label

    ###
    # load in \tilde{X} for SDS
    print "loading X_tilde_dict ... just the once though!"
    X_tilde_dict, target_text_dict = get_DS_features_for_all_data(z_dict)
    print "ok."
    

    train_rows, test_rows = [], []
    directly_supervised_indicators = np.zeros(len(domain_DS["pmids"])).astype("int8")

    unique_pmids = list(set(domain_DS["pmids"]))

    ### 
    # 6/7
    # bcw: need to fix this prevent including studies twice!!!
    pmids_already_seen = []
    prev_pmid = None
    for i, pmid in enumerate(domain_DS["pmids"]):
        if prev_pmid is None:
            pass 
        elif pmid != prev_pmid:
            pmids_already_seen.append(prev_pmid)

        cur_sentence = domain_DS["sentences"][i]

        if pmid not in testing_pmids:
            train_rows.append(i)

            ###
            # here we need to overwrite any labels for which
            # we have explicit supervision!
            if pmid in domain_supervision["pmids"]:

                cur_label = _match_direct_to_distant_supervision(
                    domain_supervision, pmid, cur_sentence)

                # if this is None, it means the current sentence
                # was not found in the candidate set, implicitly
                # this means it is a -1.
                if cur_label is None:
                    cur_label = -1
                else:
                    # we train only on 2's (highly relevant sentences), as we 
                    # are aiming for a high-precision model
                    try:
                        cur_label = _score_to_binary_lbl(cur_label, threshold=2, zero_one=False)
                    except:
                        print "Ah! something wrong with this label."
                        pdb.set_trace()
                domain_DS["y"][i] = cur_label
                directly_supervised_indicators[i] = 1

        # this is to make sure we don't wind up
        # with duplicate instances of documents
        # in the test set
        elif pmid not in pmids_already_seen:
            test_rows.append(i)

        prev_pmid = pmid

    print "huzzah -- data all set up."
    
    ###
    # assemble train and test sets
    X_train_DS = domain_DS["X"][train_rows]
    y_train_DS = np.array(domain_DS["y"])[train_rows]
    pmids_train_DS = [domain_DS["pmids"][train_row] for train_row in train_rows]
    directly_supervised_indicators_train = directly_supervised_indicators[train_rows]

    #X_test_DS = domain_DS["X"][test_rows]
    #pmids_test_DS = [domain_DS["pmids"][test_row] for test_row in test_rows]


    ###
    # first build an SDS classifier
    
    # this transforms the small amount of direct supervision we
    # have for the mapping task from candidate sentences to
    # the best sentences into feature vectors and labels with
    # which to train a model
    #
    # TODO probably X_direct should be renamed to X_tilde_direct
    ###
    X_direct, y_direct, direct_pmids, vectorizer = generate_X_y(
                                            domain_supervision,
                                            return_pmids_too=True) 
    


    # the above returns tuples of numeric and textual
    # features for each candidate. below we vectorize
    # these.
    X_train_tilde = []
    train_sentences_DS = [] 
    train_tilde_pmids = []
    for j in train_rows:
        cur_X = X_tilde_dict[domain]["X"][j]
        X_tilde_v = None
        if cur_X is not None:
            X_tilde_v = _to_vector(cur_X, vectorizer)
        X_train_tilde.append(X_tilde_v)

        train_tilde_pmids.append(X_tilde_dict[domain]['pmids'][j])
        train_sentences_DS.append(domain_DS["sentences"][j])

    # build the SDS model!
    sds_clf = build_SDS_model(X_direct, y_direct, direct_pmids,
                            directly_supervised_indicators_train,
                            X_train_tilde, train_tilde_pmids,
                            train_rows, testing_pmids,
                            X_train_DS, y_train_DS, pmids_train_DS,
                            domain, sentences=train_sentences_DS)



    ### nguyen model!
    print "OK! building Nguyen model..."
    nguyen_clf = build_nguyen_model(X_train_DS, y_train_DS,
                            directly_supervised_indicators_train)


    #pdb.set_trace()
    
    ###
    # now make predictions using both models
    #pmids_to_preds_sds = defaultdict(list)
    #pmids_to_preds_nguyen = defaultdict(list)
    #X_test_DS = domain_DS["X"][test_rows]
    #pmids_test_DS = [domain_DS["pmids"][test_row] for test_row in test_rows]
    nguyen_preds, sds_preds = [], []
    test_pmids, test_sentences = [], []
    cdsr_sentences = []
    # you'll need this to get the sentences
    bv = biviewer.PDFBiViewer()
    for test_row in test_rows:
        x = domain_DS["X"][test_row]
        nguyen_preds.append(nguyen_clf.decision_function(x)[0])
        sds_raw_pred = sds_clf.decision_function(x)[0]
        sds_preds.append(sds_raw_pred)
        cur_pmid = domain_DS["pmids"][test_row]
        test_pmids.append(cur_pmid)
        # grab the pop summary from cdsr
        study = bv.get_study_from_pmid(str(cur_pmid))
        # note that if there are multiple studies
        # with this PMID (as may happen!), then
        # we just arbitrarily select the first.
        population_sum = study[0].cochrane['CHARACTERISTICS']
        cdsr_sentences.append(population_sum)

        test_sentences.append(domain_DS["sentences"][test_row])
        #cdsr_ids.append(domain_DS["CDSR_id"])


    #pdb.set_trace()
    all_preds = pd.DataFrame({"row_id": test_rows, 
                              "pmid": test_pmids, 
                              "sentence": test_sentences,
                              "sds_pred": sds_preds,
                              "nguyen_pred": nguyen_preds,
                              "cdsr_sentences": cdsr_sentences})


    grouped = all_preds.groupby("pmid")
    preds_by_pmid = dict(list(grouped))
    
    # start assembling output
    labeling_str = [["study id", "PICO field", "CDSR sentence", 
                        "candidate sentence", "method", "rating"]]
    for pmid, results in preds_by_pmid.items():    
        nguyen_top_preds = _get_top_preds(results, "nguyen_pred", k=k)
        sds_top_preds = _get_top_preds(results, "sds_pred", k=k)
    
        # sentences in both top ranked sets
        shared = [s for s in nguyen_top_preds if s in sds_top_preds]

        # note that this is the same across all sentences here.
        #pdb.set_trace()

        indices = results.index
        cdsr_sent = results["cdsr_sentences"][indices[0]][domain] 
        set_to_label = []
        for s in shared:
            set_to_label.append([str(pmid), domain, cdsr_sent, s, "BOTH",  " "])

        for sent in nguyen_top_preds:
            if not sent in shared: 
                #set_to_label.append(sent)
                set_to_label.append(
                    [str(pmid), domain, cdsr_sent, sent, "nguyen", " "])

        for sent in sds_top_preds:
            #if sent.count("\n") > 20:
            #    pdb.set_trace()

            if not sent in shared:
                set_to_label.append(
                    [str(pmid), domain, cdsr_sent, sent, "sds", " "])

        random.shuffle(set_to_label)
        labeling_str.extend(set_to_label)


    fout = open(os.path.join(output_dir, "for-labeling-%s-2.csv" % domain), 'wb')
    writer = csv.writer(fout)
    writer.writerows(labeling_str)


# sort_key \in {"nguyen_pred", "sds_pred"}
def _get_top_preds(results, sort_key, k=3):
    ''' helper function: return top k sentences (as sorted by sort_key '''
    sorted_results_for_pmid = results.sort(sort_key, ascending=False)
    indices = sorted_results_for_pmid.index

    top_preds = []
    # only  output sentences for entire studies with '0' metric!
    for i in indices[:k]:                  
        top_preds.append(
            sorted_results_for_pmid["sentence"][i])

    return top_preds


# note: the *_1000 pickles are subsets of the available DS to
# speed things up for experimentation purposes!
def run_DS_PICO_experiments(iters=5, cv=True, test_proportion=None,
                            strategy="baseline_DS", output_dir="sds/results/",
                            y_dict_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_sentence_data10.pickle", 
                            domain_v_pickle="/Users/byron/dev/cochrane/cochranenlp/data/sds_vectorizers10.pickle", 
                            random_seed=512):

    '''
    Runs multiple (iters) experiments using the specified strategy.

    If `cv' is true, cross-validation will be performed using `iters' splits.
    Otherwise, `training_proportion' should be provided to specify the
    fraction of examples to be held out for each iter.
    '''

        

    output_path = os.path.join(output_dir,
            "%s-results-tables-pos-%s.txt" % (int(time.time()), strategy))
    print "will write results out to %s" % output_path

    # write out a preample..
    with open(output_path, 'a') as out:
        out.write("run at %s\n" % str(datetime.now()))
        out.write("strategy=%s\n" % strategy)
        if cv:
            out.write("%s-fold cross-validation." % iters)
        else:
            out.write("%s random splits using %s of the data for testing." %
                         (iters, test_proportion))
        out.write("\n\n\n")

    ## load in the available supervision
    sentences_y_dict, domain_vectorizers = None, None
    if y_dict_pickle is None:
        sentences_y_dict, domain_vectorizers = pico_DS.all_PICO_DS()
    else:
        print "unpickling DS from %s..." % y_dict_pickle
        sentences_y_dict, domain_vectorizers = _unpickle_PICO_DS(y_dict_pickle,
                                                    domain_v_pickle)
        print "ok!"

    ##
    # now load in the direct supervision we have.
    # this is kind of confusingly named; we need these
    # DS labels for evaluation here -- we don't care for
    # the features *unless* we are doing SDS!
    # @TODO refactor or rename method?
    #
    # Note that the z_dict here is a dictionary mapping
    # PICO fields to vectors comprising normalization terms
    # for each numeric feature/column.
    #
    # And domains_pmids_targets is a dictionary that
    # maps pmids and domains to target sentences
    DS_learning_tasks, z_dict, domains_pmids_targets = get_DS_features_and_labels()

    ## now we divvy up into train/test splits.
    # this will be a dictionary mapping domains to
    # lists of test PMIDs.
    test_id_lists = {}

    for domain in domains:
        labeled_pmids_for_domain = list(set(DS_learning_tasks[domain]["pmids"]))
        test_id_lists[domain] = []
        if cv:
            kfolds = cross_validation.KFold(len(labeled_pmids_for_domain),
                                                iters, shuffle=True, 
                                                random_state=random_seed)
            for train_indices, test_indices in kfolds:
                test_id_lists[domain].append(
                    [labeled_pmids_for_domain[j] for j in test_indices])
        else:
            ### TODO validate
            train_test_splits = []

            for iter_ in xrange(iters):
                cur_train_test_split = train_test_split(
                            labeled_pmids_for_domain, test_size=test_proportion,
                            random_state=random_seed+iter_)
                test_id_lists[domain].append([cur_train_test_split[1]])


    top3_indicators_2, top3_indicators_1 = [], []
    all_sentence_output_str = []

    # 6/2: much faster to just generate and cache this
    # thing -- before you were doing it every fold!
    X_tilde_dict = None
    if strategy.lower() == "sds":
        assert z_dict is not None
        print "loading X_tilde_dict; just the once though!"
        X_tilde_dict, target_text_dict = get_DS_features_for_all_data(z_dict)


    for iter_ in xrange(iters):

        ## if we're doing cross-fold validation,
        # then we need to assemble a dictionary for
        # this fold mapping domains to test PMIDs
        cur_test_pmids_dict = None
        if cv:
            test_ids_for_cur_fold = [test_id_lists[domain][iter_] for domain in domains]
            cur_test_pmids_dict = dict(zip(domains, test_ids_for_cur_fold))

        iter_results = DS_PICO_experiment(sentences_y_dict, domain_vectorizers,
                                            DS_learning_tasks, domains_pmids_targets,
                                            strategy=strategy,
                                            test_pmids_dict=cur_test_pmids_dict,
                                            test_proportion=test_proportion,
                                            z_dict=z_dict, X_tilde_dict=X_tilde_dict)

        any_really_relevant_indicators, any_kinda_relevant_indicators, cur_all_sentence_output_str, output_str = iter_results
                                        
        print u"\n".join(output_str)
        print "\n\n"

        with open(output_path, 'a') as output_f:
            output_f.write("\n\n\n\n -- fold/iter %s --\n\n" % iter_)
            output_f.write(u"\n".join(output_str))

        ##
        # setup for one core; @TODO amend to handle the cluster!
        if iter_ > 0:
            # skip headers
            cur_all_sentence_output_str = cur_all_sentence_output_str[1:]

        all_sentence_output_str.extend(cur_all_sentence_output_str)

        top3_indicators_2.extend(any_really_relevant_indicators)
        top3_indicators_1.extend(any_kinda_relevant_indicators)


    print "\n\n\n"
    avg_top3_2 = np.mean(top3_indicators_2)
    print "average fraction of articles with at least 1 really relevant sentence in top 3 sentences: %s" % avg_top3_2 
    avg_top3_1 = np.mean(top3_indicators_1)
    print "average fraction of articles with at least 1 kinda relevant sentence in top 3 sentences: %s" % avg_top3_1 

    with open(output_path.replace(".txt", "_micro.txt"), 'wb') as outf:
        outf.write("top 3 (really relevant): %s" % avg_top3_2)
        outf.write("top 3 (kinda relevant): %s" % avg_top3_1)      

    with open(output_path.replace(".txt", "_all_sentence_scores.txt"), 'wb') as outf:
        writer = csv.writer(outf)
        writer.writerows(all_sentence_output_str)


def DS_PICO_experiment(sentences_y_dict, domain_vectorizers,
                        DS_learning_tasks, domains_pmids_targets,
                        strategy="baseline_DS", test_pmids_dict=None, test_proportion=.8,
                        use_distant_supervision_only=False, z_dict=None, X_tilde_dict=None):
    '''
    This is an implementation of a naive baseline
    distantly supervised method.

    The sentences_y_dict argument is assumed to be
    a dictionary mapping domains to lists of sentences,
    their vector representations ("X") and their
    distantly derived labeld ("y").

    If test_pmids_list is not None, then the list
    of PMIDs specified by this parameter will be
    held out as testing data.

    If test_pmids_list is None, then we randomly
    select (test_proportion*100)% of the directly
    labeled data to be our held out data. Note
    that if this is 1 (as by default), then this
    means we are training entirely on DS data and
    then evaluating via the direct supervision.

    If use_distant_supervision_only is True, then
    we do not exploit the direct supervision even
    when available. (This is a useful baseline.)
    '''
    testing_pmids = None
    output_str = [""] # general output
    # pmid_, sentence_, str(pred_), str(lbl1_), str(lbl2_)]
    all_sentence_preds_output = [["domain", "pmid", "sentence", "raw pred", "lbl1", "lbl2"]] # 6/2: exhaustive!

    strategy_name = strategy.lower()

    # wait a minute: is this the same everytime??!??! why on earth 
    # don't you cache this... 
    if strategy_name == "sds" and X_tilde_dict is None:
        # we need the normalization scalars
        # for the numerical SDS features!
        assert z_dict is not None
        ### this will align with the DS_supervision.
        # note that we now *pad* this vector to make sure
        # the entries align with the DS we have.
        X_tilde_dict, target_text_dict = get_DS_features_for_all_data(z_dict)

    #elif strategy_name == "baseline_DS":
    #    print "using distant supervision *only*"

    if use_distant_supervision_only:
        print "\n\n USING DISTANT SUPERVISION ONLY!\n\n"

    #for domain_index, domain in enumerate(domains):
    for domain_index, domain in enumerate(domains):
        ##
        # here we need to grab the annotations used also
        # for SDS to evaluate our strategy.
        # note: this is the *direct* supervision!
        domain_supervision = DS_learning_tasks[domain]

        if test_pmids_dict is None:
            if p < 1:
                n_testing = int(p * len(domain_supervision["pmids"]))
                testing_pmids = random.sample(domain_supervision["pmids"], n_testing)
            else:
                testing_pmids = domain_supervision["pmids"]

        else:
            testing_pmids = test_pmids_dict[domain]
            # do we actually have labels corresponding to
            # these PMIDs???
            assert len(set(testing_pmids).intersection(
                            set(domain_supervision["pmids"]))) == len(set(testing_pmids)), \
                            '''there is a mismatch between the testing PMIDs given and the
                                supervision available!'''


            print "using provided list of %s PMIDs for testing!" % len(
                    testing_pmids)


        ###
        # Which rows correspond to studies that are in the testing data?
        # we want to exclude these.
        #
        # Note that we mutate the labels here. thus
        # we take a deepcopy so that we don't expose these
        # labels later on.
        domain_DS = copy.deepcopy(sentences_y_dict[domain])
        train_rows, test_rows = [], []
        # we maintain two sets of binary labels; one takes all
        # sentences labeled `1' *or above* as `1', while the
        # other (y2) imposes a stricter threshold of `2'.
        y_test = [] # slightly tricky
        y_test_relaxed = []
        directly_supervised_indicators = np.zeros(len(domain_DS["pmids"])).astype("int8")
        # so this is to remember which examples we actually had an explicit
        # label on, as opposed to just being part of studies
        # that were directly labeled
  
        ###
        # I think it's clear what's happening
        # pmids are not unique, but you're only checking
        # for direct supervision by cross-referencing against
        # the PMID...
        #
        # I think you can just (arbitrarily) take the first
        # instance you identify
        ###

        # make sure that domain_DS["pmids"]
        # contains redundant entries!
        # if it does, consider keeping a dictionary
        # around
        unique_pmids = list(set(domain_DS["pmids"]))

        for i, pmid in enumerate(domain_DS["pmids"]):

            cur_sentence = domain_DS["sentences"][i]

            if pmid not in testing_pmids:

                train_rows.append(i)

                ###
                # here we need to overwrite any labels for which
                # we have explicit supervision!
                if pmid in domain_supervision["pmids"] and not use_distant_supervision_only:
                    ###
                    # It is possible that this will introduce some small amount of
                    # noise, in the following way. We are checking if we have
                    # supervision for the current PMID, but this could correspond
                    # to multiple entries in the CDSR. If a sentence did not rank
                    # highly with respect to the `labeled' instance, or if it
                    # ranked highly but was considered irrelevant, *and* this assessment
                    # would not (or does not) hold for an alternative summary/target,
                    # then we may introduce a false negative here. I think this is
                    # rather unlikely and will, at the very least, be rare.
                    cur_label = _match_direct_to_distant_supervision(
                        domain_supervision, pmid, cur_sentence)


                    # if this is None, it means the current sentence
                    # was not found in the candidate set, implicitly
                    # this means it is a -1.
                    if cur_label is None:
                        cur_label = -1
                    else:
                        # we train only on 2's (highly relevant sentences), as we 
                        # deem these as the target
                        cur_label = _score_to_binary_lbl(cur_label, threshold=2, zero_one=False, 
                                        count_relevant_tables=False)


                    # 1/7/15 -- previously, we were not considering
                    # an instance directly supervised if cur_label
                    # came back None, although in seme sense
                    # it's not clear that we should be...
                    domain_DS["y"][i] = cur_label
                    directly_supervised_indicators[i] = 1


                    '''
                    ##
                    # 4/22/15 -- this line previously here
                    # 
                    #domain_DS["y2"][i] = cur_label
                    # keep track of row indices that correspond
                    # to directly supervised instances
                    #directly_supervised_indices.append(i)
                    directly_supervised_indicators[i] = 1
                    '''
            else:
                ####
                # Then this index is associated with a study to be used for
                # testing.

                '''
                Here we deal with the problem of articles being duplicated
                in the test set. This is possible because the CDSR 
                may contain multiple instances of any given article (PMID),
                extracted (e.g.) for different reviews. Thus we rely on 
                the CDSR identifier to ensure that we test on only 
                one copy of each article (and specifically, the copy that 
                we received direct supervision for).
                '''
                cur_test_index = domain_supervision['pmids'].index(pmid)
                labeled_CDSR_entry = domain_supervision['CDSR_ids'][cur_test_index]
                # only test against it matches up.
                if domain_DS["CDSR_id"][i] != labeled_CDSR_entry:
                    pass 
                    # too much output...
                    #print "skipping test entry for %s, because the CDSR id does not match." % pmid
                else:        
                    test_rows.append(i)

                    cur_label = _match_direct_to_distant_supervision(domain_supervision, 
                                                pmid, cur_sentence)
                    if cur_label is None:
                        # no match found -> this was not a labeled sentence
                        #   -> this is an irrelevant sentence
                        cur_label_strict = -1
                        cur_label_relaxed = -1
                    else:
                        ###
                        # 12/28 -- experimental; counting relevant
                        # tables here as 'positive!'
                        cur_label_strict = _score_to_binary_lbl(cur_label, threshold=2, zero_one=False, count_relevant_tables=False)
                        cur_label_relaxed = _score_to_binary_lbl(cur_label, threshold=1, zero_one=False, count_relevant_tables=False)

                    y_test.append(cur_label_strict)
                    y_test_relaxed.append(cur_label_relaxed)

                    if pmid in domain_supervision["pmids"]:
                        directly_supervised_indicators[i] = 1


        print "huzzah -- data all set up."

        ###
        # it's possible that here we end up with an empty
        # training set if we're working with a subset
        # of the DS data! another way of saying this:
        # set(domain_DS["pmids"]).intersection(set(testing_pmids))
        # may be the empty set ([]); this will happen when
        # we're not using all of the DS data.
        #
        # since this would only be for dev/testing
        # purposes, I think we can safely ignore such cases,
        # but this may cause things to break during evaluation.
        ###
        if len(y_test) == 0:
            print "no testing data!"
            pdb.set_trace()
            return ["-"*25, "\n\n no testing data! \n\n", "-"*25]

        X_train_DS = domain_DS["X"][train_rows]
        pmids_train_DS = [domain_DS["pmids"][train_row] for train_row in train_rows]
        X_test_DS = domain_DS["X"][test_rows]
        y_train_DS = np.array(domain_DS["y"])[train_rows]

        directly_supervised_indicators_train = directly_supervised_indicators[train_rows]
        clf = None

        print "strategy: %s" % strategy

        ###
        # now train a classifier
        if strategy_name == "sds":
            # this transforms the small amount of direct supervision we
            # have for the mapping task from candidate sentences to
            # the best sentences into feature vectors and labels with
            # which to train a model
            #
            # TODO probably X_direct should be renamed to X_tilde_direct
            ###


            X_direct, y_direct, direct_pmids, vectorizer = generate_X_y(
                                                    domain_supervision,
                                                    return_pmids_too=True) 
            

            # the above returns tuples of numeric and textual
            # features for each candidate. below we vectorize
            # these.
            X_train_tilde = []
            train_sentences_DS = [] 
            # train_tilde_pmids will contain the PMIDs
            # for the X_tilde instances 
            train_tilde_pmids = []
            for j in train_rows:
                cur_X = X_tilde_dict[domain]["X"][j]
                X_tilde_v = None
                if cur_X is not None:
                    X_tilde_v = _to_vector(cur_X, vectorizer)
                X_train_tilde.append(X_tilde_v)

                train_tilde_pmids.append(X_tilde_dict[domain]['pmids'][j])
                train_sentences_DS.append(domain_DS["sentences"][j])

            ''' end SDS magic '''
            clf = build_SDS_model(X_direct, y_direct, direct_pmids,
                                    directly_supervised_indicators_train,
                                    X_train_tilde, train_tilde_pmids,
                                    train_rows, testing_pmids,
                                    X_train_DS, y_train_DS, pmids_train_DS,
                                    domain, sentences=train_sentences_DS)
     

        elif strategy_name == "nguyen":
            if max(directly_supervised_indicators_train) == 0:
                print "no direct supervision provided!"
                return ["-"*25, "\n\n no directly labeled data!!! \n\n", "-"*25]

            clf = build_nguyen_model(X_train_DS, y_train_DS,
                                    directly_supervised_indicators_train)

        elif strategy_name == "direct":
            if max(directly_supervised_indicators_train) == 0:
                print "no direct supervision provided!"
                pdb.set_trace()
            clf = build_direct_only_model(X_train_DS, y_train_DS, 
                        directly_supervised_indicators_train)
        else:
            clf = get_DS_clf()
            print "standard distant supervision. fitting model..."

            clf.fit(X_train_DS, y_train_DS)


        '''
        # OK -- now make predictions!

        preds = clf.predict(X_test_DS)
        precision, recall, f, support = precision_recall_fscore_support(
                                            y_test_DS, preds)
        '''


        ###
        # make predictions for each study
        current_pmid = None
        current_target_text = None
        preds_for_current_pmid, rows_for_current_pmid = [], []
        sentences_for_current_pmid, labels_for_current_pmid = [], []

        # labels according to more forgiving 
        # criteria (>= 1 rather 2)
        labels_for_current_pmid_1 = []

        # any_relevant_indicators contains 1s where at least
        # one of the top k sentences (say k=3) is highly relevant
        any_relevant_indicators, precisions, accs = [], [], []
        # metrics using more liberal evaluation!
        any_relevant_indicators_1, precisions_1, accs_1 = [], [], []

        n_test_rows = len(test_rows)-1

        # are we misclassifying tables??
        ## bcw -- double check domension 
        # (Pdb) domain_DS["X"][test_row].shape
        # (1, 50012)
        # should be (1, 50005), i think
        #pdb.set_trace()
        for test_row_index, test_row in enumerate(test_rows):
            test_pmid = domain_DS["pmids"][test_row]
            target_text = domains_pmids_targets[domain][test_pmid] #domain_DS["targets"][test_row]
            current_sentence = domain_DS["sentences"][test_row]

            current_label = y_test[test_row_index]
            current_label_1 = y_test_relaxed[test_row_index]

            if current_label > current_label_1:
                # should not happen; sanity check
                pdb.set_trace()

            if current_pmid is None:
                current_pmid = test_pmid
                current_target_text = target_text
                sentences_for_current_pmid = [current_sentence]
                rows_for_current_pmid = [test_row]
                labels_for_current_pmid = [current_label]
                labels_for_current_pmid_1 = [current_label_1]

            elif test_pmid != current_pmid or test_row_index == n_test_rows:
                
                ###
                # 6/2: output everything; PMIDs and scores
                cur_num_sentences = len(preds_for_current_pmid)
                repeating_pmids = [str(current_pmid)] * cur_num_sentences
                for pmid_, sentence_, pred_, lbl1_, lbl2_ in zip(
                                                    repeating_pmids, 
                                                    sentences_for_current_pmid, 
                                                    preds_for_current_pmid, 
                                                    labels_for_current_pmid_1,
                                                    labels_for_current_pmid):
                                                    
                    # pmid, sentence, true label
                    all_sentence_preds_output.append(
                        [domain, pmid_, sentence_, str(pred_), str(lbl1_), str(lbl2_)])
                # 
                ### end 6/2 additions ### 

                #highest_pred = np.argmax(np.array(preds_for_current_pmid))
                sorted_pred_indices = np.array(preds_for_current_pmid).argsort()
                #highest_pred_indices = rows_for_current_pmid[highest_pred]

                highest_pred_indices = sorted_pred_indices[-3:]
                #highest_pred_indices = reversed(sorted_pred_indices)

                #true_labels = np.array(domain_DS["y"])[highest_pred_indices]
                sentences, true_labels = [], []
                true_labels_1 = []
                unique_sent_count = 0
                for sent_i in highest_pred_indices:
                    sentences.append(sentences_for_current_pmid[sent_i])
                    if len(sentences) != len(set(sentences)):
                        print "\nwarning -- there are redundant sentences within the predictions"
                        print sentences
                        # this is not necessarily a *huge* problem, e.g., it's possible that
                        # `unique' sentences are indeed exact matches. One example I have seen
                        # [u'1991.', u'1992.', u'1992.']
                        #pdb.set_trace()


                    ### why do we sometimes get duplicate sentences??!?!
                    true_labels.append(labels_for_current_pmid[sent_i])
                    true_labels_1.append(labels_for_current_pmid_1[sent_i])

                output_str.append("\n -- domain %s in study %s --\n\n" %
                                    (domain, current_pmid))
                output_str.append("-- target text --\n\n %s" % current_target_text)

                output_str.append("\n\n-- candidate sentence --\n\n")
                output_str.append("\n\n-- candidate sentence --\n\n".join(sentences))

                # should we do something special if these were not
                # directly supervised instances?
                directly_supervised_indicators[test_row]


                #precision_at_three = len(true_labels[true_labels>0])/3.0
                precision_at_three = true_labels.count(1)/3.0
                precisions.append(precision_at_three)

                precision_at_three_1 = true_labels_1.count(1)/3.0
                precisions_1.append(precision_at_three_1)

                any_relevant = 1 if true_labels.count(1) > 0 else 0
                any_relevant_indicators.append(any_relevant)

                any_relevant_1 = 1 if true_labels_1.count(1) > 0 else 0
                any_relevant_indicators_1.append(any_relevant_1)

                if any_relevant > any_relevant_1:
                    pdb.set_trace()
                top_pred_acc = 1 if true_labels[0] > 0 else 0
                accs.append(top_pred_acc)

                top_pred_acc_1 = 1 if true_labels_1[0] > 0 else 0
                accs_1.append(top_pred_acc_1)


                # domain_DS["sentences"][highest_prediction_index]
                current_pmid = test_pmid
                current_target_text = target_text
                preds_for_current_pmid = []
                sentences_for_current_pmid = [current_sentence]
                labels_for_current_pmid = [current_label]
                labels_for_current_pmid_1 = [current_label_1]
                rows_for_current_pmid = [test_row]
                pred_rows = []
            else:
                rows_for_current_pmid.append(test_row)
                sentences_for_current_pmid.append(current_sentence)
                labels_for_current_pmid.append(current_label)
                labels_for_current_pmid_1.append(current_label_1)

            current_pred = clf.decision_function(domain_DS["X"][test_row])
          
            preds_for_current_pmid.append(current_pred[0])

        '''
        ## now rankings
        raw_scores = clf.decision_function(X_test_DS)
        auc = None
        #pdb.set_trace()
        auc = sklearn.metrics.roc_auc_score(y_test_DS, raw_scores)
        '''


        output_str.append("-"*25)
        output_str.append("method: %s" % strategy)
        output_str.append("domain: %s" % domain)

        output_str.append("at least one (>=2): %s" % np.mean(any_relevant_indicators))
        output_str.append("precisions (>=2): %s" % np.mean(precisions))
        output_str.append("accuracy (>=2): %s" % np.mean(accs))
        output_str.append("--- using a less stringent criteria ---")
        output_str.append("at least one (>=1): %s" % np.mean(any_relevant_indicators_1))
        output_str.append("precisions (>=1): %s" % np.mean(precisions_1))
        output_str.append("accuracy (>=1): %s" % np.mean(accs_1))

        #output_str.append(str(sklearn.metrics.classification_report(y_test_DS, preds)))
        output_str.append("\n")
        #output_str.append("confusion matrix: %s" % str(confusion_matrix(y_test_DS, preds)))
        # output_str.append("AUC: %s" % auc)
        output_str.append("-"*25)

    return any_relevant_indicators, any_relevant_indicators_1, all_sentence_preds_output, output_str


def _to_vector(X_i, vectorizer):
    X_i_numeric, X_i_text = X_i
    X_v = vectorizer.transform([X_i_text])[0]
    X_combined = sp.sparse.hstack((X_v, X_i_numeric))
    #X.append(np.asarray(X_combined.todense())[0])
    return np.asarray(X_combined.todense())[0]

def _match_direct_to_distant_supervision(domain_supervision, pmid,
                                            cur_sentence, threshold=1):

    ###
    # this is tricky.
    # we have to identify the current sentence from
    # DS in our supervised label set. To do this, we
    # first figure out which sentences were labeled
    # for this study (pmid).
    first_index = domain_supervision["pmids"].index(pmid)
    # note: the above assumes that we have only one
    # labeled instance of a given PMID; it is theoretically
    # possible that two separate instances have been labeled
    # (corresponding to different entries in CDSR)
    study_indices = range(first_index, first_index+domain_supervision["pmids"].count(pmid))
    # now grab the actual sentences corresponding
    # to these labels
    labeled_sentences, labels = [], []
    for sent_index in study_indices:
        labeled_sentences.append(domain_supervision["sentences"][sent_index])
        labels.append(domain_supervision["y"][sent_index])

    try:
        # which of these sentences are we looking at now?
        # domain_DS["sentences"][i]
        matched_sentence_index = labeled_sentences.index(cur_sentence)

        ### 
        # let's not introduce the threshold here;
        # instead, this can happen elsewhere (after
        # the return. this allows, e.g., keeping both 
        # '1' and '2' labels as 'positive' instances
        # during evaluation.

        # and, finally, what was its (human-given) label?
        #cur_label = _score_to_binary_lbl(labels[matched_sentence_index],
        #                    threshold=threshold, zero_one=False)
        cur_label = labels[matched_sentence_index]

        return cur_label
    except:
        ###
        # NOTE here is where we are explicitly making
        # the assumption that any sentences in articles
        # *not* labeled are irrelevant. ie., every time we
        # encounter a sentence that didn't rank high
        # enough to get a label we give it a "-1" label.
        #y_test_DS.append(-1)
        return None




### TODO TODO TODO grid search over scalars here!
#
# 1 out of training PMIDs, select a small 
#       set for tuning. 
# 2 exclude this set from induction of sds model
#       and from receiving direct labels
def build_SDS_model(X_direct, y_direct, pmids_direct,
                    directly_supervised_indicators_train,
                    X_tilde_train, X_tilde_pmids,
                    train_rows, testing_pmids,
                    X_distant_train, y_distant_train,
                    pmids_distant_train,
                    domain, weight_range=None,
                    direct_weight_scalar_range=None,
                    alpha_range=None,
                    sentences=None, 
                    p_tuning=.25):
    '''
    Here we train our SDS model and return it.
    Specifically, this entails building a model
    for the `mapping' or alignment task using
    X_direct and y_direct, then applying this model
    to the DS instances.

    X_direct, y_direct - the feature vectors and labels
    provided for the candidate sentences generated via
    distant supervision.

    pmids_direct - the set of PMIDs for which we have
    direct supervision (i.e., we have labeled candidate
    sentences for these studies).

    directly_supervised_indicators_train -- this is a
    binary vector indicating which training entries
    correspond to direct supervision. This is nececessary
    because we don't want to overwrite labels that
    were explicitly provided.

    X_tilde_train is the matrix comprising all
    tilde instances for training

    train_pmids provides the PMIDs corresponding to 
    rows in the training dataset. 

    X_direct_train are tilde entries for (directly) labeled
    intances so that we can train our sds model
    '''

    # this is likely key, as it scales the instance
    # weighting for directly labeled vs indirectly 
    # labeled instances. this make since especially
    # because then the alphas will be different for
    # the two models.
    #
    # another @TODO: what if we use the Nguyen 
    # 'linear pooling' method to 'soft label'
    # the remaining instances?
    if direct_weight_scalar_range is None:
        # this will be very slow!
        
        direct_weight_scalar_range = (2, 10, 50, 100, 200, 500)#, 5,10)#np.linspace(0,1,5)

    if weight_range is None: 
        weight_range = [1]

    distant_weight = 1.0 

    if alpha_range is None: 
        # was 5,7
        #alpha_range = 10.0**-np.arange(3,7)
        alpha_range = [.00001, .0001, .001, .01, .1]

    #############################################
    # (1) Train the SDS model (M_SDS) using directly
    #  labeled data. This model is the same regardless
    #  of the scalar params to be tuned.
    #############################################
    # this is confusingly named because X_direct
    # is actually the subset of X_tilde for which we
    # have been given labels!
    #X_direct_train, y_direct_train = [], []


    ####
    # don't train on testing articles!
    ####
    direct_train_indices = []
    direct_train_pmids = []
    for i, pmid in enumerate(pmids_direct):
        #x_i, y_i = X_direct[i], y_direct[i]

        if pmid not in testing_pmids:
            #X_direct_train.append(x_i)
            #y_direct_train.append(y_i)
            direct_train_indices.append(i)
            direct_train_pmids.append(pmid)

    

    X_direct_train = []
    y_direct_train = []
    pmids_direct_train = []
    for j in direct_train_indices:
        
        X_direct_train.append(X_direct[j])

        # -1 and 1 
        y_direct_train.append(y_direct[j])

        pmids_direct_train.append(pmids_direct[j])

    #X_direct_train = [X_direct[j] for j in direct_train_indices]
    # convert to 0/1 labels
    #y_direct_train = [(y_direct[j] + 1)/2 for j in direct_train_indices] 

   
    '''
    We need a subset of the labeled data with which to 
    tune our parameters. we do this based on PMIDs.
    '''
    unique_train_pmids = list(set(direct_train_pmids))
    n_direct_train = len(unique_train_pmids)
    tuning_size = int(p_tuning * n_direct_train)
    print "tuning SDS using %s articles (pmids)." % tuning_size
    tuning_pmids = random.sample(unique_train_pmids, tuning_size)
    # get the tuning indices based on this
    tuning_indicators = np.zeros(X_distant_train.shape[0], dtype=np.int8)
    for row, pmid in enumerate(pmids_distant_train):
        if pmid in tuning_pmids:
            tuning_indicators[row] = 1
        
    tuning_indices = tuning_indicators.nonzero()[0]
    X_tune = X_distant_train[tuning_indices]
    y_tune = y_distant_train[tuning_indices]
 

    # will come in handy later!
    non_tuning_indices = np.where(tuning_indicators == 0)[0]

    '''
    n_direct_train = directly_supervised_indicators_train[directly_supervised_indicators_train>0].shape[0]

    #n_direct_train = len(direct_train_indices)
    # hold this many instances out for direct model
    ## TODO i think you should sample from the train_PMIDS???

    tuning_size = int(p_tuning * n_direct_train)
    print "tuning SDS using %s instances." % tuning_size
    tuning_indices = random.sample(range(n_direct_train), tuning_size)

    # inverse of the 'tuning indices'
    non_tuning_indices = _inverse_indices(n_direct_train, tuning_indices)
    


    X_tune = X_distant_train[tuning_indices]
    y_tune = y_distant_train[tuning_indices]
    '''
    y_tune = (y_tune + 1)/2

    #X_tune = sp.vstack([X_direct_train[j] for j in tuning_indices])
    #y_tune = np.array([y_direct_train[j] for j in tuning_indices])

    # for error scaling, below
    lambda_ = len(y_tune[y_tune<=0])/float(len(y_tune[y_tune==1]))

    # now train the SDS model that predicts
    # whether candidate sentences are indeed good
    # fits (aim for high-precision)

    # 4/10
    # class_weight="auto"
    # 4/21 - changed back to "auto"

    # maybe switch to l1?
    m_sds = get_lr_clf(class_weight="auto", scoring="f1", C_range=[1, 10, 100, 1000, 5000])


    ### 
    # note that X_direct is a misnomer; this is really
    # X_tilde, though just for training instances!

    ####
    # 4/22 -- should we be exposing the tuning indices here??
    #train_indicators_for_sds = np.ones(len(X_direct_train), dtype=np.int8)

    X_direct_train_non_tuning = []
    y_direct_train_non_tuning = []

    
    for pmid_i, pmid in enumerate(pmids_direct_train):
        if not pmid in tuning_pmids:
            try:
                #print "pmid: %s" % pmid 
                #print "index: %s" % pmid_i
                #train_indicators_for_sds[pmid_i] = 0
                X_direct_train_non_tuning.append(X_direct_train[pmid_i])
                y_direct_train_non_tuning.append(y_direct_train[pmid_i])
            except: 
                pdb.set_trace()


    # would be good to see how well we do here?
    # specifically maybe cross fold validation for f1?
    m_sds.fit(X_direct_train_non_tuning, y_direct_train_non_tuning)
    # i think you can do cross_validation.cross_val_score
  

    ##
    # here is the grid search over different instance weights
    weights_star = None
    weight_star, direct_weight_scalar_star, best_score, clf_star = None, None, np.inf, None
    alpha_star = None 

    # for debugging only
    
    flipped_str_star = None
    n_distant_train = len(y_distant_train)
    for weight, direct_weight_scalar in itertools.product(weight_range, direct_weight_scalar_range):
        #############################################
        #  (2) Generate as many DS labels as possible
        #  for PICO. For SDS, use M_SDS to either
        #  (a) filter out labels predicted to be
        #  irrelevant, or, (b) weight instances
        #  w.r.t. predicted probability of being good
        #############################################
        updated_ys = np.zeros(n_distant_train)
        weights    = np.zeros(n_distant_train)
        _to_neg_one = lambda x : -1 if x <= 0 else 1
        flipped_count, total_pos = 0, 0
        flipped_str = []

        for i, x_tilde_i in enumerate(X_tilde_train):

            # 4/22 -- I think you want to also check the tuning indicators here!
            # added "and tuning_indicators[i] == 0"
            # is X_tilde_train the same shape ??? 
            if directly_supervised_indicators_train[i] > 0 and not X_tilde_pmids[i] in tuning_pmids:
                # we do not overwrite direct labels.
                #print "directly labeled!"

                weights[i] = weight * direct_weight_scalar # or something big, since these are direct?
                # (remember that we overwrote the DS labels with
                #   the direct above)
                updated_ys[i] = y_distant_train[i]
                if y_distant_train[i] > 0:
                    pass
                    #print "directly supervised label: %s for sent %s --" % (y_distant_train[i], sentences[i])
            else:
                ### what's up with tables? is it calling them positive??!
                
                # 1. check weights of last 6 entries (should be negative!
                    # m_sds.best_estimator_.coef_[0][-5:]
                # 2. find examples of tables in sentences (sentences_i)

                # distantly supervised; weight accordingly.
                # 6/1 no sense in scaling both the distant and direct!
                weights[i] = weight #* distant_weight
                
                if x_tilde_i is not None and y_distant_train[i] > 0:

                    predicted_prob_i = m_sds.predict_proba(x_tilde_i)[0][1]
                    pred = m_sds.predict(x_tilde_i)
                    #pred = _to_neg_one(pred)
                    cur_sent = sentences[i]
                    total_pos += 1

                    # wtf
                    #line_lens = [len(line) for line in sentence.split("\n") if not line==""]

                   
                    if pred < 1:
                        #print "\n --- changed from 1 to -1 with prob %s for sent:\n %s \n" % (predicted_prob_i, cur_sent)
                        flipped_str.append("\n --- changed from 1 to -1 with prob %s for sent:\n %s \n" % (predicted_prob_i, cur_sent))
                        flipped_count += 1
                        # i wonder if we shouldn't square the weights here??
                        weights[i] = weights[i] * (1-predicted_prob_i)#**2
                    else: 
                        #print "\n\n --- did not flip label (so 1) with prob %s for sent:\n %s \n" % (predicted_prob_i, cur_sent)
                        flipped_str.append("\n\n --- did not flip label (so 1) with prob %s for sent:\n %s \n" % (predicted_prob_i, cur_sent))
                        weights[i] = weights[i] * predicted_prob_i#**2


                    updated_ys[i] = _to_neg_one(pred)
                else:
                    # if x_tilde_i is None that means this is a `padding'
                    # instance that did not score high enough to be a candidate
                    # so we just stick with our -1 label.

                    ## maybe still scale by predicted prob of being negative 
                    # here??
                    updated_ys[i] = y_distant_train[i] 
                    #weights[i] = weight * distant_weight
                   
        print "ok! updated labels (flipped %s out of %s positive), now training the actual model.." % (
                        flipped_count, total_pos)


        #clf = get_DS_clf()
        for cur_alpha in alpha_range:
            clf = SGDClassifier(alpha=cur_alpha, loss="log", 
                                    class_weight="auto", shuffle=True)

            
            clf.fit(X_distant_train[non_tuning_indices], updated_ys[non_tuning_indices], 
                        sample_weight=weights[non_tuning_indices])

            
            if len(set(updated_ys))>2:
                print "!?"
                pdb.set_trace()

            # how do we do on the held out data? 
            tune_preds = clf.predict_proba(X_tune)[:,1]
            errors = abs(y_tune - tune_preds)**2
            # upweight false negative costs
            errors[y_tune==1] = errors[y_tune==1]*lambda_
            cur_score = np.sum(errors)
            print "SDS"
            #pdb.set_trace()
            print "score for alpha %s, weight %s, direct_weight_scalar %s is: %s" % (
                               cur_alpha, weight, direct_weight_scalar, cur_score)
            if cur_score < best_score:
                best_score = cur_score 
                weight_star= weight
                direct_weight_scalar_star = direct_weight_scalar
                alpha_star = cur_alpha
                clf_star = clf
                weights_star = weights
                flipped_str_star = "\n".join(flipped_str)

                print "-- found best score so far using: weight %s and direct_weight_scalar %s, alpha %s!" % (
                        weight_star, direct_weight_scalar, alpha_star)
    
    # refit this using *all* data!
    with open("sds-flipped-sentences-str.txt", 'wb') as outf:
        outf.write(flipped_str_star)
    #print "\n\n -- for winner, here's the flipped summary --\n\n"
    #print flipped_str_star
    print "\nfitting final model!"
    clf_star = SGDClassifier(alpha=alpha_star, loss="log",  class_weight="auto", shuffle=True)
    clf_star.fit(X_distant_train, updated_ys, sample_weight=weights_star)
    #pdb.set_trace()
    return clf_star


class Nguyen:
    '''
    This is the model due to Nguyen et al. [2011],
    which is just an interpolation of probability
    estimates from two models: one trained on the
    directly supervised data and the other on the
    distantly supervised examples.
    '''
    def __init__(self, m1, m2, m1_name=None, m2_name=None):
        self.m1 = m1 
        self.m1_name = m1_name or "model 1"

        self.m2 = m2 
        self.m2_name = m2_name or "model 2"

        self.meta_clf = None

    def fit(self, X, y):
        ###
        # combine predicted probabilities from
        # our two constituent models.

        #X1 = self.m1.predict_proba(X)[:,1]
        #X2 = self.m2.predict_proba(X)[:,1]
        #XX = sp.vstack((X1,X2)).T

        #XX = self._transform(X)
        self.beta = self._minimize(X,y)


    def _minimize(self, X, y):

        p1s = self.m1.predict_proba(X)[:,1]
        p2s = self.m2.predict_proba(X)[:,1]

        yy = (y + 1)/2
        lambda_ = len(yy[yy==0])/float(len(yy[yy==1]))
        #lambda_ = 1.0

        print "lambda is: %s" % lambda_
        betas = np.linspace(0,1,50)
        
        beta_star, best_score = None, np.inf

        #for alpha in alphas:
        for beta in betas: 
            preds = beta*p1s + (1-beta)*p2s
            errors = abs(yy - preds)**2
            
            errors[yy==1] = errors[yy==1]*lambda_

            cur_score = np.sum(errors)
            #print "NGUYEN"
            #pdb.set_trace()
            print "score for beta: %s is %s" % (beta, cur_score)
            if cur_score < best_score:
                beta_star = beta 
                #alpha_star = alpha
                best_score = cur_score


        print "best beta is: %s with score: %s" % (beta_star, best_score)
        #print "best alpha (weight for %s): %s, beta (weight for %s): %s with score: %s" % (
        #    self.m1_name, alpha_star, self.m2_name, beta_star, best_score)
        #pdb.set_trace()
        return beta_star
       

    def _transform_y(self, X, y):
        pass 
        #p1s = self.m1.predict_proba(X)[:,1]
        #return y - p1s

    def _transform(self, X):
        '''
        go from raw input to `stacked' representation
        comprising m1 and m2 probability predictions.
        '''

        '''
        x1 = self.m1.predict_proba(x)[0][1]
        x2 = self.m2.predict_proba(x)[0][1]
        x_stacked = np.array([x1, x2])
        return x_stacked
        '''
        X1 = self.m1.predict_proba(X)[:,1]
        X2 = self.m2.predict_proba(X)[:,1]
        XX = sp.vstack((X1,X2)).T

        return XX


    def predict(self, X, threshold=.5):
        '''
        X_stacked = self._transform(X)
        return self.meta_clf.predict(X_stacked)
        '''
        return 1 if self.predict_proba(X) >= threshold else -1

    def predict_proba(self, X):
        # fixed???
        pred = self.beta * self.m1.predict_proba(X)[0][1] + (1-self.beta)*self.m2.predict_proba(X)[0][1]
        return [1-pred, pred] 
                

        '''
        X_stacked = self._transform(X)
        return self.meta_clf.predict_proba(X_stacked)[:,1]
        '''

    def decision_function(self, X):
        ''' just to conform to SGD API '''
        return [self.predict_proba(X)[1]]
        

def _inverse_indices(nrows, indices):
    # for when you want to grab -[list of indices].
    # this seems kinda hacky but could not find a
    # more numpythonic way of doing it...
    return list(set(range(nrows)) - set(indices))


def build_direct_only_model(X_train, y_train, direct_indices):
    directly_supervised_indices = direct_indices.nonzero()
    X_direct = X_train[directly_supervised_indices]
    y_direct = y_train[directly_supervised_indices]
    m = get_DS_clf()
    try:
        m.fit(X_direct, y_direct)
    except:
        pdb.set_trace()
    return m 

def build_nguyen_model(X_train, y_train, direct_indices, p_validation=.25):
    '''
    This implements the model of Nguyen et al., 2011.

    We assume that X_train, y_train comprise *both*
    distantly and directly supervised instances, and
    that the (row) indices of the latter are provided
    in the direct_indices list.

    p_valudation is a parameter encoding what fraction
    of the *directly supervised* instances to use to
    fit the \alpha, \beta interpolation parameters, i.e.,
    to fit the `stacked' model with. See page 737 of Nguyen
    paper (it's not super clear on this detail, actually).
    '''

    # train model 1 on direct and model 2 on
    # distant supervision

    ##############
    # model 1    #
    ##############

    # direct_indices is a binary vector spanning the 
    # whole corpus with ones indicating that direct supervision
    # was provided for corresponding sentence. 
    # here we are concerned with only these.
    directly_supervised_indices = direct_indices.nonzero()

    X_direct = X_train[directly_supervised_indices]
    y_direct = y_train[directly_supervised_indices]

    # the catch here is that we actually do
    # not want to train on *all* direct
    # supervision, because we need to use some
    # of this to fit the 'meta' or 'stacked'
    # model (else we risk letting the strictly
    # supervised model appear better than it
    # is, since it has seen the
    n_direct = X_direct.shape[0]
    validation_size = int(p_validation * n_direct)
    print "fitting Nguyen using %s instances." % validation_size
    validation_indices = random.sample(range(n_direct), validation_size)
    X_validation = X_direct[validation_indices]
    y_validation = y_direct[validation_indices]

    # now get the set to actually train on...
    
    direct_train_indices = _inverse_indices(n_direct, validation_indices)
    m1 = get_direct_clf()
    m1.fit(X_direct[direct_train_indices],
                y_direct[direct_train_indices])


    #############
    # model 2   #
    #############
    DS_indices = _inverse_indices(X_train.shape[0], list(directly_supervised_indices[0]))
    X_DS = X_train[DS_indices]
    y_DS = y_train[DS_indices]
    m2 = get_DS_clf()
    m2.fit(X_DS, y_DS)

    
    # now you need to combine these somehow.
    # i think it would suffice to run predictions
    # through a regressor?
    nguyen_model = Nguyen(m1, m2, m1_name="direct", m2_name="distant")
    print "fitting Nguyen model to validation set..."
    nguyen_model.fit(X_validation, y_validation)
    
    # now refit m1 using all data
    print "ok, found best beta; now refitting using all data"
    nguyen_model.m1.fit(X_direct, y_direct)

    return nguyen_model

def _unpickle_PICO_DS(y_dict_pickle, domain_v_pickle):

    with open(os.path.join(DATA_PATH, y_dict_pickle)) as y_dict_f:
        print "unpickling sentences and y dict (from %s)" % y_dict_pickle
        sentences_y_dict = pickle.load(y_dict_f)
        print "done unpickling."

    with open(os.path.join(DATA_PATH,domain_v_pickle)) as domain_f:
        domain_vectorizers = pickle.load(domain_f)

    return sentences_y_dict, domain_vectorizers

def get_direct_clf():
    # for now this is just the same as the
    # DS classifier; may want to revisit this though
    return get_DS_clf()

def get_DS_clf():
    # .0001, .001,
    tune_params = [{"alpha":[.00001, .0001, .001, .01, .1, 1, 10]}]
    #clf = GridSearchCV(LogisticRegression(), tune_params, scoring="accuracy", cv=5)

    ###
    # note to self: for SGDClassifier you want to use the sample_weight
    # argument to instance-weight examples!
    clf = GridSearchCV(SGDClassifier(shuffle=True, class_weight="auto", loss="log"),
             tune_params, scoring="f1")

    return clf

def run_sds_experiment(iters=10):
    # X and y for supervised distant supervision
    DS_learning_tasks = get_DS_features_and_labels()
    for domain, task in DS_learning_tasks.items():
        # note that 'task' here is comprises
        # ('raw') extracted features and labels
        X_d, y_d = generate_X_y(task)
        pmids_d = task["pmids"]

        # for experimentation, uncomment below...
        # in general these would be fed into
        # the build_clf function, below
        #return X_d, y_d, task["pmids"]
        for iter_ in xrange(iters):
            train_X, train_y, test_X, test_y = train_test_by_pmid(X_d, y_d, pmids_d)
            model = build_clf(train_X, train_y)
            model.fit(train_X, train_y)

            # To evaluate performance with respect
            # to the *true* (or primary) task, you
            # should do the following
            #   (1) Train the SDS model (M_SDS) using all train_X, train_y
            #       here.
            #   (2) Generate as many DS labels as possible
            #       for PICO. For SDS, use M_SDS to either
            #       (a) filter out labels predicted to be
            #       irrelevant, or, (b) weight instances
            #       w.r.t. predicted probability of being good
            #   (3) Other methods should basically just use
            #       the DS labels directly, possibly in combination
            #       with train_X, train_y
            #   (4) Train the `primary' model M_P using the entire
            #       distantly derived `psuedo-labeled' corpus
            #   (5) Assess performance using held out labeled
            #       data.
            pdb.set_trace()




def train_test_by_pmid(X, y, pmids, train_p=.8):
    '''
    Randomly sample 80% of the pmids as training instances;
    the rest will be testing
    '''
    unique_pmids = list(set(pmids))
    train_size = int(train_p * len(unique_pmids))
    train_pmids = random.sample(unique_pmids, train_size)
    test_pmids = list(set(pmids) - set(train_pmids))
    train_X, train_y, test_X, test_y = [], [], [], []
    for i, pmid in enumerate(pmids):
        if pmid in train_pmids:
            train_X.append(X[i])
            train_y.append(y[i])
        else:
            test_X.append(X[i])
            test_y.append(y[i])
    return train_X, train_y, test_X, test_y


## TODO probably mess with class weights; possibly undersample?
# in general we need to check to see how well calibrated our
# model is!
def build_clf(X, y):
    # .0001, .001,
    tune_params = [{"C":[.0001, .001, .01, .1, .05, 1, 2, 5, 10]}]
    clf = GridSearchCV(LogisticRegression(class_weight="auto"),
                        tune_params, scoring="f1", cv=5)
    return clf

def get_lr_clf(class_weight=None, scoring="accuracy", C_range=None):
    if C_range is None:
        C_range = [.00001, .0001, .001, .01, .1, .05, 1, 5, 10, 100]
    tune_params = [{"C":C_range}]
    clf = GridSearchCV(LogisticRegression(class_weight=class_weight),
                        tune_params, scoring=scoring, cv=5)
    return clf


def _score_to_ordinal_lbl(y_str):
    return float(y_str.strip())

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _score_to_binary_lbl(y, zero_one=True, threshold=2, count_relevant_tables=False):
    # will label anything >= threshold as '1'; otherwise 0
    # (or -1, depending on the zero_one flag).
    #
    # the 't' would indicate a table; we treat these as
    # irrelevant (-1s).
    if not _is_number(y):
        # then this is a table; we'll return 
        # 0/-1 here (as this is the assumption for now)
        assert "t" in y
        
        if count_relevant_tables and y in ("t1", "t2"): 
            return 1
    elif int(y) >= threshold:
        return 1

    return 0 if zero_one else -1

def generate_X_y(DS_learning_task, binary_labels=True,
                    y_lbl_func=_score_to_binary_lbl,
                    return_pmids_too=False, just_X=False):
    '''
    This goes from the output generated by get_DS_features_and_labels
    (below) *for a single domain* to feature vectors and scalar/binary
    labels.
    '''
    all_domain_texts = []

    for X_i in DS_learning_task["X"]:
        # the first bit of the components is the text content
        # extracted
        all_domain_texts.append(X_i[1])

    vectorizer = TfidfVectorizer(stop_words='english', min_df=3,
                                    max_features=50000, decode_error=u'ignore')
    print "fitting vectorizer ... "
    vectorizer.fit(all_domain_texts)
    print "ok."

    X, y, pmids = [], [], []

    for X_i, y_i, pmid_i in zip(
            DS_learning_task["X"], DS_learning_task["y"], DS_learning_task['pmids']):
        X_i_numeric, X_i_text = X_i
        X_v = vectorizer.transform([X_i_text])[0]
        # note that the numeric stuff is 2nd!
        X_combined = sp.sparse.hstack((X_v, X_i_numeric))
        X.append(np.asarray(X_combined.todense())[0])
        y.append(y_lbl_func(y_i))
        pmids.append(pmid_i)


    if return_pmids_too:
        # also returning the actual vectorizer for
        # later use...
        return X, y, pmids, vectorizer


    return X, y


#
# 2/2/2015 note that the mysterious '158' below reflects when we made the change
# to labeling tables
# 
# 3/19/2015 we now remove rows 1-157 in the preprocessing/label merging step!
#
'''
def get_DS_features_and_labels(candidates_path="sds/annotations/for_labeling_sharma.csv",
                                labels_path="sds/annotations/sharma-merged-labels-1-30-15.csv",

'''
# for the moment making labels and candidates the same!!! this is simpler
# and should really be the general approach
# 
# was 8-2-24.csv
# 4/17 -- was figure8-3-19.csv   

'''
12/26/15 making revisions for JMLR -- previously was was 

candidates_path="sds/annotations/master/figure8-4-20-15.csv",
                                labels_path="sds/annotations/master/figure8-4-20-15.csv",
'''
def get_DS_features_and_labels(candidates_path="sds/annotations/master/figure8-12-26-15.csv",
                                labels_path="sds/annotations/master/figure8-12-26-15.csv",
                                label_index=-1,
                                max_sentences=10, cutoff=4,
                                normalize_numeric_cols=True,
                                start_row=0):
    '''
    Load in the supervision (X,y) for DS task.

    We are making the assumption that files containing *labels* are (at least
    optionally) distinct from the file containing the corresponding labels.
    The former path is specified by the "candidates_path" argument; the latter
    by the "labels_path". This was an easy way to get out of unicode hell.
    This way, you can use the candidates file you originally generate directly
    and combine this with the labels returned (in whatever format they may be).

    We make the assumption that the 'original' file comprises
    the following fields (in this order!)

        study id,PICO field,CDSR sentence,candidate sentence

    And the labels file should have the labels in the label_index (by default,
    the last column in the sheet).

    This function returns a dictionary, where the keys are the domains of interest,
    specifically "CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS" and "CHAR_OUTCOMES".
    Each of these, in turn, contains X and y vectors (of equal cardinality).
    The X instances are tuples, where the first entry is a vector of numerical
    features, while the second is the string containing (whitespace-delimited)
    textual features. The y vectors are singletons for each instance and are
    currently strings \in {"t0", "t1", "0", "1", "2"}.
    '''
    biview = biviewer.PDFBiViewer()

    # this is just to standardize terms/strings
    pico_strs_to_domains = dict(zip(["PARTICIPANTS","INTERVENTIONS","OUTCOMES"], domains))

    X_y_dict = {}
    for d in domains:
        # X, y and pmids for each domain. the latter
        # is so we can know which studies each candidate
        # was generated for.
        X_y_dict[d] = {"X":[], "y":[], "pmids":[], "sentences":[], "CDSR_ids":[]}
        # this is a map from domains to dictionaries, which
        # in turn map from pmids to target sentences
        #domains_pmids_targets = dict(zip(domains, [{}]*3))
        domains_pmids_targets = {}
        for domain in domains:
            domains_pmids_targets[domain] = {}


    print "reading candidates from: %s" % candidates_path
    print "and labels from: %s." % labels_path

    # instantiate this outside the loop for efficiency;
    # we need an instance of this to generate certain
    # features
    pico_v = PICO_vectorizer()

    with open(candidates_path, 'rb') as candidates_file, open(labels_path, 'rU') as labels_file:
        candidates = list(unicode_csv_reader(candidates_file))
        # note that we just use a vanilla CSV reader for the
        # labels!
        labels = list(csv.reader(labels_file))

        if len(candidates) != len(labels):
            print "you have a mismatch between candidate sentences and labels!"
            pdb.set_trace()

        if start_row is None:
            # skip headers
            start_row = 1
        else:
            print "--- starting at row: %s ---" % start_row
        candidates = candidates[start_row:]
        labels = labels[start_row:]

        ###
        # note that the structure of the annotations
        # file means that studies are repeated, and
        # there are multiple annotated sentences
        # *per domain* for each study
        for candidate_line, label_line in zip(candidates, labels):
            try:
                study_id, PICO_field, target_sentence, candidate_sentence = candidate_line[:4]
                PICO_field = pico_strs_to_domains[PICO_field.strip()]
            except:
                pdb.set_trace()


            # get the study from the PMID.
            # 12/8/14. this is more complicated than originally imagined,
            # because we overlooked the detail that PMIDs are not
            # unique keys for the CDSR (!). multiple instances
            # of a given article (PMID) may exist in the database.
            ##
            studies = biview.get_study_from_pmid(study_id, all_entries=True)
            study = None

            for study_ in studies:
                if target_sentence == study_.cochrane["CHARACTERISTICS"][PICO_field].decode(
                        "utf-8", errors="ignore"):
                    study = study_
                    break
            else:
                # we should certainly never get here;
                # this would mean that none of the retreived
                # studies (studies with this PMID) match the
                # labeled candidate sentence
                print "err ... this should not happen -- something is very wrong."
                pdb.set_trace()



            ###
            # This part is kind of hacky. We go ahead and retrieve
            # all the candidate sentences here to derive additional
            # features that are not otherwise readily available
            # (e.g., the relative rank of the candidate sentence)
            ###

            pdf = study.studypdf['text']
            study_id = "%s" % study[1]['pmid']
            CDSR_id = study.cochrane['cdsr_filename']
            pdf_sents = pico_DS.sent_tokenize(pdf)

            # note that this should never return None, because we would have only
            # written out for labeling studies/fields that had at least one match.

            ranked_sentences, scores, shared_tokens = \
                        pico_DS.get_ranked_sentences_for_study_and_field(study,
                            PICO_field, pdf_sents=pdf_sents)

            # don't take more than max_sentences sentences
            num_to_keep = min(len([score for score in scores if score >= cutoff]),
                                    max_sentences)


            target_text = study.cochrane["CHARACTERISTICS"][PICO_field]
            candidates = ranked_sentences[:num_to_keep]
            scores = scores[:num_to_keep]
            shared_tokens = shared_tokens[:num_to_keep]

            try:
                cur_candidate_index = candidates.index(candidate_sentence)
            except:
                pdb.set_trace()

            X_i = extract_sds_features(candidate_sentence, shared_tokens, candidates,
                                    scores, cur_candidate_index, pico_vectorizer=pico_v)

            # @TODO we may want to do something else here
            # with the label (e.g., maybe binarize it?)
            y_i = label_line[label_index]

            # 6/23: handling issue of occasional 'missing' labels in the
            # annotations files.
            if y_i.strip() != "":
                X_y_dict[PICO_field]["X"].append(X_i)
                X_y_dict[PICO_field]["y"].append(y_i)
                X_y_dict[PICO_field]["pmids"].append(study_id)
                X_y_dict[PICO_field]["CDSR_ids"].append(CDSR_id)

                ###
                # note: studies[1].cochrane['cdsr_filename']
                # does return a unique identifier for cochrane 
                # studies... 
                ###

                # also include the actual sentences
                X_y_dict[PICO_field]["sentences"].append(candidate_sentence)
                # and the corresponding target sentence
                #X_y_dict[PICO_field]["targets"].append(target_text)

                domains_pmids_targets[PICO_field][study_id] = target_text
            else:
                print "skipping %s; no label?!" % label_line

    ###
    # @TODO refactor to use the normalize function in the
    # sklearn.preprocessing module
    ####
    if normalize_numeric_cols:
        # @TODO ugh, yeah this is not very readable
        # at the very least should factor this out into
        # separate normalizing routine...


        # record the normalizing terms for later use
        z_dict = {}
        for domain in domains:
            z_dict[domain] = []
            domain_X = X_y_dict[domain]["X"]
            #num_numeric_feats = len(X_y_dict.values()[0]["X"][0][0])
            num_numeric_feats = len(domain_X[0][0])

            col_Zs = [0]*num_numeric_feats

            for j in xrange(num_numeric_feats):

                all_vals = [X_i[0][j] for X_i in domain_X]
                # note that this will have no effect on
                # binary features (since the z terms for
                # these will be 1)
                z_j = float(max(all_vals))
                if z_j == 0:
                    z_j = 1.0

                z_dict[domain].append(z_j)
                for i in xrange(len(domain_X)):
                    # this is not cool
                    X_y_dict[domain]["X"][i][0][j] = X_y_dict[domain]["X"][i][0][j] / z_j

        return X_y_dict, z_dict, domains_pmids_targets

    return X_y_dict, domains_pmids_targets




def get_DS_features_for_all_data(numeric_col_zs,
                max_sentences=10, cutoff=4):
    '''
    basically we need to copy the above but build up the
    same dictionary for SDS!

    Crucially, by construction the feature vectors here
    will be ordered the exact same as those returned by
    all_PICO_DS in pico_DS.py.
    '''

    ###
    # 6/3 -- store the field texts, too!
    target_text_dict = {}

    # Note that by construction, *all* of these would
    # typically have `1's as corresponding labels.
    X_pmid_dict = {} # fields to feature vectors
    for pico_field in pico_DS.PICO_DOMAINS:
        X_pmid_dict[pico_field] = {"X":[], "pmids":[]}

        # this is to map from pmids to texts
        target_text_dict[pico_field] = defaultdict(dict)


    p = biviewer.PDFBiViewer()

    pico_vectorizer = PICO_vectorizer()

    # By design this will iterate in the same
    # order as `all_PICO_DS' in pico_DS. This is crucial
    # because these items must be aligned.
    for n, study in enumerate(p):
        if n % 1000 == 0:
            print "on study %s" % n

        pdf = study.studypdf['text']
        study_id = "%s" % study[1]['pmid']

        pdf_sents = pico_DS.sent_tokenize(pdf)

        for pico_field in pico_DS.PICO_DOMAINS:

            # 6/3 save the text, too
            target_text = study.cochrane["CHARACTERISTICS"][pico_field]
            target_text_dict[pico_field][study_id] = target_text

            X_i = generate_sds_feature_vectors_for_study(
                        study, pico_field, pdf_sents, 
                        pico_vectorizer=pico_vectorizer)


            if X_i:
                X_pmid_dict[pico_field]["X"].extend(X_i)
                X_pmid_dict[pico_field]["pmids"].extend([study_id]*len(X_i))


    # here you convert to count vectors

    # and now we normalize/scale numeric values via the
    # provided constants which were recorded from the
    # training data
    for domain in pico_DS.PICO_DOMAINS:
        domain_X = X_pmid_dict[domain]["X"]

        #num_numeric_feats = len(domain_X[0][0])
        #assert (len(numeric_col_zs[domain]) == num_numeric_feats)
        num_numeric_feats = None
        for i in xrange(len(domain_X)):
            X_i = domain_X[i]
            if X_i is not None:
                if num_numeric_feats is None:
                    num_numeric_feats = len(X_i[0])
                    assert len(numeric_col_zs[domain]) == num_numeric_feats

                for j in xrange(num_numeric_feats):
                    z_j = numeric_col_zs[domain][j]
                    # yuck.
                    if (X_i[0][j] / z_j) > 10:
                        pdb.set_trace()

                    X_pmid_dict[domain]["X"][i][0][j] = X_i[0][j] / z_j



    return X_pmid_dict, target_text_dict

###
# this will generate the feature vectors for the given
# study and PICO field. the task then is to iterate over
# *all* the data in the CDSR and generate such vectors
# for each field. Then apply the trained SDS model to filter
# out or weight these!
#
def generate_sds_feature_vectors_for_study(study, PICO_field, pdf_sents,
                                    max_sentences=10, cutoff=4, pico_vectorizer=None):
    '''
    This wil generate a set of feature vectors corresponding to the
    the top candidates found in pdf_sents matching the parametric
    Cochrane study object for the given field.

    More specifically, given a Cochrane study object, a PICO
    field and a tokenized (matched) PDF, we here build and return
    a feature vector for each candidate that has >= cutoff overlap
    with the corresponding study entry (up to max_sentences such
    candidates will be returned).
    '''

    DS = pico_DS.get_ranked_sentences_for_study_and_field(study,
            PICO_field, pdf_sents=pdf_sents)


    if DS is None:
        # this is possible if the Cochrane entry
        # (as encapsulated by the study object)
        # does not contain an entry for the given
        # PICO field. probably we should filter such
        # studies out?
        return False # or should this be []?

    ranked_sentences, scores, shared_tokens = DS

    # essentially we just drop sentences that don't meet
    # the cutoff. and we consider at most <= max_sentences.
    num_to_keep = min(len([score for score in scores if score >= cutoff]),
                            max_sentences)

    target_text = study.cochrane["CHARACTERISTICS"][PICO_field]
    candidates = ranked_sentences
    scores = scores
    shared_tokens = shared_tokens

    # iterate over the sentences that constitute the candidate set
    # and build a feature vector corresponding to each.
    X = []
    # also keep track of the actual sentences represented by X
    candidate_sentences = []

    if pico_vectorizer is None:
        pico_vectorizer = PICO_vectorizer()

    # remember, these are *ranked* w.r.t. (naive) similarity
    # to the target text, hence the cur_candidate_index
    for cur_candidate_index, candidate in enumerate(candidates[:num_to_keep]):
        # shared tokens for this candidate
        cur_shared_tokens = shared_tokens[cur_candidate_index]
        candidate_text = candidates[cur_candidate_index]

        # remember, this is actually a tuple where the
        # first entry comprises numeric features and
        # the second contains the textual features
        ####
        X_i = extract_sds_features(candidate_text, shared_tokens[:num_to_keep],
                                    candidates[:num_to_keep],
                                    scores[:num_to_keep], cur_candidate_index,
                                    pico_vectorizer=pico_vectorizer)

        X.append(X_i)
        # i don't think we actually need these
        candidate_sentences.append(candidate_text)

    # pad out the vector with non-candidates for indexing purposes.
    for noncandidate in candidates[num_to_keep:]:
        X.append(None)
        candidate_sentences.append(None)

    return X

###
# this returns labels and then strings + numeric feature vectors; see the 
# generate_X_y method for generation of the actual feature vectors
# used
def extract_sds_features(candidate_sentence, shared_tokens, candidates,
                                scores, cur_candidate_index, pico_vectorizer=None):

    if pico_vectorizer is None: 
        pico_vectorizer = PICO_vectorizer()

    # textual features
    X_i_text = candidate_sentence

    # extend X_i text with shared tokens (using
    # special indicator prefix "shared_")
    cur_shared_tokens = shared_tokens[cur_candidate_index]
    X_i_text = X_i_text + " ".join(["shared%s" % tok for
                                    tok in cur_shared_tokens if tok.strip() != ""])


    ## numeric features
    X_i_numeric = []

    # length should probably be a binary feature!
    # and in any case, this is already coded for elsewhere
    #X_i_numeric.append(len(candidate_sentence.split(" ")))

    

    #X_i_numeric.append(len(candidates) - cur_candidate_index)
    candidate_score = scores[cur_candidate_index]
    X_i_numeric.append(candidate_score - np.mean(scores))
    X_i_numeric.append(candidate_score - np.median(scores))
    #X_i_numeric.append(candidate_score)


    # 1/21 -- adding 'structural' features
    #structural_features = pico_DS.extract_structural_features(candidate_sentence)
    
    structural_features = pico_vectorizer.extract_structural_features(candidate_sentence)
    
    X_i_numeric.extend(structural_features.tolist())

    # note that we'll need to deal with merging these
    # textual and numeric feature sets elsewhere!
    X_i = (X_i_numeric, X_i_text)
    return X_i


'''
Routines to generate XML for entailment task (for Katrin et al.)
'''
def generate_entailment_output(candidates_path="sds/annotations/for_labeling_sharma.csv",
                                labels_path="sds/annotations/sharma-merged-labels.csv",
                                label_index=-1):
    '''
    Generate and output data for the `textual entailment'
    task.
    '''

    ###
    # @TODO should probalby make this a global var or something
    # since it's defined multiply
    pico_strs_to_domains = dict(
        zip(["PARTICIPANTS", "INTERVENTIONS","OUTCOMES"], domains))



    entailment_out = ['''<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE entailment-corpus SYSTEM "rte.dtd">\n<entailment-cdsr-corpus>''']

    with open(candidates_path, 'rb') as candidates_file, open(labels_path, 'rU') as labels_file:
        candidates = list(unicode_csv_reader(candidates_file))
        # note that we just use a vanilla CSV reader for the
        # labels!
        labels = list(csv.reader(labels_file))

        if len(candidates) != len(labels):
            print "you have a mismatch between candidate sentences and labels!"
            pdb.set_trace()

        # skip headers
        candidates = candidates[1:]
        labels = labels[1:]

        ###
        # note that the structure of the annotations
        # file means that studies are repeated, and
        # there are multiple annotated sentences
        # *per domain*.
        for i, (candidate_line, label_line) in enumerate(zip(candidates, labels)):
            #print annotation_line
            try:
                study_id, PICO_field, target_sentence, candidate_sentence = candidate_line[:4]
                PICO_field = pico_strs_to_domains[PICO_field.strip()]
            except:
                pdb.set_trace()

            y_i = label_line[label_index]

            cur_pair_str = generate_XML(i, study_id, PICO_field, target_sentence, candidate_sentence, y_i)
            #pdb.set_trace()

            entailment_out.append(cur_pair_str)

    entailment_out.append("</entailment-cdsr-corpus>")
    with open("cdsr-entailment.xml", 'wb') as outf:
        # you are going to get unicode errors here!
        outf.write("\n".join(entailment_out).encode("UTF-8", errors='ignore'))

def generate_XML(pair_id, pmid, PICO_field, t, h, label):
    '''
    Given a something like

        <pair id=5 pmid="18275573" value=2>
        <t> Australian and New Zealand Journal of Obstetrics and Gynaecology 2008; 48: DOI: </t>
        <h> 70 women presenting between 24 and 34 weeks' gestation with symptoms and signs of threatened preterm labour, where acute symptoms were arrested following use of tocolytic medication.</h>
        </pair>
    '''
    entailment_str = u'''<pair id=%s pmid='%s' pico_field='%s' value=%s>\n<t> %s </t>\n<h> %s </h>\n</pair>\n\n''' % (pair_id, pmid, PICO_field, label, t, h)
    return entailment_str


''' completely ripped off from Alex Martelli '''
def unicode_csv_reader(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        try:
            yield [unicode(cell, 'utf-8') for cell in row]
        except:
            pdb.set_trace()
