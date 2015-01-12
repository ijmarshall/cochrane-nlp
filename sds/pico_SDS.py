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

import pdb
import random
import csv
import pickle
import os 
import time
import copy 
from datetime import datetime
from collections import defaultdict

import numpy as np 
import scipy as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from readers import biviewer

# this module allows us to grab the ranked
# sentences. this is possibly not the ideal 
# location.
from experiments import pico_DS 
domains = pico_DS.PICO_DOMAINS 

### TODO TODO TODO 12/22
#       * further verification of Nguyun
#       * additional sanity checks on data/supervision/DS matching
#       * profit.
###
# note: the *_1000 pickles are subsets of the available DS to 
# speed things up for experimentation purposes!
def run_DS_PICO_experiments(iters=5, cv=True, test_proportion=None, 
                            strategy="baseline_DS", output_dir="sds/results/",
                            y_dict_pickle="sds/sentences_y_dict.pickle", 
                            domain_v_pickle="sds/vectorizers.pickle"):
    '''
    Runs multiple (iters) experiments using the specified strategy.

    If `cv' is true, cross-validation will be performed using `iters' splits. 
    Otherwise, `training_proportion' should be provided to specify the 
    fraction of examples to be held out for each iter.
    '''
    
    output_path = os.path.join(output_dir, 
            "%s-results-%s.txt" % (int(time.time()), strategy))
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
        sentences_y_dict, domain_vectorizers = _unpickle_PICO_DS(y_dict_pickle,
                                                    domain_v_pickle)


    ## now load in the direct supervision we have.
    # this is kind of confusingly named; we need these 
    # DS labels for evaluation here -- we don't care for 
    # the features *unless* we are doing SDS! 
    # @TODO refactor or rename method?
    
    # Note that the z_dict here is a dicitonary mapping
    # PICO fields to vectors comprising normalization terms
    # for each numeric feature/column. 
    DS_learning_tasks, z_dict = get_DS_features_and_labels()

    ## now we divvy up into train/test splits.
    # this will be a dictionary mapping domains to 
    # lists of test PMIDs.
    test_id_lists = {}
    ## @NOTE are you sure you're handling the
    ## train/test splits correctly here? 
    if cv:
        for domain in domains:
            labeled_pmids_for_domain = list(set(DS_learning_tasks[domain]["pmids"]))
            kfolds = cross_validation.KFold(len(labeled_pmids_for_domain), 
                                                iters, shuffle=True, random_state=512)
            
            test_id_lists[domain] = []
            for train_indices, test_indices in kfolds:
                test_id_lists[domain].append(
                    [labeled_pmids_for_domain[j] for j in test_indices])

    for iter_ in xrange(iters):

        ## if we're doing cross-fold validation,
        # then we need to assemble a dictionary for 
        # this fold mapping domains to test PMIDs
        cur_test_pmids_dict = None
        if cv:
            test_ids_for_cur_fold = [test_id_lists[domain][iter_] for domain in domains]
            cur_test_pmids_dict = dict(zip(domains, test_ids_for_cur_fold))

        output_str = DS_PICO_experiment(sentences_y_dict, domain_vectorizers, 
                                            DS_learning_tasks, strategy=strategy,
                                            test_pmids_dict=cur_test_pmids_dict, 
                                            test_proportion=test_proportion,
                                            z_dict=z_dict)

        print "\n".join(output_str) + "\n\n"
        with open(output_path, 'a') as output_f:
            output_f.write("\n\n\n\n -- fold/iter %s --\n\n" % iter_)
            output_f.write("\n".join(output_str))



def DS_PICO_experiment(sentences_y_dict, domain_vectorizers, DS_learning_tasks, 
                        strategy="baseline_DS", test_pmids_dict=None, test_proportion=1,
                        use_distant_supervision_only=False, z_dict=None):
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
    output_str = []

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
        y_test_DS = [] # slightly tricky
        directly_supervised_indicators = np.zeros(len(domain_DS["pmids"]))
    
        for i, pmid in enumerate(domain_DS["pmids"]):

            cur_sentence = domain_DS["sentences"][i]
            
            if pmid not in testing_pmids:

                train_rows.append(i)

                ### 
                # here we need to overwrite any labels for which 
                # we have explicit supervision!
                if pmid in domain_supervision["pmids"] and not use_distant_supervision_only:
                    ###
                    # It is possible that this introduces some small amount of
                    # noise, in the following way. We are checking if we have 
                    # supervision for the current PMID, but this could correspond
                    # to multiple entries in the CDSR. If a sentence did not rank
                    # highly with respect to the `labeled' instance, or if it
                    # ranked highly but was considered irrelevant, *and* this assessment
                    # would not (or does not) hold for an alternative summary/target, 
                    # then we may introduce a false negative here. I think this is 
                    # rather unlikely and will, at the very least, be rare.
                    cur_label = _match_direct_to_distant_supervision(
                        domain_supervision, pmid, cur_sentence, threshold=2)

                    # if this is None, it means the current sentence
                    # was not found in the candidate set, implicitly
                    # this means it is a -1.
                    if cur_label is None:
                        cur_label = -1
                    
                    # 1/7/15 -- previously, we were not considering
                    # an instance directly supervised if cur_label
                    # came back None. This was inconsistent with  
                    #print cur_label
                    # then overwrite the existing label
                    domain_DS["y"][i] = cur_label

       
                    #domain_DS["y2"][i] = cur_label
                    # keep track of row indices that correspond
                    # to directly supervised instances
                    #directly_supervised_indices.append(i)
                    directly_supervised_indicators[i] = 1
            else:
                ####
                # Then this index is associated with a study to be used for
                # testing. 
                test_rows.append(i)
                cur_label = _match_direct_to_distant_supervision(domain_supervision, 
                                            pmid, cur_sentence, threshold=2)
                if cur_label is None:
                    cur_label = -1

                y_test_DS.append(cur_label)

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
        if len(y_test_DS) == 0:
            return ["-"*25, "\n\n no testing data! \n\n", "-"*25]


        X_train_DS = domain_DS["X"][train_rows]
        X_test_DS = domain_DS["X"][test_rows] 
        y_train_DS = np.array(domain_DS["y"])[train_rows]
        directly_supervised_indicators_train = directly_supervised_indicators[train_rows]
        clf = None 
    
        ### 
        # now train a classifier
        strategy_name = strategy.lower()
        if strategy_name == "sds":

            # we need the normalization scalars
            # for the numerical SDS features!
            assert z_dict is not None

            # this transforms the small amount of direct supervision we
            # have for the mapping task from candidate sentences to 
            # the best sentences into feature vectors and labels with
            # which to train a model
            X_direct, y_direct, direct_pmids, vectorizer = generate_X_y(
                                                    domain_supervision, 
                                                    return_pmids_too=True)

     
            ### this will align with the DS_supervision.
            # note that we now *pad* this vector to make sure
            # the entries align with the DS we have.
            X_tilde_dict = get_DS_features_for_all_data(z_dict)

            # the above returns tuples of numeric and textual
            # features for each candidate. below we vectorize 
            # these. 
            X_train_tilde = []
            for j in train_rows:
                cur_X = X_tilde_dict[domain]["X"][j]
                X_tilde_v = None
                if cur_X is not None:
                    X_tilde_v = _to_vector(cur_X, vectorizer)
                X_train_tilde.append(X_tilde_v)

            # i think the directly supervised indices list is 
            # screwed up because these are indices w.r.t. 
            # the original list, not the train subset!

            ''' end SDS magic '''
            clf = build_SDS_model(X_direct, y_direct, direct_pmids, 
                                    directly_supervised_indicators_train,
                                    X_train_tilde,
                                    train_rows, testing_pmids, 
                                    X_train_DS, y_train_DS, 
                                    domain)
            



        elif strategy_name == "nguyen":
            if len(directly_supervised_indices) == 0:
                print "no direct supervision provided!"
                return ["-"*25, "\n\n no directly labeled data!!! \n\n", "-"*25]

            clf = build_nguyen_model(X_train_DS, y_train_DS, 
                                    directly_supervised_indices)

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
        preds_for_current_pmid, rows_for_current_pmid = [], []
        precisions, accs = [], []
        

        for test_row in test_rows:
            test_pmid = domain_DS["pmids"][test_row]
            rows_for_current_pmid.append(test_row)
            if current_pmid is None:
                current_pmid = test_pmid 
            elif test_pmid != current_pmid:
                #highest_pred = np.argmax(np.array(preds_for_current_pmid))
                sorted_pred_indices = np.array(preds_for_current_pmid).argsort()
                #highest_pred_indices = rows_for_current_pmid[highest_pred]
                highest_pred_indices = sorted_pred_indices[-3:]
                #pdb.set_trace()
                true_labels = np.array(domain_DS["y"])[highest_pred_indices]
                
                # should we do something special if these were not
                # directly supervised instances?
                directly_supervised_indicators[test_row]

                precision_at_three = len(true_labels[true_labels>0])/3.0
                precisions.append(precision_at_three)

                top_pred_acc = 1 if true_labels[0] > 0 else 0
                accs.append(top_pred_acc)

                # domain_DS["sentences"][highest_prediction_index]
                current_pmid = test_pmid 
                preds_for_current_pmid = []
                rows_for_current_pmid = []
                pred_rows = []

            current_pred = clf.decision_function(domain_DS["X"][test_row])
            preds_for_current_pmid.append(current_pred[0])

        '''
        ## now rankings
        raw_scores = clf.decision_function(X_test_DS)
        auc = None  
        #pdb.set_trace()    
        auc = sklearn.metrics.roc_auc_score(y_test_DS, raw_scores)
        '''
        #pdb.set_trace()

        output_str.append("-"*25)  
        output_str.append("method: %s" % strategy)
        output_str.append("domain: %s" % domain)
        output_str.append("precisions: %s" % np.mean(precisions))
        output_str.append("accuracy: %s" % np.mean(accs))
        #output_str.append(str(sklearn.metrics.classification_report(y_test_DS, preds)))
        output_str.append("\n")
        #output_str.append("confusion matrix: %s" % str(confusion_matrix(y_test_DS, preds)))
        # output_str.append("AUC: %s" % auc)
        output_str.append("-"*25) 
        #pdb.set_trace()

    return output_str 


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
        # and, finally, what was its (human-given) label? 
        cur_label = _score_to_binary_lbl(labels[matched_sentence_index], 
                            threshold=threshold, zero_one=False)
        #y_test_DS.append(cur_label)
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
        

    
def build_SDS_model(X_direct, y_direct, pmids_direct,
                    directly_supervised_indicators_train,
                    X_tilde_train,
                    train_rows, testing_pmids, 
                    X_distant_train, y_distant_train, 
                    domain, weight=1,
                    direct_weight_scalar=10, 
                    DS_weight_scalar=1):
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
    were explicitly provided. Further, these receive


    '''
    #############################################
    # (1) Train the SDS model (M_SDS) using directly 
    #  labeled data
    #############################################
    X_direct_train, y_direct_train = [], []
 
    # don't train on testing articles!
    for i, pmid in enumerate(pmids_direct):
        x_i, y_i = X_direct[i], y_direct[i]

        if pmid not in testing_pmids:
            X_direct_train.append(x_i)
            y_direct_train.append(y_i)
 
    # now train the SDS model that predicts 
    # whether candidate sentences are indeed good
    # fits
    m_sds = get_lr_clf()
    m_sds.fit(X_direct_train, y_direct_train)


    #############################################
    #  (2) Generate as many DS labels as possible
    #  for PICO. For SDS, use M_SDS to either 
    #  (a) filter out labels predicted to be 
    #  irrelevant, or, (b) weight instances 
    #  w.r.t. predicted probability of being good
    #############################################
    updated_ys = np.zeros(len(y_distant_train))
    weights    = np.zeros(len(y_distant_train))
    _to_neg_one = lambda x : -1 if x <= 0 else 1
    for i, x_tilde_i in enumerate(X_tilde_train):
        if x_tilde_i is not None:
            if directly_supervised_indicators_train[i] < 1:            
                weight = m_sds.predict_proba(x_tilde_i)[0][1]
                pred = m_sds.predict(x_tilde_i)
                pred = _to_neg_one(pred)

                updated_ys[i] = pred #pred[0][1]
                weights[i] = weight
            else:
                # we do not overwrite direct labels.
                print "directly labeled!"
                #pdb.set_trace()
                weights[i] = weight*direct_weight_scalar # or something big, since these are direct?
                # (remember that we overwrote the DS labels with 
                #   the direct above)
                updated_ys[i] = y_distant_train[i]
        else:
            # if x_tilde_i is None that means this is a `padding'
            # instance that did not score high enough to be a candidate
            # so we just stick with our -1 label.
            updated_ys[i] = y_distant_train[i]
            weights[i] = weight * DS_weight_scalar # arbitrary

    print "ok! updated labels, now training the actual model.."

    clf = get_DS_clf()
    clf.fit(X_distant_train, updated_ys, sample_weight=weights)
    #clf.fit(X_distant_train, updated_ys)   
    return clf



class Nguyen:
    '''
    This is the model due to Nguyen et al. [2011],
    which is just an interpolation of probability 
    estimates from two models: one trained on the 
    directly supervised data and the other on the 
    distantly supervised examples.
    '''
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2 

        self.meta_clf = None

    def fit(self, X, y):
        ###
        # combine predicted probabilities from
        # our two constituent models.
     
        #X1 = self.m1.predict_proba(X)[:,1]
        #X2 = self.m2.predict_proba(X)[:,1]
        #XX = sp.vstack((X1,X2)).T

        XX = self._transform(X)

        # maybe we shouldn't even do regularization?
        self.meta_clf = LogisticRegression()
        self.meta_clf.fit(XX, y)

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


    def predict(self, X):
        X_stacked = self._transform(X)
        return self.meta_clf.predict(X_stacked)

    def predict_proba(self, X):
        X_stacked = self._transform(X)
        return self.meta_clf.predict_proba(X_stacked)[:,1]

    def decision_function(self, X):
        ''' just to conform to SGD API '''
        return self.predict_proba(X)

def build_nguyen_model(X_train, y_train, direct_indices, p_validation=.5):
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

    def _inverse_indices(nrows, indices):
        # for when you want to grab -[list of indices]. 
        # this seems kinda hacky but could not find a 
        # more numpythonic way of doing it...
        return list(set(range(nrows)) - set(indices))

    # train model 1 on direct and model 2 on 
    # distant supervision 

    ##############
    # model 1    #
    ##############
    X_direct = X_train[direct_indices]
    y_direct = y_train[direct_indices]

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
    DS_indices = _inverse_indices(X_train.shape[0], direct_indices)
    X_DS = X_train[DS_indices]
    y_DS = y_train[DS_indices]
    m2 = get_DS_clf()
    m2.fit(X_DS, y_DS)

    # now you need to combine these somehow. 
    # i think it would suffice to run predictions 
    # through a regressor? 
    nguyen_model = Nguyen(m1, m2)
    print "fitting Nguyen model to validation set..."
    nguyen_model.fit(X_validation, y_validation)
    return nguyen_model

def _unpickle_PICO_DS(y_dict_pickle, domain_v_pickle):
    with open(y_dict_pickle) as y_dict_f:
        print "unpickling sentences and y dict (from %s)" % y_dict_pickle
        sentences_y_dict = pickle.load(y_dict_f)
        print "done unpickling."

    with open(domain_v_pickle) as domain_f:
        domain_vectorizers = pickle.load(domain_f)

    return sentences_y_dict, domain_vectorizers

def get_direct_clf():
    # for now this is just the same as the 
    # DS classifier; may want to revisit this though
    return get_DS_clf()

def get_DS_clf():
    # .0001, .001, 
    tune_params = [{"alpha":[.00001, .0001, .01, .1, 1, 10]}]
    #clf = GridSearchCV(LogisticRegression(), tune_params, scoring="accuracy", cv=5)

    ###
    # note to self: for SGDClassifier you want to use the sample_weight
    # argument to instance-weight examples!
    clf = GridSearchCV(SGDClassifier(shuffle=True, class_weight="auto", loss="log"), 
             tune_params, scoring="precision")

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
    tune_params = [{"C":[.001, .001, .01, .1, .05, 1, 2, 5, 10]}]
    clf = GridSearchCV(LogisticRegression(class_weight="auto"), 
                        tune_params, scoring="f1", cv=5)
    return clf

def get_lr_clf():
    tune_params = [{"C":[.001, .001, .01, .1, .05, 1, 2, 5, 10]}]
    clf = GridSearchCV(LogisticRegression(class_weight="auto"), 
                        tune_params, cv=5)
    return clf


def _score_to_ordinal_lbl(y_str):
    return float(y_str.strip())

def _score_to_binary_lbl(y_str, zero_one=True, threshold=1):
    # will label anything >= threshold as '1'; otherwise 0
    # (or -1, depending on the zero_one flag).
    # 
    # the 't' would indicate a table; we treat these as 
    # irrelevant (-1s).
    if not "t" in y_str and float(y_str.strip()) >= threshold:
        return 1

    return 0 if zero_one else -1

def generate_X_y(DS_learning_task, binary_labels=True, 
                    y_lbl_func=_score_to_binary_lbl, 
                    return_pmids_too=False, just_X=False):
    '''
    This goes from the output generated by get_DS_features_and_labels
    (below) *for a single domain* to actual vectors and scalar/binary 
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
        X_combined = sp.sparse.hstack((X_v, X_i_numeric))
        X.append(np.asarray(X_combined.todense())[0])
        y.append(y_lbl_func(y_i))
        pmids.append(pmid_i)

    if return_pmids_too:
        # also returning the actual vectorizer for 
        # later use...
        return X, y, pmids, vectorizer

    return X, y


# "sds/annotations/for_labeling_sharma.csv"
def get_DS_features_and_labels(candidates_path="sds/annotations/for_labeling_sharma.csv",
                                labels_path="sds/annotations/sharma-merged-labels-1-2-15.csv",
                                label_index=-1,
                                max_sentences=10, cutoff=4, normalize_numeric_cols=True):
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
    currently strings \in {"0", "1", "2"}.
    '''
    biview = biviewer.PDFBiViewer() 

    # this is just to standardize terms/strings
    pico_strs_to_domains = dict(zip(["PARTICIPANTS", "INTERVENTIONS","OUTCOMES"], domains))

    X_y_dict = {}
    for d in domains:
        # X, y and pmids for each domain. the latter
        # is so we can know which studies each candidate
        # was generated for.
        X_y_dict[d] = {"X":[], "y":[], "pmids":[], "sentences":[]}


    print "reading candidates from: %s" % candidates_path
    print "and labels from: %s." % labels_path

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
            pdf_sents = pico_DS.sent_tokenize(pdf)

            # note that this should never return None, because we would have only
            # written out for labeling studies/fields that had at least one match.
            ranked_sentences, scores, shared_tokens = pico_DS.get_ranked_sentences_for_study_and_field(study, 
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
                                    scores, cur_candidate_index)

            # @TODO we may want to do something else here
            # with the label (e.g., maybe binarize it?)
            y_i = label_line[label_index]
            X_y_dict[PICO_field]["X"].append(X_i)
            X_y_dict[PICO_field]["y"].append(y_i)
            X_y_dict[PICO_field]["pmids"].append(study_id)

            # also include the actual sentences
            X_y_dict[PICO_field]["sentences"].append(candidate_sentence)

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
                z_j = float(max(all_vals))
                z_dict[domain].append(z_j)
                for i in xrange(len(domain_X)):
                    # this is not cool
                    X_y_dict[domain]["X"][i][0][j] = X_y_dict[domain]["X"][i][0][j] / z_j
        return X_y_dict, z_dict 

    return X_y_dict




def get_DS_features_for_all_data(numeric_col_zs, 
                max_sentences=10, cutoff=4):
    '''
    basically we need to copy the above but build up the 
    same dictionary for SDS! 

    Crucially, by construction the feature vectors here 
    will be ordered the exact same as those returned by 
    all_PICO_DS in pico_DS.py. 
    '''

    # Note that by construction, *all* of these would 
    # typically have `1's as corresponding labels. 
    X_pmid_dict = {} # fields to feature vectors
    for pico_field in pico_DS.PICO_DOMAINS:
        X_pmid_dict[pico_field] = {"X":[], "pmids":[]}

    p = biviewer.PDFBiViewer()
    # By design this will iterate in the same
    # order as `all_PICO_DS' in pico_DS. This is crucial
    # because these items must be aligned.
    for n, study in enumerate(p):
        if n % 100 == 0:
            print "on study %s" % n 

        # should it be > or >= ?!
        #if n >= 1000:
        #    break 

        pdf = study.studypdf['text']
        study_id = "%s" % study[1]['pmid']

        '''
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
        '''

        pdf_sents = pico_DS.sent_tokenize(pdf)

        for pico_field in pico_DS.PICO_DOMAINS:
            
            X_i = generate_sds_feature_vectors_for_study(
                        study, pico_field, pdf_sents)
            #if study_id == u'15050264':
            #    print "ok here's the issue."
            #    pdb.set_trace()
            if X_i: 
                X_pmid_dict[pico_field]["X"].extend(X_i)
                X_pmid_dict[pico_field]["pmids"].extend([study_id]*len(X_i))

    # and now we normalize/scale numeric values via the
    # provided constants which were recorded from the 
    # training data
    #pdb.set_trace()
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
                    X_pmid_dict[domain]["X"][i][0][j] = X_i[0][j] / z_j


    return X_pmid_dict 

### 
# @TODO 1/2/2015 -- this is the key method to work on.
# this will generate the feature vectors for the given 
# study and PICO field. the task then is to iterate over
# *all* the data in the CDSR and generate such vectors
# for each field. Then apply the trained SDS model to filter
# out or weight these!
#
def generate_sds_feature_vectors_for_study(study, PICO_field, pdf_sents, 
                                    max_sentences=10, cutoff=4):
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

    # remember, these are *ranked* w.r.t. (naive) similarity
    # to the target text, hence the cur_candidate_index
    for cur_candidate_index, candidate in enumerate(candidates[:num_to_keep]):
        # shared tokens for this candidate
        cur_shared_tokens = shared_tokens[cur_candidate_index]
        candidate_text = candidates[cur_candidate_index]

        # remember, this is actually a tuple where the 
        # first entry comprises numeric features and 
        # the second contains the textual features
        X_i = extract_sds_features(candidate_text, shared_tokens[:num_to_keep], candidates, 
                                    scores[:num_to_keep], cur_candidate_index)

        X.append(X_i)
        # i don't think we actually need these
        candidate_sentences.append(candidate_text)

    # pad out the vector with non-candidates for indexing purposes.
    for noncandidate in candidates[num_to_keep:]:
        X.append(None)
        candidate_sentences.append(None)

    return X 

def extract_sds_features(candidate_sentence, shared_tokens, candidates, 
                                scores, cur_candidate_index):
    # textual features
    X_i_text = candidate_sentence

    # extend X_i text with shared tokens (using 
    # special indicator prefix "shared_")
    cur_shared_tokens = shared_tokens[cur_candidate_index]
    X_i_text = X_i_text + " ".join(["shared_%s" % tok for 
                                    tok in cur_shared_tokens if tok.strip() != ""])


    ## numeric features
    X_i_numeric = []
    X_i_numeric.append(len(candidate_sentence.split(" ")))

    X_i_numeric.append(len(candidates) - cur_candidate_index)
    candidate_score = scores[cur_candidate_index]
    X_i_numeric.append(candidate_score - np.mean(scores))
    X_i_numeric.append(candidate_score - np.median(scores))
    X_i_numeric.append(candidate_score)

    # note that we'll need to deal with merging these 
    # tesxtual and numeric feature sets elsewhere!
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

            