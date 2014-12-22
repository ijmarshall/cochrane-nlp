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
    # the features in this case, though. 
    # @TODO refactor or rename method?
    DS_learning_tasks = get_DS_features_and_labels()

    ## now we divvy up into train/test splits.
    # this will be a dictionary mapping domains to 
    # lists of test PMIDs.
    test_id_lists = {}
    ## @NOTE are you sure you're handling the
    ## train/test splits correctly here? 
    if cv:
        for domain in domains:
            labeled_pmids_for_domain = DS_learning_tasks[domain]["pmids"]
            kfolds = cross_validation.KFold(len(labeled_pmids_for_domain), iters, shuffle=True)
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
                                            test_proportion=test_proportion)

        print "\n".join(output_str) + "\n\n"
        with open(output_path, 'a') as output_f:
            output_f.write("\n\n\n\n -- fold/iter %s --\n\n" % iter_)
            output_f.write("\n".join(output_str))

###
# note: the *_1000 pickles are subsets of the available DS to 
# speed things up for experimentation purposes!
def DS_PICO_experiment(sentences_y_dict, domain_vectorizers, DS_learning_tasks, 
                        strategy="baseline_DS", test_pmids_dict=None, test_proportion=1,
                        use_distant_supervision_only=False):
    '''
    This is an implementation of a naive baseline 
    distantly supervised method. 

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
        # here we need to grab the annotations used also
        # for SDS to evaluate our strategy
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
        # which rows correspond to studies that 
        # are in the testing data? we want to 
        # exclude these. 
        # note that we mutate the labels here. thus 
        # we take a deepcopy so that we don't expose these 
        # labels later on.
        domain_DS = copy.deepcopy(sentences_y_dict[domain])

        train_rows, test_rows = [], []
        y_test_DS = [] # tricky
        directly_supervised_indices = []
        for i, pmid in enumerate(domain_DS["pmids"]):
            cur_sentence = domain_DS["sentences"][i]
            if pmid not in testing_pmids:
                train_rows.append(i)

                ### 
                # here we need to overwrite any labels for which 
                # we have explicit supervision here!
                if pmid in domain_supervision["pmids"] and not use_distant_supervision_only:
                    cur_label = _match_direct_to_distant_supervision(
                        domain_supervision, pmid, cur_sentence)
                    if cur_label is not None:
                        print cur_label
                        # then overwrite the existing label
                        domain_DS["y"][i] = cur_label
                        # keep track of row indices that correspond
                        # to directly supervised instances
                        directly_supervised_indices.append(i)
            else:
                test_rows.append(i)
                cur_label = _match_direct_to_distant_supervision(domain_supervision, 
                                            pmid, cur_sentence)
                if cur_label is None:
                    cur_label = -1
                y_test_DS.append(cur_label)

        print "huzzah!"

        # do we have roughly the expected amount of supervision?
        # I think so (per domain, anyway)
        pdb.set_trace()

        ### 
        # it's possible that here we end up with an empty 
        # training set if we're working with a subset 
        # of the DS data! since this would only be for dev/testing
        # purposes, I think we can safely ignore such cases,
        # but this may cause things to break during evaluation..
        ###
        if len(y_test_DS) == 0:
            return ["-"*25, "\n\n no testing data! \n\n", "-"*25]

        X_train_DS = domain_DS["X"][train_rows]
        # the tricky part is going to be to get the
        # labels for this 
        X_test_DS = domain_DS["X"][test_rows] 
        y_train_DS = np.array(domain_DS["y"])[train_rows]
       
        clf = None 
        ###
        # @TMP the second bit of this conditional is temporary 
        # and for debugging purposes only!!!
        if strategy.lower() == "nguyen" and len(directly_supervised_indices) > 0:
            clf = build_nguyen_model(X_train_DS, y_train_DS, 
                                    directly_supervised_indices)
        else:
            clf = get_DS_clf()
            print "fitting model..."
            clf.fit(X_train_DS, y_train_DS)
            preds = clf.predict(X_test_DS)

        precision, recall, f, support = precision_recall_fscore_support(
                                            y_test_DS, preds)
        
        ## now rankings
        raw_scores = clf.decision_function(X_test_DS)
        
        auc = None      
        auc = sklearn.metrics.roc_auc_score(y_test_DS, raw_scores)
        
        output_str.append("-"*25)  
        output_str.append("method: %s" % strategy)
        output_str.append("domain: %s" % domain)
        output_str.append(str(sklearn.metrics.classification_report(y_test_DS, preds)))
        output_str.append("\n")
        output_str.append("confusion matrix: %s" % str(confusion_matrix(y_test_DS, preds)))
        output_str.append("AUC: %s" % auc)
        output_str.append("-"*25) 

    return output_str 


def _match_direct_to_distant_supervision(domain_supervision, pmid, cur_sentence):
  
    ###
    # this is tricky.
    # we have to identify the current sentence from 
    # DS in our supervised label set. To do this, we 
    # first figure out which sentences were labeled
    # for this study (pmid).
    first_index = domain_supervision["pmids"].index(pmid)
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
                            threshold=1, zero_one=False)
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
        

## @TODO finish this
class Nguyen():
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
        # maybe we shouldn't even do regularization?
        self.meta_clf = LogisticRegression()

        ###
        # these will be vectors! you'll need to 
        # combine them
        X1 = self.m1.predict_proba_(X)
        X2 = self.m2.predict_proba_(X)
        pdb.set_trace()
        self.meta_clf.fit(X, y)

    def transform(self, x):
        '''
        go from raw input to `stacked' representation
        comprising m1 and m2 probability predictions.
        '''
        pass 

    def predict(self, x):
        pass 

    def predict_proba_(self, x):
        pass 


def build_nguyen_model(X_train, y_train, direct_indices, p_validation=.6):
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
    
    print "nguyen!!"
    pdb.set_trace()
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
    pdb.set_trace()

def _unpickle_PICO_DS(y_dict_pickle, domain_v_pickle):
    with open(y_dict_pickle) as y_dict_f:
        print "unpickling sentences and y dict..."
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
    tune_params = [{"alpha":[.00001, .0001, .001, .01, .1]}]
    #clf = GridSearchCV(LogisticRegression(), tune_params, scoring="accuracy", cv=5)

    ###
    # note to self: for SGDClassifier you want to use the sample_weight
    # argument to instance-weight examples!
    clf = GridSearchCV(SGDClassifier(
             shuffle=True, class_weight="auto", loss="log"), 
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

        #pdb.set_trace()
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
    clf = GridSearchCV(LogisticRegression(), tune_params, scoring="accuracy", cv=5)
    return clf


def _score_to_ordinal_lbl(y_str):
    return float(y_str.strip())

def _score_to_binary_lbl(y_str, zero_one=True, threshold=1):
    # will label anything >= threshold as '1'; otherwise 0
    # (or -1, depending on the zero_one flag).
    if float(y_str.strip()) >= threshold:
        return 1

    return 0 if zero_one else -1

def generate_X_y(DS_learning_task, binary_labels=True, y_lbl_func=_score_to_binary_lbl):
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

    vectorizer = TfidfVectorizer(stop_words='english', decode_error=u'ignore', min_df=1)
    print "fitting vectorizer ... "
    #pdb.set_trace()
    vectorizer.fit(all_domain_texts)
    print "ok."

    X, y = [], []
 
    for X_i, y_i in zip(DS_learning_task["X"], DS_learning_task["y"]):
        X_i_numeric, X_i_text = X_i
        X_v = vectorizer.transform([X_i_text])[0]
        X_combined = sp.sparse.hstack((X_v, X_i_numeric))
        X.append(np.asarray(X_combined.todense())[0])
        y.append(y_lbl_func(y_i))
        
    return X, y


# "sds/annotations/for_labeling_sharma.csv"
def get_DS_features_and_labels(candidates_path="sds/annotations/for_labeling_sharma.csv",
                                labels_path="sds/annotations/sharma-merged-labels.csv",
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
        # *per domain*. 
        for candidate_line, label_line in zip(candidates, labels):
            #print annotation_line
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

            X_i_text = candidate_sentence

            ## numeric features
            # @TODO add more!
            X_i_numeric = []
            X_i_numeric.append(len(candidate_sentence.split(" ")))

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

            # shared tokens for this candidate
            cur_shared_tokens = shared_tokens[cur_candidate_index]
            # extend X_i text with shared tokens (using 
            # special indicator prefix "shared_")
            X_i_text = X_i_text + " ".join(["shared_%s" % tok for 
                                            tok in cur_shared_tokens if tok.strip() != ""])

            X_i_numeric.append(len(candidates) - cur_candidate_index)
            candidate_score = scores[cur_candidate_index]
            X_i_numeric.append(candidate_score - np.mean(scores))
            X_i_numeric.append(candidate_score - np.median(scores))
            
            # @TODO add additional features, e.g., difference from next 
            # highest candidate score..


            # note that we'll need to deal with merging these 
            # tesxtual and numeric feature sets elsewhere!
            X_i = (X_i_numeric, X_i_text)
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

        for domain in domains:
            domain_X = X_y_dict[domain]["X"]
            #num_numeric_feats = len(X_y_dict.values()[0]["X"][0][0])
            num_numeric_feats = len(domain_X[0][0])

            col_Zs = [0]*num_numeric_feats
            for j in xrange(num_numeric_feats):
                all_vals = [X_i[0][j] for X_i in domain_X] 
                z_j = float(max(all_vals))

                for i in xrange(len(domain_X)):
                    # this is not cool
                    X_y_dict[domain]["X"][i][0][j] = X_y_dict[domain]["X"][i][0][j] / z_j
    
    return X_y_dict


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

            