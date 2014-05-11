'''
Do 'vanilla' supervised learning over labeled citations.

> import supervised_learner
> reader = supervised_learner.LabeledAbstractReader()
> sl = supervised_learner.SupervisedLearner(reader)
> X,y = sl.features_from_citations()

To actually vectorize, something like:
> vectorizer = DictVectorizer(sparse=True)
> vectorizer.fit_transform(X)
'''

# std lib
import string
import pdb
import random 
import re
import os
import cPickle as pickle

# sklearn, &etc
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import scipy
import numpy


# homegrown
from indexnumbers import swap_num
import bilearn
from taggedpipeline import TaggedTextPipeline
from journalreaders import LabeledAbstractReader
import tokenizer
from tokenizer import MergedTaggedAbstractReader, FullTextReader
import progressbar


# a useful helper.
def punctuation_only(s):
    return s.strip(string.punctuation).strip() == ""

def integer_filter(w):
    return w['num'] == True

def is_sample_size(w):
    return "n" in w["tags"]


class SupervisedLearner:
    def __init__(self, abstract_reader, target="n", 
                        hold_out_a_test_set=False, test_set_p=None):
        '''
        abstract_reader: a LabeledAbstractReader instance.
        target: the tag of interest (i.e., to be predicted)
        '''
        self.abstract_reader = abstract_reader
        self.target = target
        # this is a special target because we 
        # enforce the additional constraint that
        # there be only one 'yes' vote per citation
        self.predicting_sample_size = target == "n"

        # reserve some data for testing?
        self.holding_out_a_test_set = hold_out_a_test_set
        if self.holding_out_a_test_set:
            assert test_set_p is not None

        self.test_set_p = test_set_p
        self.n_citations = len(self.abstract_reader)

    def is_target(self, w):
        return self.target in w["tags"]

    def plot_preds(self, preds, y):
        # (preds, y) = sl.cv()
        # sklearn wraps up the predicted results
        
        pos_indices = [i for i in xrange(len(y)) if y[i]>0]
        all_preds = [preds[i][1] for i in xrange(len(y))]
        pos_preds = [preds[i][1] for i in pos_indices]

    def generate_features(self):
        print "generating feature vectors"

        # I don't think we ever want to flatten abstracts.
        self.features, self.y = self.features_from_citations(
                    flatten_abstracts=False)
        self.vectorizer = DictVectorizer(sparse=True)

        #if self.predicting_sample_size:
        # note that we keep structure around that keeps features 
        # in citations together. specifically, features will be a 
        # list of feature vectors representing words
        # in abstracts comprising distinct citations
        all_features = []
        for citation_fvs in self.features:
            all_features.extend(citation_fvs)

        self.vectorizer.fit(all_features) 
        #else:
        #    self.vectorizer.fit(self.features)

        self.X_fv = []
        no_abstracts = 0
        for X_citation in self.features:
            if len(X_citation) > 0:
                #pdb.set_trace()
                self.X_fv.append(self.vectorizer.transform(X_citation))
            else:
                self.X_fv.append(None)
                no_abstracts += 1
        print "({0} had no abstracts!)".format(no_abstracts)
        #self.X_fv = [self.vectorizer.transform(X_citation) for X_citation in self.features if len(X_citation) > 0]

        if self.holding_out_a_test_set:
            self.set_held_out_indices()

    def set_held_out_indices(self):
        test_set_size = int(self.test_set_p*self.n_citations)
        print "setting aside a test set of size {0}".format(test_set_size)
        #import pdb; pdb.set_trace()
        self.test_indices = random.sample(range(self.n_citations), test_set_size)

    def select_train_citation_indices(self, train_p):
        '''
        this is somewhat confusing, but the idea here is to allow one to
        have a single, consistent test set and to increase the training
        set to see how this affects performance on said set. 
        '''
        # first remove the held out indices.
        self.train_indices = [
                i for i in range(self.n_citations) if not i in self.test_indices]
        # now draw a sample from the remaining (train) abstracts.
        train_set_size = int(train_p*len(self.train_indices))
        print "going to train on {0} citations".format(train_set_size)
        self.train_indices = random.sample(self.train_indices, train_set_size)

    '''
    @TODO this method is meant to supplant the following routine.
    The idea is that is more general, i.e., allows us to
    assess performance on <tx>, etc; not just <n>
    '''
    def train_and_test(self, test_size=.2, train_p=None):
        test_citation_indices = None
        train_citation_indices = None
        if self.holding_out_a_test_set:
            print "using the held-out test set!"
            test_size = len(self.test_indices)
            test_citation_indices = self.test_indices
            train_citation_indices = self.train_indices
        else:
            test_size = int(test_size*self.n_citations)
            test_citation_indices = random.sample(range(self.n_citations), test_size)

        print "test set of size {0} out of {1} total citations".format(
                                    test_size, self.n_citations)

    @staticmethod
    def max_index(self, a):
        return max((v, i) for i, v in enumerate(a))[1]

    def train_and_test_sample_size(self, test_size=.2, train_p=None):
        '''
        @TODO need to amend for predicting things other than sample size
        in retrospect, should probably never flatten abstracts; at test
        time we'll want to enforce certain constraints

        @TODO refactor -- this method is too long.
        '''
        test_citation_indices = None
        train_citation_indices = None
        if self.holding_out_a_test_set:
            print "using the held-out test set!"
            test_size = len(self.test_indices)
            test_citation_indices = self.test_indices
            train_citation_indices = self.train_indices
        else:
            test_size = int(test_size*self.n_citations)
            test_citation_indices = random.sample(range(self.n_citations), test_size)

        print "test set of size {0} out of {1} total citations".format(
                                    test_size, self.n_citations)
        
        X_train, y_train = [], []
        X_test, y_test = [], []
        test_citation_indices.sort() # not necessary; tmp
        for i in xrange(self.n_citations):
            if self.X_fv[i] is not None:
                is_a_training_instance = (
                        train_citation_indices is None or 
                        i in train_citation_indices)
                if not i in test_citation_indices and is_a_training_instance:
                    # we flatten these for training.
                    X_train.extend(self.X_fv[i])
                    y_train.extend(self.y[i])

                elif i in test_citation_indices:
                    # these we keep structured, though.
                    X_test.append(self.X_fv[i])
                    y_test.append(self.y[i])

        clf = SupervisedLearner._get_SVM()
        X_train = scipy.sparse.vstack(X_train)
        clf.fit(X_train, y_train)
        print "ok -- testing!"
        max_index = lambda a: max((v, i) for i, v in enumerate(a))[1]
        
        '''
        @TODO refactor. note that this will have to change for other
        targets (TX's, etc.)
        '''
        TPs, FPs, N_pos = 0, 0, 0
        no_truth = 0
        for test_citation_i, citation_fvs in enumerate(X_test):
            true_lbls_i = y_test[test_citation_i]

            # we set the index corresponding to the max 
            # val (most likely entry) to 1; all else are 0
            preds_i = clf.best_estimator_.decision_function(citation_fvs)
            preds_i_max = max_index(preds_i)
            preds_i = [-1]*len(preds_i)
            preds_i[preds_i_max] = 1

            # *abstract level* predictions. 
            
            if not 1 in true_lbls_i:
                cit_n = test_citation_indices[test_citation_i]
                print "-- no target for abstract (biview_id) {0}!".format(
                            self.abstract_reader[cit_n]["biview_id"])
                # since we force a prediction for every abstract right now,
                # i'll penalize us here. this is an upperbound on precision.
                FPs += 1
                no_truth += 1 
            else:
                N_pos += 1 
                if preds_i.index(1) == true_lbls_i.index(1):
                    TPs +=1
                else:
                    FPs += 1


        N = len(X_test)
        print "no labels available for %s citations!" % no_truth
        return TPs, FPs, N_pos, N


    def cv(self, predict_probs=False):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            self.X_fv, self.y, test_size=0.1)
        clf = SupervisedLearner._get_SVM()
        clf.fit(X_train, y_train)

        preds = None
        if predict_probs:
            # well, *log* probs, anyway
            preds = [p[1] for p in clf.predict_log_proba(X_test)]
        else:
            preds = clf.predict(X_test)
        return preds, y_test
    

    @staticmethod 
    def _get_SVM():
        tune_params = [{"C":[.01, .1, 1,5,10]}]
        return GridSearchCV(LinearSVC(), tune_params, scoring="f1")
        #return LinearSVC(C=.1)


    def train(self):
        features, y = self.features_from_citations()
        self.vectorizer = DictVectorizer(sparse=True)
        X_fv = self.vectorizer.fit_transform(self.features)
        
        self.clf = _get_SVM()

        ##
        # @TODO grid search over c?
        self.clf.fit(X_fv, y)


    def features_from_citations(self, flatten_abstracts=False):
        X, y = [], []

        pb = progressbar.ProgressBar(len(self.abstract_reader), timer=True)
        for cit_id in range(len(self.abstract_reader)):
            # first we perform feature extraction over the
            # abstract text (X)

            merged_tags = self.abstract_reader.get(cit_id)     
            #pdb.set_trace()
            p = TaggedTextPipeline(merged_tags, window_size=4)
            p.generate_features()


            # @TODO will eventually want to exploit sentence 
            # structure, I think 


            ####
            # IM: 'punct' = token has all punctuation
            # filter here is a lambda function used on the
            # individual word's hidden features
            ###
            # X_i = p.get_features(flatten=True, filter=lambda w: w['punct']==False)
            # y_i = p.get_answers(flatten=True, answer_key=lambda w: "n" in w["tags"], filter=lambda w: w['punct']==False)

            ####
            # IM: xml annotations are now all available in w["tags"] for each word in the features list
            ####


            if self.predicting_sample_size:            
                ###
                # restrict to integers only
                ###

                #X_i = p.get_features(flatten=True, filter=lambda w: w['num']==True)
                X_i = p.get_features(flatten=True, filter=integer_filter)
                y_i = p.get_answers(flatten=True, 
                        answer_key=is_sample_size, 
                        filter=integer_filter)
            else: 
                X_i = p.get_features(flatten=True)
                y_i = p.get_answers(flatten=True, 
                        answer_key=self.is_target)

            if flatten_abstracts:
                X.extend(X_i)
                y.extend(y_i)
            else:
                X.append(X_i)
                y.append(y_i)

            pb.tap()    

        return X, y

    def train_on_all_data(self):
        X_train, y_train = [], []
        for i in xrange(self.n_citations):
            if self.X_fv[i] is not None:            
                # we flatten these for training.
                X_train.extend(self.X_fv[i])
                y_train.extend(self.y[i])

        clf = SupervisedLearner._get_SVM()
        X_train = scipy.sparse.vstack(X_train)
        print "fitting...."
        clf.fit(X_train, y_train)
        print "success!"

        return clf, self.vectorizer
        #print "ok -- testing!"
        #max_index = lambda a: max((v, i) for i, v in enumerate(a))[1]





def average_learning_curve(nruns=5):
    y_total = numpy.array([])
    for i in xrange(nruns):
        print "\n\n--on run %s\n\n" % i
        x_i, y_i, lows, highs = learning_curve()
        if y_total.shape[0] == 0:
            y_total = numpy.array(y_i)
        else:
            y_total += numpy.array(y_i)
            print "\n\n---\naverage so far (after %s iters): %s \n\n" % (
                                    i, y_total / float(i+1))
    return x_i, y_total / float(nruns)

def calc_metrics(TPs, FPs, N_pos, N):
    TPs, FPs, N_pos, N = float(TPs), float(FPs), float(N_pos), float(N)
    recall = TPs /  N_pos
    precision = TPs / (TPs + FPs)
    f = 0
    if precision + recall > 0:
        f = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f

def learning_curve():
    nruns = 5

    reader = MergedTaggedAbstractReader()
    sl = SupervisedLearner(reader, hold_out_a_test_set=True, test_set_p=.2)
    sl.generate_features()

    average_fs = []
    lows, highs = [], []
    pb = progressbar.ProgressBar(nruns, timer=True)
    train_ps = numpy.linspace(.15,.95,4)
    for train_p in train_ps:
        cur_avg_f = 0
        cur_low, cur_high = numpy.inf, -numpy.inf
        for i in xrange(nruns):
            sl.select_train_citation_indices(train_p)
            TPs, FPs, N_pos, N = sl.train_and_test_sample_size(train_p=train_p)
            r, p, f = calc_metrics(TPs, FPs, N_pos, N)
            print "precision: {0}; recall: {1}; f: {2}".format(p, r, f)
            cur_avg_f += f
            if f < cur_low:
                cur_low = f
            if f > cur_high:
                cur_high = f
        lows.append(cur_low)
        highs.append(cur_high)
            
        pb.tap()
        avg_f = cur_avg_f/float(nruns)
        print "\naverage f-score for {0}:{1}".format(train_p, avg_f)
        average_fs.append(avg_f)

    # factor in the held out data
    train_citations = sl.n_citations - (sl.n_citations * sl.test_set_p)
    return [int(sl.n_citations*p) for p in train_ps], average_fs, lows, highs



def train_and_pickle_full_text(path="cache/labeled/"):
    # get full text reader here, 
    #tagged_texts = [tokenizer.tag_words(text) for text in texts]

    reader = FullTextReader()   
    sl = SupervisedLearner(reader, target="n")
    sl.generate_features()
    clf, vectorizer = sl.train_on_all_data()

    with open("sample_size_predictor_ft.pickle", "wb") as out_f:
        pickle.dump(clf.best_estimator_, out_f)
        #pickle.dump(clf, out_f)

    with open("sample_size_vectorizer_ft.pickle", "wb") as out_f:
        pickle.dump(vectorizer, out_f)

    return clf, vectorizer

def train_and_pickle():
    """ intended to be plugged into SPA, eventually. note that we
        pickle the actual clf model (and vectorizer) rather than 
        the SupervisedLearner class, in part because this is arguably
        cleaner, and in part because python kept yelling at me about
        pickling functions, and I didn't get to the bottom of it.
    """
    print "-- treatment model --"
    target = "tx"
    #reader = MergedTaggedAbstractReader( merge_function=lambda a,b: a or b)
    reader = FullTextReader()
    sl = SupervisedLearner(reader, target=target)
    sl.generate_features()
    clf, vectorizer = sl.train_on_all_data()
    with open("tx_predictor_ft.pickle", "wb") as out_f:
        pickle.dump(clf, out_f)

    with open("tx_vectorizer_ft.pickle", "wb") as out_f:
        pickle.dump(vectorizer, out_f)
    pdb.set_trace()
    return clf, vectorizer

if __name__ == "__main__":

    # @TODO make these args.
    nruns = 1
    predict_probs = False
    target = "n"

    #reader = MergedTaggedAbstractReader()
    reader = FullTextReader()

    sl = SupervisedLearner(reader, target=target)
    sl.generate_features()

    p_sum, r_sum, f_sum, np_sum = [0]*4
    auc, apr = 0, 0
    print "running models"
    pb = progressbar.ProgressBar(nruns, timer=True)
    for i in xrange(nruns):
        TPs, FPs, N_pos, N = sl.train_and_test_sample_size()
       
        if predict_probs:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
            auc += metrics.auc(fpr, tpr)
            prec, recall, thresholds = metrics.precision_recall_curve(y_test, preds)
            apr += metrics.auc(recall, prec)
        else:
            r, p, f = calc_metrics(TPs, FPs, N_pos, N)
            print "\n--precision: {0} / recall {1} / f {2}\n".format(p, r, f)
            p_sum += p
            r_sum += r
            f_sum += f
  
        pb.tap()

    avg = lambda x: x / float(nruns)
 
    if predict_probs:
        print "averages\nAUC: {0}\nArea under precision-recall curve {1}".format(avg(auc), avg(apr))
    else:
        print "averages\n precision: {0:.2}\n recall: {1:.2}\n f: {2:.2}".format(
                    avg(p_sum), avg(r_sum), avg(f_sum))






