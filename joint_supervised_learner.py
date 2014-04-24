'''
Fancier than supervised_learner, or will be. 

The idea here is to implement a model that incorporates
all fields of interest at once (since these are not independent).

To run vanilla/baseline CRF:

# note that I'm using the union of tags here!
> import tokenizer
> import joint_supervised_learner
> reader = tokenizer.MergedTaggedAbstractReader(merge_function=lambda a,b: a or b)
# important to pass in the 'tx', because otherwise it will think its
# predicting sample size! 
> sl = joint_supervised_learner.SupervisedLearner(reader, target="tx")
> sl.train_and_test_mallet()
'''

# std lib
import pickle
import string
import os
import pdb
import random 
import re
import subprocess
from itertools import izip
tx_tag = re.compile("tx[0-9]+")

# sklearn, &etc
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
import scipy
import numpy
import csv

# homegrown
from indexnumbers import swap_num
import bilearn
from taggedpipeline import TaggedTextPipeline
from journalreaders import LabeledAbstractReader
from tokenizer import MergedTaggedAbstractReader, FullTextReader
import progressbar



### assuming this is effectively 'constant'.
cur_dir = os.getcwd()

##################
# for CRF stuff! #
##################
MALLET_PATHS = "/Users/bwallace/dev/eclipse-workspace/mallet/bin/:/Users/bwallace/dev/eclipse-workspace/mallet/lib/mallet-deps.jar"
JAVA_PATH = "/usr/bin/java"
MALLET_OUTPUT_DIR = os.path.join(cur_dir, "mallet-output")
if not os.path.exists(MALLET_OUTPUT_DIR):
    os.makedirs(MALLET_OUTPUT_DIR)


# a useful helper.
punctuation_only = lambda s: s.strip(string.punctuation).strip() == ""

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

        self.X_fv, self.y = None, None

    def plot_preds(self, preds, y):
        # (preds, y) = sl.cv()
        # sklearn wraps up the predicted results
        
        pos_indices = [i for i in xrange(len(y)) if y[i]>0]
        all_preds = [preds[i][1] for i in xrange(len(y))]
        pos_preds = [preds[i][1] for i in pos_indices]

    def generate_features(self):
        print "generating feature vectors"

        # I don't think we ever want to flatten abstracts.
        self.features, self.y, self.tokens = self.features_from_citations(
                    flatten_abstracts=False, return_tokens=True)

        self.vectorizer = DictVectorizer(sparse=True)

        # note that we keep structure around that keeps features 
        # in citations together. specifically, features will be a 
        # list of feature vectors representing words
        # in abstracts comprising distinct citations
        all_features = []
        for citation_fvs in self.features:
            all_features.extend(citation_fvs)
        
        self.vectorizer.fit(all_features) 


        self.X_fv = []
        # for later look up
        self.tokens_fv = []
        no_abstracts = 0
        for X_citation in self.features:
            if len(X_citation) > 0:
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
    CRF via Mallet
    '''
    @staticmethod
    def to_mallet(X, y, test_file=False):
        to_lbl = lambda bool_y: "1" if bool_y else "-1"

        mallet_out = []
        y_test_out = []
        # all nonzero features
        # X represents a single abstract
        # X_i is word i in said abstract
        for x_i, y_i in izip(X, y):
            x_features = x_i.nonzero()[1] # we only care for the columns
            lbl_str = ""
            if not test_file:
                # then we include the label
                lbl_str = " " + to_lbl(y_i)
            else:
                y_test_out.append(to_lbl(y_i))

            cur_str = " ".join(
                    [str(f_j) for f_j in x_features]) + " " + lbl_str 
            
            mallet_out.append(cur_str.lstrip() + "\n")
        
        mallet_out.append("\n") # blank line separating instances
        if test_file:
            y_test_out.append("\n")

        mallet_str = "".join(mallet_out)
        if test_file:
            return (mallet_str, "\n".join(y_test_out))
        return mallet_str

    @staticmethod
    def train_mallet(train_f, model_f_name):
        full_train_path = os.path.join(MALLET_OUTPUT_DIR, train_f)
        full_model_path = os.path.join(MALLET_OUTPUT_DIR, model_f_name)
        
        p = subprocess.Popen([JAVA_PATH, '-Xmx2g', '-cp', 
                               MALLET_PATHS, 'cc.mallet.fst.SimpleTagger',
                               "--train", "true", 
                               "--iterations", "500",
                               "--model-file", full_model_path, 
                               full_train_path], stdout=subprocess.PIPE)
        output, errors = p.communicate()

        print "ok! model trained and written to %s!" % full_model_path

        return full_model_path


    @staticmethod
    def test_mallet(test_f, full_model_path):
        full_test_path = os.path.join(MALLET_OUTPUT_DIR, test_f)
        print "making predictions for instances @ "
        p = subprocess.Popen([JAVA_PATH, '-Xmx2g', '-cp', 
                               MALLET_PATHS, 'cc.mallet.fst.SimpleTagger',
                               "--model-file", full_model_path, 
                               full_test_path,
                               ], stdout=subprocess.PIPE)
        
        predictions, errors = p.communicate()
        return predictions, errors


    def write_files_to_disk_for_mallet(self, test_citation_indices, fpath):

        test_size = len(test_citation_indices)
        print "test set of size {0} out of {1} total citations".format(
                                        test_size, self.n_citations)

        ''' mallet! '''
        print "assembling mallet str..."
        train_str, test_str, test_lbls_str = [], [], []
        test_tokens_str = [] # also build a tokens string dict
        # check if it's a test instance.
        # create strings...
        for i in xrange(self.n_citations):
            if self.X_fv[i] is not None:
                if not i in test_citation_indices:
                    # we flatten these for training.
                    train_str.append(
                        SupervisedLearner.to_mallet(self.X_fv[i], self.y[i]))

                else:
                    # in the test case, we generate separate file strings for 
                    # the instances and the lables
                    mallet_str, test_lbls_i = SupervisedLearner.to_mallet(
                                    self.X_fv[i], self.y[i], test_file=True)

                    test_str.append(mallet_str.strip() + "\n\n")
                    test_lbls_str.append(test_lbls_i.strip() + "\n\n")
                    test_tokens_str.append("\n".join(self.tokens[i]) + "\n\n")

        test_indices_out = os.path.join(MALLET_OUTPUT_DIR, fpath + "test-indices")
        with open(test_indices_out, 'w') as test_out:
            test_out.write("".join("\n".join([str(i) for i in test_citation_indices])))

        train_out_path = os.path.join(MALLET_OUTPUT_DIR, fpath + "train")
        with open(train_out_path, 'w') as train_out:
            train_out.write("".join(train_str))

        test_lbls_path = os.path.join(MALLET_OUTPUT_DIR, fpath + "test-lbls")
        with open(test_lbls_path, 'w') as test_lbls_out:
            test_lbls_out.write("".join(test_lbls_str))

        test_out_path = os.path.join(MALLET_OUTPUT_DIR, fpath + "test")
        with open(test_out_path, 'w') as test_out:
            test_out.write("".join(test_str))

        test_tokens_str = "".join(test_tokens_str)
        test_tokens_out_path = os.path.join(MALLET_OUTPUT_DIR, fpath + "test-tokens")
        with open(test_tokens_out_path, 'w') as test_tokens_out:
            test_tokens_out.write(test_tokens_str)

        print "train and test files written to disk."

        return train_out_path, test_out_path, test_lbls_path, test_tokens_out_path

    @staticmethod
    def limit_to_two_txs(predictions, max_TXs=2):
        n_TXs = 0
        in_TX = False 

        new_preds = []
        for pred in predictions:
            if pred.strip() == "1":
                if in_TX:
                    new_preds.append(pred)
                else:
                    if n_TXs < max_TXs:
                        new_preds.append(pred)
                        in_TX = True
                    else:
                        new_preds.append("-1\n")
                    n_TXs += 1
            else:
                # -1 
                new_preds.append(pred)
                in_TX = False
        return new_preds


    def train_mallet_on_all_data(self, train_file_path = "tx.mallet."):
        if self.X_fv is None:
            print "features not yet generated! taking a stab at it..."
            self.generate_features()
            print "ok."

        train_path, test_path, test_y_path, test_tokens_out_path = \
                    self.write_files_to_disk_for_mallet(
                        test_citation_indices=[], 
                        fpath=train_file_path)

        model_f = "fully.train.crf.mallet"
        full_model_path = SupervisedLearner.train_mallet(train_path, model_f)

        ### also dump vectorizer
        vectorizer_out="crf.vectorizer.pickle"
        with open(vectorizer_out, 'wb') as v_out:
            pickle.dump(self.vectorizer, v_out)


    def train_and_test_mallet(self, fpath="CRF.", force_two_TXs=False):
        if self.X_fv is None:
            print "features not yet generated! taking a stab at it..."
            self.generate_features()
            print "ok."

        folds = cross_validation.ShuffleSplit(self.n_citations, n_iter=10)
        F_scores, precs, recalls = [], [], []
        for i, (train_indices, test_indices) in enumerate(folds):
            fpath_i = fpath + "{0}.".format(i)

            train_path, test_path, test_y_path, test_tokens_out_path =\
                 self.write_files_to_disk_for_mallet(
                        test_citation_indices=test_indices, fpath=fpath_i)

            '''
            train and test in mallet
            '''
            model_f = fpath + "model"
            full_model_path = SupervisedLearner.train_mallet(train_path, model_f)
            predictions, errors = SupervisedLearner.test_mallet(test_path, full_model_path)
            #predictions = [l.strip() for l in predictions.split("\n")]
            #predictions = open(preds_path).readlines()
            predictions = [pred.strip() + "\n" for pred in predictions.split("\n")]
            if force_two_TXs:
                new_predictions = SupervisedLearner.limit_to_two_txs(predictions)
                predictions = new_predictions

            #true_lbls = [l.strip() for l in open(test_y_path).readlines()]
            true_lbls = open(test_y_path).readlines()
            #test_tokens = [l.strip() for l in open(test_tokens_path).readlines()]
            test_tokens = open(test_tokens_out_path).readlines()
            
            ###
            # sanity check and convert to ints, as this is what sklearn
            # demands.
            y_true, y_pred, abstract_tokens = [], [], []
            
            #if len(true_lbls) != len(predictions):
            #    raise Exception("lengths of predictions and true do not match!")
            ## assert that segments are aligned
            
            for i, (y_i, pred_i) in enumerate(izip(true_lbls, predictions)):
                if (y_i == "\n" and pred_i != "\n") or (pred_i == "\n" and y_i != "\n"):
                    pdb.set_trace()
                    raise Exception("segments are not aligned! (index: {0})".format(i))
                elif y_i == pred_i == "\n":
                #if y_i == pred_i == "":
                    pass
                elif y_i != "":
                    y_true.append(int(y_i))
                    y_pred.append(int(pred_i))

                

            print "\n----- results for fold {0} ------- \n".format(i)
            print confusion_matrix(y_true, y_pred)
            print classification_report(y_true, y_pred)
            F_scores.append(f1_score(y_true, y_pred))
            prec_i, recall_i = precision_recall_fscore_support(y_true, y_pred)[:2]
            precs.append(prec_i[1])
            recalls.append(recall_i[1])
            print "\n------------ \n"

            # dump them to disk, too.
            predictions_out = os.path.join(MALLET_OUTPUT_DIR, fpath_i + "predictions")
            with open(predictions_out, 'w') as preds_out:
                preds_out.write("".join(predictions))

            #tokens_str = " ".join(tokens)
            
            merged_output = os.path.join(MALLET_OUTPUT_DIR, fpath_i + "crf.preds.csv")
            with open(merged_output, 'w') as merged_out_file:
                merged_writer = csv.writer(merged_out_file)
                merged_writer.writerow(["token", "y (true)", "y (predicted)"])
                #merged_out_file.write("\t".join(["token", "y (true)", "y (predicted)"]))
                #pdb.set_trace()
                for merged_line in izip(test_tokens, true_lbls, predictions):
                    #for merged_line in izip(tokens_i.split("\n"), abstract_i, preds_i):
                    if "\n" in merged_line:
                        #pdb.set_trace()
                        # two blank lines
                        merged_writer.writerow(["", "", ""])
                        merged_writer.writerow(["", "", ""])
                    else:
                        merged_writer.writerow([str(s) for s in merged_line])
                    #merged_out_file.write("\t".join([str(s) for s in merged_line]))
                    #merged_writer.writerow("") # blank row to demarcate abstracts

            
        print sum(F_scores)/float(len(F_scores))
        pdb.set_trace()


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
        for test_citation_i, citation_fvs in enumerate(X_test):
            true_lbls_i = y_test[test_citation_i]
            preds_i = [p[1] for p in clf.predict_log_proba(citation_fvs)]
            # we set the index corresponding to the max 
            # val (most likely entry) to 1; all else are 0
            preds_i_max = max_index(preds_i)
            preds_i = [-1]*len(preds_i)
            preds_i[preds_i_max] = 1

            # *abstract level* predictions. 
            if not 1 in true_lbls_i:
                cit_n = test_citation_indices[test_citation_i]
                print "-- no sample size for abstract (biview_id) {0}!".format(
                            self.abstract_reader[cit_n]["biview_id"])
                # since we force a prediction for every abstract right now,
                # i'll penalize us here. this is an upperbound on precision.
                FPs += 1 
            else:
                N_pos += 1 
                if preds_i.index(1) == true_lbls_i.index(1):
                    TPs +=1
                else:
                    FPs += 1


        N = len(X_test)
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
        return SVC(probability=True, kernel='linear', C=3)

    def train(self):
        features, y = self.features_from_citations()
        self.vectorizer = DictVectorizer(sparse=True)
        X_fv = self.vectorizer.fit_transform(self.features)
        
        self.clf = _get_SVM()

        ##
        # @TODO grid search over c?
        self.clf.fit(X_fv, y)


    def features_from_citations(self, flatten_abstracts=False, 
                                        flatten_sentences=True, return_tokens=False):
        '''
        Notes on structure: there are two 'levels' of possible structure
        here, sentence-level and abstract-level. On one extreme,
        X will comprise a completely flat list of vectors representing each 
        x_i; the other extreme is that X comprises nested lists of lists,
        where X[j] corresponds to abstract j and X[j][k] maps to the x[i]
        (word instance) in abstract i 
        '''
        X, y = [], []
        tokens = []
        pb = progressbar.ProgressBar(len(self.abstract_reader), timer=True)
        for cit_id in range(len(self.abstract_reader)):
            # first we perform feature extraction over the
            # abstract text (X)

            merged_tags = self.abstract_reader.get(cit_id)     
            p = TaggedTextPipeline(merged_tags, window_size=4)
            p.generate_features()

            ####
            # IM: xml annotations are now all available in w["tags"] for each word in the features list
            ####


            if self.predicting_sample_size:            
                ###
                # restrict to integers only
                ###
                X_i = p.get_features(flatten=flatten_sentences, 
                            filter=lambda w: w['num']==True)
                y_i = p.get_answers(flatten=flatten_sentences, 
                        answer_key=lambda w: "n" in w["tags"], 
                        filter=lambda x: x['num']==True)
            else: 
                ### *this* flatten refers to flattening sentences!
                X_i = p.get_features(flatten=flatten_sentences)
                y_i = p.get_answers(flatten=flatten_sentences, 
                        answer_key=
                        lambda w: any (
                            [tx_tag.match(tag_i) for tag_i in w["tags"]]))
                tokens.append(p.get_words(flatten=flatten_sentences))

            # see comments above regarding possible
            # structures
            if flatten_abstracts:
                X.extend(X_i)
                y.extend(y_i)
            else:
                X.append(X_i)
                y.append(y_i)

            pb.tap()

        if return_tokens:
            #pdb.set_trace()
            return X, y, tokens
        return X, y

        




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

    reader = MergedTaggedAbstractReader(merge_function=lambda a,b: a or b)
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



if __name__ == "__main__":

    # @TODO make these args.
    nruns = 10
    predict_probs = False
    target = "tx1"

    reader = MergedTaggedAbstractReader()

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






