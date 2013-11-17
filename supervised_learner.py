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

# sklearn, &etc
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import scipy
import numpy

# homegrown
from indexnumbers import swap_num
import bilearn
from bilearn import bilearnPipeline
import agreement as annotation_parser

import progressbar

# tmp @TODO use some aggregation of our 
# annotations as the gold-standard.
annotator_str = "BCW"
# a useful helper.
punctuation_only = lambda s: s.strip(string.punctuation).strip() == ""

class SupervisedLearner:
    def __init__(self, abstract_reader, target="n"):
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

    def plot_preds(self, preds, y):
        # (preds, y) = sl.cv()
        # sklearn wraps up the predicted results
        
        pos_indices = [i for i in xrange(len(y)) if y[i]>0]
        all_preds = [preds[i][1] for i in xrange(len(y))]
        pos_preds = [preds[i][1] for i in pos_indices]

    def generate_features(self):
        print "generating feature vectors"
        self.features, self.y = self.features_from_citations(flatten_abstracts=not self.predicting_sample_size)
        self.vectorizer = DictVectorizer(sparse=True)

        if self.predicting_sample_size:
            # then features will be a list feature vectors representing words
            # in utterances comprising distinct citations
            all_features = []
            for citation_fvs in self.features:
                all_features.extend(citation_fvs)
       
            self.vectorizer.fit(all_features) 
            self.X_fv = []
            no_abstracts = 0
            for X_citation in self.features:
                if len(X_citation) > 0:
                    self.X_fv.append(self.vectorizer.transform(X_citation))
                else:
                    self.X_fv.append(None)
                    no_abstracts += 1
            print "({0} had no abstracts!)".format(no_abstracts)
            #self.X_fv = [self.vectorizer.transform(X_citation) for X_citation in self.features if len(X_citation) > 0]
        else:
            self.X_fv = self.vectorizer.fit_transform(self.features)

    def train_and_test_sample_size(self, test_size=.1):
        n_citations = len(self.X_fv)
        test_size = int(test_size*n_citations)
        print "test set of size {0} out of {1} total citations".format(test_size, n_citations)
        test_citation_indices = random.sample(range(n_citations), test_size)
        X_train, y_train = [], []
        X_test, y_test = [], []
        for i in xrange(n_citations):
            if self.X_fv[i] is not None:
                if not i in test_citation_indices:
                    # we flatten these for training.
                    X_train.extend(self.X_fv[i])
                    y_train.extend(self.y[i])
                else:
                    # these we keep structured, though.
                    X_test.append(self.X_fv[i])
                    y_test.append(self.y[i])

        clf = SupervisedLearner._get_SVM()
        X_train = scipy.sparse.vstack(X_train)
        clf.fit(X_train, y_train)
        #pdb.set_trace()
        print "ok -- testing!"
        max_index = lambda a: max((v, i) for i, v in enumerate(a))[1]
        test_preds, test_true = [], [] # these will be flat!
        for test_citation_i, citation_fvs in enumerate(X_test):
            true_lbls_i = y_test[test_citation_i]
            preds_i = [p[1] for p in clf.predict_log_proba(citation_fvs)]
            # we set the index corresponding to the max 
            # val (most likely entry) to 1; all else are 0
            preds_i_max = max_index(preds_i)
            preds_i = [-1]*len(preds_i)
            preds_i[preds_i_max] = 1

            test_preds.extend(preds_i)
            test_true.extend(true_lbls_i)

        return test_preds, test_true


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
        return SVC(probability=True, kernel='linear')

    def train(self):
        features, y = self.features_from_citations()
        self.vectorizer = DictVectorizer(sparse=True)
        X_fv = self.vectorizer.fit_transform(self.features)
        
        self.clf = _get_SVM()

        ##
        # @TODO grid search over c?
        self.clf.fit(X_fv, y)


    @staticmethod
    def filter_tags(tags):
        '''
        returns the subset of tags which are not referring to
        punctuation
        '''
        kept_tags = []
        for tag in tags:
            if not punctuation_only (tag.keys()[0]):
                kept_tags.append(tag)
        return kept_tags

    @staticmethod
    def filter_words(words_to_annotations, words, feature_vectors):
        '''
        return the subset of words in words that appears in 
        words_to_annotations. also filter out puncutation-only 
        tokens. note that we take in the feature_vectors list, 
        too, so that we may return the right X_i's (i.e., those
        corresponding to the word that are kept)
        '''

        annotated_words = []
        for w_to_tags in words_to_annotations:
            # w_to_tags is a dict {word: [tag set]}
            word = w_to_tags.keys()[0]
            annotated_words.append(word)

        kept_words, kept_fvs = [], []
        for i,w in enumerate(words):
            if not punctuation_only(w) and w in annotated_words:
                kept_words.append(w)
                kept_fvs.append(feature_vectors[i])
                annotated_words.remove(w)

        return (kept_words, kept_fvs)


    def features_from_citations(self, flatten_abstracts=True):
        X, y = [], []

        pb = progressbar.ProgressBar(len(self.abstract_reader), timer=True)
        for cit in self.abstract_reader:
            # first we perform feature extraction over the
            # abstract text (X)

            abstract_text = cit["abstract"] 

            ###
            # IM: added, removes all tags before sending to the pipeline
            abstract_text = re.sub('(<\/([a-z0-9_]+)>|<([a-z0-9_]+)>)', "", abstract_text) 
            
            p = bilearnPipeline(abstract_text)
            p.generate_features()
            #filter=lambda x: x["w[0]"].isdigit()
  
            # @TODO will eventually want to exploit sentence 
            # structure, I think 
            X_i = p.get_features(flatten=True)
            words = p.get_answers(flatten=True)
   
            # now construct y vector based on parsed tags
            cit_file_id = cit["file_id"]
         
            # aahhhh stupid zero indexing confusion
            abstract_tags = annotation_parser.get_annotations(cit_file_id-1, annotator_str, convert_numbers=True)
            
            # we only keep words for which we have annotations
            # and that are not, e.g., just puncutation.
            training_words, training_fvs = SupervisedLearner.filter_words(
                                    abstract_tags, words, X_i)


            # IM: swap_num needs to work on the untokenized abstract (since numbers may be multiple words)
            # training_words = [swap_num(w_i) for w_i in training_words]
            training_tags = SupervisedLearner.filter_tags(abstract_tags)

            X_cit, y_cit = [], [] # only useful if we are not flattening abstracts
            for j, w in enumerate(training_words):
                tags = None

                for tag_index, tag in enumerate(training_tags):
                    ####
                    # IM: this runs through whole abstract to see if
                    #     a word is tagged, but could this return errors
                    #     if same word is tagged differently in different
                    #     places (e.g. 'mg' in '200 mg ibuprofen versus 500 mg paracetamol')
                    #     since it always returns the tag of the first tagged instance
                    #
                    #  BCW: I do not think so, because the training_tags
                    #    are ordered in the same way as the training_words.
                    #    note that we break when we find the first instance
                    #    of the word, and then remove this word from the 
                    #    training_tags. 
                    ###
                    if tag.keys()[0] == w:
                        tags = tag.values()[0]
                        break

                if tags is None:
                    # uh-oh.
                    raise Exception, "no tags???"


                w_lbl = 1 if self.target in tags else -1
                X_cit.append(training_fvs[j])
                y_cit.append(w_lbl)


                # remove this tag from the list -- remember,
                # words can appear multiple times!
                training_tags.pop(tag_index) ## IM: first instance of tagged word removed here
            
            if flatten_abstracts:
                X.extend(X_cit)
                y.extend(y_cit)
            else:
                X.append(X_cit)
                y.append(y_cit)
            pb.tap()

        return X, y

        

class LabeledAbstractReader:
    ''' 
    Parses labeled citations from the provided path. Assumes format is like:

        Abstract 1 of 500
        Prothrombin fragments (F1+2) ...
            ...
        BiviewID 42957; PMID 11927130

    '''
    def __init__(self, path_to_data="data/drug_trials_in_cochrane_BCW.txt", num_labeled_abstracts=150):
        # @TODO probably want to read things in lazily, rather than
        # reading everything into memory at once...
        self.abstracts = []
        self.abstract_index = 1 # for iterator
        self.path_to_abstracts = path_to_data
        print "parsing data from {0}".format(self.path_to_abstracts)

        self.num_abstracts = num_labeled_abstracts
        self.parse_abstracts()

        self.num_citations = len(self.citation_d) 
        print "ok."


    def __iter__(self):
        self.abstract_index = 0
        return self

    def __len__(self):
        return self.num_citations

    def next(self):
        if self.abstract_index >= self.num_citations:
            raise StopIteration
        else:
            self.abstract_index += 1
            return self.citation_d[self.abstract_index-1]
            

    def _is_demarcater(self, l):
        '''
        True iff l is a line separating two citations.
        Demarcating lines look like "BiviewID 42957; PMID 11927130"
        '''

        # reasonably sure this will not give any false positives...
        return l.startswith("BiviewID") and "PMID" in l

    def _get_IDs(self, l):
        ''' Assumes l is a demarcating line; returns Biview and PMID ID's '''
        grab_id = lambda s : s.lstrip().split(" ")[1].strip()
        biview_id, pmid = [grab_id(s) for s in l.split(";")]
        return biview_id, pmid

    def _is_new_citation_line(self, l):
        return l.startswith("Abstract ")

    def parse_abstracts(self):
        self.citation_d = {}
        in_citation = False
        # abstract_num is the arbitrary, per-file, sequentially
        # incremented id assigned abstracts. this is *not*
        # zero-indexed and varies across files. we need to hold
        # on to this to cross-ref with the annotations_parser 
        # (agreement.py) module -- this is the ID that lib uses!
        #
        # abstract_index is used only internally; the difference
        # is that we only hold on to abstracts that have annotations
        # (many do not, especially when using just, e.g., my file)
        abstract_num, abstract_index = 1, 0 
        with open(self.path_to_abstracts, 'rU') as abstracts_file:
            cur_abstract = ""
            
            for line in abstracts_file.readlines():
                line = line.strip()
                if self._is_demarcater(line):
                    biview_id, pmid = self._get_IDs(line)
                    if LabeledAbstractReader.is_annotated(abstract_num):
                        self.citation_d[abstract_index] = {"abstract":cur_abstract, 
                                                "Biview_id":biview_id,
                                                "pubmed_id":pmid,
                                                "file_id":abstract_num} # yes, redundant
                        abstract_index += 1
                    else:
                        print "no annotations for {0} -- ignoring!".format(abstract_num)
                    cur_abstract = ""
                    in_citation = False
                    abstract_num += 1
                elif in_citation and line:
                    # then this is the abstract
                    cur_abstract = line
                elif self._is_new_citation_line(line):
                    in_citation = True

                if abstract_num > self.num_abstracts:
                    return self.citation_d

        return self.citation_d

    @staticmethod
    def is_annotated(cit_file_id):
        # aahhhh stupid zero indexing confusion
        abstract_tags = annotation_parser.get_annotations(cit_file_id-1, annotator_str, convert_numbers=True)

        for w in abstract_tags:
            if len(w.values()[0]) > 0:
                return True
        return False

    def get_text(self):
        return [cit["abstract"] for cit in self.citation_d.values()]


def learning_curve():
    nruns = 10

    reader = LabeledAbstractReader()
    sl = SupervisedLearner(reader)
    sl.generate_features()
    average_fs = []
    pb = progressbar.ProgressBar(nruns, timer=True)
    train_ps = numpy.linspace(.1,.8,5)
    for train_p in train_ps:
        cur_avg_f = 0
        for i in xrange(nruns):
            preds, y_test = sl.train_and_test_sample_size(test_size=1-train_p)
            p, r, f, s = precision_recall_fscore_support(y_test, preds, average="micro")
            cur_avg_f += f
            pb.tap()
        avg_f = cur_avg_f/float(nruns)
        print "\naverage f-score for {0}:{1}".format(train_p, avg_f)
        average_fs.append(avg_f)
    return train_ps, average_fs



if __name__ == "__main__":
    # @TODO make these args.
    nruns = 10
    predict_probs = False

    reader = LabeledAbstractReader()
    sl = SupervisedLearner(reader)#, target="tx")
    sl.generate_features()

    p_sum, r_sum, f_sum, np_sum = [0]*4
    auc, apr = 0, 0
    print "running models"
    pb = progressbar.ProgressBar(nruns, timer=True)
    for i in xrange(nruns):
        #preds, y_test = sl.cv(predict_probs=predict_probs)
        preds, y_test = sl.train_and_test_sample_size()
       
        if predict_probs:
            fpr, tpr, thresholds = metrics.roc_curve(y_test, preds)
            auc += metrics.auc(fpr, tpr)
            prec, recall, thresholds = metrics.precision_recall_curve(y_test, preds)
            apr += metrics.auc(recall, prec)
            #pdb.set_trace()
        else:
            p, r, f, s = precision_recall_fscore_support(y_test, preds, average="micro")

            print "\n--precision: {0} / recall {1} / f {2}\n".format(p, r, f)
            p_sum += p
            r_sum += r
            f_sum += f
            np_sum += s
        pb.tap()

    avg = lambda x: x / float(nruns)

    if predict_probs:
        print "averages\nAUC: {0}\nArea under precision-recall curve {1}".format(avg(auc), avg(apr))
    else:
        print "averages\n # of target words: {0:.2}\n precision: {1:.2}\n recall: {2:.2}\n f: {3:.2}".format(
                    avg(np_sum), avg(p_sum), avg(r_sum), avg(f_sum))






