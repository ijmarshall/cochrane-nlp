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

import string
import pdb

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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

    def plot_preds(self, preds, y):
        # (preds, y) = sl.cv()
        # sklearn wraps up the predicted results
        
        pos_indices = [i for i in xrange(len(y)) if y[i]>0]
        all_preds = [preds[i][1] for i in xrange(len(y))]
        pos_preds = [preds[i][1] for i in pos_indices]

    def cv(self):
        features, y = self.features_from_citations()
        self.vectorizer = DictVectorizer(sparse=True)
        X_fv = self.vectorizer.fit_transform(features)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X_fv, y, test_size=0.1)
        clf = SupervisedLearner._get_SVM()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return preds, y_test

    @staticmethod 
    def _get_SVM():
        return SVC(probability=True, kernel='linear')

    def train(self):
        features, y = self.features_from_citations()
        self.vectorizer = DictVectorizer(sparse=True)
        X_fv = self.vectorizer.fit_transform(features)
        
        self.clf = _get_SVM()

        ##
        # @TODO grid search over c?
        self.clf.fit(X_fv, y)


    @staticmethod
    def filter_tags(tags):
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


    def features_from_citations(self):
        X, y = [], []
        for cit in self.abstract_reader:
            # first we perform feature extraction over the
            # abstract text (X)
            abstract_text = cit["abstract"]
            p = bilearnPipeline(abstract_text)
            p.generate_features()
            #filter=lambda x: x["w[0]"].isdigit()
            ###
            # note that the pipeline segments text into
            # sentences. so X_i will comprise k lists, 
            # where k is the number of sentences. each 
            # of these lists contain the feature vectors
            # for the words comprising the respective 
            # sentences.
            ###
            X_i = p.get_features()
            words = p.get_answers()
   
            # now construct y vector based on parsed tags
            cit_file_id = cit["file_id"]
         
            # aahhhh stupid zero indexing confusion
            abstract_tags = annotation_parser.get_annotations(
                                cit_file_id-1, annotator_str, convert_numbers=True)


            # for now we'll flatten the sentences 
            words_flat, X_i_flat = [], []
            for sentence_i in xrange(len(X_i)):
                X_i_flat.extend(X_i[sentence_i])
                # one more step; swap in numbers!
                words_flat.extend([swap_num(w_ij) for w_ij in words[sentence_i]])


            # we only keep words for which we have annotations
            # and that are not, e.g., just puncutation.
            training_words, training_fvs = SupervisedLearner.filter_words(
                                    abstract_tags, words_flat, X_i_flat)

            training_words = [swap_num(w_i) for w_i in training_words]

            training_tags = SupervisedLearner.filter_tags(abstract_tags)
            y_i = []
            
            for j, w in enumerate(training_words):
                tags = None

                for tag_index, tag in enumerate(training_tags):
                    if tag.keys()[0] == w:
                        tags = tag.values()[0]
                        break

                if tags is None:
                    # uh-oh.
                    raise Exception, "no tags???"


                w_lbl = 1 if self.target in tags else -1
                X.append(training_fvs[j])

                y.append(w_lbl)
                # remove this tag from the list -- remember,
                # words can appear multiple times!
                training_tags.pop(tag_index)

        return X, y

        

class LabeledAbstractReader:
    ''' 
    Parses labeled citations from the provided path. Assumes format is like:

        Abstract 1 of 500
        Prothrombin fragments (F1+2) ...
            ...
        BiviewID 42957; PMID 11927130

    '''
    def __init__(self, path_to_data="data/drug_trials_in_cochrane_BCW.txt"):
        # @TODO probably want to read things in lazily, rather than
        # reading everything into memory at once...
        self.abstracts = []
        self.abstract_index = 1 # for iterator
        self.path_to_abstracts = path_to_data
        print "parsing data from {0}".format(self.path_to_abstracts)

        self.parse_abstracts()
        self.num_citations = len(self.citation_d) 
        print "ok."


    def __iter__(self):
        self.abstract_index = 1
        return self

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
        # zero-indexed and varies across files.
        abstract_num = 1
        with open(self.path_to_abstracts, 'rU') as abstracts_file:
            cur_abstract = ""
            
            for line in abstracts_file.readlines():
                line = line.strip()
                if self._is_demarcater(line):
                    biview_id, pmid = self._get_IDs(line)
                    self.citation_d[abstract_num] = {"abstract":cur_abstract, 
                                                "Biview_id":biview_id,
                                                "pubmed_id":pmid,
                                                "file_id":abstract_num} # yes, redundant
                    in_citation = False
                    abstract_num += 1
                elif in_citation and line:
                    # then this is the abstract
                    cur_abstract = line
                elif self._is_new_citation_line(line):
                    in_citation = True

        return self.citation_d

    def get_text(self):
        return [cit["abstract"] for cit in self.citation_d.values()]


if __name__ == "__main__":
    nruns = 10
    reader = LabeledAbstractReader()
    sl = SupervisedLearner(reader)
    p_sum, r_sum, f_sum, np_sum = [0]*4
    pb = progressbar.ProgressBar(nruns)
    for i in xrange(nruns):
        #print "on iter {0}".format(i)
        
        preds, y_test = sl.cv()
        p, r, f, s = precision_recall_fscore_support(y_test, preds, average="micro")
        p_sum += p
        r_sum += r
        f_sum += f
        np_sum += s
        pb.tap()

    avg = lambda x: x / float(nruns)

    print "averages\n # of target words: {0}\n precision: {1}\n recall: {2}\n f: {3}".format(
                    avg(np_sum), avg(p_sum), avg(r_sum), avg(f_sum))






