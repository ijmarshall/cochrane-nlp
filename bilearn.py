#
# version 2 of bilearner
#
#

# requires numpy, sklearn
#
#


import cPickle as pickle
import cPickle as pickle
from collections import defaultdict
import logging
import math
from pprint import pprint
import re

import biviewer
from indexnumbers import swap_num
import indexnumbers
import numpy as np
import pipeline
import progressbar
from scipy import stats
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


with open('data/brill_pos_tagger.pck', 'rb') as f:
    pos_tagger = pickle.load(f)

logging.basicConfig(level=logging.INFO)
logging.info("Importing python modules")


def show_most_informative_features(vectorizer, clf, n=25):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    c_f = [(math.exp(w), i) for (w, i) in c_f if (w < 0) or (w > 0)]
    if n == 0:
        n = len(c_f)/2
    top = zip(c_f[:n], c_f[:-(n+1):-1])
    for (c1, f1), (c2, f2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2)



class bilearnPipeline(pipeline.Pipeline):

    def __init__(self, text):
        self.functions = [[{"w": word, "p": pos} for word, pos in pos_tagger.tag(self.word_tokenize(sent))] for sent in self.sent_tokenize(swap_num(text))]
        self.load_templates()
        # self.stem = PorterStemmer()

    def load_templates(self):
        self.templates = (
                          (("w", 0),),
                          (("w", 1),),
                          (("w", 2),),
                          (("w", 3),),
                          (("w", -1),),
                          (("w", -2),),
                          (("w", -3),),
                          (('w', -2), ('w',  -1)),
                          # (('stem', -1), ('stem',  0)),
                          # (('stem',  0), ('stem',  1)),
                          (('w',  1), ('w',  2)),
                          # (('p',  0), ('p',  1)),
                          (('p',  1), ('p',  2)),
                          (('p',  -1), ('p',  -2)),
                          # (('stem', -2), ('stem',  -1), ('stem',  0)),
                          # (('stem', -1), ('stem',  0), ('stem',  1)),
                          # (('stem', 0), ('stem',  1), ('stem',  2)),
                          (('p', -2), ),
                          (('p', -1), ),
                          (('p', 1), ),
                          (('p', 2), ),
                          (('num', -1), ), 
                          (('num', 1), ),
                          (('cap', -1), ),
                          (('cap', 1), ),
                          (('sym', -1), ),
                          (('sym', 1), ),
                          # (('p1', 1), ),
                          # (('p2', 1), ),
                          # (('p3', 1), ),
                          # (('p4', 1), ),
                          # (('s1', 1), ),
                          # (('s2', 1), ),
                          # (('s3', 0), ),
                          # (('s4', 0), ),
                          (('wi', 0), ),
                          (('si', 0), ),
                          (('next_noun', 0), ),
                          )

        self.answer_key = "w"

    def run_functions(self, show_progress=False):
        for i, sent_function in enumerate(self.functions):
            words = {"BOW" + word["w"]: True for word in self.functions[i]}

            last_noun_index = 0

            for j, function in enumerate(sent_function):
                word = self.functions[i][j]["w"]
                features = {"num": word.isdigit(),
                            "cap": word[0].isupper(),
                            "sym": word.isalnum(),
                            "p1": word[0],
                            "p2": word[:2],
                            "p3": word[:3],
                            "p4": word[:4],
                            "s1": word[-1],
                            "s2": word[-2:],
                            "s3": word[-3:],
                            "s4": word[-4:],
                            # "stem": self.stem.stem(word),
                            "wi": j,
                            "si": i}
                
                self.functions[i][j].update(features)
                self.functions[i][j].update(words)

                # if pos is a noun, back fill the previous words
                pos = self.functions[i][j]["p"]
                if re.match("NN*", pos):
                    for k in range(last_noun_index, j):
                        self.functions[i][k]["next_noun"] = word
                    last_noun_index = j

            for k in range(last_noun_index, len(sent_function)):
                self.functions[i][k]["next_noun"] = "END_OF_SENTENCE"



class BiLearner():

    def __init__(self):

        logging.info("Initialising Bilearner")
        self.data = {}
        self.numberswapper = indexnumbers.NumberTagger()
        # self.stem = PorterStemmer()

    def reset(self):
        " sets the y_lookup back to the original regex rule results "
        self.y_lookup = self.data["y_lookup_init"].copy()

        # define empty arrays for the answers to both sides
        self.y_cochrane_no = np.empty(shape=(len(self.data["words_cochrane"]),))
        self.y_pubmed_no = np.empty(shape=(len(self.data["words_pubmed"]),))

        # make a new metrics dict
        self.metrics = defaultdict(list)
        # with 0 = the seed rule iteration
        # 1, 2, 3, n... = iterations
        # defaultdict can accept non-lists fine, for non changing properties


    def generate_features(self, test_mode=False):
        """
        generate the variables for the learning problem
        each row represents a candidate answer
        """        
        if test_mode:
            logging.warning("In test mode: not processing all data")
        logging.info("Loading biview data")

        # load cochrane and pubmed parallel text viewer
        self.biviewer = biviewer.BiViewer(in_memory=False, test_mode=test_mode)

        # feature variables
        # each item is a candidate answer (here meaning an integer from the text)
        # the X variables are static
        X_cochrane_l = []
        X_pubmed_l = []

        # store the words (here=numbers) of interest in a separate list
        # since these may not always be used as a feature, but still needed
        words_cochrane_l = []
        words_pubmed_l = []

        # these find the corresponding article (biviewer id) which the
        # candidate answer comes from, converted to numpy arrays later
        study_id_lookup_cochrane_l = []
        study_id_lookup_pubmed_l = []

        # answer variable PER STUDY (assumes one population per study)
        # used to generate y which is used for training
        # the y variables change at each iteration of the algorithm
        # depending which answers are most probable
        # key variable for the biviewer, 'correct' answers are passed
        # between the two views with it

        # called init since may be reused from stating position
        # should make a copy
        self.data["y_lookup_init"] = {}

        logging.info("Generating features, and finding seed data")
        p = progressbar.ProgressBar(len(self.biviewer), timer=True)

        counter = 0  # number of studies initially found

        for study_id, study in enumerate(self.biviewer):

            p.tap()

            cochrane_dict, pubmed_dict = study
            cochrane_text = cochrane_dict["CHAR_PARTICIPANTS"]
            pubmed_text = pubmed_dict["abstract"]


            # use simple rule to identify population sizes (low sens/recall, high spec/precision)
            matches = re.findall('([1-9][0-9]*) (?:participants|men|women|patients) were (?:randomi[sz]ed)', self.numberswapper.swap(pubmed_text))
            if len(matches) == 1:
                self.data["y_lookup_init"][study_id] = int(matches[0])
                counter += 1
            else:
                # -1 signifies population not known (at this stage)
                self.data["y_lookup_init"][study_id] = -1

            # generate features for all studies
            # the generate_features functions only generate features for integers
            # words are stored separately since they are not necessarily used as features
            #     but are needed for the answers
            X_cochrane_study, words_cochrane_study = self.generate_features_cochrane(cochrane_text)
            X_pubmed_study, words_pubmed_study = self.generate_features_pubmed(pubmed_text)

            # these lists will be made into array to be used as lookup dicts
            study_id_lookup_cochrane_l.extend([study_id] * len(X_cochrane_study))
            study_id_lookup_pubmed_l.extend([study_id] * len(X_pubmed_study))

            # Add features to the feature lists
            X_cochrane_l.extend(X_cochrane_study)
            # pprint(X_cochrane_study)
            X_pubmed_l.extend(X_pubmed_study)

            # Add words to the word lists
            words_cochrane_l.extend((int(word) for word in words_cochrane_study))
            words_pubmed_l.extend((int(word) for word in words_pubmed_study))

        logging.info("Creating NumPy arrays")

        # create np arrays for fast lookup of corresponding answer
        self.data["study_id_lookup_cochrane"] = np.array(study_id_lookup_cochrane_l)
        self.data["study_id_lookup_pubmed"] = np.array(study_id_lookup_pubmed_l)

        # create vectors for the 'words' which are the candidate answers
        self.data["words_cochrane"] = np.array(words_cochrane_l)
        self.data["words_pubmed"] = np.array(words_pubmed_l)

        # set up vectorisers for cochrane and pubmed
        self.data["vectoriser_cochrane"] = DictVectorizer(sparse=True)
        self.data["vectoriser_pubmed"] = DictVectorizer(sparse=True)

        # train vectorisers
        self.data["X_cochrane"] = self.data["vectoriser_cochrane"].fit_transform(X_cochrane_l)
        self.data["X_pubmed"] = self.data["vectoriser_pubmed"].fit_transform(X_pubmed_l)

        self.reset()
 
    def save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self, filename):
        logging.info("loading %s", filename)
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        logging.info("%s loaded successfully", filename)

    def learn(self, iterations=1, C=2, aperture=0.95, aperture_type="probability", test_abstract_file="data/test_abstracts.pck"):

        # save the current settings
        (self.metrics["iterations"], self.metrics["C"], self.metrics["aperture"],
            self.metrics["aperture_type"]) = (iterations, C, aperture, aperture_type)

        # open the test data
        with open(test_abstract_file, 'rb') as f:
            raw_data = pickle.load(f)

        test_data = []
        counter = 0  # number of matches using regular expression rule
        # convert to feature vector
        for entry in raw_data:
            matches = re.findall('([1-9][0-9]*) (?:participants|men|women|patients) were (?:randomi[sz]ed)', self.numberswapper.swap(entry["text"]))
            if len(matches) == 1:
                counter += 1

            features, words = self.text_to_features(entry["text"])
            test_data.append({"features":features, "words": words, "answer": entry["answer"], "text": entry['text']})

        logging.info("%d/%d accuracy with seed rule", counter, len(raw_data))
        self.metrics["study_accuracy"] = [counter]

        # .get_shape()[1] of the sparse matrix returns the number of columns
        # i.e. the total number of features (get_shape() = (nrows, ncolumns))
        self.metrics["samples_cochrane"], self.metrics["features_cochrane"] = self.data["X_cochrane"].get_shape()
        self.metrics["samples_pubmed"], self.metrics["features_pubmed"] = self.data["X_pubmed"].get_shape()

        p = progressbar.ProgressBar(iterations)

        for i in xrange(iterations):
            p.tap()
            confusion = []
            self.learn_cochrane(C=C, aperture=aperture, aperture_type=aperture_type)
            self.learn_pubmed(C=C, aperture=aperture, aperture_type=aperture_type)
            # logging.info("End of iteration %d" % (i, ))

            score = 0

            score2 = 0 # score positive and negative results
            denominator2 = 0

            for entry in test_data:
                if entry["features"] is not None:
                    prediction = self.predict_population_features(entry["features"], entry["words"])
                    denominator2 += len(entry["words"])
                    if int(prediction[0])==entry["answer"]:
                        score += 1
                        score2 += len(entry["words"])
                    else:
                        score2 += len(entry["words"]) - 2
                        confusion.append(entry["text"])
            logging.info("%d/%d accuracy after %d iterations", score, len(test_data), i)
            self.metrics["study_accuracy"].append(score)
            self.metrics["study_accuracy_2"].append(score2)
        self.metrics["study_denominator_2"].append(denominator2)
            

        with open('confused.txt', 'wb') as f:
            f.write("\n\n".join(confusion))

        
    def learn_cochrane(self, C=2.5, aperture=0.90, aperture_type='probability'):

        

        # create answer vectors with the seed answers
        for word_id in xrange(len(self.y_cochrane_no)):
            self.y_cochrane_no[word_id] = self.y_lookup[self.data["study_id_lookup_cochrane"][word_id]]

        self.y_cochrane = (self.y_cochrane_no == self.data["words_cochrane"])




        # set filter vectors (-1 = unknown)
        filter_train = (self.y_cochrane_no != -1).nonzero()[0]
        filter_test = (self.y_cochrane_no == -1).nonzero()[0]

        self.metrics["cochrane_training_examples"].append(len(filter_train))
        self.metrics["cochrane_test_examples"].append(len(filter_test))

        if len(filter_test)==0:
            print "leaving early - run out of data!"
            raise IndexError("Out of data from Cochrane side")

        # set training vectors
        X_cochrane_train = self.data["X_cochrane"][filter_train]
        y_cochrane_train = self.y_cochrane[filter_train]

        # and test vectors as the rest
        X_cochrane_test = self.data["X_cochrane"][filter_test]
        y_cochrane_test = self.y_cochrane[filter_test]

        # and the numbers to go with it for illustration purposes
        words_cochrane_test = self.data["words_cochrane"][filter_test]
        study_id_lookup_cochrane_test = self.data["study_id_lookup_cochrane"][filter_test]

        # make and fit new LR model
        model = LogisticRegression(C=C, penalty='l1')
        logging.debug("fitting model to cochrane data...")
        model.fit(X_cochrane_train, y_cochrane_train)





        # assign predicted probabilities of being a population size to result
        result = model.predict_proba(X_cochrane_test)[:,1]

        self.metrics["cochrane_prob_dist"].append(np.array(np.percentile(np.absolute(0.5-result), list(np.arange(0, 1, 0.1)))).round(3))
        

        # set the cut off for picking the most confident results
        
        if aperture_type == "percentile":
            top_pc_score = stats.scoreatpercentile(result, aperture)
            top_result_indices = (result > top_pc_score).nonzero()[0]
        elif aperture_type == "absolute":
            top_result_indices = np.argsort(result)[-aperture:]
        else:
            top_pc_score = aperture
            top_result_indices = (result > top_pc_score).nonzero()[0]


        self.metrics["cochrane_no_top_indices"].append(len(top_result_indices))

        # # exclude the answer if another answer in the same study scores > 0.5 probability
        # exclude_result_indices = (result > 0.95).nonzero()[0]
        # print len(top_result_indices), len(self.biviewer)

                # number of times each study is represented
        # instances = np.bincount(study_id_lookup_cochrane_test[exclude_result_indices])

        for i in top_result_indices:
            # if instances[study_id_lookup_cochrane_test[i]] == 1:
            self.y_lookup[study_id_lookup_cochrane_test[i]] = words_cochrane_test[i]
            # print "Number %d, with %.2f probability" % (words_cochrane_test[i], result[i])
            # print
            # print self.biviewer[study_id_lookup_cochrane_test[i]][0]
            # print
            # print
            # else:
            #     print "multiple high probability results for no %d" % (i,)

        # show_most_informative_features(self.data["vectoriser_cochrane"], model)

    def learn_pubmed(self, C=2.5, aperture=0.90, aperture_type='probability'):

        for word_id in xrange(len(self.y_pubmed_no)):
            self.y_pubmed_no[word_id] = self.y_lookup[self.data["study_id_lookup_pubmed"][word_id]]

        self.y_pubmed = (self.y_pubmed_no == self.data["words_pubmed"])

        # set filter vectors (-1 = unknown)
        filter_train = (self.y_pubmed_no != -1).nonzero()[0]
        filter_test = (self.y_pubmed_no == -1).nonzero()[0]

        if len(filter_test)==0:
            print "leaving early - run out of data!"
            raise IndexError("Out of data from Pubmed side")


        self.metrics["pubmed_training_examples"].append(len(filter_train))
        self.metrics["pubmed_test_examples"].append(len(filter_test))


        # set training vectors
        X_pubmed_train = self.data["X_pubmed"][filter_train]
        y_pubmed_train = self.y_pubmed[filter_train]

        # and test vectors as the rest
        X_pubmed_test = self.data["X_pubmed"][filter_test]
        y_pubmed_test = self.y_pubmed[filter_test]

        # and the numbers to go with it for illustration purposes
        words_pubmed_test = self.data["words_pubmed"][filter_test]
        study_id_lookup_pubmed_test = self.data["study_id_lookup_pubmed"][filter_test]

        # make and fit new LR model
        self.pubmed_model = LogisticRegression(C=C, penalty='l1')
        logging.debug("fitting model to pubmed data...")
        self.pubmed_model.fit(X_pubmed_train, y_pubmed_train)

        # assign predicted probabilities of being a population size to result
        result = self.pubmed_model.predict_proba(X_pubmed_test)[:,1]

        
        self.metrics["pubmed_prob_dist"].append(np.array(np.percentile(np.absolute(0.5-result), list(np.arange(0, 1, 0.1)))).round(3))

        # set the cut off for picking the most confident results in both directions
        
        if aperture_type == "percentile":
            top_pc_score = stats.scoreatpercentile(result, aperture)
            top_result_indices = (result > top_pc_score).nonzero()[0]
        elif aperture_type == "absolute":
            top_result_indices = np.argsort(result)[-aperture:]
        else:
            top_pc_score = aperture
            top_result_indices = (result > top_pc_score).nonzero()[0]

        self.metrics["pubmed_no_top_indices"].append(len(top_result_indices))

        if len(top_result_indices) == 0:
            raise IndexError("Iteration not identifying any further high probability candidates")

        # top_result_indices = np.argsort(result)[-10:]

        # exclude the answer if another answer in the same study scores > 0.5 probability
        # exclude_result_indices = (result > 0.95).nonzero()[0]


        # number of times each study is represented
        # instances = np.bincount(study_id_lookup_pubmed_test[exclude_result_indices])

        for i in top_result_indices:
            # if instances[study_id_lookup_pubmed_test[i]] == 1:
            self.y_lookup[study_id_lookup_pubmed_test[i]] = words_pubmed_test[i]
            # print "Number %d, with %.2f probability" % (words_pubmed_test[i], result[i])
            # print
            # print self.biviewer[study_id_lookup_pubmed_test[i]][1]
            # print
            # print
            # else:
            #     print "multiple high probability results for no %d" % (i,)

        # show_most_informative_features(self.data["vectoriser_pubmed"], self.pubmed_model)

    def generate_features_cochrane(self, text):
        "generate and return features for Cochrane review"
        p = bilearnPipeline(text)
        p.generate_features()
        X = p.get_features(filter=lambda x: x["w[0]"].isdigit())
        words = p.get_answers(filter=lambda x: x["w"].isdigit())

        
        return X, words

    def generate_features_pubmed(self, text, answer_fn=None, include_word=False):
        "generate features for pubmed abstract"
        p = bilearnPipeline(text)
        p.generate_features()
        X = p.get_features(filter=lambda x: x["w[0]"].isdigit())
        words = p.get_answers(filter=lambda x: x["w"].isdigit())

        return X, words

    def predict_population_text(self, text):
        X_pubmed_study_vec, words_pubmed_study = self.text_to_features(text)
        return self.predict_population_features(X_pubmed_study_vec, words_pubmed_study)

    def predict_population_features(self, X_pubmed_study_vec, words_pubmed_study):
        # assign predicted probabilities of being a population size to result
        result = self.pubmed_model.predict_proba(X_pubmed_study_vec)[:,1]
        # get index of maximum probability
        amax = np.argmax(result)
        # return (winning number, probability)
        return (words_pubmed_study[amax], result[amax])

    def text_to_features(self, text):
        # load features and get words
        X_pubmed_study, words_pubmed_study = self.generate_features_pubmed(text)
        # change features to sparse matrix
        if X_pubmed_study:
            X_pubmed_study_vec = self.data["vectoriser_pubmed"].transform(X_pubmed_study)
        else:
            X_pubmed_study_vec = None
        return X_pubmed_study_vec, words_pubmed_study

    def evaluate_test(self, filename):
        with open(filename, 'rb') as f:
            test_data = pickle.load(f)

        score = 0
        for entry in test_data:
            if int(self.predict_population_text(entry["text"])[0])==entry["answer"]:
                score += 1
        print "final score %d/%d" % (score, len(test_data))










def main():
    test()










def test():

    b = BiLearner()
    b.generate_features(test_mode=True) # test_mode just uses the first 250 cochrane reviews for speed
    b.learn(iterations=10, C=3, aperture=100, aperture_type="absolute")
    pprint(b.metrics)
    

if __name__ == '__main__':
    main()

