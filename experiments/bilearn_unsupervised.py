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

import numpy as np
import pipeline
import progressbar
from scipy import stats
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizer import tag_words, MergedTaggedAbstractReader
# from journalreaders import LabeledAbstractReader

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import RandomizedLogisticRegression

from sklearn.cross_validation import ShuffleSplit

from scipy.sparse import vstack


with open('data/brill_pos_tagger.pck', 'rb') as f:
    pos_tagger = pickle.load(f)

logging.basicConfig(level=logging.INFO)
logging.info("Importing python modules")


def show_most_informative_features(vectorizer, clf, n=50):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    c_f = [(math.exp(w), i) for (w, i) in c_f if (w < 0) or (w > 0)]
    if n == 0:
        n = len(c_f)/2
    top = zip(c_f[:n], c_f[:-(n+1):-1])
    for (c1, f1), (c2, f2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2)



class bilearnPipeline(pipeline.Pipeline):

    def __init__(self, text, window_size):
        self.text = re.sub('(?:[0-9]+)\,(?:[0-9]+)', '', text)
        self.functions = [[{"w": word, "p": pos} for word, pos in pos_tagger.tag(self.word_tokenize(sent))] for sent in self.sent_tokenize(swap_num(text))]
        self.load_templates()        
        self.w_pos_window = window_size
        self.text = text  
        

    def load_templates(self):
        self.templates = (
                            (("w_int", 0),),
                            # (("w", 1),),
                            # (("w", 2),),
                            # (("w", 3),),
                            # # (("wl", 4),),
                            # (("w", -1),),
                            # (("w", -2),),
                            # (("w", -3),),
                            # (("wl", -4),),
                            # (('w', -2), ('w',  -1)),
                            # (('wl',  -1), ('wl',  -2), ('wl',  -3)),
                            # (('stem', -1), ('stem',  0)),
                            # (('stem',  0), ('stem',  1)),
                            # (('w',  1), ('w',  2)),
                            # (('wl',  1), ('wl',  2), ('wl',  3)),
                            # (('p',  0), ('p',  1)),
                            # (('p',  1),),
                            # (('p',  2),),
                            # (('p',  -1),),
                            # (('p',  -2),),
                            # (('p',  1), ('p',  2)),
                            # (('p',  -1), ('p',  -2)),
                            # (('stem', -2), ('stem',  -1), ('stem',  0)),
                            # (('stem', -1), ('stem',  0), ('stem',  1)),
                            # (('stem', 0), ('stem',  1), ('stem',  2)),
                            # (('p', -2), ),
                            # (('p', -1), ),
                            # (('p', 1), ),
                            # (('p', 2), ),
                            # (('num', -1), ), 
                            # (('num', 1), ),
                            # (('cap', -1), ),
                            # (('cap', 1), ),
                            # (('sym', -1), ),
                            # (('sym', 1), ),
                            (('div10', 0), ),
                            (('>10', 0), ),
                            (('numrank', 0), ),
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
                            # (('next_noun', 0), ),
                            # (('next_verb', 0), ),
                            # (('last_noun', 0), ),
                            # (('last_verb', 0), ),
                            )

        self.answer_key = "w"
        # self.w_pos_window = window_size # set 0 for no w_pos window features
 
    def run_functions(self, show_progress=False):

        # make dict to look up ranking of number in abstract
        num_list_nest = [[int(word["w"]) for word in sent if word["w"].isdigit()] for sent in self.functions]
        num_list = [item for sublist in num_list_nest for item in sublist] # flatten
        num_list.sort(reverse=True)
        num_dict = {num: len(num_list)+2-rank for rank, num in enumerate(num_list)}
        num_index = 0
        for i, sent_function in enumerate(self.functions):

            last_noun_index = 0
            last_noun = "BEGINNING_OF_SENTENCE"

            last_verb_index = 0
            last_verb = "BEGINNING_OF_SENTENCE"

            for j, function in enumerate(sent_function):
                # print j
                word = self.functions[i][j]["w"]
                features = {"num": word.isdigit(),
                            "cap": word[0].isupper(),
                            "sym": not word.isalnum(),
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
                            "si": i,
                            "wl": word.lower(),
                            "punct": not any(c.isalnum() for c in word) # all 
                            }
                if word.isdigit():
                    num = int(word)
                    num_index += 1
                    features[">10"] = num > 10
                    features["w_int"] = num
                    features["div10"] = ((num % 10) == 0)
                    features["numrank"] = num_dict[num]
                    features["numindex"] = num_index
                
                self.functions[i][j].update(features)


                # self.functions[i][j].update(words)

                # if pos is a noun, back fill the previous words with 'next_noun'
                # and the rest as 'last_noun'
                pos = self.functions[i][j]["p"]
                
                if re.match("NN.*", pos):

                    for k in range(last_noun_index, j):
                        self.functions[i][k]["next_noun"] = word
                        self.functions[i][k]["last_noun"] = last_noun
                    last_noun_index = j
                    last_noun = word
                    
                # and the same for verbs
                elif re.match("VB.*", pos):

                    for k in range(last_verb_index, j):
                        
                        self.functions[i][k]["next_verb"] = word
                        self.functions[i][k]["last_verb"] = last_verb
                    last_verb_index = j
                    last_verb = word

            for k in range(last_noun_index, len(sent_function)):
                self.functions[i][k]["next_noun"] = "END_OF_SENTENCE"
                self.functions[i][k]["last_noun"] = last_noun

            for k in range(last_verb_index, len(sent_function)):
                self.functions[i][k]["next_verb"] = "END_OF_SENTENCE"
                self.functions[i][k]["last_verb"] = last_verb

class bilearnPipelineCochrane(bilearnPipeline):

    def __init__(self, text_dict, window_size):

        self.functions = []
        for key, value in text_dict.iteritems():
            self.functions.extend([[{"w": word, "p": pos, "cochrane_part":key} for word, pos in pos_tagger.tag(self.word_tokenize(sent))] for sent in self.sent_tokenize(swap_num(value))])

        self.load_templates()        
        self.w_pos_window = window_size
        # self.text = text  
        
    def load_templates(self):
        self.templates = (
                        (("w_int", 0),),
                        # (("w", 1),),
                        # (("w", 2),),
                        # (("w", 3),),
                        # # (("wl", 4),),
                        # (("w", -1),),
                        # (("w", -2),),
                        # (("w", -3),),
                        # (("wl", -4),),
                        # (('w', -2), ('w',  -1)),
                        # (('wl',  -1), ('wl',  -2), ('wl',  -3)),
                        # (('stem', -1), ('stem',  0)),
                        # (('stem',  0), ('stem',  1)),
                        # (('w',  1), ('w',  2)),
                        # (('wl',  1), ('wl',  2), ('wl',  3)),
                        # (('p',  0), ('p',  1)),
                        # (('p',  1),),
                        # (('p',  2),),
                        # (('p',  -1),),
                        # (('p',  -2),),
                        # (('p',  1), ('p',  2)),
                        # (('p',  -1), ('p',  -2)),
                        # (('stem', -2), ('stem',  -1), ('stem',  0)),
                        # (('stem', -1), ('stem',  0), ('stem',  1)),
                        # (('stem', 0), ('stem',  1), ('stem',  2)),
                        # (('p', -2), ),
                        # (('p', -1), ),
                        # (('p', 1), ),
                        # (('p', 2), ),
                        # (('num', -1), ), 
                        # (('num', 1), ),
                        # (('cap', -1), ),
                        # (('cap', 1), ),
                        # (('sym', -1), ),
                        # (('sym', 1), ),
                        (('div10', 0), ),
                        (('>10', 0), ),
                        (('numrank', 0), ),
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
                        # (('cochrane_part', 0), ),
                        # (('next_noun', 0), ),
                        # (('next_verb', 0), ),
                        # (('last_noun', 0), ),
                        # (('last_verb', 0), ),
                        )

class BiLearner():

    def __init__(self, test_mode=True, window_size=4):

        logging.info("Initialising Bilearner")
        self.data = {}
        # load cochrane and pubmed parallel text viewer
        self.biviewer = biviewer.BiViewer(in_memory=False, test_mode=test_mode)
        self.window_size = window_size

        
        # self.stem = PorterStemmer()

    def initialise(self, seed="regex"):

        
        self.data["y_lookup_init"] = {i: -1 for i in range(len(self.biviewer))}

        # if seed == "annotations":
        #     self.seed_y_annotations() # generate y vector from regex
        # else:
        #     self.seed_y_regex() # generate y vector from regex

        print "Resetting joint predictions"
        self.pred_joint = self.data["y_lookup_init"].copy()

        # define empty arrays for the answers to both sides
        self.y_cochrane_no = np.empty(shape=(len(self.data["words_cochrane"]),))
        self.y_pubmed_no = np.empty(shape=(len(self.data["words_pubmed"]),))

        # make a new metrics dict
        self.metrics = defaultdict(list)
        # with 0 = the seed rule iteration
        # 1, 2, 3, n... = iterations
        # defaultdict can accept non-lists fine, for non changing properties


    def generate_features(self, test_mode=False, filter_uniques=True):
        """
        generate the variables for the learning problem
        each row represents a candidate answer
        filter_uniques = exclude candidates where the number is not replicated in both Cochrane + Pubmed
        """        
        if test_mode:
            logging.warning("In test mode: not processing all data")
        logging.info("Loading biview data")

        
        # feature variables
        # each item is a candidate answer (here meaning an integer from the text)
        # the X variables are static
        X_cochrane_l = []
        X_pubmed_l = []
        X_pubmed_external_l = [] # so that predictions on 'unseen' abstracts can't use the internal cochrane features

        # store the words (here=numbers) of interest in a separate list
        # since these may not always be used as a feature, but still needed
        words_cochrane_l = []
        words_pubmed_l = []

        # these find the corresponding article (biviewer id) which the
        # candidate answer comes from, converted to numpy arrays later
        study_id_lookup_cochrane_l = []
        study_id_lookup_pubmed_l = []

        # filter the training/test data to integers which appear in both datasets
        # during training
        # models for assessment run on the whole dataset
        cochrane_distant_filter_l = []
        pubmed_distant_filter_l = []

        # answer variable PER STUDY (assumes one population per study)
        # used to generate y which is used for training
        # the y variables change at each iteration of the algorithm
        # depending which answers are most probable
        # key variable for the biviewer, 'correct' answers are passed
        # between the two views with it

        # called init since may be reused from stating position
        # should make a copy
        # self.data["y_lookup_init"] = {}

        

        logging.info("Generating features")
        p = progressbar.ProgressBar(len(self.biviewer), timer=True)

        counter = 0  # number of studies initially found

        for study_id, study in enumerate(self.biviewer):

            p.tap()

            cochrane_dict, pubmed_dict = study
            cochrane_dict_subset = {k: cochrane_dict.get(k, "") for k in ('CHAR_PARTICIPANTS', 'CHAR_INTERVENTIONS', 'CHAR_OUTCOMES', 'CHAR_NOTES')}
            
            pubmed_text = pubmed_dict.get("abstract", "")


            
            # generate features for all studies
            # the generate_features functions only generate features for integers
            # words are stored separately since they are not necessarily used as features
            #     but are needed for the answers




            # first generate features from the parallel texts

            p_cochrane = bilearnPipelineCochrane(cochrane_dict_subset, self.window_size)
            words_cochrane_study = p_cochrane.get_words(filter=lambda x: x["w"].isdigit(), flatten=True) # get the integers

            p_pubmed = bilearnPipeline(pubmed_text, self.window_size)
            words_pubmed_study = p_pubmed.get_words(filter=lambda x: x["w"].isdigit(), flatten=True) # get the integers


            # generate filter vectors for integers which match between Cochrane + Pubmed
            common_ints = set(words_cochrane_study) &  set(words_pubmed_study)


            


            # add presence in both texts as a common feature
            p_cochrane.add_feature(feature_id="shared_num", feature_fn=lambda x: x["w"] in common_ints)
            p_pubmed.add_feature(feature_id="shared_num", feature_fn=lambda x: x["w"] in common_ints)


            # p_cochrane = bilearnPipeline(cochrane_dict_subset["CHAR_PARTICIPANTS"], self.window_size)
            p_cochrane.generate_features()
            p_pubmed.generate_features()
            
            






            cochrane_filter_study = p_cochrane.get_answers(answer_key=lambda x: x["w"] in common_ints, filter=lambda x: x["num"], flatten=True)
            pubmed_filter_study = p_pubmed.get_answers(answer_key=lambda x: x["w"] in common_ints, filter=lambda x: x["num"], flatten=True)


            # get filtered + flattened feature dicts
            X_cochrane_study = p_cochrane.get_features(filter=lambda x: x["num"], flatten=True)
            X_pubmed_study = p_pubmed.get_features(filter=lambda x: x["num"], flatten=True)


            # print X_pubmed_study

            
            
            # these lists will be made into array to be used as lookup dicts
            study_id_lookup_cochrane_l.extend([study_id] * len(X_cochrane_study))
            study_id_lookup_pubmed_l.extend([study_id] * len(X_pubmed_study))

            # Add features to the feature lists
            X_cochrane_l.extend(X_cochrane_study)
            X_pubmed_l.extend(X_pubmed_study)

            # Add words to the word lists
            words_cochrane_l.extend((int(word) for word in words_cochrane_study))
            words_pubmed_l.extend((int(word) for word in words_pubmed_study))

            # Add filters to the filter lists
            cochrane_distant_filter_l.extend(cochrane_filter_study)
            pubmed_distant_filter_l.extend(pubmed_filter_study)


        logging.info("Creating NumPy arrays")

        # create np arrays for fast lookup of corresponding answer
        self.data["study_id_lookup_cochrane"] = np.array(study_id_lookup_cochrane_l)
        self.data["study_id_lookup_pubmed"] = np.array(study_id_lookup_pubmed_l)

        # create vectors for the 'words' which are the candidate answers
        self.data["words_cochrane"] = np.array(words_cochrane_l)
        self.data["words_pubmed"] = np.array(words_pubmed_l)

        # create filter vectors for training
        self.data["distant_filter_cochrane"] = np.array(cochrane_distant_filter_l)
        self.data["distant_filter_pubmed"] = np.array(pubmed_distant_filter_l)

        # set up vectorisers for cochrane and pubmed
        self.data["vectoriser_cochrane"] = DictVectorizer(sparse=True)
        self.data["vectoriser_pubmed"] = DictVectorizer(sparse=True)

        # train vectorisers
        self.data["X_cochrane"] = self.data["vectoriser_cochrane"].fit_transform(X_cochrane_l)
        self.data["X_pubmed"] = self.data["vectoriser_pubmed"].fit_transform(X_pubmed_l)
        self.data["X_pubmed_external"] = self.data["vectoriser_pubmed"].fit_transform(X_pubmed_external_l)


        # self.reset()



    def seed_y_annotations(self, annotation_viewer, hide_reader_ids, test_reader_ids):
        """
        initialises the joint y vector with data from manually annotated abstracts
        filter_ids = ids of the MergedTaggedAbstractReader to pay attention to
        """

        self.initialise()

        self.annotation_viewer = annotation_viewer

        self.annotator_viewer_to_biviewer = {}

        logging.info("Generating seed data from annotated abstracts")
        
        p = progressbar.ProgressBar(len(self.annotation_viewer), timer=True)

        hide_biviewer_ids = []
        test_biviewer_ids = []

        for study in range(len(self.annotation_viewer)):
            p.tap()

            biview_id = annotation_viewer[study]["biview_id"]
            self.annotation_viewer_to_biviewer[study] = biview_id

            parsed_tags = [item for sublist in annotation_viewer.get(study) for item in sublist] # flatten list
            tagged_numbers = [w[0] for w in parsed_tags if 'n' in w[1]] # then get any tagged numbers

            if tagged_numbers:
                number = int(tagged_numbers[0])
            else:
                number = -2

            self.data["y_lookup_init"][biview_id] = number


    def seed_y_regex(self, annotation_viewer):
        """
        initialises the joint y vector with data from manually annotated abstracts
        filter_ids = ids of the MergedTaggedAbstractReader to pay attention to
        """

        self.initialise()

        self.annotation_viewer_to_biviewer = {}
        self.answers = {}

        self.annotation_viewer = annotation_viewer

        logging.info("Generating answers for test set")
        
        p = progressbar.ProgressBar(len(self.annotation_viewer), timer=True)


        for study in range(len(self.annotation_viewer)):
            p.tap()

            biview_id = annotation_viewer[study]["biview_id"]
            self.annotation_viewer_to_biviewer[study] = biview_id

            # set answers
            parsed_tags = [item for sublist in annotation_viewer.get(study) for item in sublist] # flatten list
            tagged_numbers = [w[0] for w in parsed_tags if 'n' in w[1]] # then get any tagged numbers

            if tagged_numbers:
                number = int(tagged_numbers[0])
            else:
                number = -2

            self.answers[biview_id] = number


        logging.info("Generating seed data from regular expression")

        p = progressbar.ProgressBar(len(self.biviewer), timer=True)
        counter = 0  # number of studies initially found
        for study_id, (cochrane_dict, pubmed_dict) in enumerate(self.biviewer):
            p.tap()
            pubmed_text = pubmed_dict.get("abstract", "")
            # use simple rule to identify population sizes (low sens/recall, high spec/precision)
            pubmed_text = swap_num(pubmed_text)
            matches = re.findall('([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients|children|people) were (?:randomi[sz]ed)', pubmed_text)
            # matches += re.findall('(?:[Ww]e randomi[sz]ed )([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients)', pubmed_text)
            # matches += re.findall('(?:[Aa] total of )([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients)', pubmed_text)
            if len(matches) == 1:
                self.data["y_lookup_init"][study_id] = int(matches[0])
                counter += 1

        self.seed_abstracts = counter
        logging.info("%d seed abstracts found", counter)

     


      
    def reset(self, hide_reader_ids, test_reader_ids):

        self.test_reader_ids = test_reader_ids
        self.hide_reader_ids = hide_reader_ids

        hide_biviewer_ids = []
        test_biviewer_ids = []

        for study_id in test_reader_ids:
            test_biviewer_ids.append(self.annotation_viewer_to_biviewer[study_id])

        for study_id in hide_reader_ids:
            hide_biviewer_ids.append(self.annotation_viewer_to_biviewer[study_id])


        self.visible_biviewer_ids = np.array(list(set(range(len(self.biviewer))) - set(hide_biviewer_ids)))
        self.test_biviewer_ids = np.array(test_biviewer_ids)

        
        self.pred_joint = self.data["y_lookup_init"].copy()

        # these weird bits of code find the indices of the pubmed and cochrane
        # training data vectors which correspond to the biviewer ids found in 
        # the loop above
        self.visible_cochrane_ids = np.arange(self.data["study_id_lookup_cochrane"].shape[0])[np.in1d(self.data["study_id_lookup_cochrane"], self.visible_biviewer_ids)]
        self.visible_pubmed_ids = np.arange(self.data["study_id_lookup_pubmed"].shape[0])[np.in1d(self.data["study_id_lookup_pubmed"], self.visible_biviewer_ids)]
        self.test_pubmed_ids = np.arange(self.data["study_id_lookup_pubmed"].shape[0])[np.in1d(self.data["study_id_lookup_pubmed"], self.test_biviewer_ids)]
     


    def save_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def load_data(self, filename):
        logging.info("loading %s", filename)
        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        logging.info("%s loaded successfully", filename)

    def learn(self, iterations=1, C=2, aperture=0.95, sample_weight=1, aperture_type="probability", test_abstract_file="data/test_abstracts.pck"):

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
            # matches = re.findall('([1-9][0-9]*) (?:participants|men|women|patients) were (?:randomi[sz]ed)', self.numberswapper.swap(entry["text"]))
            # if len(matches) == 1:
            #     counter += 1

            features, words = self.generate_prediction_features(entry["text"])
            test_data.append({"features":features, "words": words, "answer": entry["answer"], "text": entry['text']})


        # logging.info("%d/%d accuracy with seed rule", counter, len(raw_data))
        # self.metrics["study_accuracy"] = [counter]

        # .get_shape()[1] of the sparse matrix returns the number of columns
        # i.e. the total number of features (get_shape() = (nrows, ncolumns))
        self.metrics["samples_cochrane"], self.metrics["features_cochrane"] = self.data["X_cochrane"].get_shape()
        self.metrics["samples_pubmed"], self.metrics["features_pubmed"] = self.data["X_pubmed"].get_shape()

        p = progressbar.ProgressBar(iterations)

        p_filter, c_filter = self.data["distant_filter_pubmed"], self.data["distant_filter_cochrane"]


        




        for i in xrange(iterations):
            p.tap()
            confusion = []
            # self.learn_cochrane(C=C, aperture=aperture, aperture_type=aperture_type)

            if i > 0:

                # # if i % 2 == 0:
                # cochrane_model_f = self.learn_view(self.data["X_cochrane"][c_filter], self.data["words_cochrane"][c_filter], self.data["study_id_lookup_cochrane"][c_filter],
                #                 C=C, aperture=aperture, aperture_type=aperture_type, update_joint=True)

                # # else:
                # pubmed_model_f = self.learn_view(self.data["X_pubmed"][p_filter], self.data["words_pubmed"][p_filter], self.data["study_id_lookup_pubmed"][p_filter],
                #                 C=C, aperture=aperture, aperture_type=aperture_type, update_joint=True)

                # print p_filter
                # print c_filter

                # print len(p_filter), len(c_filter)

                cochrane_model_f = self.learn_view(self.data["X_cochrane"], self.data["words_cochrane"], self.data["study_id_lookup_cochrane"],
                                C=C, aperture=aperture, aperture_type=aperture_type, update_joint=True, sample_weight=sample_weight)

                # print show_most_informative_features(self.data["vectoriser_pubmed"], cochrane_model_f)


                pubmed_model = self.learn_view(self.data["X_pubmed"], self.data["words_pubmed"], self.data["study_id_lookup_pubmed"],
                                C=C, aperture=aperture, aperture_type=aperture_type, update_joint=True, sample_weight=sample_weight)

                


            else:


                pubmed_model = self.learn_view(self.data["X_pubmed"], self.data["words_pubmed"], self.data["study_id_lookup_pubmed"],
                            C=0.8, aperture=aperture, aperture_type=aperture_type, update_joint=False, sample_weight=sample_weight)
            
            
            
            


            score = 0

            score2 = 0 # score positive and negative results
            denominator2 = 0

            for entry in test_data:
                if entry["features"] is not None:
                    prediction = self.predict_population_features(entry["features"], entry["words"], pubmed_model)
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


    def learn_pubmed(self, C=1.0, update=False, aperture_type="absolute", aperture=20, sample_weight=1):

        pubmed_model = self.learn_view(self.data["X_pubmed"][self.visible_pubmed_ids], self.data["words_pubmed"][self.visible_pubmed_ids], self.data["study_id_lookup_pubmed"][self.visible_pubmed_ids],
                        C=C, aperture=aperture, aperture_type=aperture_type, update_joint=update, sample_weight=sample_weight)
        
        return pubmed_model

    def learn_cochrane(self, C=1.0, update=False, aperture_type="absolute", aperture=20, sample_weight=1):


        cochrane_model = self.learn_view(self.data["X_cochrane"][self.visible_cochrane_ids], self.data["words_cochrane"][self.visible_cochrane_ids], self.data["study_id_lookup_cochrane"][self.visible_cochrane_ids],
                        C=C, aperture=aperture, aperture_type=aperture_type, update_joint=update, sample_weight=sample_weight)
        
        return cochrane_model


    def learn_view(self, X_view, words_view, joint_from_view_index,
                    C=1.0, aperture=0.90, aperture_type='probability', update_joint=True,
                    sample_weight=1):
        

        initial_test_filter = np.empty(shape=(len(words_view),), dtype=bool)

        for word_id in xrange(len(words_view)):
            y = self.data["y_lookup_init"][joint_from_view_index[word_id]]
            initial_test_filter[word_id] = (y != -1)


        # print "self.pred_joint info"
        # no_known = len([i for i in self.pred_joint.values() if i != -1])
        # print "no known = %d/%d" % (no_known, len(self.pred_joint.values()))
        # print






        
        # extend all the variables appropriately

        X_view_w = X_view
        words_view_w = words_view
        joint_from_view_index_w = joint_from_view_index

        

        # print "X VIEW"
        # print X_view.get_shape()
        # print X_view
        # print "X VIEW WINDOW - pre"
        # print X_view_w.get_shape()
        # print X_view_w

        initial_test_filter = initial_test_filter.nonzero()[0]

        # print "initial test filter"
        # print initial_test_filter
        # print len(initial_test_filter)

        for i in range(sample_weight-1):
            X_view_w = vstack((X_view_w, X_view[initial_test_filter]), format="csr")
            words_view_w = np.concatenate((words_view_w, words_view[initial_test_filter]))
            joint_from_view_index_w = np.concatenate((joint_from_view_index_w, joint_from_view_index[initial_test_filter]))

        # print "X VIEW WINDOW - post"
        # print X_view_w.get_shape()
        # print X_view_w
        



        pred_view_w = np.empty(shape=(len(words_view_w),), dtype=int) # make a new empty vector for predicted values
        # (pred_view is predicted population sizes; not true/false)

        # print self.pred_joint


        



        # create answer vectors with the seed answers
        for word_id in xrange(len(pred_view_w)):
            pred_view_w[word_id] = self.pred_joint[joint_from_view_index_w[word_id]]
            
        



        y_view_w = (pred_view_w == words_view_w) #* 2 - 1 # set Trues to 1 and Falses to -1

        # set filter vectors (-1 = unknown)



        filter_train = (pred_view_w != -1).nonzero()[0]
        filter_test = (pred_view_w == -1).nonzero()[0]

        # print len(filter_train), len(filter_test)

        # print filter_train, len(filter_train)
        # print filter_train, len(filter_test)


        # self.metrics["cochrane_training_examples"].append(len(filter_train))
        # self.metrics["cochrane_test_examples"].append(len(filter_test))


        if len(filter_test)==0:
            print "leaving early - run out of data!"
            raise IndexError("out of data")


        # set training vectors
        X_train = X_view_w[filter_train]
        y_train = y_view_w[filter_train]
        

        # and test vectors as the rest
        X_test = X_view_w[filter_test]
        y_test = y_view_w[filter_test]

        # and the numbers to go with it for illustration purposes
        words_test = words_view_w[filter_test]
        joint_from_view_index_test = joint_from_view_index_w[filter_test]

        # make and fit new LR model
        # model = LogisticRegression(C=C, penalty='l1')
        model = self.model(C=C)
        logging.debug("fitting model to cochrane data...")
        model.fit(X_train, y_train)






        if update_joint:

            preds = model.predict_proba(X_test)[:,1] # predict unknowns

            # get top results (by aperture type selected)
            if aperture_type == "percentile":
                top_pc_score = stats.scoreatpercentile(preds, aperture)
                top_result_indices = (preds > top_pc_score).nonzero()[0]
            elif aperture_type == "absolute":
                top_result_indices = np.argsort(preds)[-aperture:]
            else:
                top_pc_score = aperture
                top_result_indices = (preds > top_pc_score).nonzero()[0]

            # extend the joint predictions
            for i in top_result_indices:

                self.pred_joint[joint_from_view_index_test[i]] = words_test[i]
                # pubmed = self.biviewer[joint_from_view_index_test[i]][1]['abstract']
                # cochrane = self.biviewer[joint_from_view_index_test[i]][0]['CHAR_PARTICIPANTS']
                # # print
                # print pubmed
                # print
                # print cochrane
                # print words_test
                # print words_test[i]

        return model


    def test_pubmed(self, model, display_preds=False, default_stats=False):

        # print "train on"
        # print len(self.visible_pubmed_ids)
        # print self.visible_pubmed_ids
        # print "test on"
        # print len(self.test_pubmed_ids)
        # print self.test_pubmed_ids

        assert len(set(self.visible_pubmed_ids) & set(self.test_pubmed_ids)) == 0 # visible and test must not overlap

        

        return self.test(model, self.data["X_pubmed"][self.test_pubmed_ids], self.data["words_pubmed"][self.test_pubmed_ids], self.data["study_id_lookup_pubmed"][self.test_pubmed_ids], display_preds=display_preds, default_stats=default_stats)


    def test(self, model, X_test, words_test, joint_from_view_index, display_preds, default_stats=False):

        
        preds = model.predict_proba(X_test)[:,1] # predict unknowns
        
        metrics = {}

        accuracy_count = 0
        total_abstracts = len(set(joint_from_view_index))
        total_candidates = 0
        per_integer_accuracy_count = 0
        potential_correct_answers = 0


        true_pos_preds = []
        false_pos_preds = []


        # enforce one pop size per abstract

        for id in list(set(joint_from_view_index)):



            max_index = preds[joint_from_view_index==id].argmax()
            
            no_candidates = len(preds[joint_from_view_index==id])
            pred_probs_neg = np.delete(preds[joint_from_view_index==id], max_index)
            std_neg = np.std(pred_probs_neg)
            mean_neg = np.mean(pred_probs_neg)


            prob_pos = preds[joint_from_view_index==id][max_index]



            # print preds[joint_from_view_index==id][~max_index]
            # print np.std(preds[joint_from_view_index==id][~max_index])

            std_from_mean = (prob_pos - mean_neg) / std_neg
            prob_mass = prob_pos / np.sum(preds[joint_from_view_index==id])
            
            expected_ratio = prob_mass / (float(1)/float(no_candidates))




            if default_stats: # default tagger selects the first integer
                y_pred = words_test[joint_from_view_index==id][0]
            else:
                y_pred = words_test[joint_from_view_index==id][max_index]

            # print
            # print self.biviewer[id][1]['abstract']
            # print

            y_actual = self.answers[id]

            # print y_pred, y_actual
            # print type(y_pred), type(y_actual)


            if y_actual != -2:
                potential_correct_answers += 1
                

            if display_preds:
                print "Abstract %d; answer=%d" % (id, y_actual)
                print "Predicted answer %d\n%f SDs from negative examples\n%f percent of probability mass\n%f times expected probability mass" % (y_pred, std_from_mean, prob_mass, expected_ratio)
                result = np.around(np.vstack((preds[joint_from_view_index==id], words_test[joint_from_view_index==id])).T, decimals=2)
                np.set_printoptions(precision=3, suppress=True)
                print result

            total_candidates += no_candidates


            if y_pred == y_actual:
                # print "matched!"
                accuracy_count += 1
                per_integer_accuracy_count += no_candidates
                true_pos_preds.append(prob_pos)

            elif no_candidates > 1:
                false_pos_preds.append(prob_pos)
                per_integer_accuracy_count += (no_candidates-2) # the wrongly guessed and the actual right answer were both mistakes
            elif no_candidates > 0:
                false_pos_preds.append(prob_pos)






            # print y_pred, y_actual


        metrics["accuracy"] = float(accuracy_count)/float(total_abstracts)
        metrics["per_integer_accuracy"] = float(per_integer_accuracy_count)/float(total_candidates)
        metrics["recall"] = float(accuracy_count)/float(potential_correct_answers)
        metrics["precision"] = float(accuracy_count)/float(total_abstracts)
        if (metrics["precision"] + metrics["recall"]) > 0:
            metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1"] = 0
        # metrics["true_pos_preds"] = true_pos_preds
        # metrics["false_pos_preds"] = false_pos_preds

        return metrics





    def features_from_text(self, text):
        "generate and return features for a piece of text"
        p = bilearnPipeline(text, self.window_size)
        p.generate_features()
        X = p.get_features(filter=lambda x: x["num"], flatten=True)
        words = p.get_words(filter=lambda x: x["num"], flatten=True)
        
        return X, words


    def model(self, C=1.0):
        # clf = Pipeline([
        #                 ('feature_selection', RandomizedLogisticRegression()),
        #                 ('classification', SVC(probability=True))
        #                ])
        # clf = SVC(C=C, kernel='linear', probability=True)
        clf = LogisticRegression(C=C, penalty="l1")
        
        
        return clf




    def predict_population_text(self, text, clf):
        X_pubmed_study_vec, words_pubmed_study = self.generate_prediction_features(text)
        return self.predict_population_features(X_pubmed_study_vec, words_pubmed_study, clf)

    def predict_population_features(self, X_pubmed_study_vec, words_pubmed_study, clf):
        # assign predicted probabilities of being a population size to result
        result = clf.predict_proba(X_pubmed_study_vec)[:,1]
        # get index of maximum probability
        amax = np.argmax(result)
        # return (winning number, probability)
        return (words_pubmed_study[amax], result[amax])

    def generate_prediction_features(self, text):
        # load features and get words
        X_study, words_study = self.features_from_text(text)
        # change features to sparse matrix
        if X_study:
            X_study_vec = self.data["vectoriser_pubmed"].transform(X_study)
        else:
            X_study_vec = None
        return X_study_vec, words_study










def main():
    test()


def cross_validate():
    """
    find optimal C and number of iterations using cross validation
    using annotated abstracts
    """

    metrics = []

    annotated_abstracts = MergedTaggedAbstractReader()    

    bilearn = BiLearner(test_mode=False)
    bilearn.load_data('new_features.pck') # regenerate this with current feature set before running
    # bilearn.generate_features() # regenerate this with current feature set before running
    # bilearn.save_data('new_features.pck')
    bilearn.seed_y_regex(annotation_viewer=annotated_abstracts)

    # set up parameters
    C_candidates = np.logspace(-1, 1, 5)
    folds = 4
    fold_size = 0.25
    # fold_size = float(1) / float(folds)

    # C_candidates = np.logspace(-1, 1, 3)
    # (approx [0.1, 0.22, 0.46, 1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0])

    scale_C = True # whether to use constant C or scale
    # scaling factor True = 1/n_samples; False = 1

    n_iter_max = 60 # for distant supervision
    # optimisation_type = "accuracy" # which metric to optimise C on
    optimisation_type = "recall" # which metric to optimise C on

    # split annotated abstracts into: A - test, hidden 20%; B - train, visible 80%
    CV_main = ShuffleSplit(len(annotated_abstracts), n_iter=folds, test_size=fold_size)



    distant_metrics_l = []
    
    for B_indices, A_indices in CV_main:

        print len(B_indices), len(A_indices)
        

        print "DISTANT"
        distant_metrics_l_C_axis = []

        for C_i in C_candidates:
            distant_metrics_l_n_axis = []
            
            bilearn.reset(hide_reader_ids=A_indices,
                          test_reader_ids=A_indices)

            for n_i in range(n_iter_max):
                print
                print "C=%f, iteration %d" % (C_i, n_i)
                print

                if n_i % 2:
                    bilearn.learn_cochrane(C=C_i, update=True)
                elif n_i > 0:
                    # NB first run of loop runs this branch of the if.. else (0 % 2 == 0)
                    bilearn.learn_pubmed(C=C_i, update=True)
                    
                    

                model = bilearn.learn_pubmed(C=C_i, update=False)

                metrics_i = bilearn.test_pubmed(model, default_stats=True)
                
                print "default f1", metrics_i["f1"]
                print "default precision", metrics_i["precision"]
                print "default recall", metrics_i["recall"]
                print "default accuracy", metrics_i["per_integer_accuracy"]



                metrics_i = bilearn.test_pubmed(model)
                distant_metrics_l_n_axis.append(metrics_i[optimisation_type])

                print "f1", metrics_i["f1"]
                print "precision", metrics_i["precision"]
                print "recall", metrics_i["recall"]
                print "accuracy", metrics_i["per_integer_accuracy"]

            distant_metrics_l_C_axis.append(distant_metrics_l_n_axis)


        distant_metrics = np.array(distant_metrics_l_C_axis)


        ####
        ##  iteration 1-n_all (=distant supervised part)
        ##   
        ##  n_all = all data used up
        ##
        ##   iterate through C candidates, training on B2 and testing on B1
        ##   repeat on each CV_tune split
        ##
        ##   score as maximum performance at *any* iteration
        ##   record max performing C, and n
        ##
        ##   record mean highest performing C and n (C_iter_max, n_iter_max)
        ####

        print "FINAL RESULTS"


        print "distant metrics"
        # print "Iteration %d" % (i,)
        print distant_metrics

        C_max_index, n_max =  np.unravel_index(distant_metrics.argmax(), distant_metrics.shape)
        C_max = C_candidates[C_max_index]

        print "Max C=%f, max n=%d" % (C_max, n_max)

        # then test on the tuned parameters
        bilearn.reset(hide_reader_ids=B_indices,
                      test_reader_ids=B_indices)


        test_metrics = []
        for n_i in range(n_max):
            if n_i % 2:
                bilearn.learn_cochrane(C=C_max, update=True)
            elif n_i > 0:
                # NB first run of loop runs this branch of the if.. else (0 % 2 == 0)
                bilearn.learn_pubmed(C=C_max, update=True)
                

            model = bilearn.learn_pubmed(C=C_max, update=False)
            metrics_i = bilearn.test_pubmed(model)

            metrics_i["C_max"] = C_max
            metrics_i["n_max"] = n_max
            metrics_i["n"] = n_i

            print "f1", metrics_i["f1"]
            print "precision", metrics_i["precision"]
            print "recall", metrics_i["recall"]
            print "accuracy", metrics_i["per_integer_accuracy"]

            test_metrics.append(metrics_i)

        distant_metrics_l.append(test_metrics)

    output = {"C_candidates": C_candidates,
              "folds": folds,
              "fold_size": fold_size,
              "n_iter_max": n_iter_max,
              "optimisation_type": optimisation_type,
              "metrics": distant_metrics_l}

    with open('bilearn11output.log', 'wb') as f:
        pprint(output, f)







def test():
    cross_validate()

    # b = BiLearner(test_mode=True, window_size=4)
    # b.generate_features() # test_mode just uses the first 250 cochrane reviews for speed


            

if __name__ == '__main__':
    main()

print "2"