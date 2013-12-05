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
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from tokenizer import tag_words
from journalreaders import LabeledAbstractReader

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import RandomizedLogisticRegression


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

    def __init__(self, text):
        self.functions = [[{"w": word, "p": pos} for word, pos in pos_tagger.tag(self.word_tokenize(sent))] for sent in self.sent_tokenize(swap_num(text))]
        self.load_templates()        
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
                          (('in_num_list', 0), ),
                          )

        self.answer_key = "w"
        self.w_pos_window = 6 # set 0 for no w_pos window features
 
    def run_functions(self, show_progress=False):

        # make dict to look up ranking of number in abstract
        num_list_nest = [[int(word["w"]) for word in sent if word["w"].isdigit()] for sent in self.functions]
        num_list = [item for sublist in num_list_nest for item in sublist] # flatten
        num_list.sort(reverse=True)
        num_dict = {num: rank for rank, num in enumerate(num_list)}

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
                            "wl": word.lower()}
                if word.isdigit():
                    num = int(word)
                    features[">10"] = num > 10
                    features["w_int"] = num
                    features["div10"] = ((num % 10) == 0)
                    features["numrank"] = num_dict[num]
                
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



class BiLearner():

    def __init__(self, test_mode=True):

        logging.info("Initialising Bilearner")
        self.data = {}
        # load cochrane and pubmed parallel text viewer
        self.biviewer = biviewer.BiViewer(in_memory=False, test_mode=test_mode)

        
        # self.stem = PorterStemmer()

    def reset(self, seed="regex"):
        " sets the y_lookup back to the original regex rule results "

        
        self.data["y_lookup_init"] = {i: -1 for i in range(len(self.biviewer))}

        if seed == "annotations":
            self.init_y_annotations() # generate y vector from regex
        else:
            self.init_y_regex() # generate y vector from regex

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
            cochrane_text = cochrane_dict.get("CHAR_PARTICIPANTS", "") + cochrane_dict.get("CHAR_INTERVENTIONS", "") + cochrane_dict.get("CHAR_OUTCOMES", "") + cochrane_dict.get("CHAR_NOTES", "")
            pubmed_text = pubmed_dict.get("abstract", "")


            
            # generate features for all studies
            # the generate_features functions only generate features for integers
            # words are stored separately since they are not necessarily used as features
            #     but are needed for the answers




            # first generate features from the parallel texts

            p_cochrane = bilearnPipeline(cochrane_text)
            p_cochrane.generate_features()
            


            words_cochrane_study = p_cochrane.get_words(filter=lambda x: x["num"], flatten=True) # get the integers



            p_pubmed = bilearnPipeline(pubmed_text)
            p_pubmed.generate_features()
            
            words_pubmed_study = p_pubmed.get_words(filter=lambda x: x["num"], flatten=True) # get the integers


            # generate filter vectors for integers which match between Cochrane + Pubmed
            common_ints = list(set(words_cochrane_study) &  set(words_pubmed_study))



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

        # self.reset()

    def init_y_regex(self):
        """
        initialises the joint y vector with data from a simple seed regex rule
        """
        logging.info("Identifying seed data from regular expression")
        p = progressbar.ProgressBar(len(self.biviewer), timer=True)
        counter = 0  # number of studies initially found
        for study_id, (cochrane_dict, pubmed_dict) in enumerate(self.biviewer):
            p.tap()
            pubmed_text = pubmed_dict.get("abstract", "")
            # use simple rule to identify population sizes (low sens/recall, high spec/precision)
            pubmed_text = swap_num(pubmed_text)
            matches = re.findall('([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients) were (?:randomi[sz]ed)', pubmed_text)
            # matches += re.findall('(?:[Ww]e randomi[sz]ed )([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients)', pubmed_text)
            # matches += re.findall('(?:[Aa] total of )([1-9][0-9]*) (?:\w+ )*(?:participants|men|women|patients)', pubmed_text)
            if len(matches) == 1:
                self.data["y_lookup_init"][study_id] = int(matches[0])
                counter += 1

        self.seed_abstracts = counter
        logging.info("%d seed abstracts found", counter)

    def init_y_annotations(self):
        """
        initialises the joint y vector with data from manually annotated abstracts
        """
        logging.info("Identifying seed data from annotated data")
        p = progressbar.ProgressBar(len(self.biviewer), timer=True)
        annotation_viewer = LabeledAbstractReader()

        counter = 0
        for study in annotation_viewer:
            study_id = int(study["Biview_id"])
            text = swap_num(annotation_viewer.get_biview_id(study_id)['abstract'])                

            parsed_tags = tag_words(text, flatten=True)
            tagged_number = [w[0] for w in parsed_tags if 'n' in w[1]]
            if tagged_number:
                number = re.match("[Nn]?=?([1-9]+[0-9]*)", tagged_number[0])

                if number:
                    self.data["y_lookup_init"][study_id] = int(number.group(1))
                    counter += 1
                else:
                    raise TypeError('Unable to convert tagged number %s to integer', tagged_number[0])

        self.seed_abstracts = counter
        logging.info("%d seed abstracts found", counter)


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
                                C=1, aperture=aperture, aperture_type=aperture_type, update_joint=True)

                # print show_most_informative_features(self.data["vectoriser_pubmed"], cochrane_model_f)


                pubmed_model = self.learn_view(self.data["X_pubmed"], self.data["words_pubmed"], self.data["study_id_lookup_pubmed"],
                                C=C, aperture=aperture, aperture_type=aperture_type, update_joint=True)

                


            else:


                pubmed_model = self.learn_view(self.data["X_pubmed"], self.data["words_pubmed"], self.data["study_id_lookup_pubmed"],
                            C=C, aperture=aperture, aperture_type=aperture_type, update_joint=False)
            
            
            
            


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


    def learn_view(self, X_view, words_view, joint_from_view_index,
                    C=2.5, aperture=0.90, aperture_type='probability', update_joint=True):


        pred_view = np.empty(shape=(len(words_view),)) # make a new empty vector for predicted values
        # (pred_view is predicted population sizes; not true/false)

        # create answer vectors with the seed answers
        for word_id in xrange(len(pred_view)):
            pred_view[word_id] = self.pred_joint[joint_from_view_index[word_id]]

        y_view = (pred_view == words_view) * 2 - 1 # set Trues to 1 and Falses to -1

        # set filter vectors (-1 = unknown)
        filter_train = (pred_view != -1).nonzero()[0]
        filter_test = (pred_view == -1).nonzero()[0]



        # self.metrics["cochrane_training_examples"].append(len(filter_train))
        # self.metrics["cochrane_test_examples"].append(len(filter_test))


        if len(filter_test)==0:
            print "leaving early - run out of data!"
            raise IndexError("out of data")


        # set training vectors
        X_train = X_view[filter_train]
        y_train = y_view[filter_train]

        # and test vectors as the rest
        X_test = X_view[filter_test]
        y_test = y_view[filter_test]

        # and the numbers to go with it for illustration purposes
        words_test = words_view[filter_test]
        joint_from_view_index_test = joint_from_view_index[filter_test]

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

        return model





    def features_from_text(self, text):
        "generate and return features for a piece of text"
        p = bilearnPipeline(text)
        p.generate_features()
        X = p.get_features(filter=lambda x: x["num"], flatten=True)
        words = p.get_words(filter=lambda x: x["num"], flatten=True)
        
        return X, words


    def model(self, C=1.0):
        # clf = Pipeline([
        #                 ('feature_selection', RandomizedLogisticRegression()),
        #                 ('classification', SVC(probability=True))
        #                ])
        clf = RandomForestClassifier()
        #SVC(C=C, kernel='linear')
        # clf = LogisticRegression(C=C, penalty="l1")
        
        
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










def test():

    b = BiLearner(test_mode=True)
    b.generate_features() # test_mode just uses the first 250 cochrane reviews for speed

    b.reset(seed='regex')
    b.learn(iterations=30, C=3.5, aperture=10, aperture_type="absolute")
    pprint(b.metrics)
    

if __name__ == '__main__':
    main()

