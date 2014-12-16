'''
Supervised distant supervision for PICO extraction.

Here we aim to go from the information (direct 
distant supervision for PICO task) contained in 
the annotations file to feature vectors and labels for the 
candidate filtering task. In the writeup nomenclature,
this is to generate \tilde{x} and \tilde{y}.
'''

import pdb
import random
import csv

import numpy as np 
import scipy as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from readers import biviewer

# this module allows us to grab the ranked
# sentences. this is possibly not the ideal 
# location.
from experiments import pico_DS 


def run_experiment(iters=10):
    # X and y for supervised distant supervision
    DS_learning_tasks = get_DS_features_and_labels()
    for domain, task in DS_learning_tasks.items():
        # note that 'task' here is comprises
        # ('raw') extracted features and labels
        X_d, y_d = generate_X_y(task)
        pmids_d = task["pmid"]

        #pdb.set_trace()
        # for experimentation, uncomment below...
        # in general these would be fed into 
        # the build_clf function, below
        #return X_d, y_d, task["pmid"]
        for iter_ in xrange(iters):   
            train_X, train_y, test_X, test_y = train_test_by_pmid(X_d, y_d, pmids_d)
            model = build_clf(train_X, train_y)
            model.fit(train_X, train_y)

            # To evaluate performance with respect
            # to the *true* (or primary) task, you 
            # should do the following
            #   (1) Train the DS model (M_DS) using all train_X, train_y
            #       here. 
            #   (2) Generate as many DS labels as possible
            #       for PICO. For SDS, use M_DS to either 
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
    randomly sample 80% of the pmids as training instances; 
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
    X1 = []
    
    for X_i, y_i in zip(DS_learning_task["X"], DS_learning_task["y"]):
        X_i_numeric, X_i_text = X_i
        X_v = vectorizer.transform([X_i_text])[0]
        #X_i_numeric = np.matrix(X_i_numeric)
        ##pdb.set_trace()

        X1.append(X_v)
        X_combined = sp.sparse.hstack((X_v, X_i_numeric))
        #pdb.set_trace()
        X.append(np.asarray(X_combined.todense())[0])
        y.append(y_lbl_func(y_i))
        
    return X, y

# "sds/annotations/for_labeling_sharma.csv"
def get_DS_features_and_labels(candidates_path="sds/annotations/for_labeling_sharma.csv",
                                labels_path="sds/annotations/sharma-merged-labels.csv",
                                label_index=-1,
                                max_sentences=10, cutoff=4, normalize_numeric_cols=True):
    '''
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

    domains = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]
    # this is just to standardize terms/strings
    pico_strs_to_domains = dict(zip(["PARTICIPANTS", "INTERVENTIONS","OUTCOMES"], domains))

    X_y_dict = {}
    for d in domains:
        # X, y and pmids for each domain. the latter
        # is so we can know which studies each candidate
        # was generated for.
        X_y_dict[d] = {"X":[], "y":[], "pmid":[]}


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
            num_to_keep = min(len([score for score in scores if score >= cutoff]), max_sentences)


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
            X_y_dict[PICO_field]["pmid"].append(study_id)

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


''' completely ripped off from Alex Martelli '''
def unicode_csv_reader(utf8_data, **kwargs):
    csv_reader = csv.reader(utf8_data, **kwargs)
    for row in csv_reader:
        try:
            yield [unicode(cell, 'utf-8') for cell in row]
        except:
            pdb.set_trace()

            