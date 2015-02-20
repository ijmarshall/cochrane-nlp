import sys, os, logging, csv
import cPickle as pickle
import os.path
logging.basicConfig(level=logging.INFO)

reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import scipy as sp
import sklearn


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, precision_recall_fscore_support
from sklearn.grid_search import ParameterGrid

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords

sys.path.insert(0, os.getcwd())
import cochranenlp
from cochranenlp.readers import biviewer

'''
get the text
- tokenize the text into sentences
- vectorize the sentences using HashingVectorizer
TODO: add more features

Per PICO
- Iterate over all CDSR studies
  * Get the PICO text and compute cosine distance with the LSH
  * If distance > threshold set y = 1, 0 otherwise

Predict [X,y] with 5-fold crossvalidation per PICO using SGD

'''

DATA_PATH = cochranenlp.config["Paths"]["base_path"]

# PICO_DOMAINS = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

viewer = biviewer.PDFBiViewer()

sentence_tokenizer = PunktSentenceTokenizer()

def get_sentences(held_out):
    pmids = set()
    sentences = []
    for study in viewer:
        pmid = study[1]['pmid']
        if pmid in held_out:
            continue
        if pmid not in pmids:  # Not Cached
            logging.debug("parsing sentences for %s" % pmid)
            text = study.studypdf["text"].decode("utf-8", errors="ignore")
            sentences.append(sentence_tokenizer.tokenize(text))
            pmids.add(pmid)
        else:
            logging.debug("skipping, already parsed %s" % pmid)
    return [{"pmid": k, "sentence": v} for k, t in zip(pmids, sentences) for v in t]



vectorizer = HashingVectorizer(stop_words=stopwords.words('english'), norm="l2", ngram_range=(5, 5), analyzer="char_wb", decode_error="ignore")

def vectorize(sentences):
    return vectorizer.transform(sentences)

def get_X(sentences):
    return vectorize([x["sentence"] for x in sentences])


def get_characteristic_fragments(pmid, domain):
    studies = viewer.get_study_from_pmid(pmid)
    char = [s[0]["CHARACTERISTICS"][domain] or "" for s in studies]
    return " ".join(char).decode("utf-8", errors="ignore")


def __get_similarity(domain, sentences, pmid):
    s2 = sentence_tokenizer.tokenize(get_characteristic_fragments(pmid, domain))
    y1 = vectorize(s2) if s2 else vectorize([""])

    # determine the cosine similarity of the sentences, and marking as relevant if exceeding threshold
    y2 = vectorize(sentences)
    return (y1 * y2.T)


def get_y(domain, sentences, threshold):
    y = np.zeros(len(sentences), 'bool')

    pmid_ptr = None
    tmp = []
    idx_ptr = 0
    idxs = []
    for idx, s in enumerate(sentences):
        if not pmid_ptr:
            pmid_ptr = s['pmid']
        elif pmid_ptr != s['pmid']:
            # next pmid
            logging.debug("distilling essence of %s for %s at %s" % (pmid_ptr, domain, threshold))
            R = __get_similarity(domain, tmp, pmid_ptr)
            y[idxs] = sum(np.any(R > threshold)).A[0, :]

            tmp = []
            idxs = []
            pmid_ptr = s['pmid']
            idx_ptr = idx

        idxs.append(idx)
        tmp.append(s['sentence'])

    return y


def get_test_data(file, domain):
    out = []
    test_domain = domain.replace("CHAR_", "")
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['PICO field'].strip() == test_domain:
                candidate = row['candidate sentence'].decode("utf-8", errors="ignore")
                sentences = sentence_tokenizer.tokenize(candidate)
                for s in sentences:
                    out.append({"sentence": s, "rating": row['rating'], 'pmid': row['study id']})

    held_out = set([t['pmid'] for t in out])
    return out, held_out


def scorer_factory(test_data):
    X_test = vectorize([t['sentence'] for t in test_data])
    y_true = np.array([1 if t['rating'] in set(['1', '2', 't1']) else 0 for t in test_data])

    def scorer(estimator, X, y):
        logging.info("Estimating %s %s" % (len(y_true), sum(y_true)))
        y_pred = estimator.predict(X_test)
        logging.info("Predicted %s %s" % (len(y_pred), sum(y_pred)))
        return precision_recall_fscore_support(y_true, y_pred, average="micro")

    return scorer


def run_experiment(X, domain, sentences, scorer):
    logging.debug("running experiment for %s" % domain)
    tune_params = ParameterGrid([
        {"alpha": [.00001, .001, 1, 10],
         "threshold": [0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5]}])

    best_estimator = None
    best_score = 0

    for params in tune_params:
        logging.info("running %s with alpha=%s, threshold=%s" % (domain, params["alpha"], params["threshold"]))
        logging.info("getting y...")
        y = get_y(domain, sentences, threshold=params["threshold"])
        logging.info("Number of samples %s, of which positive %s" % (len(y), sum(y)))
        sgd = SGDClassifier(shuffle=True, loss="hinge", penalty="l2", alpha=params["alpha"])
        logging.info("fitting...")
        sgd.fit(X, y)
        precision, recall, f1, support = scorer(sgd, None, None)
        logging.info("precision %s, recall %s, f1 %s" % (precision, recall, f1))
        if(precision >= best_score):
            logging.info("this estimator was better!")
            best_estimator = sgd
            best_score = precision
            logging.debug("storing %s" % domain)
            with open(DATA_PATH + domain + ".pck", "wb") as f:
                pickle.dump(best_estimator, f)


def run_experiments(domain):
    logging.info("setting up")
    ratings = DATA_PATH + "../sds/annotations/master/figure8-2-15.csv"
    test, held_out = get_test_data(ratings, domain)
    scorer = scorer_factory(test)
    sentences = get_sentences(held_out)
    train_X = get_X(sentences)

    logging.info("starting experiments")
    run_experiment(train_X, domain, sentences, scorer)

if __name__ == '__main__':
    domain = sys.argv[1]
    logging.info(domain)
    run_experiments(domain)
