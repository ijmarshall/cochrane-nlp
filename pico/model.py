import sys, os, logging, csv, collections, functools
import cPickle as pickle
import os.path
logging.basicConfig(level=logging.DEBUG)

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
  * Get the PICO text and compute cosine similarity with the LSH
  * If similarity > threshold set y = 1, 0 otherwise

'''

DATA_PATH = cochranenlp.config["Paths"]["base_path"]

# PICO_DOMAINS = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

viewer = biviewer.PDFBiViewer()

sentence_tokenizer = PunktSentenceTokenizer()

domain = sys.argv[1]


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def persist(file_name):
    file_name_with_extension = file_name + ".pck"
    def func_decorator(func):
        def func_wrapper(*args, **kwargs):
            if os.path.isfile(file_name_with_extension):
                with open(file_name_with_extension, 'rb') as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with open(file_name_with_extension, 'wb') as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                    f.close()
                    return result
        return func_wrapper
    return func_decorator


@persist(DATA_PATH + "sentences")
def get_sentences():
    logging.info("getting sentences")
    pmids = set()
    sentences = []
    for study in viewer:
        pmid = study[1]['pmid']
        if pmid not in pmids:  # Not Cached
            logging.debug("parsing sentences for %s" % pmid)
            text = study.studypdf["text"].decode("utf-8", errors="ignore")
            sentences.append(sentence_tokenizer.tokenize(text))
            pmids.add(pmid)
        else:
            logging.debug("skipping, already parsed %s" % pmid)
    return [{"pmid": k, "sentence": v} for k, t in zip(pmids, sentences) for v in t]


vectorizer = HashingVectorizer(stop_words=stopwords.words('english'),
                               norm="l2", ngram_range=(3, 3),
                               analyzer="char_wb",
                               decode_error="ignore",
                               strip_accents="ascii")

def vectorize(sentences):
    return vectorizer.transform(sentences)

def get_X(sentences, held_out):
    logging.debug("vectorizing sentences")
    return vectorize([s["sentence"] for s in sentences if not s["pmid"] in held_out])

@memoized
def get_characteristic_fragment(pmid, domain):
    logging.debug("getting %s fragment for %s" % (domain, pmid))
    studies = viewer.get_study_from_pmid(pmid)
    char = [s[0]["CHARACTERISTICS"][domain] or "" for s in studies]
    return " ".join(char).decode("utf-8", errors="ignore")


@persist(DATA_PATH + "fragments_" + domain)
def get_characteristic_fragments(sentences, domain, held_out):
    logging.info("getting CDSR fragments")
    return [get_characteristic_fragment(s['pmid'], domain) or "" for s in sentences if not s['pmid'] in held_out]


def get_characteristic_fragment_vector(fragments):
    logging.info("vectorizing CDSR fragments")
    return vectorize(fragments)


@persist(DATA_PATH + "R_" + domain)
def get_R(X, y):
    assert X.shape == y.shape
    logging.info("computing similarity ...")
    R = np.zeros(y.shape[0], 'float')
    for idx in range(len(R)):  # we're using a loop here to save memory
        R[idx] = (y[idx,:] * X[idx,:].T)[0,0]
    return R


def get_y(R, threshold):
    return (R >= threshold)

def get_test_data(file_name, domain):
    out = []
    test_domain = domain.replace("CHAR_", "")
    with open(file_name) as f:
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
    y_true = np.array([True if t['rating'] in set(['1', '2', 't1']) else False for t in test_data])

    def scorer(estimator, X, y):
        logging.info("Estimating %s %s" % (len(y_true), sum(y_true)))
        y_pred = estimator.predict(X_test)
        logging.info("Predicted %s %s" % (len(y_pred), sum(y_pred)))
        return precision_recall_fscore_support(y_true, y_pred, average="micro")

    return scorer


def run_experiments(X, R, scorer):
    logging.debug("running experiment for %s" % domain)
    tune_params = ParameterGrid([
        {"alpha": [.00001, .001, 1, 10],
         "threshold": [0.01, 0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]}])

    best_estimator = None
    best_score = 0

    for params in tune_params:
        logging.info("running %s with alpha=%s, threshold=%s" % (domain, params["alpha"], params["threshold"]))
        logging.info("getting y...")
        y = get_y(R, threshold=params["threshold"])
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


def start(domain, is_cached):
    logging.info("setting up (cached: %s)" % is_cached)
    ratings = DATA_PATH + "../sds/annotations/master/figure8-2-15.csv"
    test, held_out = get_test_data(ratings, domain)
    scorer = scorer_factory(test)

    if not is_cached:
        sentences = get_sentences()

        X = get_X(sentences, held_out)

        fragments = get_characteristic_fragments(sentences, domain, held_out)

        y = get_characteristic_fragment_vector(fragments)
        R = get_R(X, y)
    else:
        X = get_X()  # Cached
        R = get_R()  # Cached

    logging.info("starting experiments")
    run_experiments(X, R, scorer)

if __name__ == '__main__':
    domain = sys.argv[1]
    is_cached = sys.argv[2] == "1"
    logging.info(domain)
    start(domain, is_cached)
