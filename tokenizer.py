import re

from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.metrics.agreement import AnnotationTask
from itertools import chain
from functools import wraps
from scipy.stats import describe

from indexnumbers import swap_num

import configparser # easy_install configparser
config = configparser.ConfigParser()
config.read('CNLP.INI')
base_path = config["Paths"]["base_path"]


def memo(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap


@memo
def get_abstracts(annotator):
    """
    Take the annotators abstract files
    """
    annotated_abstracts_file = base_path + annotator + ".txt"
    with open (annotated_abstracts_file, "r") as file:
        data=file.read()

    def clean(abstract):
        return (re.split("BiviewID [0-9]*; PMID ?[0-9]*", abstract)[0]).strip()

    return [clean(abstract) for abstract in re.split('Abstract \d+ of \d+', data)][1:]

# Tokenize an abstract
open_tag = '<[a-z0-9_]+>'
close_tag = '<\/[a-z0-9_]+>'
tag_def = "(" + open_tag + "|" + close_tag + ")" # more convinient than '<\/?[a-z0-9_]+>'

def tokenize_abstract(abstract, tag_def, convert_numbers=False):
    """
    Takes an abstact (string) and converts it to a list of words or tokens
    For example "A <tx>treatment</tx>, of" -> ['A', '<tx>', 'treatment', '</tx>', ',' 'of']
    This uses regexes and not a proper (context-free) DOM parser, so beware.
    """
    if convert_numbers:
        abstract = swap_num(abstract)
    tokens_by_tag = re.split(tag_def, abstract)
    def tokenize(token):
        if not re.match(tag_def, token):
            return word_tokenize(token)
        else:
            return [token]
    return list(chain.from_iterable([tokenize(token) for token in tokens_by_tag])) # flatten

def annotations(tokens):
    """
    Process tokens into a list with {word -> [tokens]} items
    The value is a list, since tokens can be annotated several times
    """
    mapping = []
    curr = []
    for token in tokens:
        if re.match(open_tag, token):
            curr.append(re.match('<([a-z0-9_]+)>',token).group(1))
        elif re.match(close_tag, token):
            tag = re.match('<\/([a-z0-9_]+)>',token).group(1)
            try:
                curr.remove(tag)
            except ValueError:
                pass
        else:
            mapping.append({token: list(curr)})
    return mapping


def combine_annotations(annotations_A, annotations_B):
    """
    Build a list of [annotator, word, annotation] for two annotators
    """
    a = [['A', idx, "&".join(x.values()[0])] for idx, x in enumerate(annotations_A)]
    b = [['B', idx, "&".join(x.values()[0])] for idx, x in enumerate(annotations_B)]
    return a + b

def get_annotations(abstract_nr, annotator, convert_numbers=False):
    '''
    if convert_numbers is True, numerical strings (e.g., "twenty-five")
    will be converted to number ("25").
    '''
    return annotations(tokenize_abstract(get_abstracts(annotator)[abstract_nr], tag_def, convert_numbers=convert_numbers))

def agreement(abstract_nr):
    """
    Figure out who annotator A and B should be in a round-robin fashion
    Returns the combined annotations for the abstract_nr
    """
    annotators = ["IJM", "BCW", "JKU"]
    annotator_A = annotators[abstract_nr % len(annotators)]
    annotator_B = annotators[(abstract_nr + 1) % len(annotators)]
    annotations_A = get_annotations(abstract_nr, annotator_A)
    annotations_B = get_annotations(abstract_nr, annotator_B)
    return { "annotations" : combine_annotations(annotations_A, annotations_B),
             "annotator_A" : annotator_A,
             "annotator_B" : annotator_B }

def eliminate_order(tag):
    return re.sub('([0-9])((?:_a)?)', 'X\g<2>', tag)

def agreement_fn(a,b):
    # ignore the treatment number, tx1 is the same as tx2 (and tx1_a to tx2_a)
    def get_tag_set(string):
        return set([eliminate_order(x) for x in string.split('&')] if string else [])
    a = get_tag_set(a)
    b = get_tag_set(b)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    else:
        # linearly scale (all agree = 0) (none agree = 1)
        return len(a.difference(b)) * (1 / float(max(len(a), len(b))))

def calc_agreements(nr_of_abstracts=100):
    # Loop over the abstracts and caluclate the kappa and alpha per abstract
    aggregate = []
    for i in range(0, nr_of_abstracts):
        _agreement = agreement(i)
        try:
            a = AnnotationTask(_agreement['annotations'], agreement_fn)
            aggregate.append({
                "kappa" : a.kappa(),
                "alpha" : a.alpha(),
                "annotator_A" : _agreement['annotator_A'],
                "annotator_B" : _agreement['annotator_B'] })
        except:
            print("Could not calculate kappa for abstract %i between %s and %s" % (i + 1, _agreement['annotator_A'], _agreement['annotator_B']))
            pass


    # Summary statistics
    kappa = describe([a['kappa'] for a in aggregate])
    print("number of abstracts %i" % kappa[0])
    print("[kappa] mean: " + str(kappa[2]))
    print("[kappa] variance: " + str(kappa[3]))
    alpha = describe([a['alpha'] for a in aggregate])
    print("[alpha] mean: " + str(alpha[2]))
    print("[alpha] variance: " + str(alpha[3]))

def merge_annotations(a, b, strategy = lambda a,b: a & b, preprocess = lambda x: x):
    """"
    Returns the merging of a and b
    based on strategy (defaults to set intersection) Optionally takes
    a preprocess argument which takes a tag as argument and must
    return the processed tag

    example usage:
    JKU = get_abstracts("JKU")
    BCW = get_abstracts("BCW")

    JKU1 = annotations(tokenize_abstract(JKU[1], tag_def))
    BCW1 = annotations(tokenize_abstract(BCW[1], tag_def))

    print(merge_annotations(JKU1, BCW1, preprocess = eliminate_order))
    """
    assert(len(a) == len(b))
    result = []
    for i in range(0, len(a)):
        key = a[i].keys()[0]
        first = set([preprocess(x) for x in a[i].values()[0]])
        second = set([preprocess(x) for x in b[i].values()[0]])
        result.append({key : list(strategy(first, second))})
    return result


if __name__ == "__main__":
    calc_agreements()
