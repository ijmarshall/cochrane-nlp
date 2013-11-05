import re

from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.metrics.agreement import AnnotationTask
from itertools import chain
from functools import wraps
from scipy.stats import describe

def memo(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

base_path = "/Users/joelkuiper/Desktop/drug_trials_in_cochrane_"

@memo
def get_abstracts(annotator):
    """
    Take the annotators abstract files
    """
    annotated_abstracts_file = base_path + annotator + ".txt"
    with open (annotated_abstracts_file, "r") as file:
        data=file.read()

    return [abstract.strip() for abstract in re.split('Abstract \d+ of \d+', data)][1:]

# Tokenize an abstract
open_tag = '<[a-z0-9_]+>'
close_tag = '<\/[a-z0-9_]+>'
tag_def = "(" + open_tag + "|" + close_tag + ")" # more convinient than '<\/?[a-z0-9_]+>'

def tokenize_abstract(abstract, tag_def):
    """
    Takes an abstact (string) and converts it to a list of words or tokens
    For example "A <tx>treatment</tx>, of" -> ['A', '<tx>', 'treatment', '</tx>', ',' 'of']
    This uses regexes and not a proper (context-free) DOM parser, so beware.
    """
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
    stack = []
    for token in tokens:
        if re.match(open_tag, token):
            tag = re.match('<([a-z0-9_]+)>',token).group(1)
            # ignore the treatment number, tx1 is the same as tx2 (and tx1_a to tx2_a)
            stack.append(re.sub('([0-9])((?:_a)?)', '_x\g<2>', tag))
        elif re.match(close_tag, token):
            stack.pop()
        else:
            mapping.append({token: list(stack)})
    return mapping


def combine_annotations(annotations_A, annotations_B):
    """
    Build a list of [annotator, word, annotation] for two annotators
    """
    a = [['A', idx, "&".join(x.values()[0])] for idx, x in enumerate(annotations_A)]
    b = [['B', idx, "&".join(x.values()[0])] for idx, x in enumerate(annotations_B)]
    return a + b

def get_annotations(abstract_nr, annotator):
    return annotations(tokenize_abstract(get_abstracts(annotator)[abstract_nr], tag_def))

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

def agreement_fn(a,b):
    a = set(a.split('&') if a else [])
    b = set(b.split('&') if b else [])
    if len(a) == 0 and len(b) == 0:
        return 0.0
    else:
        return len(a.difference(b)) * (1 / float(max(len(a), len(b))))


# Loop over the abstracts and caluclate the kappa and alpha per abstract
aggregate = []
nr_of_abstracts = 25
for i in range(0, nr_of_abstracts):
    _agreement = agreement(i)
    a = AnnotationTask(_agreement['annotations'], agreement_fn)
    aggregate.append({
        "kappa" : a.kappa(),
        "alpha" : a.alpha(),
        "annotator_A" : _agreement['annotator_A'],
        "annotator_B" : _agreement['annotator_B'] })

# Summary statistics
kappa = describe([a['kappa'] for a in aggregate])
print("[kappa] mean: " + str(kappa[2]))
print("[kappa] variance: " + str(kappa[3]))
alpha = describe([a['alpha'] for a in aggregate])
print("[alpha] mean: " + str(alpha[2]))
print("[alpha] variance: " + str(alpha[3]))
