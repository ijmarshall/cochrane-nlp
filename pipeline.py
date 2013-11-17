#
#   Pipeline 6
#



import cPickle as pickle
from collections import defaultdict
from functools import wraps
from itertools import izip

from indexnumbers import swap_num
from nltk import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from progressbar import ProgressBar



def filters(func):
    """
    used as decorator
    allows pipeline functions to return helpful
    views/permetations of output data - flattened lists
    and filters based on the base (hidden) features
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        
        flatten = kwargs.pop("flatten", False)
        filter = kwargs.pop("filter", None)

        raw_output = func(self, *args, **kwargs)
        if filter:
            filtered_output = [[raw_word for raw_word, base_word in izip(raw_sent, base_sent) if filter(base_word)]
                                for raw_sent, base_sent in izip(raw_output, self.functions)]
        else:
            filtered_output = raw_output

        if flatten:
            return [item for sublist in filtered_output for item in sublist]
        else:
            return filtered_output
    return wrapper



class Pipeline(object):

    def __init__(self, text):
        self.text = text
        self.functions = [[{"w": word} for word in self.word_tokenize(sent)] for sent in self.sent_tokenize(swap_num(text))]
        self.load_templates()


    def load_templates(self):
        " used in subclasses to load template tuples "
        pass

    def sent_tokenize(self, text):
        return sent_tokenize(text)

    def word_tokenize(self, sent):
        return word_tokenize(sent)

    def generate_features(self, templates=None, show_progress=False):
        if not templates:
            templates = self.templates

        # 1. run base functions
        self.run_functions(show_progress=show_progress)
        # 2. apply templates
        self.X = self.apply_templates(templates, show_progress=show_progress)

    def run_functions(self, show_progress=False):
        " used in subclasses to chain together feature functions "
        raise NotImplemented

    def apply_templates(self, templates=None, show_progress=False):
        """
        based on crfutils
        """
        if not templates:
            templates = self.templates
        X = [[{} for word in sent] for sent in self.functions]
        if show_progress:
            pb = ProgressBar(len(templates) * len(X), timer=True)
        for template in templates:
            name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
            for sent_index, X_sent in enumerate(X):
                if show_progress:
                    pb.tap()
                sent_len = len(X_sent)
                for word_index, X_word in enumerate(X_sent):
                    values = []
                    for field, offset in template:
                        p = word_index + offset
                        if p < 0 or p > (sent_len - 1):
                            values = []
                            break
                            values.append("_OUT_OF_RANGE_")
                        else:
                            values.append(self.functions[sent_index][p][field])
                    if values:    
                        X[sent_index][word_index][name] = '|'.join([str(value) for value in values])
        return X


    def get_text(self):
        return self.text

    @filters
    def get_words(self):
        return [[word["w"] for word in sent] for sent in self.functions]

    @filters
    def get_base_functions(self):
        return self.functions

    @filters
    def get_answers(self, answer_key=lambda x: True):
        """
        returns y vectors for each sentence, where the answer_key
        is a lambda function which derives the answer from the 
        base (hidden) features
        """
        return [[answer_key(word) for word in sent] for sent in self.functions]

    @filters
    def get_features(self, filter=None, flatten=False):
        return self.X

    @filters
    def get_crfsuite_features(self):
        return [[["%s=%s" % (key, value) for key, value in word.iteritems()] for word in sent] for sent in self.X]
        





def main():
    from biviewer import BiViewer
    from bilearn import bilearnPipeline

    
    b = BiViewer()


    p = bilearnPipeline(b[1][1]['abstract'])

    # p.generate_features()
    # print p.get_features(filter=lambda x: x["w[0]"].isdigit())
    # # print p.get_answers(filter=lambda x: x["w"].isdigit())

    # p2 = bilearnPipeline("")
    # p2.generate_features()
    # # print p2.get_answers()
    # print p2.get_features(filter = lambda x: x["w"].isdigit())    


    p2 = bilearnPipeline("No numbers in this sentence! Or this one which has a number of 123 either.")
    p2.generate_features()
    
    # print p2.get_crfsuite_features(flatten=True, filter=lambda x: x["sym"]==False)
    print p2.get_features(flatten=True, filter=lambda x: x["num"])
    print p2.get_answers(flatten=False, answer_key=lambda x: x["num"], filter=lambda x: not x["sym"])

    
    # print p2.get_features(filter = lambda x: x["w[0]"].isdigit())


if __name__ == '__main__':
    main()




