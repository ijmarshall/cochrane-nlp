#
#   Pipeline 5
#


"""

V5 of Pipeline

changes:
    - Pipeline is now only used for chaining together functions
    - New functions are added by subclassing Pipeline
    - Baseline function is only "w" for word

"""

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from progressbar import ProgressBar

from collections import defaultdict
from itertools import izip
from nltk import PorterStemmer

from indexnumbers import swap_num

import cPickle as pickle


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
        pass

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

    # def get_features(self, filter=None):
    #     if filter:
    #         return [item for sublist in self.X for item in sublist if filter(item)]
    #     else:
    #         return self.X


    def get_features(self, filter=None, flatten=False):

        if filter:
            output = []
            for sent_X in self.X:
                output.extend([word for word in sent_X if filter(word)])
            return output
        else:
            if flatten:
                return [item for sublist in self.X for item in sublist]
            else:
                return self.X

    def get_words(self, filter=None, flatten=False):
        if filter:
            output = []
            for sent in self.functions:
                output.extend([word["w"] for word in sent if filter(word)])
            return output
        else:
            words = [[word["w"] for word in sent] for sent in self.functions]
            if flatten:
                return [item for sublist in words for item in sublist]
            else:
                return words

    def get_text(self):
        return self.text


    def get_answers(self, answer_key=None, filter=None, flatten=False):
        if not answer_key:
            answer_key = self.answer_key
        if filter:
            return [item[answer_key] for sublist in self.functions for item in sublist if filter(item)]
        else:
            answers = [[word[answer_key] for word in sent] for sent in self.functions]
            if flatten:
                return [item for sublist in answers for item in sublist]
            else:
                return answers

    def get_crfsuite_features(self, flatten=False):

        features =  [[["%s=%s" % (key, value) for key, value in word.iteritems()] for word in sent] for sent in self.X]
        if flatten:
            return [item for sublist in features for item in sublist]
        else:
            return features

        





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


    p2 = bilearnPipeline("No numbers in this sentence! Or this one either.")
    p2.generate_features()
    print p2.get_crfsuite_features(flatten=False)
    # print p2.get_features(filter = lambda x: x["w[0]"].isdigit())


if __name__ == '__main__':
    main()




