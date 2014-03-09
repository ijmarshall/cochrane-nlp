#
#   pipeline.py
#
#   N.B. predictive modelling requires a trained model in pickle form:
#   - get `quality_models.pck` from `Dropbox/cochranetech/quality-prediction-results/models`
#   - put in the `models/` directory
#

from abstract_pipeline import Pipeline
import pdb

import random

# custom tokenizers based on NLTK
from tokenizers import word_tokenizer, sent_tokenizer

import collections

import cPickle as pickle

import quality3
import sklearn


from pprint import pprint

import logging
logger = logging.getLogger(__name__)

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]



class MockPipeline(Pipeline):
    pipeline_title = "Dummy predictions"

    def predict(self, full_text):
        logger.info("running hybrid predict!")
        return self.document_predict(full_text), self.sentence_predict(full_text)

    def document_predict(self, full_text):
        return {domain: random.choice([1, 0]) for domain in CORE_DOMAINS}

    def sentence_predict(self, full_text):
        # first get sentence indices in full text
        sent_indices = self._sent_spans(full_text)

        # then the strings (for internal use only)
        sent_text = [full_text[start:end] for start, end in sent_indices]

        # for this example, randomly assign as relevant or not
        # with a 1 in 50 chance of being positive for each domain
        sent_predict = [{domain: random.choice([1] + ([-1] * 10)) for domain in CORE_DOMAINS} for sent in sent_text]
        # return dict like:
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}

        return collections.OrderedDict(zip(sent_indices, sent_predict))

    def _sent_spans(self, text):
        return sent_tokenizer.span_tokenize(text)



class RegularPipeline(MockPipeline):
    pipeline_title = "Predict everything as randomization - true sent tokenizing"

    def sentence_predict(self, full_text):
        # first get sentence indices in full text
        sent_indices = sent_tokenizer.span_tokenize(full_text)

        # then the strings (for internal use only)
        sent_text = [full_text[start:end] for start, end in sent_indices]

        # for this example, assign every 7th div as being positive
        # sentence 0 with domain 0 (Random sequence generation) should be positive
        sent_predict = [{domain: rating for domain, rating in zip(CORE_DOMAINS, [1, 0, 0, 0, 0, 0])} for sent in sent_text]
    
        return collections.OrderedDict(zip(sent_indices, sent_predict))


class RegularFakeSentPipeline(RegularPipeline):
    pipeline_title = "Predict everything as randomization - fake sent tokenizing"
    def _sent_spans(self, text):
        """
        returns fake sentence spans starting at zero, sentence boundary every 50 chars
        """
        return [(start, start + 50) for start in range(len(text), 50)] + [(len(text) - (len(text) % 50), len(text))]




class RoBPipeline(Pipeline):
    """
    Predicts risk of bias document class + relevant sentences
    """
    pipeline_title = "Risk of Bias"

    def __init__(self):
        logger.info("loading models")
        self.doc_models, self.doc_vecs, self.sent_models, self.sent_vecs = self.load_models('models/quality_models.pck')
        logger.info("done loading models")

    def load_models(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def predict(self, full_text):

        logger.debug("starting prediction code")
        # first get sentence indices in full text
        sent_indices = sent_tokenizer.span_tokenize(full_text)

        # then the strings (for internal use only)
        sent_text = [full_text[start:end] for start, end in sent_indices]

        sent_preds_by_domain = [] # will rejig this later to make a list of dicts
        doc_preds = {}

        for test_domain, doc_model, doc_vec, sent_model, sent_vec in zip(CORE_DOMAINS, self.doc_models, self.doc_vecs, self.sent_models, self.sent_vecs):

            ####
            ## PART ONE - get the predicted sentences with risk of bias information
            ####

            # vectorize the sentences
            X_sents = sent_vec.transform(sent_text)


            # get predicted 1 / -1 for the sentences
            # bcw -- addint type conversion patch for numpy.int64 weirdness
            pred_sents = [int(x_i) for x_i in sent_model.predict(X_sents)]
            sent_preds_by_domain.append(pred_sents) # save them for later highlighting

            # for internal feature generation, get the sentences which are predicted 1
            positive_sents = [sent for sent, pred in zip(sent_text, pred_sents) if pred==1]

            # make a single string per doc
            summary_text = " ".join(positive_sents)

            print test_domain
            print "=" *60
            print
            print "\n\n".join(positive_sents)
            print
            print



            ####
            ##  PART TWO - integrate summarized and full text, then predict the document class
            ####

            doc_vec.builder_clear()
            doc_vec.builder_add_docs([full_text])
            doc_vec.builder_add_docs([summary_text], prefix="high-prob-sent-")

            X_doc = doc_vec.builder_transform()

            # change the -1s to 0s for now (TODO: improve on this)
            # done because the viewer has three classes, and we're only predicting two here
            doc_preds[test_domain] = 1 if doc_model.predict(X_doc)[0] == 1 else 0


        # rejig to correct output format
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}
        sent_preds_values = [{domain: rating for domain, rating in zip(CORE_DOMAINS, sent_ratings)} for sent_ratings in zip(*sent_preds_by_domain)]

        # make a dict; filter only rows with at least one positive prediction
        sent_preds = dict([row for row in zip(sent_indices, sent_preds_values) if (1 in row[1].values())])

        return doc_preds, sent_preds
