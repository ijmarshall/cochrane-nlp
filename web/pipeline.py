#
#   pipeline.py
#
#   N.B. predictive modelling requires a trained model in pickle form:
#   - get `quality_models.pck` from `Dropbox/cochranetech/quality-prediction-results/models`
#   - put in the `models/` directory
#


from abc import ABCMeta, abstractmethod
import random

# custom tokenizers based on NLTK
from tokenizers import word_tokenizer, sent_tokenizer 

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *

import cPickle as pickle

import quality3
import sklearn

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]





class Pipeline(object):
    __metaclass__ = ABCMeta

    def parse(self, pages):
        # we need to do two things, create a single string for each page
        # and establish a interval-tree to figure out the original nodes
        parsed = []
        for idx, page in enumerate(pages):
            if page is not None:
                textNodes = [node["str"] for node in page]

                total = 0
                ranges = []
                for txt in textNodes:
                    start = total
                    total += len(txt) + 1 # we're adding an extra space
                    ranges.append((start, total))
                interval_tree = SortedSet(ranges, key_type = (int, int), updator = OverlappingIntervalsUpdator)
                page_str = " ".join(textNodes)

                parsed.append({"str": page_str,
                               "length": total,
                               "intervals": interval_tree,
                               "ranges": ranges})
            else:
                parsed.append({})

        return parsed

    def get_page_offsets(self, page_lengths):
        " takes list of page lengths, returns cumulative list for offsets "
        # we store this because we want per document, not per page
        def accumulate(x, l=[0]):
            # since list l held by reference, value is stored between function calls!
            l[0] += x
            return l[0]
        return map(accumulate, [0] + page_lengths)

    def predict(self, input):

        parsed_pages = self.parse(input)

        # get the page lengths, and the page offsets in the whole doc string
        page_lengths = [page["length"] for page in parsed_pages]
        total_length = self.get_page_offsets(page_lengths)

        full_text = ' '.join(page["str"] for page in parsed_pages)
        
        # get the predictions

        print "getting predictions"
        document_predictions, sentence_predictions = self.hybrid_predict(full_text)

        print "done!"

        # Now we need to get /back/ to the page and node indexes
        annotations = []
        for sentence_bound, labels in sentence_predictions.iteritems():
            page_nr = next(i for i,v in enumerate(total_length) if v >= sentence_bound[0]) - 1
            page = parsed_pages[page_nr]
            offset = total_length[page_nr]
            bound = (sentence_bound[0] - offset, sentence_bound[1] - offset)
            nodes = [page["ranges"].index(x) for x in page["intervals"].overlap(bound)]

            annotations.append({
                "page": page_nr,
                "nodes": nodes,
                "labels": labels})

        return {"document": document_predictions,
                "annotations": annotations}




class MockPipeline(Pipeline):

    def hybrid_predict(self, full_text):
        print "running hybrid predict!"
        return self.document_predict(full_text), self.sentence_predict(full_text)

    def document_predict(self, full_text):
        return {domain: random.choice([1, 0]) for domain in CORE_DOMAINS}

    def sentence_predict(self, full_text):
        # first get sentence indices in full text
        sent_indices = sent_tokenizer.span_tokenize(full_text)

        # then the strings (for internal use only)
        sent_text = [full_text[start:end] for start, end in sent_indices]

        # for this example, randomly assign as relevant or not
        # with a 1 in 50 chance of being positive for each domain
        sent_predict = [{domain: random.choice([1] + ([-1] * 10)) for domain in CORE_DOMAINS} for sent in sent_text]
        
        # return dict like:
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}

        return dict(zip(sent_indices, sent_predict))



class RoBPipeline(Pipeline):
    """
    Predicts risk of bias document class + relevant sentences
    """

    def __init__(self):
        print "loading models... please wait..."
        self.doc_models, self.doc_vecs, self.sent_models, self.sent_vecs = self.load_models('models/quality_models.pck')
        print "done!!!"

    def load_models(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def hybrid_predict(self, full_text):

        print 'starting prediction code'
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
            pred_sents = sent_model.predict(X_sents)



            sent_preds_by_domain.append(pred_sents) # save them for later highlighting

            # for internal feature generation, get the sentences which are predicted 1
            positive_sents = [sent for sent, pred in zip(sent_text, pred_sents) if pred==1]

            # make a single string per doc
            summary_text = " ".join(positive_sents)


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

        sent_preds = dict(zip(sent_indices, sent_preds_values))

        return doc_preds, sent_preds

            





