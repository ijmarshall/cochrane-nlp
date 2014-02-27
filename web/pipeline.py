from abc import ABCMeta, abstractmethod
import random

# custom tokenizers based on NLTK
from tokenizers import word_tokenizer, sent_tokenizer 

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *

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
        document_predictions, sentence_predictions = self.hybrid_predict(full_text)

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

    def hybrid_predict(self, parsed_pages):


        return self.document_predict(parsed_pages), self.sentence_predict(parsed_pages)

    def document_predict(self, full_text):
        return {domain: random.choice([1, 0]) for domain in CORE_DOMAINS}

    def sentence_predict(self, full_text):
        # first get sentence indices in full text
        sent_indices = sent_tokenizer.span_tokenize(full_text)

        # then the strings (for internal use only)
        sent_text = [full_text[start:end] for start, end in sent_indices]

        # for this example, randomly assign as relevant or not
        # with a 1 in 50 chance of being positive for each domain
        sent_predict = [{domain: random.choice([1] + ([-1] * 300)) for domain in CORE_DOMAINS} for sent in sent_text]

        # return dict like:
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}

        print dict(zip(sent_indices, sent_predict))

        return dict(zip(sent_indices, sent_predict))



# class RoBPipeline(Pipeline):
#     """
#     Predicts risk of bias document class + relevant sentences
#     """

#     def hybrid_predict(self, parsed_pages):
        





