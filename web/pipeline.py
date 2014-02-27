from abc import ABCMeta, abstractmethod
import random

# custom tokenizers based on NLTK
from tokenizers import word_tokenizer, sent_tokenizer 

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *

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
        
        # get the predictions
        document_predictions, sentence_predictions = self.hybrid_predict(parsed_pages)

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

    def document_predict(self, parsed_pages):
        return {"Domain 1": 1, "Domain 2": -1, "Domain 3": 0}

    def sentence_predict(self, parsed_pages):
        return self.random_annotations(50, document_length=sum([page["length"] for page in parsed_pages]))

    def random_annotations(self, nr_simulated, document_length):
        # Mock sentence level predictions by generating random annotations
        def randinterval(start, stop):
            lower = random.randrange(start, stop)
            upper = random.randrange(lower, lower + 100)
            return (lower, upper)

        # Simulate a dict like
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}
        # This is the expected return format for any /real/ sentence prediction system
        sentence_bounds = [randinterval(0, document_length) for i in range(nr_simulated)]
        labels = [{"Domain 1": random.randint(-1, 1),
                   "Domain 2": random.randint(-1, 1),
                   "Domain 3": random.randint(-1, 1)} for i in range(nr_simulated)]
        return dict(zip(sentence_bounds, labels))







