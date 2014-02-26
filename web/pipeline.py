from abc import ABCMeta, abstractmethod
import random

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *

class Pipeline(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, input):
        pass

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
                               "total": total,
                               "intervals": interval_tree,
                               "ranges": ranges})
            else:
                parsed.append({})

        return parsed


class MockPipeline(Pipeline):

    def random_annotations(self, nr_simulated, document_length=1000):
        # Mock sentence level predictions by generating random annotations
        def randinterval(start, stop):
            lower = random.randrange(start, stop)
            upper = random.randrange(lower, lower + 70) # 70 = average letters in sentence
            return (lower, upper)

        # Simulate a dict like
        # {(13, 33): {'Domain 1': 1, 'Domain 2': -1, 'Domain 3': -1},
        #  (27, 77): {'Domain 1': 1, 'Domain 2': 0, 'Domain 3': 1}}
        # This is the expected return format for any sentence /real/ prediction system
        sentence_bounds = [randinterval(0, document_length) for i in range(nr_simulated)]
        labels = [{"domain_1": random.randint(-1, 1),
                   "domain_2": random.randint(-1, 1),
                   "domain_3": random.randint(-1, 1)} for i in range(nr_simulated)]
        return dict(zip(sentence_bounds, labels))

    def predict(self, input):
        parsed_pages = self.parse(input)
        # we store this because we want per document, not per page
        page_bounds = [page["total"] for page in parsed_pages]

        # Mock document level predictions, this can be done on
        document_predictions = {"domain_1": 1, "domain_2": -1, "domain_3": 0}

        # Mock sentence level predictions
        sentence_predictions = self.random_annotations(10, document_length=page_bounds[-1])

        # Now we need to get /back/ to the page and node indexes
        annotations = []
        for sentence_bound, labels in sentence_predictions.iteritems():
            page_nr = next(i for i,v in enumerate(page_bounds) if v >= sentence_bound[0])
            page = parsed_pages[page_nr]
            bound = (sentence_bound[0], sentence_bound[1] - 1)
            nodes = [page["ranges"].index(x) for x in page["intervals"].overlap(sentence_bound)]
            annotations.append({
                "page": page_nr,
                "nodes": nodes,
                "labels": labels })

        return { "document": document_predictions,
                 "annotations": annotations }
