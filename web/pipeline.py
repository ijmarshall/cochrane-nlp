from abc import ABCMeta, abstractmethod
import pprint

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *

pp = pprint.PrettyPrinter(indent=4)

class Pipeline(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def predict(self, input):
        pass
    
    def parse(self, pages):
        # we need to do two things, create a single string for each page
        # and establish a interval-tree to figure out the original nodes

        parsed = [] # stores the text and the interval tree per page
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

                parsed.append({"str": page_str, "intervals": interval_tree})
            else:
                parsed.append({"str" : None, "intervals": None})

        # We return both the full string and the interval tree, per page
        return parsed


class MockPipeline(Pipeline):
    def predict(self, input):
        parsed_pages = self.parse(input)
        return {"annotations": [x["str"] for x in parsed_pages]}
