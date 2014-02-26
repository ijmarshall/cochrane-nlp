from abc import ABCMeta, abstractmethod
# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *
from itertools import *
import pprint

pp = pprint.PrettyPrinter(indent=4)

class Pipeline(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def predict(self, input):
        pass
        
    # we need to do two things, create a single string for each page
    # and establish a interval-tree to figure out the original nodes
    
    def parse(self, pages):
        parsedPages = [] # stores the text and the interval tree per page
        for idx, page in enumerate(pages):
            if page is not None:
                textNodes = [node["str"] for node in page] 

                total = 0
                ranges = []
                for txt in textNodes:
                    start = total
                    total += len(txt) + 1 # we're adding an extra space
                    ranges.append((start, total))
                intervalTree = SortedSet(ranges, updator = OverlappingIntervalsUpdator)
                pageStr = " ".join(textNodes)

                parsedPages.append({"str": pageStr, "intervals": intervalTree})
            else:
                parsedPages.append({"str" : None, "intervals": None})

        return parsedPages


class MockPipeline(Pipeline):
    def predict(self, input):
        parsedPages = self.parse(input)
        return {"annotations": [x["str"] for x in parsedPages]}
