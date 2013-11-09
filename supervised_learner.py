'''
Do 'vanilla' supervised learning over labeled citations.
'''

import bilearn
from bilearn import bilearnPipeline


class SupervisedLearner:
    def __init__(self, abstract_reader):
        self.abstract_reader = abstract_reader

    def features_from_citations(self):
        X, y = [], []
        for cit in self.abstract_reader:
            abstract_text = cit["abstract"]
            p = bilearnPipeline(text)
            p.generate_features()
            #filter=lambda x: x["w[0]"].isdigit()
            X_i = p.get_features()
            words = p.get_answers()
            X.append(X_i)
            y.append() ##????
        return X, y

        

class LabeledAbstractReader:
    ''' 
    Parses labeled citations from the provided path. Assumes format is like:

        Abstract 1 of 500
        Prothrombin fragments (F1+2) ...
            ...
        BiviewID 42957; PMID 11927130

    '''
    def __init__(self, path_to_data="data/drug_trials_in_cochrane_BCW.txt"):
        # @TODO probably want to read things in lazily, rather than
        # reading everything into memory at once...
        self.abstracts = []
        self.abstract_index = 0 # for iterator
        self.path_to_abstracts = path_to_data
        print "parsing data from {0}".format(self.path_to_abstracts)
        self.parse_abstracts()
        self.num_citations = len(self.citation_d) 
        print "ok."


    def __iter__(self):
        self.abstract_index = 0
        return self

    def next(self):
        if self.abstract_index >= self.num_citations:
            raise StopIteration
        else:
            self.abstract_index += 1
            return self.citation_d.values()[self.abstract_index-1]

    def _is_demarcater(self, l):
        '''
        True iff l is a line separating two citations.
        Demarcating lines look like "BiviewID 42957; PMID 11927130"
        '''

        # reasonably sure this will not give any false positives...
        return l.startswith("BiviewID") and "PMID" in l

    def _get_IDs(self, l):
        ''' Assumes l is a demarcating line; returns Biview and PMID ID's '''
        grab_id = lambda s : s.lstrip().split(" ")[1].strip()
        biview_id, pmid = [grab_id(s) for s in l.split(";")]
        return biview_id, pmid

    def _is_new_citation_line(self, l):
        return l.startswith("Abstract ")

    def parse_abstracts(self):
        self.citation_d = {}
        in_citation = False
        with open(self.path_to_abstracts, 'rU') as abstracts_file:
            cur_abstract = ""
            
            for line in abstracts_file.readlines():
                line = line.strip()
                if self._is_demarcater(line):
                    biview_id, pmid = self._get_IDs(line)
                    self.citation_d[pmid] = {"abstract":cur_abstract, 
                                                "Biview_id":biview_id,
                                                "pubmed_id":pmid} # yes, redundant
                    in_citation = False
                elif in_citation and line:
                    # then this is the abstract
                    cur_abstract = line
                elif self._is_new_citation_line(line):
                    in_citation = True

        return self.citation_d

    def get_text(self):
        return [cit["abstract"] for cit in self.citation_d.values()]





