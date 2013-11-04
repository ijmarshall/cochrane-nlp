'''
Do 'vanilla' supervised learning over labeled citations.
'''

import bilearn
from bilearn import bilearnPipeline


class SupervisedLearner:
    def __init__(self):
        pass

    def features_from_citations(self):
        p = bilearnPipeline(text)
        p.generate_features()
        X = p.get_features(filter=lambda x: x["w[0]"].isdigit())
        words = p.get_answers(filter=lambda x: x["w"].isdigit())

        return X, words

class LabeledAbstractReader:
    ''' 
    Parses labeled citations from the provided path. Assumes format is like:

        Abstract 1 of 500
        Prothrombin fragments (F1+2) ...
            ...
        BiviewID 42957; PMID 11927130

    '''
    def __init__(self, path_to_data="data/drug_trials_in_cochrane_BCW.txt"):
        # @TODO probably want to make this entire class an iterator,
        # rather than loading everything into memory!
        self.abstracts = []
        self.path_to_abstracts = path_to_data
        print "parsing data from {0}".format(self.path_to_abstracts)
        self.parse_abstracts()

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
        biview_id, pmid_id = [grab_id(s) for s in l.split(";")]
        return biview_id, pmid_id

    def _is_new_citation_line(self, l):
        return l.startswith("Abstract ")

    def parse_abstracts(self):
        self.citation_d = {}
        in_citation = False
        with open(self.path_to_abstracts, 'rU') as abstracts_file:
            cur_abstract = ""
            
            for line in abstracts_file.readlines():
                if self._is_demarcater(line):
                    biview_id, pmid_id = self._get_IDs(line)
                    self.citation_d[pmid_id] = {"abstract":cur_abstract, 
                                                "Biview_id":biview_id}
                    in_citation = False
                elif in_citation and line.strip():
                    # then this is the abstract
                    cur_abstract = line
                elif self._is_new_citation_line(line):
                    in_citation = True

        return self.citation_d






