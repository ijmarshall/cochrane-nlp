#
#   Pubmed reader
#
from glob import glob
import codecs 
from pprint import pprint
import re
from xmlbase import XMLReader


import xml.etree.cElementTree as ET


def list_bounds(input_list, index, boundary):
    """
    returns indexed word with words surrounding
    useful function in many places
    for displaying collocations
    """
    index_lower = index - boundary
    if index_lower < 0:
        index_lower = 0
        
    index_upper = index + boundary + 1
    if index_upper > len(input_list):
        index_upper = len(input_list)
        
    return input_list[index_lower:index_upper]
    





class NLMCorpusReader(XMLReader):
    pass





        
class PMCCorpusReader(NLMCorpusReader):
    #
    # not fully functioning yet - nxml files are not really valid xml - they contain HTML within some fields
    #
    def __init__(self, filename):
        NLMCorpusReader.__init__(self, filename)
        self.section_map["title"] = 'front/article-meta/title-group/article-title'
        self.section_map["abstract"] = 'front/article-meta/abstract'
        
        
   
        

class PubmedCorpusReader(NLMCorpusReader):

    def __init__(self, filename):
        NLMCorpusReader.__init__(self, filename)
        self.section_map["title"] = 'Article/ArticleTitle'
        self.section_map["abstract"] = 'Article/Abstract'
        self.section_map["linkedIds"] = 'OtherID'
        self.section_map["pmid"] = 'PMID'
        self.section_map["mesh"] = 'MeshHeadingList/MeshHeading/DescriptorName'
        
    def is_pmc_linked(self):
        els = self.data.findall('OtherID')
        if len(els) > 0:
            for el in els:
                text = self._ET2unicode(el)
                result = re.search("PMC[0-9]+", text)
                if result is not None:
                    return text.strip()
        return None


def main():
    PATH = "/users/iain/Code/data/cdsrpubmed"
    
    
    print "finding files"
    files = glob(PATH + '/*.xml')
    print "found!"
    counter = 0
    
    for f in files[:10]:
        
        test = PubmedCorpusReader(f)
        print test.text_filtered_all("mesh")

    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()

