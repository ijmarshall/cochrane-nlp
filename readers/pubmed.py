#
#   Pubmed reader
#
from glob import glob
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




class Reader(XMLReader):

    def __init__(self, filename):
        NLMCorpusReader.__init__(self, filename)
        self.section_map["title"] = 'Article/ArticleTitle'
        self.section_map["abstract"] = 'Article/Abstract'
        self.section_map["linkedIds"] = 'OtherID'
        self.section_map["pmid"] = 'PMID'
        self.section_map["mesh"] = 'MeshHeadingList/MeshHeading/DescriptorName'
        self.section_map["language"] = 'Article/Language'
        self.section_map["affiliation"] = 'Article/Affiliation'
        self.section_map["ptype"] = 'Article/PublicationTypeList/PublicationType'
        
        

    def text_all(self):
        output = NLMCorpusReader.text_all(self) # get the normal output dict
        output["PMCid"] = self.is_pmc_linked()
        return output

    def is_pmc_linked(self):
        els = self.data.findall('OtherID')
        if len(els) > 0:
            for el in els:
                text = self._ET2unicode(el)
                result = re.match("PMC[0-9]+", text)
                if result is not None:
                    return result.group(0)
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
