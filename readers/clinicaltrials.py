 #
 #  connection to clinicaltrials.gov test
 #

from xmlbase import XMLReader
import requests
import xml.etree.cElementTree as ET

class Connection:

    def __init__(self):
        pass

    def get(self, ctid):
        url = "https://clinicaltrials.gov/show/" + ctid
        r = requests.get(url, params = {"resultsxml": "true"})
        return Reader(xml_string=r.content)


class Reader(XMLReader):

    def __init__(self, filename=None, xml_string=None):
        XMLReader.__init__(self, filename, xml_string)
        self.section_map["title"] = 'brief_title'
        self.section_map["design"] = 'study_design'
        self.section_map["population"] = 'eligibility'
        self.section_map["interventions"] = 'intervention'
        self.section_map["outcomes"] = 'clinical_results/outcome_list/outcome'
        self.section_map["arms"] = 'number_of_arms'
        self.section_map["results-pmids"] = 'results_reference/PMID'
        self.section_map["pmids"] = 'reference/PMID'
        # self.section_map["abstract"] = 'Article/Abstract'
        # self.section_map["linkedIds"] = 'OtherID'
        # self.section_map["pmid"] = 'PMID'
        # self.section_map["mesh"] = 'MeshHeadingList/MeshHeading/DescriptorName'
        # self.section_map["language"] = 'Article/Language'



def main():

    c = Connection()
    response = c.get('NCT00423098')

    # print response.title
    # print response.sponsors
    print response.pmids









if __name__ == '__main__':
    main()
