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
        self.section_map["interventions"] = 'intervention/intervention_name'
        self.section_map["outcomes"] = 'clinical_results/outcome_list/outcome'
        self.section_map["num_arms"] = 'number_of_arms'
        self.section_map["arms"] = 'arm_group/arm_group_label'
        self.section_map["resultspmids"] = 'results_reference/PMID'
        self.section_map["pmids"] = 'reference/PMID'
        self.section_map["nct_id"] = 'id_info/nct_id'

        self.section_map["study_type"] = 'study_type'

        self.root = self.data.getroot()


        # self.section_map["abstract"] = 'Article/Abstract'
        # self.section_map["linkedIds"] = 'OtherID'
        # self.section_map["pmid"] = 'PMID'
        # self.section_map["mesh"] = 'MeshHeadingList/MeshHeading/DescriptorName'
        # self.section_map["language"] = 'Article/Language'

    @property
    def study_type(self):
        return self.root.find('study_type').text

    @property
    def phase(self):
        return self.root.find('phase').text

def main():

    c = Connection()
    response = c.get('NCT00423098')

    # print response.title
    # print response.sponsors
    print response.pmids









if __name__ == '__main__':
    main()
