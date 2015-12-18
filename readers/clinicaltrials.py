 #
 #  connection to clinicaltrials.gov test
 #

import re

from xmlbase import XMLReader
import requests
import xml.etree.cElementTree as ET

def clean_study_design_str(study_design):
    """Removes intra-paren commas

    study_design: str of comma-separated key:value pairs

    This is a preprocessing step so we can later split on commas and get key:value pairs

    """
    IN_PAREN, OUT_PAREN = range(2)

    state = OUT_PAREN
    for c in study_design:
        if state == OUT_PAREN:
            yield c
            state = IN_PAREN if c == '(' else OUT_PAREN

        elif state == IN_PAREN:
            if c == ',':
                continue # don't yield commas inside pairs of parentheses
            yield c
            state = OUT_PAREN if c == ')' else IN_PAREN

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

        # self.section_map["abstract"] = 'Article/Abstract'
        # self.section_map["linkedIds"] = 'OtherID'
        # self.section_map["pmid"] = 'PMID'
        # self.section_map["mesh"] = 'MeshHeadingList/MeshHeading/DescriptorName'
        # self.section_map["language"] = 'Article/Language'

        self.extract_clf_fields()

    def extract_clf_fields(self):
        """Extracts fields of interest for prediction from the ct.gov XML entry"""

        root = self.data.getroot()

        # Fields of interest for prediction
        keys = ('allocation', 'endpoint_classification', 'intervention_model', 'masking', 'primary_purpose',
                'condition', 'gender', 'healthy_volunteers', 'maximum_age', 'minimum_age', 'phase', 'study_type')

        # Initialize fields just in case they are not present
        self.fields = {}
        for key in keys:
            self.fields[key] = None

        self.fields['condition'] = root.find('condition').text
        self.fields['phase'] = root.find('phase').text
        self.fields['study_type'] = root.find('study_type').text
        self.fields['gender'] = root.find('eligibility/gender').text
        self.fields['minimum_age'] = root.find('eligibility/minimum_age').text
        self.fields['maximum_age'] = root.find('eligibility/maximum_age').text

        # Healthy volunteers may be missing
        healthy_volunteers = root.find('eligibility/healthy_volunteers')
        if healthy_volunteers is not None: # truthiness oddity
            self.fields['healthy_volunteers'] = healthy_volunteers.text

        # Study design is comma-separated key:value pairs string
        study_design = root.find('study_design').text
        study_design = ''.join(list(clean_study_design_str(study_design))) # delete intra-paren commas
        for pair in study_design.split(','):
            key, value = pair.split(':')
            key = re.sub('\s+', '_', key.strip()).lower() # lowercase and get rid of spaces
            self.fields[key] = value.strip()


def main():

    c = Connection()
    response = c.get('NCT00423098')

    # print response.title
    # print response.sponsors
    print response.pmids









if __name__ == '__main__':
    main()
