"""
metamap.py
Simple metamap wrapper which uses the XML output format

N.B.
This requires Metamap to be set up in a specific way - see the METAMAP_INSTALL.txt file for the instructions
"""


from subprocess import Popen, PIPE, STDOUT
import cochranenlp
import xml.etree.cElementTree as ET
import unidecode

METAMAP_PATH = cochranenlp.config.get('Paths', 'METAMAP_PATH') # TO PUBMed pdfs


def metamap(string_list, term_processing=True):
    """
    By default accepts list of terms (i.e. noun phrases)
    and returns a list of mappings corresponding to each
    term.
    
    >>> metamap(['myocardial infarction', 'aspirin'])
    [[{'cui': 'C0027051',
       'name': 'Myocardial Infarction',
       'neg': '0',
       'raw_text': 'myocardial infarction'},
      {'cui': 'C0428953',
       'name': 'Electrocardiogram: myocardial infarction (finding)',
       'neg': '0',
       'raw_text': 'myocardial infarction'}],
     [{'cui': 'C0004057',
       'name': 'Aspirin',
       'neg': '0',
       'raw_text': 'aspirin'}]]
    """

    if not isinstance(string_list, list):
        string_list = [string_list]
                
    string_list = [unidecode.unidecode(s) for s in string_list if s]
    
    mm_command = [METAMAP_PATH, '--XMLn1', '--silent', '-R', 'ATC,MDR,RXNORM,SNOMEDCT_US,MSH', '-Z', '2015AB', '-V', 'NLM', '--sldi']

    if term_processing:
        mm_command.append('-z')
    
    p = Popen(mm_command, stdout=PIPE, stdin=PIPE, stderr=STDOUT)

    grep_stdout = p.communicate(input='\n'.join(string_list)+'\n')

    raw = grep_stdout[0]
    xml_start_i = raw.find('<?xml')
    xml_str = raw[xml_start_i:]

    root = ET.fromstring(xml_str)
    mmos = root.findall('MMO')

    output = []
    for s, mmo in zip(string_list, mmos):

        cuis = mmo.findall('Utterances/Utterance/Phrases/Phrase/Mappings/Mapping/MappingCandidates/Candidate')


        output.append([{"name": cui.find('CandidatePreferred').text, 
                       "cui": cui.find('CandidateCUI').text, 
                       "neg": cui.find('Negated').text,
                       "raw_text": s} for cui in cuis]
                       )
    return output
    
