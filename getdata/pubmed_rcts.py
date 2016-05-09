#
#    get all RCT data from PubMed
#

from glob import glob
import xml.etree.cElementTree as ET
from cochranenlp.output.progressbar import ProgressBar
from cochranenlp.readers import pmlib
import cochranenlp
import cPickle as pickle
import os
import shelve
from contextlib import closing

### set local paths from CNLP.INI

PUBMED_ABSTRACTS_PATH = cochranenlp.config.get("Paths","rct_abstracts_path") # to pubmed xml
CLINICAL_TRIALS_PATH = cochranenlp.config.get("Paths","clinical_trials_path") # to pubmed pdfs
BASE_PATH = cochranenlp.config.get("Paths","base_path")

pubmed = pmlib.Pubmed()
ET2unicode = pubmed._ET2unicode


def main():

    print CLINICAL_TRIALS_PATH
    print PUBMED_ABSTRACTS_PATH

    print "Collecting additional PMIDs from the clinicaltrials.gov data"
    with closing(shelve.open('rct_pmids.temp')) as db:
        target_pmids = db.get('ids')
        if not target_pmids:
            s = pmlib.IterSearch("Randomized Controlled Trial[ptyp]")
            target_pmids = [i for i in s.itersearch(show_progress=True)]
            db['ids'] = target_pmids





    target_pmids = set(target_pmids)


    filenames = glob(PUBMED_ABSTRACTS_PATH + '*.xml')

    print "%d documents found" % len(filenames)

    pmids = []

    p = ProgressBar(len(filenames), timer=True)
        
    print "%d RCTs with pubmed" % len(pmids)

    existing_pmids = set([os.path.split(filename)[1][:-4] for filename in glob(PUBMED_ABSTRACTS_PATH + '*.xml')])

    to_get = target_pmids.difference(existing_pmids)

    

    print "%d to be retrieved" % len(to_get)
    

    print "Retrieving abstracts for identified studies"

    search = pmlib.IterFetch(list(to_get), retmax=250)
    
    for (index, abstract) in enumerate(search.iterfetch()):
        id = ET2unicode(abstract.find("PMID"))
        if index % 250 == 0:
            print "%d abstracts retrieved" % (index, )
        xml = ET.ElementTree(abstract)
        xml.write(PUBMED_ABSTRACTS_PATH + str(id).strip() + ".xml")

    print "writing link file"

    with open(BASE_PATH + 'clinicaltrials_links.pck', 'wb') as f:
        pickle.dump(linkfile, f)




        
    

    


if __name__ == '__main__':
    main()