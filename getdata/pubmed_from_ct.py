#
#    get a list of pubmed IDs from the clinicaltrials.gov data
#

from glob import glob
import xml.etree.cElementTree as ET
from cochranenlp.output.progressbar import ProgressBar
from cochranenlp.readers import clinicaltrials, pmlib
import cochranenlp
import cPickle as pickle
import os

### set local paths from CNLP.INI

PUBMED_ABSTRACTS_PATH = cochranenlp.config["Paths"]["pubmed_abstracts_path"] # to pubmed xml
CLINICAL_TRIALS_PATH = cochranenlp.config["Paths"]["clinical_trials_path"] # to pubmed pdfs
BASE_PATH = cochranenlp.config["Paths"]["base_path"]


pubmed = pmlib.Pubmed()
ET2unicode = pubmed._ET2unicode




def main():

    print CLINICAL_TRIALS_PATH
    print PUBMED_ABSTRACTS_PATH

    print "Collecting additional PMIDs from the clinicaltrials.gov data"
    filenames = glob(CLINICAL_TRIALS_PATH + '*.xml')

    print "%d documents found" % len(filenames)

    pmids = []

    p = ProgressBar(len(filenames), timer=True)

    linkfile = []

    for filename in filenames:
        p.tap()
        ctr = clinicaltrials.Reader(filename)
        ids = ctr.resultspmids
        if ids:
            pmids.append(ids[0])
            linkfile.append({'clinicaltrials.gov': os.path.split(filename)[1], 'pmid': ids[0], 'pm_filename':ids[0]+'.xml'})
        

    print "%d documents with pubmed links" % len(pmids)
    pmids_set = set(pmids)
    print "of which %d are unique" % len(pmids_set)

    print "=" * 40


    existing_pmids = set([os.path.split(filename)[1][:-4] for filename in glob(PUBMED_ABSTRACTS_PATH + '*.xml')])

    to_get = pmids_set.difference(existing_pmids)

    print existing_pmids

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