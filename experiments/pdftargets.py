#
#	PDFs to obtain
#

import csv
import riskofbias
from cochranenlp.readers import biviewer
from cochranenlp.output.progressbar import ProgressBar

from collections import Counter

data = riskofbias.RoBData(test_mode=False)
data.generate_data(doc_level_only=False)

all_pmids = Counter()

b = biviewer.BiViewer()
print "getting all pubmed ids in CDSR..."

for doc in b:
	all_pmids[doc.pubmed['pmid']] += 1

multi_assessed = [pmid for pmid, count in all_pmids.iteritems() if count >1]

print "Total multi-assessed = %d" % len(multi_assessed)


docs = riskofbias.DocFilter(data)
uids = filtered_data.get_ids() # what we have PDFs for

print "PDFs we have = %d" % len(uids)

to_get = set(multi_assessed).difference(uids)

print "no to get = %d" % len(to_get)


