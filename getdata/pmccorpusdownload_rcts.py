# pmccorpusdownload_rcts.py
# tries to identify as many RCTs from Pubmed Central as possible, # where they have PDF and NXML files
# and downloads them


from biviewer import BiViewer
from StringIO import StringIO
import cPickle as pickle
import glob
import os.path
from pprint import pprint
import re
import tarfile

from ftplib import FTP
from io import BytesIO
from progressbar import ProgressBar


OUTPUT_PATH = '/users/iain/Code/data/cdsrpdfs/'

def getfile(ftp, filename):
    with open(OUTPUT_PATH + os.path.basename(filename), 'wb') as f:
        # print "writing file to %s" % (OUTPUT_PATH + os.path.basename(filename),)
        ftp.retrbinary("RETR " + filename, f.write)


def getaddresses(filename):

    lookup = {}

    with open(filename, 'rb') as f:

        for line in f:
            ids = line.strip().split('\t')
            
            lookup[ids[-1]] = ids[0]#.split('/')[-1].split('.')[0]

    return lookup

            # if len(ids) > 0:
    
            #     if ids[-1] in pmc_ids_in_cdsr and ids[0].split('/')[-1].split('.')[0] not in dirs:
            #         refs.append(ids[0])




def main():


    print "Loading Cochrane and Pubmed linked data..."
    with open('data/biviewer_links_pmc_oa.pck', 'rb') as f:
        viewer = pickle.load(f)
    
    print "Loading PMC index..."
    lookup = getaddresses('data/pmc_pdf_list.txt')



    print "Connecting to Pubmed Central FTP..."



    ftp = FTP('ftp.ncbi.nlm.nih.gov')
    ftp.login()
    print ftp.getwelcome()
    print    
    print "Downloading pdfs"

    notfound = 0
    found = 0

    pdf_present_links = []

    p = ProgressBar(len(viewer), timer=True)

    for study in viewer:
        p.tap()
        pmc_id = study['pmc_id']
        # print pmc_id
        if pmc_id in lookup:
            pdf_filename = lookup[pmc_id]
            getfile(ftp, 'pub/pmc/%s' % (pdf_filename,))
            # print "yes"
            study["pdf_filename"] = os.path.basename(pdf_filename)
            pdf_present_links.append(study)
            found += 1
        else:
            notfound += 1

    with open('data/pdf_present_links.pck', 'wb') as f:
        pickle.dump(pdf_present_links, f)

    print "done! %d pdfs not found; %d found" % (notfound, found)

if __name__ == '__main__':
    main()
