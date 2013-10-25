# pmccorpusdownload.py
# finds out which RCTs in Cochrane are in the open access subset (file_list.txt)
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


OUTPUT_PATH = '/users/iain/Code/data/cdsrpmc2/'

def gettar(ftp, filename):

    data = BytesIO()
    ftp.retrbinary("RETR " + filename, data.write)
    data.seek(0)
    tar = tarfile.open(mode="r:gz", fileobj=data)
    # print tar.getmembers()
    tar.extractall(OUTPUT_PATH)

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
    viewer = BiViewer(linkfile='data/biviewer_links_pmc_oa.pck')

    print "Loading PMC index..."
    lookup = getaddresses('data/pmc_file_list.txt')



    print "Connecting to Pubmed Central FTP..."


    p = ProgressBar(len(viewer))

    ftp = FTP('ftp.ncbi.nlm.nih.gov')
    ftp.login()
    print ftp.getwelcome()
    print    
    print "Downloading pdfs"

   
    for study in viewer:
        p.tap()
        pmc_id = study[1]['PMCid']
        gettar(ftp, 'pub/pmc/%s' % (lookup[pmc_id],))



if __name__ == '__main__':
    main()
