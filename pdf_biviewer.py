#
#    pdf_biviewer.py
#
        

#
#   FOR TESTING PURPOSES, TO INTEGRATE INTO biviewer.py WHEN WORKING
#

"""

Class for iterating through PDF full text papers with associated Cochrane data

"""

from journalreaders import PdfReader
from rm5reader import *
from pmreader import *
import cPickle as pickle
from pprint import pprint
from time import time
from progressbar import ProgressBar
import collections
import sys
import random
import re


### set local paths here

COCHRANE_REVIEWS_PATH = "/users/iain/code/data/cdsr2013/" # to revman files
PUBMED_ABSTRACTS_PATH = "/users/iain/code/data/cdsrpubmed/" # to pubmed xml
PDF_PATH = "/users/iain/code/data/cdsrpdfs/" # to pubmed pdfs



class BiViewer():
    """
    Class for accessing parallel data from pubmed and cochrane
    Indexed as a list with each item containing data from a single RCT

        designed to be - very fast when working from memory
                       - quite quick if accessing from disk sequentially
                       - and caches if random access from disk (last n reviews)
                       
    """


    def __init__(self, in_memory=False, cdsr_cache_length=20,
                 cdsr_filter=None, test_mode=False, linkfile="data/pdf_present_links.pck"):

        self.import_data(filename=linkfile, test_mode=test_mode)

        self.data = []
        self.cdsr_cache_length = cdsr_cache_length
        self.cdsr_cache_index = collections.deque()
        self.cdsr_cache_data = {}
        # self.pm_filter = pm_filter # not needed at present since PDFs return one block of text
        self.cdsr_filter = cdsr_filter

        if in_memory==True:
            self.load_data_in_memory()


    def import_data(self, filename, test_mode=False):
        "loads the pubmed-cdsr correspondance list"
        with open(filename, 'rb') as f:
            self.index_data = pickle.load(f)
        if test_mode:
            # get the first 2500 studies only (more since now per study, not per review)
            self.index_data = self.index_data[:2500]
                    
    def __len__(self):
        "returns number of RCTs in index (not number of Cochrane reviews)"
        return len(self.index_data)

    def __getitem__(self, key):
        """
        returns (cochrane_data, pubmed_data) tuple relating to study index number
        implements a cache of last cochrane reviews accessed to avoid reparsing for
        when using file based access rather than memory based
        """
        try:
            return self.data[key] # first return data if loaded in memory
        except IndexError: # if not in memory will retrieve from file
            
            study = self.index_data[key]
            
            
            if study['cdsr_filename'] in self.cdsr_cache_data:
                # if review in cache return it
                cr = self.cdsr_cache_data[study['cdsr_filename']] 
            else:
                # else load from file, save to end of cache, and delete oldest cached review
                cr = RM5(COCHRANE_REVIEWS_PATH + study['cdsr_filename']).refs(full_parse=True, return_dict=True)
                self.cdsr_cache_data[study['cdsr_filename']] = cr # save to cache
                self.cdsr_cache_index.append(study['cdsr_filename']) # and add to index deque

                if len(self.cdsr_cache_index) > self.cdsr_cache_length:
                    self.cdsr_cache_data.pop(self.cdsr_cache_index.popleft())
                    # removes the oldest review from the data cache
                    # (the one indexed 0 in the index deque)
                    # and simultaneously removes from the index (popleft)

            pm = PdfReader(PDF_PATH + study['pdf_filename'])
            if self.cdsr_filter and self.pm_filter:
                return (cr[study['cdsr_refcode']].get(self.cdsr_filter, ""), pm.get_text())
            else:
                return (cr[study['cdsr_refcode']], pm.get_text())

    def load_data_in_memory(self):
        self.data = []
        p = ProgressBar(len(self), timer=True)
        for i in range(len(self)):
            p.tap()
            self.data.append(self[i])
        
    

    
def main():
    # example
    # set paths at top of script before running



    bv = BiViewer()

    # bv[corpus_study_index][0=cochrane;1=pubmed][part_to_retrieve]

    print "Title:"
    print bv[0][1]["title"] # print the pubmed title of the first study
    print
    print "Abstract:"
    print bv[0][1]["abstract"] # find out the pubmed abstract of the first study
    print
    print "Intervention description in Cochrane:"
    print bv[0][0]["CHAR_INTERVENTIONS"]

    # the biviewer essentially returns a list of tuples (cochrane, pubmed), with cochrane and pubmed being dicts
    #  showing interesting parts of the studies
    #
    # cochrane is the cochrane review parsed by reference (and only the parts related to the current study returned)
    # - specifically it returns RM5.refs()["unique_id"] from rm5reader.py
    #
    # and pubmed is pm.text_all() from pmreader.py

    import random

    print "Select a random study, show the pubmed title, and cochrane risk of bias info"

    study = random.choice(bv)

    print "Title"
    print study[1]["title"]
    print

    print "Risk of bias"
    print
    for i, domain in enumerate(study[0]["QUALITY"]):
        print "Domain number %d" % (i, )
        print "Name\t\t" + domain["DOMAIN"]
        print "Description\t" + domain["DESCRIPTION"]
        print "Rating\t\t" + domain["RATING"]







    
if __name__ == '__main__':
    main()