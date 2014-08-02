#
#    biviewer.py
#
        

"""

Class for iterating through pubmed abstracts with associated Cochrane data

"""

import cPickle as pickle
import collections

from journalreaders import PdfReader
from pmreader import *
from progressbar import ProgressBar
from rm5reader import *

import os

import cochranenlp

### set local paths from CNLP.INI



COCHRANE_REVIEWS_PATH = cochranenlp.config["Paths"]["cochrane_reviews_path"] # to revman files
PUBMED_ABSTRACTS_PATH = cochranenlp.config["Paths"]["pubmed_abstracts_path"] # to pubmed xml
PDF_PATH = cochranenlp.config["Paths"]["pdf_path"] # to pubmed pdfs
DATA_PATH = cochranenlp.config["Paths"]["base_path"] # to pubmed pdfs



class BiViewer():
    """
    Class for accessing parallel data from pubmed and cochrane
    Indexed as a list with each item containing data from a single RCT

        designed to be - very fast when working from memory
                       - quite quick if accessing from disk sequentially
                       - and caches if random access from disk (last n reviews)
                       
    """


    def __init__(self, **kwargs):
        
        self.init_common_variables(**kwargs)
        self.BiviewerView = collections.namedtuple('BiViewer_View', ['cochrane', 'pubmed'])


    def init_common_variables(self, in_memory=False, cdsr_cache_length=20,
                 test_mode=False, linkfile=os.path.join(DATA_PATH, "biviewer_links_all.pck")):
        "set up variables used in all subclasses"
        self.import_data(filename=linkfile, test_mode=test_mode)
        self.data = []
        self.cdsr_cache_length = cdsr_cache_length
        self.cdsr_cache_index = collections.deque()
        self.cdsr_cache_data = {}

        if in_memory==True:
            self.load_data_in_memory()


    def import_data(self, filename, test_mode=False):
        "loads the pubmed-cdsr correspondance list"
        with open(filename, 'rb') as f:
            self.index_data = pickle.load(f)
        if test_mode:
            # get the first 2500 studies only (more since now per study, not per review)
            self.index_data = self.index_data[:2500]

        # # TEMPORARY - limit index to first instance of any PDF 
        # # TODO make this better
        # filtered = []
        # pmids_already_encountered = set()

        # for study in self.index_data:
        #     if study["pmid"] not in pmids_already_encountered:
        #         filtered.append(study)
        #         pmids_already_encountered.add(study["pmid"])

        # print "skipped %d (from the whole of Cochrane)" % (len(self.index_data) - len(filtered))
        # self.index_data = filtered
        # # END TEMPORARY CODE


        # end temporary code

                    

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
                cr = RM5(COCHRANE_REVIEWS_PATH + study['cdsr_filename']).full_parse()
                self.cdsr_cache_data[study['cdsr_filename']] = cr # save to cache
                self.cdsr_cache_index.append(study['cdsr_filename']) # and add to index deque

                if len(self.cdsr_cache_index) > self.cdsr_cache_length:
                    self.cdsr_cache_data.pop(self.cdsr_cache_index.popleft())
                    # removes the oldest review from the data cache
                    # (the one indexed 0 in the index deque)
                    # and simultaneously removes from the index (popleft)

            # return self.BiviewerView(cr[study['cdsr_refcode']], self.second_view(study))
            return self.BiviewerView(dict(cr[study['cdsr_refcode']].items() + {"cdsr_filename": study['cdsr_filename']}.items()), self.second_view(study))


    def second_view(self, study):
        """ parses pubmed abstract file for base class
        or can be overridden in subclass to return other
        data, e.g. PDF text"""
        pm = PubmedCorpusReader(PUBMED_ABSTRACTS_PATH + study['pmid'] + ".xml")
        return pm.text_all()

    
    def load_data_in_memory(self):
        self.data = []
        p = ProgressBar(len(self), timer=True)
        for i in range(len(self)):
            p.tap()
            self.data.append(self[i])




class PDFBiViewer(BiViewer):
    """
    Accesses parallel data from Cochrane reviews and associated
    full text PDFs.
    PDFs are converted to plain text using pdftotext (done in
    PdfReader class located in journalreaders.py)
    """

    def __init__(self, **kwargs):
        self.BiviewerView = collections.namedtuple('BiViewer_View', ['cochrane', 'studypdf'])
        BiViewer.init_common_variables(self, **kwargs)
        self.pdf_index = self.get_pdf_index()

        # limit self index to those studies with linked PDFs
        self.index_data = [item for item in self.index_data if item["pmid"] in self.pdf_index]




    def get_pdf_index(self):
    	"""
    	Makes an index (dict) of pdfs by pubmed id
    	RULES:
    	PDF files must be named where the last consecutive number string
    	is the pubmed ID (or just named PMID.pdf)
    	"""

    	pdf_filenames_all = glob(PDF_PATH + "*.pdf")
    	pdf_index = {}

    	for filename in pdf_filenames_all:
            
            pmids = re.search("([1-9][0-9]*)\.pdf", filename
            # pmids = re.search("_([1-9][0-9]*)\.pdf", filename) # uncomment this to retrive just the initial 2,200 or so PDFs (ignore the new ones)


            if pmids:
    			pdf_index[pmids.group(1)] = filename
    	return pdf_index


    def second_view(self, study, cachepath=os.path.join(DATA_PATH, "cache")):
        """ overrides code which gets pubmed abstract
        and instead returns the full text of an associated PDF"""
        try:

            # try to read first as plain text from the cache if exists
            with open(os.path.join(cachepath,  os.path.splitext(os.path.basename(self.pdf_index[study['pmid']]))[0] + '.txt'), 'rb') as f:
                text = f.read()

            return {"text": text, "pmid": study['pmid']}
        except:
            # otherwise run through pdftotext
            pm = PdfReader(self.pdf_index[study['pmid']])
            return {"text": pm.get_text(), "pmid": study['pmid']}

    def cache_pdfs(self, cachepath=os.path.join(DATA_PATH, "cache"), refresh=False):
        if not os.path.exists(cachepath):
            os.makedirs(cachepath)
        
        all_pdfs = set(os.path.splitext(os.path.basename(self.pdf_index[entry['pmid']]))[0] for entry in self.index_data)

        if not refresh:
            already_done = set(os.path.splitext(os.path.basename(filename))[0] for filename in glob(os.path.join(cachepath, "*.txt")))
            todo = list(all_pdfs - already_done)
        else:
            todo = list(all_pdfs)

        if not todo:
            print "cache up to date"
        else:
            pb = ProgressBar(len(todo), timer=True)

        for pdf_filename in todo:
            
            pb.tap()



            pm = PdfReader(os.path.join(PDF_PATH, pdf_filename + '.pdf'))
            text = pm.get_text()

            with open(os.path.join(cachepath, pdf_filename + '.txt'), 'wb') as f:
                f.write(text)




    

    
def main():
    # example
    # set paths in CNLP.INI before running



    bv = BiViewer()

    # bv[corpus_study_index][0=cochrane;1=pubmed][part_to_retrieve]

    print "Title:"
    print bv[0].pubmed["title"] # print the pubmed title of the first study
    print
    print "Abstract:"
    print bv[0].pubmed["abstract"] # find out the pubmed abstract of the first study
    print
    print "Intervention description in Cochrane:"
    print bv[0].cochrane["CHAR_INTERVENTIONS"]

    # the biviewer essentially returns a list of named tuples (cochrane, pubmed), with cochrane and pubmed being dicts
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
    print study.pubmed["title"]
    print

    print "Risk of bias"
    print
    for i, domain in enumerate(study.cochrane["QUALITY"]):
        print "Domain number %d" % (i, )
        print "Name\t\t" + domain["DOMAIN"]
        print "Description\t" + domain["DESCRIPTION"]
        print "Rating\t\t" + domain["RATING"]







    
if __name__ == '__main__':
    main()