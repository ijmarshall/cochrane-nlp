cochrane-nlp
============

files for systematic review automation project

bilearn.py
----------
the main algorithm for the co-training/distant supervision

pipeline.py
-----------
does the NLP stuff, takes in text and outputs dicts of features
which can be used by sklearn algorithms

indexnumbers.py
---------------
some code I wrote for something else, but contains a function
to efficiently convert numbers in words to integers
not perfect (won't handle big ordinals correctly yet, e.g. "one hundred and second" is converted too "100 and second" and probably other things too)

pmlib.py
--------
Manages eutils connections and batch downloads

rm5reader.py
------------
Parses Cochrane review XML files

pmreader.py
-----------
Parses pubmed eutils XML output

biviewer.py
-----------
Contains BiViewer class which manages use of the parallel corpora
by creating a 'list' of (cochrane, pubmed) tuples
which can be accessed from memory or disk

data/biviewer\_links\_all.pck
-----------------------------
Pickle file containing linkage data between Cochrane and Pubmed
in the format (used by biviewer.py):

    [{"CDSRfilename": cdsr_filename1, "CDSRrefcode": cdsr_refcode1A, "PMfilename": pm_filename1B},
    {"CDSRfilename": cdsr_filename1, "CDSRrefcode": cdsr_refcode1B, "PMfilename": pm_filename1B},
    {"CDSRfilename": cdsr_filename1, "CDSRrefcode": cdsr_refcode1C, "PMfilename": pm_filename1C},
    {"CDSRfilename": cdsr_filename2, "CDSRrefcode": cdsr_refcode2A, "PMfilename": pm_filename2B},
     ...
    ]

data/test\_abstracts.pck
------------------------
Pickle file containing 137 abstracts with population size manually tagged,
as a list of dicts:

    [{"test": abstract1_as_str,
      "answer": population_size1_as_int},
      {"test": abstract2_as_str,
      "answer": population_size2_as_int},
      ...
    ]

data/brill\_pos\_tagger.pck
---------------------------
Brill POS tagger from NLTK for temporary, to be replaced by CRFsuite version trained on medpost corpus soon...

