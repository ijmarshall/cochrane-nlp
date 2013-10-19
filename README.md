cochrane-nlp
============

files for systematic review automation project

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
---------------------------
Pickle file containing linkage data between Cochrane and Pubmed
in the format (used by biviewer.py):

    [{"CDSRfilename": cdsr_filename_1,
      "refs": [{"CDSRrefcode": cdsr_refcode1A, "PMfilename": pm_filename1A},
               {"CDSRrefcode": cdsr_refcode1B, "PMfilename": pm_filename1B},
               {"CDSRrefcode": cdsr_refcode1C, "PMfilename": pm_filename1C}]},
     {"CDSRfilename": cdsr_filename_2,
      "refs": [{"CDSRrefcode": cdsr_refcode2A, "PMfilename": pm_filename2A},
               {"CDSRrefcode": cdsr_refcode2B, "PMfilename": pm_filename2B},
               {"CDSRrefcode": cdsr_refcode2C, "PMfilename": pm_filename2C}]}
      ]

