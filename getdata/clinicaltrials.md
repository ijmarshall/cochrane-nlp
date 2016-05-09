Getting started with the ClinicalTrials.gov BiViewer
====================================================

1.  Copy the ClinicalTrials.gov data folder from our dropbox to your
    local data folder

2.  Add a reference to it to your CNLP.INI file. Mine looks like:

    CLINICAL\_TRIALS\_PATH = /Users/iain/Code/data/clinicaltrials.gov/

3.  Run `python pubmed_from_ct.py` from the `getdata` folder in
    cochranenlp
    -   This will automatically download the PubMed XML of trials with
        published results


