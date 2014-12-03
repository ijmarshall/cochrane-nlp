import pdb
import random
import csv

import numpy as np 
import sklearn

from readers import biviewer

# this module allows us to grab the ranked
# sentences. this is possibly not the ideal 
# location.
from experiments import pico_DS 

'''
Here we aim to go from the information (direct 
distant supervision for PICO task) contained in 
the annotations file to feature vectors and labels for the 
candidate filtering task. In the writeup nomenclature,
this is to generate \tilde{x} and \tilde{y}.
'''
def generate_DS_X_y(annotations_path="sds/annotations/for_labeling_sharma.csv", max_sentences=10, cutoff=4):
    '''
    We make the assumption that the annotations file comprises
    the following fields (in this order!)

        study id,PICO field,CDSR sentence,candidate sentence,rating
    '''
    biview = biviewer.PDFBiViewer() 

    domains = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]
    # this is just to standardize terms/strings
    pico_strs_to_domains = dict(zip(["PARTICIPANTS", "INTERVENTIONS","OUTCOMES"], domains))

    X_y_dict = {}
    for d in domains:
        # X, y for each domain.
        X_y_dict[d] = {"X":[], "y":[]}

    with open(annotations_path, 'rU') as labels_file:
        annotations_reader = csv.reader(labels_file)
        annotations_reader.next() # skip header 
        ###
        # note that the structure of the annotations
        # file means that studies are repeated, and
        # there are multiple annotated sentences
        # *per domain*. 
        for annotation_line in annotations_reader:
            try:
                study_id, PICO_field, target_sentence, \
                    candidate_sentence, label = annotation_line
                PICO_field = pico_strs_to_domains[PICO_field.strip()]
            except:
                pdb.set_trace()


            # get the study from the PMID
            study = biview.get_study_from_pmid(study_id)

            X_i_text = candidate_sentence

            ## numeric features
            # @TODO add more!
            X_i_numeric = []
            X_i_numeric.append(len(candidate_sentence.split(" ")))

            ###
            # This part is kind of hacky. We go ahead and retrieve
            # all the candidate sentences here to derive additional 
            # features that are not otherwise readily available
            # (e.g., the relative rank of the candidate sentence)
            ###
            pdf = study.studypdf['text']
            study_id = "%s" % study[1]['pmid']
            pdf_sents = pico_DS.sent_tokenize(pdf)

            # note that this should never return None, because we would have only
            # written out for labeling studies/fields that had at least one match.
            ranked_sentences, scores, shared_tokens = pico_DS.get_ranked_sentences_for_study_and_field(study, 
                        PICO_field, pdf_sents=pdf_sents)
            
            # don't take more than max_sentences sentences
            num_to_keep = min(len([score for score in scores if score >= cutoff]), max_sentences)

            # TMP TMP TMP
            #candidates1 = list(candidates)
            scores1 = list(scores)
            shared_tokens1 = list(shared_tokens)

            target_text = study.cochrane["CHARACTERISTICS"][PICO_field]
            candidates = ranked_sentences[:num_to_keep]
            scores = scores[:num_to_keep]
            shared_tokens = shared_tokens[:num_to_keep]
            
            try:
                cur_candidate_index = candidates.index(candidate_sentence)
            except:
                pdb.set_trace()
            # shared tokens for this candidate
            cur_shared_tokens = shared_tokens[cur_candidate_index]
            # extend X_i text with shared tokens (using 
            # special indicator prefix "shared_")
            X_i_text = X_i_text + " ".join(["shared_%s" % tok for 
                                            tok in cur_shared_tokens if tok.strip() != ""])

            X_i_numeric.append(len(candidates) - cur_candidate_index)
            candidate_score = scores[cur_candidate_index]
            X_i_numeric.append(candidate_score - np.mean(scores))
            X_i_numeric.append(candidate_score - np.median(scores))
            
            # @TODO add additional features, e.g., difference from next 
            # highest candidate score..


            # note that we'll need to deal with merging these 
            # tesxtual and numeric feature sets elsewhere!
            X_i = (X_i_numeric, X_i_text)
            # @TODO we may want to do something else here
            # with the label (e.g., maybe binarize it?)
            y_i = label
            X_y_dict[PICO_field]["X"].append(X_i)
            X_y_dict[PICO_field]["y"].append(y_i)

    return X_y_dict
            