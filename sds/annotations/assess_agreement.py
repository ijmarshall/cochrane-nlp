
import pdb
import csv

import nltk 
from nltk.metrics import AnnotationTask

def get_brians_lbls():
    # list of (coder, example id, value) triples
    return get_label_triples("brian_labels.csv", "brian", 0)

def get_aakash_lbls():
    return get_label_triples("for_labeling_sharma.csv", "aakash", 4)

def get_label_triples(path, labeler_str, label_index):
    label_triples = []

    # note that we are assuming that rows are the same
    # between files!!!
    with open(path, 'rU') as csv_file:
        next(csv_file) # skip headers
        reader = csv.reader(csv_file)

        for i, row in enumerate(reader):
            #pdb.set_trace()
            lbl = row[label_index].strip()
            # this is to feed to the AnnotationTask
            # class eventually; see: 
            # http://www.nltk.org/_modules/nltk/metrics/agreement.html
            triple = (labeler_str, str(i), lbl)
            label_triples.append(triple)

    return label_triples

def _simple_merge_metric(lbl1, lbl2):
    ''' '''
    if lbl1 == lbl2 or all(
            [lbl in ("1", "2") for lbl in [lbl1, lbl2]]):
        return 0
    else:
        # then lbl1 != lbl2 and one of them is a 0.
        return 1

def _twos_v_all_metric(lbl1, lbl2):
    ''' '''
    if lbl1 == lbl2 or not any(
        [lbl == "2" for lbl in [lbl1, lbl2]]):
        return 0
    else:
        return 1


def calc_agreement():
    brian = get_brians_lbls()
    print "\nnumber of labels from brian: %s" % len(brian)
    aakash = get_aakash_lbls()
    print "number of labels from aakash: %s" % len(aakash)

    all_triples = brian + aakash

    task = AnnotationTask(all_triples)

    # @TODO do not treat disagreement between 2s and 1s 
    #   the same as 0s and 1s... we can pass in a distant
    #   metric to AnnotationTask that has to return a 
    #   value between 0 and 1 for labels -- but not sure
    #   what's appropriate here..
    print "\n -- simple average overall agreement --"
    print task.avg_Ao()

    print "\n -- kappa --"
    print task.kappa()

    #### what about if we treat 1s and 2s as agreements?
    task = AnnotationTask(all_triples, distance=_simple_merge_metric)
    print "\nok, now grouping.."

    print "\n -- average overall agreement, grouping 1s and 2s --"
    print task.avg_Ao()

    print "\n -- and kappa (with grouping) --"
    print task.kappa()

    #### and if we only pay attention to whether there is 
    #### agreement on 2's?
    task = AnnotationTask(all_triples, distance=_twos_v_all_metric)
    print "\nok, now 2s v all.."

    print "\n -- average overall agreement, 2s v all --"
    print task.avg_Ao()

    print "\n -- and kappa (2s v all) --"
    print task.kappa()

if __name__ == '__main__':
    calc_agreement()
