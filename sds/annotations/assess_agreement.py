
import pdb
import csv

import nltk 
from nltk.metrics import AnnotationTask

def get_brians_lbls():
    # list of (coder, example id, value) triples
    return get_label_triples("sharma__and_brian_ratings_5.csv", "brian", -1, start_index=276)
    #return get_label_triples("rating compare3.csv", "brian", 11, start_index=42)
    #return get_label_triples("rating_compare_2.csv", "brian", 6)
    #return get_label_triples("brian_labels.csv", "brian", 0)

def get_aakash_lbls():
    return get_label_triples("sharma__and_brian_ratings_5.csv", "aakash", -2, start_index=276)
    #return get_label_triples("for_labeling_sharma_round2.csv", "aakash", 4, start_index=276)
    #return get_label_triples("rating compare3.csv", "aakash", 10, start_index=42)
    #return get_label_triples("for_labeling_sharma.csv", "aakash", 4)

def get_label_triples(path, labeler_str, label_index, start_index=0):
    print "reading labels at %s for %s" % (path, labeler_str)
    label_triples = []

    # note that we are assuming that rows are the same
    # between files!!!
    with open(path, 'rU') as csv_file:
        next(csv_file) # skip headers
        reader = csv.reader(csv_file)

        for i, row in enumerate(reader):
            print i
            #pdb.set_trace()
            if i >= start_index:
                lbl = row[label_index].strip()
                # this is to feed to the AnnotationTask
                # class eventually; see: 
                # http://www.nltk.org/_modules/nltk/metrics/agreement.html
                triple = (labeler_str, str(i), lbl)
                label_triples.append(triple)

    return label_triples

'''
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
'''

def _group_0s_and_1s(tuples):
    new_tuples = []
    for t in tuples:
        if t[-1] in ("0", "1", "t0", "t1"):
            new_tuples.append(t[:2]+("0",))
        else:
            new_tuples.append(t)
    return new_tuples

def _group_1s_and_2s(tuples):
    new_tuples = []
    for t in tuples:
        if t[-1] in ("1", "2"):
            new_tuples.append(t[:2]+("1",))
        else:
            new_tuples.append(t[:2]+("0",))
    return new_tuples


def calc_agreement():
    brian = get_brians_lbls()
    print "\nnumber of labels from brian: %s" % len(brian)
    aakash = get_aakash_lbls()
    print "number of labels from aakash: %s" % len(aakash)

    #pdb.set_trace()

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
    #task = AnnotationTask(all_triples, distance=_simple_merge_metric)
    print "\nok, now grouping.."
    brian_0s_v_all = _group_1s_and_2s(brian)
    aakash_0s_v_all = _group_1s_and_2s(aakash)
    triples_0s_v_all = brian_0s_v_all + aakash_0s_v_all 
    task = AnnotationTask(triples_0s_v_all)
    print "\n -- average overall agreement, grouping 1s and 2s --"
    print task.avg_Ao()

    print "\n -- and kappa (with grouping) --"
    print task.kappa()

    #### and if we only pay attention to whether there is 
    #### agreement on 2's?
    #task = AnnotationTask(all_triples, distance=_twos_v_all_metric)
    brian_2s_v_all = _group_0s_and_1s(brian)
    aakash_2s_v_all = _group_0s_and_1s(aakash)
    triples_2s_v_all = brian_2s_v_all + aakash_2s_v_all
    task = AnnotationTask(triples_2s_v_all)

    print "\nok, now 2s v all.."

    print "\n -- average overall agreement, 2s v all --"
    print task.avg_Ao()

    print "\n -- and kappa (2s v all) --"
    print task.kappa()

if __name__ == '__main__':
    calc_agreement()

