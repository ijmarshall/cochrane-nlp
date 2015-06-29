import csv
import pdb
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

import numpy as np 
import scipy as sp

from sklearn import metrics

import matplotlib.pyplot as plt
import pylab
import seaborn as sns
sns.set_style("whitegrid")

fields = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

_neg_to_0 = lambda x : 0 if x <= 0 else 1.0


# e.g., 
# sds_study_scores1, sds_study_scores2, sds_sorted_results = results_reporter.individual_runs("1433386342-results-sds_all_sentence_scores.txt")
def individual_runs(sent_results_path):
    study_scores1, study_scores2 = [], []
    sorted_results = []

    per_study_results = _split_by_pmids(sent_results_path)
    for pmid, results in per_study_results.items():
        sorted_results_for_pmid = results.sort("raw pred", ascending=False)
        #pdb.set_trace()
        #sorted_results[pmid] = sorted_results_for_pmid
        sorted_results.append(sorted_results_for_pmid)
        l1_exists = _neg_to_0(sorted_results_for_pmid[:3]["lbl1"].max())
        l2_exists = _neg_to_0(sorted_results_for_pmid[:3]["lbl2"].max())

        study_scores1.append(l1_exists)
        study_scores2.append(l2_exists)

    return study_scores1, study_scores2, sorted_results


def dump_false_positives(sent_results_path, lbl_col="lbl1"):
    per_study_results = _split_by_pmids(sent_results_path)
    false_positives_str = [["pmid", "sentence"]]

    for pmid, results in per_study_results.items():
        sorted_results_for_pmid = results.sort("raw pred", ascending=False)
        indices = sorted_results_for_pmid.index
        # only  output sentences for entire studies with '0' metric!
        if sorted_results_for_pmid[lbl_col][:3].max() < 1: 
            for i in indices[:3]:                  
                false_positives_str.append(
                    ["%s" % sorted_results_for_pmid["pmid"][i],
                     "%s" % sorted_results_for_pmid["sentence"][i]])

    with open(sent_results_path.replace(".txt", "_false_positives.csv"), 'wb') as outf:
        csv_w = csv.writer(outf)    
        csv_w.writerows(false_positives_str)

def _split_by_pmids(sent_results_path, target_domain="CHAR_PARTICIPANTS"):
    all_sent_scores = pd.read_csv(sent_results_path)
    all_sent_scores = all_sent_scores[all_sent_scores["domain"] == target_domain]
    grouped = all_sent_scores.groupby("pmid")
    by_pmid = dict(list(grouped))
    return by_pmid


def generate_plots(nguyen_path, sds_path, domain="CHAR_PARTICIPANTS"):
    lbls_n, lbls2_n, scores_n, pmids_nguyen = _read_lbls_and_scores(nguyen_path, target_domain=domain)
    fpr_n, tpr_n, thresholds_n = metrics.roc_curve(lbls_n, scores_n)
    auc_n =  metrics.auc(fpr_n, tpr_n)

    lbls_sds, lbls2_sds, scores_sds, pmids_sds = _read_lbls_and_scores(sds_path, target_domain=domain)
    fpr_sds, tpr_sds, thresholds_sds = metrics.roc_curve(lbls_sds, scores_sds)
    auc_sds =  metrics.auc(fpr_sds, tpr_sds)

    ### 
    # ok, plotting time
    plt.figure()
    plt.plot(fpr_n, tpr_n, label='Nguyen (area = %0.2f)' % auc_n)
    plt.plot(fpr_sds, tpr_sds, label='SDS (area = %0.2f)' % auc_sds)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s ROCs (for level 1 relevance)' % domain)
    plt.savefig("%s_ROCs.pdf" % domain)

'''
I wonder if here you shouldn't do AUCs *per article*
'''

# e.g., fpath="1433334082-results-sds_all_sentence_scores.txt"
def AUCs(fpath, domain="CHAR_PARTICIPANTS"):
    lbls, lbls2, scores, pmids = _read_lbls_and_scores(fpath, target_domain=domain)
    fpr, tpr, thresholds = metrics.roc_curve(lbls, scores)
    auc =  metrics.auc(fpr, tpr)
    print "auc 1 (%s): %s" % (domain, auc)

    fpr, tpr, thresholds = metrics.roc_curve(lbls2, scores)
    auc =  metrics.auc(fpr, tpr)
    print "auc 2 (%s): %s" % (domain, auc)

# fpath_sds = "1435172852-results-sds_all_sentence_scores.txt"
# fpath_n = "1435156741-results-nguyen_all_sentence_scores.txt"
def draw_aucs_per_pmid(fpath_sds, fpath_n, domain="CHAR_PARTICIPANTS", lbl_str="lbl1",
                        marker_n="x", marker_sds="o", color_n="black", color_sds="blue"):
    print "assuming path to SDS is: %s and path to nguyen is: %s" % (fpath_sds, fpath_n)

    aucs_sds, fprs_sds, tprs_sds = aucs_per_pmid(fpath_sds, domain=domain, lbl_str=lbl_str)
    aucs_n, fprs_n, tprs_n = aucs_per_pmid(fpath_n, domain=domain, lbl_str=lbl_str)

    assert len(aucs_sds) == len(fprs_sds)

    #plt.clf()
    #plt.figure()

    points_sds, points_n = [], []

    #pdb.set_trace()
    for i in xrange(len(aucs_sds)):
        points_sds.extend(zip(fprs_sds[i], tprs_sds[i]))
        points_n.extend(zip(fprs_sds[i], tprs_sds[i]))

        #if i == 0:
        #plt.plot(fprs_n, tprs_n), label='Nguyen (area = %0.2f)' % auc_n)
        #plt.plot(fprs_n[i], tprs_n[i], c=color_n, marker=marker_n)#label='Nguyen (area = %0.2f)' % auc_n)
        #plt.plot(fprs_sds[i], tprs_sds[i], c=color_sds, marker=marker_sds)#label='SDS (area = %0.2f)' % auc_sds)

    #lbls_sds, lbls2_sds, scores_sds, pmids_sds = _read_lbls_and_scores(sds_path, target_domain=domain)
    #fpr_sds, tpr_sds, thresholds_sds = metrics.roc_curve(lbls_sds, scores_sds)
    #auc_sds =  metrics.auc(fpr_sds, tpr_sds)
    #plt.savefig("test.pdf")
    #for auc, fpr, tpr in zip(aucs, fprs, tprs):
    return points_sds, points_n


def all_results():
    for d in ["CHAR_PARTICIPANTS", "CHAR_OUTCOMES", "CHAR_INTERVENTIONS"]:
        print "*"*20 + d + "*"*20

        sig_test_etc(domain=d)
        print "\n\n"



def sig_test_etc(domain="CHAR_OUTCOMES"):
    fpath_n = "1435156741-results-nguyen_all_sentence_scores.txt"
    aucs_n, fprs_sds, tprs_sds = aucs_per_pmid(fpath_n, domain=domain, lbl_str="lbl1")
    aucs_n = np.array(aucs_n)

    fpath_sds = "1435172852-results-sds_all_sentence_scores.txt"
    aucs_sds, fprs_sds, tprs_sds = aucs_per_pmid(fpath_sds, domain=domain, lbl_str="lbl1")
    aucs_sds = np.array(aucs_sds)

    #pdb.set_trace()

    pretty_str = "-"*50
    print "\n".join([pretty_str, "t-test"])
    print sp.stats.ttest_rel(aucs_sds, aucs_n)[-1]
    print "\n".join([pretty_str, "wilcoxon"])
    print sp.stats.wilcoxon(aucs_sds, aucs_n)[-1]

    draw_auc_stuff(aucs_n, aucs_sds, name=domain)

def prettify_canvas():
    ax1 = pylab.axes()
    pylab.setp(ax1.get_xticklabels(), size=14)
    pylab.setp(ax1.get_yticklabels(), size=14)
    pylab.setp(ax1.xaxis.label, size=16)
    pylab.setp(ax1.yaxis.label, size=16)
    pylab.rcParams['font.family'] = 'sans-serif'
    pylab.rcParams['font.sans-serif'] = ['Helvetica']
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    pylab.axes().xaxis
    pylab.axes().yaxis

def draw_auc_stuff(aucs_n, aucs_sds, name="yo", add_percentiles=False):
    pylab.clf()
    prettify_canvas()
    sns.kdeplot(np.array(aucs_n), shade=True, color="red")
    sns.kdeplot(np.array(aucs_sds), shade=True, color="blue")

    alpha_lvl = .5
    if add_percentiles:
        pylab.axvline(x=np.median(aucs_sds), color="blue", ls="--", alpha=alpha_lvl)
        pylab.axvline(x=np.percentile(aucs_sds, 20), color="blue", ls="--", alpha=alpha_lvl)
        pylab.axvline(x=np.percentile(aucs_sds, 80), color="blue", ls="--", alpha=alpha_lvl)

        pylab.axvline(x=np.median(aucs_n), color="red", ls="--", alpha=alpha_lvl)
        pylab.axvline(x=np.percentile(aucs_n, 20), color="red", ls="--", alpha=alpha_lvl)
        pylab.axvline(x=np.percentile(aucs_n, 80), color="red", ls="--", alpha=alpha_lvl)

   
    pylab.xlabel("AUC (per article)")
    pylab.ylabel("density")
    
    pretty_name = name.replace("CHAR_", "")
    pretty_name = pretty_name.lower()
    pylab.title(pretty_name)
    pylab.xlim([.5,1])
    pylab.yticks([])
    pylab.savefig(pretty_name+".pdf")

    # paired t-test!
    #sp.stats.ttest_rel( aucs_sds, aucs_n)
    # (array(1.955538662719973), 0.05372893761667645)

# aucs, fprs, tprs = 
def aucs_per_pmid(fpath, domain="CHAR_PARTICIPANTS", lbl_str="lbl1"):
    by_pmid = _split_by_pmids(fpath, target_domain=domain)
    aucs, fprs, tprs = [], [], []
    for pmid, df in by_pmid.items():
        if df[lbl_str].max() == -1:
            # this means there were no positive instances
            # for this example! so we just ignore.
            pass 
        else:
            fpr, tpr, thresholds = metrics.roc_curve(df[lbl_str], df["raw pred"])
            
            auc =  metrics.auc(fpr, tpr)

            aucs.append(auc)
            fprs.append(fpr)
            tprs.append(tpr)
    print "average auc: %s" % (sum(aucs)/float(len(aucs)))
    return aucs, fprs, tprs


        


def _read_lbls_and_scores(fpath, target_domain="CHAR_PARTICIPANTS"):
    lbls, lbls2, scores, pmids = [], [], [], []
    with open(fpath, 'rU') as all_scores_f:
        all_scores = csv.reader(all_scores_f)
        all_scores.next() # headers
        # 6/25: including domain now!
        for domain, pmid, sentence, raw_pred, lbl1, lbl2 in all_scores:
            if domain == target_domain:
                lbls.append(int(lbl1))
                lbls2.append(int(lbl2))
                scores.append(float(raw_pred))    
                pmids.append(pmid)

    return np.array(lbls), np.array(lbls2), np.array(scores), pmids

def average_results(results_file_path, target_variable="at least one (>=1):", 
                        field="CHAR_PARTICIPANTS"):
    results = []
    relevant_field = False
    with open(results_file_path, 'rU') as results_file:
        for l in results_file:
            if "domain: %s" % field in l:
                relevant_field = True
            
            if target_variable in l and (field is None or relevant_field): 
                cur_result = float(l.split(target_variable)[1])
                results.append(cur_result)
                relevant_field = False 
    #pdb.set_trace()
    return results

def make_report(results_file_path, target_variable="at least one (>=1):"):
    out_str = ["results for %s" % results_file_path]
    for field in fields:
        result_set = average_results(results_file_path, field=field, target_variable=target_variable)
        #pdb.set_trace()
        out_str.append("%s -- mean: %s, sd: %s" % (field, np.mean(result_set), np.std(result_set)))
    print "\n".join(out_str)
    #return out_str 

def dump(output, csv_path):
    with open(csv_path, 'wb') as output_f:
        csv_writer = csv.writer(output_f)
        csv_writer.writerows(output)

def gen_output_file(results_file_path, output_to_file=True):
    output_str = [["study id", "domain", "target text", "candidate text"]]
    results_stream = open(results_file_path, 'rb')

    cur_field = None
    cur_study_id = None 

    in_target_text = False 
    cur_target_text = ""

    in_candidate_text = False 
    cur_candidate_text = ""

    n = 0 
    for i,l in enumerate(results_stream.readlines()):
        if "-- domain" in l:
            if n > 0:
                output_str.append([cur_study_id, cur_field, cur_target_text, cur_candidate_text])
            n += 1
            in_candidate_text = False
            in_target_text = False
            cur_candidate_text = ""
            cur_target_text = ""

            if "CHAR_PARTICIPANTS" in l:
                cur_field = "population"
            elif "CHAR_INTERVENTIONS" in l:
                cur_field = "interventions"
            elif "CHAR_OUTCOMES" in l: 
                cur_field = "outcomes"
            else:
                print "wtf field is this???"
                pdb.set_trace()

            # grab the study id
            cur_study_id = l.split("in study ")[-1].split(" --")[0]


        elif "target text" in l: 
            in_target_text = True
            cur_target_text = ""

        elif "candidate sentence" in l:
            in_target_text = False
            in_candidate_text = True
            if cur_candidate_text != "":
                output_str.append([cur_study_id, cur_field, cur_target_text, cur_candidate_text])
            cur_candidate_text = ""

        elif in_target_text:
            cur_target_text += l 

        elif in_candidate_text:
            cur_candidate_text += l


    if output_to_file:
        outpath = results_file_path.replace(".txt", ".csv")
        csv_writer = csv.writer(open(outpath, 'wb'))
        csv_writer.writerows(output_str)

    return output_str
