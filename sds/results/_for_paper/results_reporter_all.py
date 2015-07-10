import csv
import pdb
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import numpy as np 
import scipy as sp

import statsmodels
from statsmodels.formula.api import logit
import statsmodels.stats.proportion as smprop
from pandas import DataFrame 
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import ratings_analysis 

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.stats.proportion as smprop

fields = ["CHAR_PARTICIPANTS", "CHAR_INTERVENTIONS", "CHAR_OUTCOMES"]

_neg_to_0 = lambda x : 0 if x <= 0 else 1.0


def all_individual_runs_results(sent_results_path, k=3):
    for domain in fields:
        individual_runs(sent_results_path, domain=domain, k=k)

# e.g., 
# sds_study_scores1, sds_study_scores2, sds_sorted_results = results_reporter.individual_runs("1433386342-results-sds_all_sentence_scores.txt")
def individual_runs(sent_results_path, domain="CHAR_PARTICIPANTS", k=3):
    study_scores1, study_scores2 = [], []
    sorted_results = []

    n, N = 0, 0
    per_study_results = _split_by_pmids(sent_results_path, target_domain=domain)
    for pmid, results in per_study_results.items():
        sorted_results_for_pmid = results.sort("raw pred", ascending=False)
        #pdb.set_trace()
        #sorted_results[pmid] = sorted_results_for_pmid
        

        #sorted_results.append(sorted_results_for_pmid)

        #sorted_results_for_pmid[:k]["lbl1"]
        #cur_l1_lbls = sorted_results_for_pmid[:k]["lbl1"].values
        #cur_l1_lbls = cur_l1_lbls.clip(0)
        #study_scores1.extend(cur_l1_lbls)
        #pdb.set_trace()
        
        #n += sum(cur_l1_lbls)
        #N += k
        l1_exists = _neg_to_0(sorted_results_for_pmid[:k]["lbl1"].max())
        l2_exists = _neg_to_0(sorted_results_for_pmid[:k]["lbl2"].max())
        #pdb.set_trace()
        # convert to 0/1
        l1_exists = max(l1_exists, 0)
        #study_scores1.append(l1_exists)

        #l2_exists = max(l2_exists, 0)
        #study_scores2.append(l2_exists)

        if l1_exists > 0:
            n += 1
        N += 1
    # yo!
    lower, upper = smprop.proportion_confint(n,N)
    #point_est = sum(study_scores1)/float(len(study_scores1))
    point_est = float(n)/float(N)


    print "%s results for %s" % (domain, sent_results_path)
    print "%s l1: %s (%s, %s)" % (domain, point_est, lower, upper)
    #print "l2: %s" % (sum(study_scores2)/float(len(study_scores2)))
    return study_scores1, study_scores2, sorted_results



def _est_and_l_u(sent_results_path, domain,k):
    n, N = 0, 0
    per_study_results = _split_by_pmids(sent_results_path, target_domain=domain)
    for pmid, results in per_study_results.items():
        sorted_results_for_pmid = results.sort("raw pred", ascending=False)
        l1_exists = _neg_to_0(sorted_results_for_pmid[:k]["lbl1"].max())

        if l1_exists > 0:
            n += 1
        N += 1
    # yo!
    lower, upper = smprop.proportion_confint(n,N)
    #point_est = sum(study_scores1)/float(len(study_scores1))
    point_est = float(n)/float(N)   
    return point_est, lower, upper 

def all_individual_results():
    individual_runs_all_methods("CHAR_PARTICIPANTS", k=10, lower_x=.8)
    individual_runs_all_methods("CHAR_PARTICIPANTS", k=3, upper_x=.85)

    individual_runs_all_methods("CHAR_INTERVENTIONS", lower_x=.8, k=10)
    individual_runs_all_methods("CHAR_INTERVENTIONS", k=3, lower_x=.5, upper_x=.9)

    individual_runs_all_methods("CHAR_OUTCOMES", k=10,lower_x=.5, upper_x=.9)
    individual_runs_all_methods("CHAR_OUTCOMES", k=3,lower_x=.15, upper_x=.7)



###
# @TODO 
# build/output a TeX table with all results (including AUC averages and top-3, top-10
# estimates and intervals)
####
def kitchen_sink():
    # ---- DS ---- #

    '''
    point estimates and CIs for different ks
    '''
    p_10 = individual_runs_all_methods("CHAR_PARTICIPANTS", k=10, lower_x=.8)
    p_3 = individual_runs_all_methods("CHAR_PARTICIPANTS", k=3, upper_x=.85)

    i_10 = individual_runs_all_methods("CHAR_INTERVENTIONS", lower_x=.8, k=10)
    i_3 = individual_runs_all_methods("CHAR_INTERVENTIONS", k=3, lower_x=.6, upper_x=.9)

    o_10 = individual_runs_all_methods("CHAR_OUTCOMES", k=10,lower_x=.6, upper_x=.9)
    o_3 = individual_runs_all_methods("CHAR_OUTCOMES", k=3,lower_x=.15, upper_x=.7)

    
    '''
    AUC stuff 
    '''
    pylab.clf()

    sns.set_style("white")

    fpath_n = "1435156741-results-nguyen_all_sentence_scores.txt"
    fpath_sds = "1435172852-results-sds_all_sentence_scores.txt"
    fpath_direct = "1436271841-results-direct_all_sentence_scores.txt"
    fpath_bp = "1435953510-results-baseline_plus_DS_all_sentence_scores.txt"

    aucs_meta_d = {}
    for domain in fields:
        aucs_n, fprs_n, tprs_n = aucs_per_pmid(fpath_n, domain=domain, lbl_str="lbl1")
        aucs_n = np.array(aucs_n)

        aucs_sds, fprs_sds, tprs_sds = aucs_per_pmid(fpath_sds, domain=domain, lbl_str="lbl1")
        aucs_sds = np.array(aucs_sds)
        
        aucs_d, fprs_d, tprs_d = aucs_per_pmid(fpath_direct, domain=domain, lbl_str="lbl1")
        aucs_d = np.array(aucs_d)

        # plus because it used supervision where available (just overwriting labels)
        aucs_bp, fprs_bp, tprs_bp = aucs_per_pmid(fpath_bp, domain=domain, lbl_str="lbl1")
        aucs_bp = np.array(aucs_bp)

        aucs_meta_d[domain] = draw_auc_stuff(aucs_n, aucs_sds, aucs_bp, aucs_d, name="DS-AUC-%s" % domain)


    '''
    @TODO now build your table!

            method | average AUC | top-3 est (lower, upper) | top-10 est (lower, upper)
            ----------------------------------------------------------------------------
            DS     | .83         | ...
            Nguyen | .85         |
    '''
    latex_str = [r"\begin{table} \centering \begin{tabular} { c c c c } {\bf method} & {\bf mean AUC} & {\bf top-3 mean (CI)} & {\bf top-10 mean (CI)} \\ \hline \multicolumn{4}{c}{\it Population} \\"]
    # so population first
    cur_domain = "CHAR_PARTICIPANTS"

    method_order = ['Direct only', 'DS', 'Nguyen', 'SDS']

    for method in method_order:
        
        est3, lower3, upper3 = p_3[method]
        est10, lower10, upper10 = p_10[method]
        
        cur_tex_str = r" & ".join([method, "%0.3f" % aucs_meta_d[cur_domain][method], 
                                "%0.3f (%0.3f, %0.3f)" % (est3, lower3, upper3),
                                "%0.3f (%0.3f, %0.3f) \\\\" % (est10, lower10, upper10)])

        #pdb.set_trace()
        latex_str.append(cur_tex_str)


    latex_str.append(r"\multicolumn{4}{c}{\it Interventions} \\")

    # now interventions
    # so population first
    cur_domain = "CHAR_INTERVENTIONS"
    for method in method_order:
        
        est3, lower3, upper3 = i_3[method]
        est10, lower10, upper10 = i_10[method]
        
        cur_tex_str = r" & ".join([method, "%0.3f" % aucs_meta_d[cur_domain][method], 
                                "%0.3f (%0.3f, %0.3f)" % (est3, lower3, upper3),
                                "%0.3f (%0.3f, %0.3f) \\\\" % (est10, lower10, upper10)])

        #pdb.set_trace()
        latex_str.append(cur_tex_str)

    latex_str.append(r"\multicolumn{4}{c}{\it Outcomes} \\")
    cur_domain = "CHAR_OUTCOMES"
    for method in method_order:
        
        est3, lower3, upper3 = o_3[method]
        est10, lower10, upper10 = o_10[method]
        
        cur_tex_str = r" & ".join([method, "%0.3f" % aucs_meta_d[cur_domain][method], 
                                "%0.3f (%0.3f, %0.3f)" % (est3, lower3, upper3),
                                "%0.3f (%0.3f, %0.3f) \\\\" % (est10, lower10, upper10)])

        #pdb.set_trace()

        latex_str.append(cur_tex_str)


    latex_str.append(r"\end{tabular} \end{table}")

    print "here it comes!"
    #pdb.set_trace()
    print "\n".join(latex_str)


'''

6/6/2015

results_reporter.individual_runs_all_methods("CHAR_PARTICIPANTS", k=10, lower_x=.8)
results_reporter.individual_runs_all_methods("CHAR_PARTICIPANTS", k=3, upper_x=.85)

results_reporter.individual_runs_all_methods("CHAR_INTERVENTIONS", lower_x=.8, k=10)
results_reporter.individual_runs_all_methods("CHAR_INTERVENTIONS", k=3, lower_x=.6, upper_x=.9)

results_reporter.individual_runs_all_methods("CHAR_OUTCOMES", k=10,lower_x=.6, upper_x=.9)
results_reporter.individual_runs_all_methods("CHAR_OUTCOMES", k=3,lower_x=.15, upper_x=.7)



'''
# e.g., 
# sds_study_scores1, sds_study_scores2, sds_sorted_results = results_reporter.individual_runs("1433386342-results-sds_all_sentence_scores.txt")
def individual_runs_all_methods(domain="CHAR_PARTICIPANTS", lower_x=.5, upper_x=1,
        nguyen_fpath='1435156741-results-nguyen_all_sentence_scores.txt',
        sds_fpath="1435172852-results-sds_all_sentence_scores.txt",
        baseline_fpath="1435953510-results-baseline_plus_DS_all_sentence_scores.txt", 
        direct_fpath = "1436271841-results-direct_all_sentence_scores.txt",
        k=3, k2=10):
    

    pylab.clf()
    sns.set_style("darkgrid")
    ys = [2.25,1.75,1.25,.75]

    results_d = {}
    methods_names = ["Direct only", "DS", "Nguyen", "SDS"]
    colors = ["green", "gray", "red", "blue"]
    offset = 0#.05
    for i, sent_results_path in enumerate([direct_fpath, baseline_fpath, nguyen_fpath, sds_fpath]):

        point_est, lower, upper = _est_and_l_u(sent_results_path, domain, k)
        print "%s results for %s" % (domain, methods_names[i])
        print "%s l1: %s (%s, %s)" % (domain, point_est, lower, upper)

        results_d[methods_names[i]] = (point_est, lower, upper)

        pylab.plot([lower, upper], [ys[i]+offset, ys[i]+offset], alpha=.5, color=colors[i])
        pylab.scatter([point_est], [ys[i]+offset], color=colors[i], s=200, alpha=.75)

        '''
        point_est2, lower2, upper2 = _est_and_l_u(sent_results_path, domain, k2)
        print "%s 2: %s (%s, %s)" % (domain, point_est2, lower2, upper2)

        pylab.plot([lower2, upper2], [ys[i]-offset, ys[i]-offset], alpha=.5, color=colors[i])
        pylab.scatter([point_est2], [ys[i]-offset], color=colors[i], marker="^", s=200, alpha=.75)
        '''

    ylocs = ys
    ylabels = methods_names
 
    pylab.ylim((0.5,2.5))
    pylab.xlim((lower_x, upper_x))

    locs, labels = pylab.yticks(ylocs, ylabels)
    # 7/9 no label?
    #pylab.xlabel(r"proportion of articles for which at least one sentence rated $\geq 1$ ranked among the top $k=%s$" % k)
    prettify_canvas(turn_off_y=False)
    pylab.savefig("ds-results-%s-%s.pdf" % (domain, k))

    return results_d


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

def metrics_redux(fpath, domain="CHAR_PARTICIPANTS", k=3):
    #lbls, lbls2, scores, pmids = _read_lbls_and_scores(fpath, target_domain=domain)
    by_pmid = _split_by_pmids(fpath, target_domain=domain)
    at_least_1, at_least_2 = [], []
    for pmid, df in by_pmid.items():
        if df[lbl_str].max() == -1:
            # this means there were no positive instances
            # for this example! so we just ignore.
            pass 
        else:
            pass 

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
# fpath_baseline = "'1435953510-results-baseline_DS_all_sentence_scores.txt'"
def draw_aucs_per_pmid(fpath_sds, fpath_n, fpath_baseline, domain="CHAR_PARTICIPANTS", lbl_str="lbl1",
                        marker_n="x", marker_sds="o", color_n="black", color_sds="blue"):
    
    
    print "assuming path to SDS is: %s and path to nguyen is: %s" % (fpath_sds, fpath_n)

    aucs_sds, fprs_sds, tprs_sds = aucs_per_pmid(fpath_sds, domain=domain, lbl_str=lbl_str)
    aucs_n, fprs_n, tprs_n = aucs_per_pmid(fpath_n, domain=domain, lbl_str=lbl_str)

    assert len(aucs_sds) == len(fprs_sds) == len(fpath_baseline)

    #plt.clf()
    #plt.figure()

    points_sds, points_n, points_baseline = [], [], []

    #pdb.set_trace()
    for i in xrange(len(aucs_sds)):
        points_sds.extend(zip(fprs_sds[i], tprs_sds[i]))
        points_n.extend(zip(fprs_n[i], tprs_n[i]))

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
    aucs_n, fprs_n, tprs_n = aucs_per_pmid(fpath_n, domain=domain, lbl_str="lbl1")
    aucs_n = np.array(aucs_n)

    fpath_sds = "1435172852-results-sds_all_sentence_scores.txt"
    aucs_sds, fprs_sds, tprs_sds = aucs_per_pmid(fpath_sds, domain=domain, lbl_str="lbl1")
    aucs_sds = np.array(aucs_sds)


    '''
    fpath_b = "1436189976-results-baseline_DS_all_sentence_scores.txt"
    aucs_b, fprs_b, tprs_b = aucs_per_pmid(fpath_b, domain=domain, lbl_str="lbl1")
    aucs_b = np.array(aucs_b)
    '''
    fpath_direct = "1436271841-results-direct_all_sentence_scores.txt"
    aucs_d, fprs_d, tprs_d = aucs_per_pmid(fpath_direct, domain=domain, lbl_str="lbl1")
    aucs_d = np.array(aucs_d)

    # plus because it used supervision where available (just overwriting labels)
    fpath_bp = "1435953510-results-baseline_plus_DS_all_sentence_scores.txt"
    aucs_bp, fprs_bp, tprs_bp = aucs_per_pmid(fpath_bp, domain=domain, lbl_str="lbl1")
    aucs_bp = np.array(aucs_bp)

    pretty_str = "-"*50
    print "\n".join([pretty_str, "t-test (sds v baseline)"])
    print sp.stats.ttest_rel(aucs_sds, aucs_bp)[-1]
    print "\n".join(["", "wilcoxon (sds v baseline)"])
    print sp.stats.wilcoxon(aucs_sds, aucs_bp)[-1]


    print "\n".join([pretty_str, "t-test (sds v nguyen)"])
    print sp.stats.ttest_rel(aucs_sds, aucs_n)[-1]
    print "\n".join(["", "wilcoxon (sds v nguyen)"])
    print sp.stats.wilcoxon(aucs_sds, aucs_n)[-1]

    #pdb.set_trace()
    draw_auc_stuff(aucs_n, aucs_sds, aucs_bp, aucs_d, name=domain)

def prettify_canvas(big_font=True, turn_off_y=False):
    ax1 = pylab.axes()
    if big_font:
        pylab.setp(ax1.get_xticklabels(), size=16)
        pylab.setp(ax1.get_yticklabels(), size=18)
        pylab.setp(ax1.xaxis.label, size=16)
        pylab.setp(ax1.yaxis.label, size=18)

    if turn_off_y:
        pylab.axes().get_yaxis().set_visible(False)

    #pylab.rcParams['font.family'] = 'sans-serif'
    #pylab.rcParams['font.sans-serif'] = ['Helvetica']
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    pylab.axes().xaxis
    pylab.axes().yaxis
    pylab.tight_layout()

def draw_auc_stuff(aucs_n, aucs_sds, aucs_bp, aucs_d, name="yo", add_percentiles=False, alpha_lvl = .5):
    pylab.clf()
    prettify_canvas(big_font=True)
    
    auc_d = {"Direct only":np.mean(aucs_d),
             "DS":np.mean(aucs_bp),
             "Nguyen":np.mean(aucs_n),
             "SDS":np.mean(aucs_sds)}

    direct_dens = sns.kdeplot(np.array(aucs_d), shade=True, color="green", label="Direct only (mean AUC {0:.3f})".format(np.mean(aucs_d)), legend=True)

    ds_dens = sns.kdeplot(np.array(aucs_bp), shade=True, color="gray", label="DS (mean AUC {0:.3f})".format(np.mean(aucs_bp)), legend=True)

    n_dens = sns.kdeplot(np.array(aucs_n), shade=True, color="red", label="Nguyen (mean AUC {0:.3f})".format(np.mean(aucs_n)), legend=True)

    sds_dens = sns.kdeplot(np.array(aucs_sds), shade=True, color="blue", label="SDS (mean AUC {0:.3f})".format(np.mean(aucs_sds)), legend=True)

  
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
    #pylab.title(pretty_name)
    pylab.xlim([.5,1])
    pylab.yticks([])
    #pylab.legend([ds_dens, n_dens, sds_dens], ["DS", "Nguyen", "SDS"])
    pylab.legend(loc="upper left", prop={'size':18})
    prettify_canvas()
    pylab.savefig("DS-AUCs-2" + pretty_name+".pdf")

    return auc_d

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
