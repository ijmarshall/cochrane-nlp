import pdb 

import statsmodels
from statsmodels.formula.api import logit


import numpy as np 
import statsmodels.stats.proportion as smprop

from pandas import DataFrame 
import pandas as pd 

import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# cd /Users/byron/dev/cochrane/cochranenlp/sds/annotations/sds-labeling-exp


def simple_analyze_results(methods_path="CHAR_POPULATION-key.csv", 
                            ratings_path="brian-ratings-population.csv",
                            study_ids_path="CHAR_POPULATION-study-ids.csv",
                            title="Population", grouped=True):

    #print result.summary()
    res_d = {}
    M = load_results(methods_path, ratings_path, study_ids_path)
    
    if grouped:
        study_results = M.groupby("study_id")
        study_results_d = dict(list(study_results))
        n_sds, N_sds = 0, 0
        n_n, N_n = 0, 0 

        for study, results in study_results_d.items():
            if results[(results["sds_any"]==1) & (results["y"]==1)].shape[0] > 0:
                n_sds += 1
            else:
                print "nope for sds! (%s)" % study

            if results[(results["nguyen_any"]==1) & (results["y"]==1)].shape[0] > 0:
                n_n += 1
            else:
                print "nope for nguyen! (%s)" % study

            N_n += 1
            N_sds += 1
            #pdb.set_trace()


    else:
        n_sds = M[(M["sds_any"]==1) & (M["y"]==1)].shape[0]
        N_sds = M[(M["sds_any"]==1)].shape[0]

        n_n = M[(M["nguyen_any"]==1) & (M["y"]==1)].shape[0]
        N_n = M[(M["nguyen_any"]==1)].shape[0]

    print N_n
    #pdb.set_trace()
    print "est and CI for sds:"
    lower_sds, upper_sds = smprop.proportion_confint(n_sds,N_sds)
    est_sds = float(n_sds)/N_sds
    print "%s (%s)" % (est_sds, smprop.proportion_confint(n_sds,N_sds))
    res_d["SDS"] = [est_sds, lower_sds, upper_sds]
    


    print "est and CI for nguyen:"
    lower_n, upper_n = smprop.proportion_confint(n_n,N_n)
    est_n = float(n_n)/N_n
    print "%s (%s)" % (est_n, smprop.proportion_confint(n_n,N_n))
    res_d["Nguyen"] = [est_n, lower_n, upper_n]

    pylab.clf()
    sns.set_style("darkgrid")
    y = 1.5
    pylab.plot([lower_sds, upper_sds], [y, y], alpha=.5, color="blue")
    pylab.scatter([est_sds], [y], s=200, alpha=.75)

    pylab.plot([lower_n, upper_n], [1, 1], alpha=.5, color="red") 
    pylab.scatter([est_n], [1], s=200, color="red", alpha=.75)

    ylocs = [1, y]
    ylabels = ["Nguyen", "SDS"]

    locs, labels = pylab.yticks(ylocs, ylabels)
    pylab.ylim((0.5,2))
    if "Population" in title:
        pylab.xlim((.84, 1))
        xticks = [.85, .9, .95, 1.0]
        pylab.xticks(xticks)
    else:
        pylab.xlim((.75, 1))
    

    if grouped:
        pylab.xlabel(r"proportion of articles for which at least one sentence rated $\geq 1$ ranked among the top $k=3$")
        pylab.title("%s" % title)
        pylab.savefig("%s-grouped.pdf" % title)
    else:
        pylab.xlabel("proportion of top-3 sentences deemed relevant")
        pylab.title("%s" % title)
        pylab.savefig("%s-overall.pdf" % title)

    return res_d

def prettify_canvas():
    ax1 = pylab.axes()
    #pylab.setp(ax1.get_xticklabels(), size=14)
    #pylab.setp(ax1.get_yticklabels(), size=14)
    #pylab.setp(ax1.xaxis.label, size=16)
    #pylab.setp(ax1.yaxis.label, size=16)
    #pylab.rcParams['font.family'] = 'sans-serif'
    #pylab.rcParams['font.sans-serif'] = ['Helvetica']
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    pylab.axes().xaxis
    pylab.axes().yaxis

def load_results(methods_path, ratings_path, study_ids_path):
    methods = DataFrame.from_csv(methods_path, index_col=None)

    # ratings
    brian_lbls = DataFrame.from_csv(ratings_path, index_col=None)
    
    # study IDs
    study_ids = DataFrame.from_csv(study_ids_path, index_col=None)

    print "methods path: %s" % methods_path
    print "ratings path: %s" % ratings_path
    print "study IDs path: %s" % study_ids_path

    # merge these
    joined = methods.join(brian_lbls)
    joined = joined.join(study_ids)
    #pdb.set_trace()
    # get indices for nguyen results
    nguyen_values = {"method":["BOTH", "nguyen"]}
    nguyen_all_mask = joined.isin(nguyen_values)['method']
    # all entries where the method was nguyen or BOTH
    nguyen = joined[nguyen_all_mask]

    # and for sds results.
    sds_values = {"method":["BOTH", "sds"]}
    sds_all_mask = joined.isin(sds_values)['method']
    sds = joined[sds_all_mask]

    '''
    assemble design matrix (with dummy coding for methods). 
    this will result in something like: 

               method_BOTH  method_nguyen  method_sds  nguyen_any  sds_any   y
        0            1              0           0           1        1       1
        1            0              1           0           1        0       0

    note that we care primarily about nguyen_any and sds_any, which 
    are slightly awkward names for indicator variables encoding whether
    both methods or exclusively one picked a given rated sentence. 
    '''
    design_matrix = pd.get_dummies(joined["method"], prefix="method")
    # but now replace 'both' with indicators for both methods
    design_matrix["nguyen_any"] = design_matrix["method_nguyen"] + design_matrix["method_BOTH"]
    design_matrix["sds_any"] = design_matrix["method_sds"] + design_matrix["method_BOTH"]

    # add dependent var
    design_matrix['rating'] = joined["rating"]
    design_matrix['y'] = joined["rating"] 
    design_matrix['study_id']  = joined["study id"]
    #pdb.set_trace()
    #design_matrix.loc[(design_matrix['y'] == 't0'),'y'] = 0
    #design_matrix.loc[(design_matrix['y'] == 't1'),'y'] = 1
    # convert binaries to explicit integers (0/1)
    design_matrix['y'] = design_matrix['y'].astype(int)
    #pdb.set_trace()
    design_matrix['y'] = design_matrix['y'] > 0
    #pdb.set_trace()
    #df[df['A'] > 0]


    return design_matrix

def analyze_results(methods_path="CHAR_POPULATION-key.csv", ratings_path="brian-ratings-population.csv",
                        study_ids_path="CHAR_POPULATION-study-ids.csv"):
    # key
    design_matrix = load_results(methods_path, ratings_path, study_ids_path)

    # model the relevance (binarized) relevance rating as dependent
    # on the method(s)
    result = logit(formula = 'y ~ sds_any + nguyen_any', data=design_matrix).fit()
    #result = logit(formula = 'y ~ sds_any + nguyen_any', data=design_matrix).fit()
    #result = logit(formula = 'y ~ sds_any', data=design_matrix).fit()



    #print smprop.proportion_confint(n_n, N_n)
    #smprop.proportion_confint(n_sds,N_sds)

    ''' 
    Optimization terminated successfully.
             Current function value: 0.286221
             Iterations 7
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  205
    Model:                          Logit   Df Residuals:                      202
    Method:                           MLE   Df Model:                            2
    Date:                Mon, 22 Jun 2015   Pseudo R-squ.:                 0.03769
    Time:                        09:11:54   Log-Likelihood:                -58.675
    converged:                       True   LL-Null:                       -60.973
                                            LLR p-value:                    0.1005
    ==============================================================================
                     coef    std err          z      P>|z|      [95.0% Conf. Int.]
    ------------------------------------------------------------------------------
    Intercept      1.3516      0.783      1.726      0.084        -0.183     2.886
    sds_any        1.1939      0.586      2.038      0.042         0.046     2.342
    nguyen_any     0.3224      0.694      0.465      0.642        -1.037     1.682
    ==============================================================================
    '''
    return design_matrix, result



def plot(res):
    '''
    assumes order is 
        Intercept
        sds_any
        nguyen_any
    '''
    point_ests = res.params
    CIs = res.conf_int()

    ests, lowers, uppers = [], [], []
    for i in xrange(len(point_ests)):
        lower = CIs[0][i]
        upper = CIs[1][i]

        est = point_ests[i]
        print "%s (%s, %s)" % (est, lower, upper)

        ests.append(est)
        lowers.append(lower)
        uppers.append(upper)

 
def anova_lm(*args, **kwargs):
    """
    ANOVA table for one or more fitted linear models.

    Parmeters
    ---------
    args : fitted linear model results instance
        One or more fitted linear models

    **kwargs**

    scale : float
        Estimate of variance, If None, will be estimated from the largest
        model. Default is None.
    test : str {"F", "Chisq", "Cp"} or None
        Test statistics to provide (Why not just give all). Default is "F".

    Returns
    -------
    anova : DataFrame
        A DataFrame containing.

    Notes
    -----
    Model statistics are given in the order of args. Models must have
    a formula_str attribute.

    See Also
    --------
    model_results.compare_f_test, model_results.compare_lm_test
    """
    from pandas import DataFrame
    import numpy as np
    from scipy import stats

    test = kwargs.get("test", "F")
    scale = kwargs.get("scale", None)
    n_models = len(args)
 
    model_formula = []
    pr_test = "PR(>%s)" % test
    names = ['df_resid', 'ssr', 'df_diff', 'ss_diff', test, pr_test]
    table = DataFrame(np.empty((n_models, 6)), columns = names)
 
    if not scale: # assume biggest model is last
        scale = args[-1].scale
 
    table["ssr"] = map(getattr, args, ["ssr"]*n_models)
    table["df_resid"] = map(getattr, args, ["df_resid"]*n_models)
    table.ix[1:]["df_diff"] = np.diff(map(getattr, args, ["df_model"]*n_models))
    table["ss_diff"] = -table["ssr"].diff()
    if test == "F":
        table["F"] = table["ss_diff"] / table["df_diff"] / scale
        table[pr_test] = stats.f.sf(table["F"], table["df_diff"],
                             table["df_resid"])
 
    return table








