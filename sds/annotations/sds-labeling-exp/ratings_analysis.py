import statsmodels
from statsmodels.formula.api import logit


import numpy as np 

from pandas import DataFrame 
import pandas as pd 

# cd /Users/byron/dev/cochrane/cochranenlp/sds/annotations/sds-labeling-exp

def analyze_results():
    # key
    methods = DataFrame.from_csv("method-key-6-9.csv", index_col=None)

    # ratings
    brian_lbls = DataFrame.from_csv("brian-ratings.csv", index_col=None)

    # merge these
    joined = methods.join(brian_lbls)

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
    design_matrix['y'] = joined["rating"] 
    design_matrix['y'] = design_matrix['y'] > 0
    # convert binaries to explicit integers (0/1)
    design_matrix['y'] = design_matrix['y'].astype(int)

    # model the relevance (binarized) relevance rating as dependent
    # on the method(s)
    result = logit(formula = 'y ~ sds_any + nguyen_any', data=design_matrix).fit()
    print result.summary()

    '''
    result = logit(formula = 'y ~ sds_any', data=design_matrix).fit()

    # average both
    np.mean((joined[joined['method'] == 'BOTH'])['rating'])

    # average sds

    # average nguyen
    '''
