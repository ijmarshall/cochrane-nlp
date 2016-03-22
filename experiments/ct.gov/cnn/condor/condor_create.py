import itertools

from collections import OrderedDict

from IPython import get_ipython

def args_generator(args):
    """Generates list of tuples corresponding to args hyperparam dict
    
    Parameters
    ----------
    args : dict of hyperparam values
    
    """
    od = OrderedDict(args)
    keys = [param for param in od]

    for vals in itertools.product(*[value for param, value in od.items()]):
        yield zip(keys, [str(val) for val in vals])
        
def make_exp(exp_group, args):
    """Perform setup work for a condor experiment
    
    Parameters
    ----------
    exp_group : name of this experiment group
    args : list of (arg_str, arg_val) tuples

    1. Remove any output from a previous experiment with the same name
    2. Do the same thing for weights
    3. Create the condor job file
    
    """
    equalized_args = ['='.join(tup) for tup in args]
    stripped_args = [arg.lstrip('-') for arg in equalized_args]
    exp_name = '+'.join(stripped_args)
    
    unrolled_args = [arg for arg_tup in args for arg in arg_tup]
    arg_str = ' '.join(unrolled_args)

    get_ipython().system(u'rm -rf ../output/$exp_group/$exp_name')
    
    get_ipython().system(u'mkdir -p ../output/$exp_group/$exp_name')
    get_ipython().system(u'mkdir -p ../weights/$exp_group/')
    
    get_ipython().system(u"sed 's/ARGUMENTS/$arg_str/g' job_template > /tmp/tmp1")
    get_ipython().system(u"sed 's/EXP_GROUP/$exp_group/g' /tmp/tmp1 > /tmp/tmp2")
    get_ipython().system(u"sed 's/EXPERIMENT/$exp_name/g' /tmp/tmp2 > /tmp/tmp3")
    
    get_ipython().system(u'mkdir -p $exp_group')
    get_ipython().system(u'cp /tmp/tmp3 $exp_group/$exp_name')
    
def make_exps(exp_group, args):
    """Wrapper around make_exp()
    
    Call make_exp() with every setting of the arguments.
    
    """
    args_list = list(args_generator(args))

    for args_setting in args_list:
        make_exp(exp_group, args_setting)


if __name__ == '__main__':
    # Example experiment group!

    args = {'-nb-filter': [1, 2, 3, 4, 5], '-hidden-dim': [32, 64]}
    make_exps(exp_group='test', args=args)
