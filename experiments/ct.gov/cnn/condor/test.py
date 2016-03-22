from condor_create import make_exps

args = {'-nb-filter': [1, 2, 3, 4, 5], '-hidden-dim': [32, 64]}

make_exps(exp_group='test', args=args)
