import os

import numpy as np

from condor_create import make_exps
from support import const_generator, my_range, cycle_generator


exp_group = os.path.basename(__file__).split('.')[-2]


bit = ['True', 'False']

args = {'-exp-id': my_range(start=0, end=100),
        '-labels': const_generator('primary_purpose'),
        '-nb-epoch': const_generator(50),
        '-composite-labels': const_generator(True),
        '-class-weight': cycle_generator(bit),
}

make_exps(exp_group, args, num_exps=2)
