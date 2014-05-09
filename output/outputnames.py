#
#	outputnames.py
#

#	generates helpful filenames to record the output of arbitrarily named scripts

import __main__
import time
import os

def filename(label="", extension="csv"):

	name = os.path.splitext(os.path.basename(__main__.__file__))[0]
	time_stamp = time.strftime("%Y-%h-%d--%H%M")

	return name + '-' + time_stamp + "."+ extension

