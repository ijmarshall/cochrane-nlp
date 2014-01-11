#
#   parse results
#

import sys
from numpy import array
import csv

def max_performance(metrics, param):
	return array([metric[-1][param] for metric in metrics])

def max_performance_sup(metrics, param):
	return array([metric[param] for metric in metrics])

def parse_supervised():

	log_file = sys.argv[1]
	with open(log_file, 'rb') as f:
		results = eval(f.read())
	
	# print results

	metrics = results["metrics"]
	means = {param: max_performance_sup(metrics, param).mean() for param in ["f1", "precision", "recall", "per_integer_accuracy"]}
	mins = {param: max_performance_sup(metrics, param).min() for param in ["f1", "precision", "recall", "per_integer_accuracy"]}
	maxes = {param: max_performance_sup(metrics, param).max() for param in ["f1", "precision", "recall", "per_integer_accuracy"]}
	print means
	print mins
	print maxes




def main():
	
	log_file = sys.argv[1]
	with open(log_file, 'rb') as f:
		results = eval(f.read())
	
	print results

	metrics = results["metrics"]
	means = {param: max_performance_sup(metrics, param).mean() for param in ["f1", "precision", "recall", "per_integer_accuracy"]}
	print means

	metrics = results["metrics"]
	# means = {param: max_performance(metrics, param).mean() for param in ["f1", "precision", "recall", "per_integer_accuracy"]}

	# print means

	print metrics

	output = []

	for i, run in enumerate(metrics):

		for item in run:

			item["fold"] = i + 1
			output.append(item)


		# print run
		# recalls = [str(result["recall"]) for result in run]
		# print
		# print "run %d" % (i,)
		# print "\n".join(recalls)


	with open('results/mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
	    w = csv.DictWriter(f, output[0].keys())
	    w.writeheader()
	    w.writerows(output)





if __name__ == '__main__':
	parse_supervised()