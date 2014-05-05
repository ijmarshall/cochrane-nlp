
############################################################
#   
#   metrics
#
############################################################

import sklearn.metrics

def aggregate_fold_results(pickled_fold_metrics_dir, nfolds=5, 
                base_metrics_str="metrics_<FOLD>.pickle", out_file="aggregated_metrics.csv"):
    """
    Aggregates results from independently run folds,
    i.e., when folds have been run in parallel. returns

    Reads in all nfolds sets of metrics (assumed to be located
    in the pickled_fold_metrics_dir directory) and returns
    an aggregated BinaryMetricsRecorder object.
    """
    aggregated_metrics = None
    for fold_number in xrange(nfolds):
        with open(os.path.join(pickled_fold_metrics_dir,
                base_metrics_str.replace("<FOLD>", str(fold_number)))) as fold_metrics_f:
            fold_binary_metrics = pickle.load(fold_metrics_f)

            if fold_number == 0:
                # instantiate aggregate metrics using info from 
                # the first fold; we will assume that this information 
                # is constant across the folds!
                aggregated_metrics = BinaryMetricsRecorder(domains=fold_binary_metrics.fold_metrics.keys())
                aggregated_metrics.title = fold_binary_metrics.title
            for domain in fold_binary_metrics.domains:
                #pdb.set_trace()
                aggregated_metrics.fold_metrics[domain].extend(
                    fold_binary_metrics.fold_metrics[domain])
                
    aggregated_metrics.save_csv(os.path.join(pickled_fold_metrics_dir, out_file))
    return aggregated_metrics

class BinaryMetricsRecorder(object):
    """
    records results of folds, and outputs them in various formats

    """

    METRIC_NAMES = ["n", "f1", "precision", "recall", "accuracy"]
    METRIC_FUNCTIONS = [lambda preds, test: len(preds),
                        sklearn.metrics.f1_score, sklearn.metrics.precision_score,
                        sklearn.metrics.recall_score, sklearn.metrics.accuracy_score]


    def __init__(self, title="Untitled", domains=["default"]):
        self.title = title
        self.domains = domains
        self.fold_metrics = {k: [] for k in domains}
        

    def add_preds_test(self, preds, test, domain="default"):
        """
        add a fold of data
        """
        fold_metric = {metric_name: metric_fn(test, preds) for metric_name, metric_fn
                    in zip (self.METRIC_NAMES, self.METRIC_FUNCTIONS)}

        fold_metric["domain"] = domain
        fold_metric["fold"] = len(self.fold_metrics[domain])
        
        self.fold_metrics[domain].append(fold_metric)

    def _means(self, domain):

        summaries = {k: [] for k in self.METRIC_NAMES}

        for row in self.fold_metrics[domain]:
            for metric_name in self.METRIC_NAMES:
                summaries[metric_name].append(row[metric_name])

        means = {metric_name: np.mean(summaries[metric_name]) for metric_name in self.METRIC_NAMES}

        means["domain"] = domain
        means["fold"] = "mean"
        means["n"] = np.sum(summaries["n"]) # overwrite with the sum

        return means


    def save_csv(self, filename):

        output = []
        for domain in self.domains:
            output.extend(self.fold_metrics[domain])
            output.append(self._means(domain))
            output.append({}) # blank line

        with open(filename, 'wb') as f:
            w = csv.DictWriter(f, ["domain", "fold"] + self.METRIC_NAMES)
            w.writeheader()
            w.writerows(output)

        