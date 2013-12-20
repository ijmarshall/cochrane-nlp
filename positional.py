import re
tx_tag = re.compile("tx[0-9]+")

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist
import seaborn as sns
# "muted"
sns.set(palette="muted", style="nogrid")

import numpy as np

from indexnumbers import swap_num
import bilearn
from taggedpipeline import TaggedTextPipeline
from journalreaders import LabeledAbstractReader
from tokenizer import MergedTaggedAbstractReader
import progressbar

def _sentence_contains(tagged_sentence, tag="n"):
    return any([tag in t[1] for t in tagged_sentence])

def count_treatments(citation):
    tx_ns = [0]
    for sentence_num, sentence in enumerate(citation):
        for token, tag_set in sentence:
            for tag in tag_set:
                matches = tx_tag.findall(tag)
                for match in matches:
                    # remove _a (e.g., tx1_a)
                    tx_ns.append(int(match[2:]))

    return max(tx_ns)

def hist_of_tx_counts(plot_them=True):
    reader = MergedTaggedAbstractReader()
    n_txs = []
    for citation in reader:
        n_txs.append(count_treatments(citation))

    if plot_them:
        plt.clf()
        max_num = max(n_txs)+1
        #counts = [0]*max_num
        #x = range(max_num)
        nbins = max_num
        counts, bins, patches = hist(
            n_txs, bins=nbins, alpha=.8, normed=True, 
            align='mid', label=["1"])
        offset = abs(bins[1]-bins[0])/2.0
        #import pdb; pdb.set_trace()
        plt.xticks([b+offset for b in bins], [0, 1, 2, 3, 4])
        plt.xlabel("number of treament groups")
        #plt.xticks(bins)

    return n_txs

def pos_deltas(tag1="tx1", tag2="tx2", plot_them=True):
    reader = MergedTaggedAbstractReader()
    deltas = []
    for citation_index in xrange(len(reader)):
        pos1, pos2 = None, None
        citation = reader.get(citation_index)
        # number of citations
        citation_length = len(citation)
        for sentence_num, sentence in enumerate(citation):
            #print float(sentence_num+1)/citation_length
            if _sentence_contains(sentence, tag1):
                pos1 = float(sentence_num+1)/citation_length
            
            if _sentence_contains(sentence, tag2):
                pos2 = float(sentence_num+1)/citation_length

        if pos1 is not None and pos2 is not None:
            deltas.append(abs(pos1-pos2))


    if plot_them:
        plt.clf()
        hist(deltas, alpha=.8)
        plt.xlim(0,1)
        plt.xlabel("(normalized) sentence distance between '%s' and '%s' tokens" % 
                        (tag1, tag2))
    return deltas

def tag_positions(tag="n"):
    reader = MergedTaggedAbstractReader()
    positions = []
    for citation_index in xrange(len(reader)):
        citation = reader.get(citation_index)
        # number of citations
        citation_length = len(citation)
        for sentence_num, sentence in enumerate(citation):
            if _sentence_contains(sentence, tag):
                pos = float(sentence_num)/citation_length
                positions.append(pos)
    return positions

def txs_histos():
    plt.clf()
    tx1_positions = tag_positions(tag="tx1")
    tx2_positions = tag_positions(tag="tx2")
    hist(tx1_positions, alpha=1, label="tx1")
    hist(tx2_positions, alpha=.5, label="tx2")
    plt.xlim(0,1)
    plt.legend()

def positional_histo(tag="n"):
    positions = tag_positions(tag)
    hist(positions)