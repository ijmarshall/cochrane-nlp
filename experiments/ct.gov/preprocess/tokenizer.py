# wrapper around keras.preprocessing.text.Tokenizer

import pickle

import numpy as np
import pandas as pd

from nltk import sent_tokenize, word_tokenize

import keras
from keras.preprocessing import sequence

from support import abstract2idxs_generator, length_generator


class Tokenizer:

    def __init__(self, abstracts, maxlen_ratio=.95, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n'):
        """Convert abstracts into indexes

        1. Record unique words
        2. Build word2idx and idx2word dicts
        3. Indexify abstracts
        4. Compute maxlen based on the ratio and pad abstracts

        """
        tok = keras.preprocessing.text.Tokenizer(filters=filters)

        tok.fit_on_texts(abstracts)
        abstracts_idxed = tok.texts_to_sequences(abstracts)

        word2idx = tok.word_index
        word2idx['<MASK>'] = 0
        idx2word = {idx: word for word, idx in word2idx.items()}

        # Compute maxlen based on the ratio and pad abstracts
        
        lengths = pd.Series(list(length_generator(abstracts)))
        for length in range(min(lengths), max(lengths)):
            num_lengths = len(lengths[lengths <= length])

            if num_lengths / float(len(abstracts)) >= maxlen_ratio:
                maxlen = length

                break

        abstracts_padded = sequence.pad_sequences(abstracts_idxed, maxlen=maxlen)

        self.abstracts = abstracts
        self.word2idx, self.idx2word = word2idx, idx2word
        self.abstracts_idxed = abstracts_idxed
        self.maxlen = maxlen
        self.abstracts_padded = abstracts_padded

    def build_embeddings(self, model):
        """Generate mini word2vec vectors for each word"""

        vector_size = model.vector_size

        embeddings = np.zeros([len(self.idx2word), vector_size])

        for i, word in sorted(self.idx2word.items()):
            embeddings[i] = model[word] if word in model else 0 # just yield all zeros for OOV words (including the mask)

        self.embeddings = embeddings
        self.vector_size = vector_size

    def do_pickle(self, pickle_name):
        embeddings_info = {
                    'abstracts': self.abstracts,
                    'abstracts_padded': self.abstracts_padded,
                    'embeddings': {'pubmed': self.embeddings},
                    'word_dim': self.vector_size,
                    'maxlen': self.maxlen,
                    'vocab_size': len(self.embeddings),
                    'word2idx': self.word2idx,
                    'idx2word': self.idx2word,
        }

        pickle.dump(embeddings_info, open(pickle_name, 'wb'))

    def test(self, abstract_idx):
        print self.abstracts[abstract_idx]
        print

        print ' '.join(self.idx2word[idx] for idx in self.abstracts_padded[abstract_idx])
        print


if __name__ == '__main__':
    pass # do a little demo here!
