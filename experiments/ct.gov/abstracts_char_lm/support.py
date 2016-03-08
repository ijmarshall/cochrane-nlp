
# coding: utf-8

# # Support
# 
# Helper functions in support of character-level language modelling.

import numpy as np

def sample(a, temperature=1.0):
    """Helper function to sample an index from a probability array"""

    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))

    return np.argmax(np.random.multinomial(1, a, 1))

def sample_sentences(model, num_abstracts, vocab_dim, char2idx, idx2char, num_seqs, sent_len=1, temperature=.5):
    """Sample a sequence from the char nn
    
    Since we're stuck feeding in 2,093 samples at each iteration, we might as
    well keep these 2,093 samples around and see which ones are the best
    sounding in the end.

    This is not as sophisticated as a beam search, but it is better than just
    doing a DFS.
    
    """
    model.reset_states()
    
    X = np.zeros(shape=[num_abstracts, 1], dtype=np.int)
    X[:, 0] = char2idx['START'] # forget about the rest after `num_abstracts`

    ys = np.zeros([num_seqs, sent_len])
    
    for t in range(sent_len):
        # Predict distribution over next character
        #
        probs = model.predict(X, batch_size=num_abstracts)
        for i in range(num_seqs): # only update the first `num_abstracts` sequences
            predicted_idx = sample(probs[i, 0], temperature)
            # predicted_idx = np.argmax(probs[i, 0])
            ys[i, t] = predicted_idx

        X[:, 0] = predicted_idx # update X so the next char we feed in is the one predicted by the rnn

    return ys
