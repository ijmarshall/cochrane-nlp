
# coding: utf-8

# # Support
# 
# Helper functions in support of character-level language modelling.

import numpy as np

def sample_sentences(model, num_abstracts, vocab_dim, char2idx, idx2char, sent_len=100):
    """Sample a sequence from the char nn
    
    Since we're stuck feeding in 2,093 samples at each iteration, we might as
    well keep these 2,093 samples around and see which ones are the best
    sounding in the end.

    This is not as sophisticated as a beam search, but it is better than just
    doing a DFS.
    
    """
    model.reset_states()
    
    X = np.zeros(shape=[num_abstracts, sent_len], dtype=np.int)
    X[:, 0] = char2idx['START']
    
    for t in range(sent_len):
        # Predict distribution over next character
        #
        probs = model.predict(X, batch_size=num_abstracts)
        probs /= probs.sum(axis=1, keepdims=True) # numerical errors sometimes make np.random.choice complain
        
        for i, sample in enumerate(X):
            predicted_idx = np.random.choice(vocab_dim, p=probs[i])
            X[i, t] = predicted_idx

    return X
