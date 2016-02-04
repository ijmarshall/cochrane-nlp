import numpy as np

import theano
import theano.tensor as T

class Cell:
    def step(self, h_prev, x):
        """Compute new hidden state from previous hidden state and input"""

        raise NotImplementedError

class VanillaCell(Cell):

    def __init__(self, H, word_dim):
        """Initialize learnable weights of cell
        
        Parameters
        ----------
        H : dimensionality of hidden state
        word_dim : dimensionality of word vectors
        
        """
        self.H, self.word_dim = H, word_dim

        # Random weight initialization
        Wh = np.random.uniform(-1, 1, size=[H, H]) # hidden to hidden function
        Wx = np.random.uniform(-1, 1, size=[word_dim, H]) # input to hidden function
        
        # Theano variables
        self.Wh = theano.shared(value=Wh, name='Wh') # hidden-hidden function
        self.Wx = theano.shared(value=Wx, name='Wx') # input-hidden function

        self.params = [self.Wh, self.Wx]

    def step(self, x, h_prev, *params):
        """Defines one step of a vanilla RNN

        Parameters
        ----------
        x : element of original sequence passed by scan()
        h_prev : accumulation of result thus far
        Wh, Wx : non-sequences passed in every iteration

        Take the input at the current time step, along with the previous hidden
        state and combine them to produce a new hidden state.

        """
        Wh, Wx = params

        z = T.dot(h_prev, Wh) + T.dot(x, Wx)
        hidden = T.tanh(z)

        return hidden
