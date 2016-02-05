import numpy as np

import theano
import theano.tensor as T

class VanillaCell(object):

    def __init__(self, hidden_dim, word_dim):
        """Initialize learnable weights of a vanilla RNN cell
        
        Parameters
        ----------
        hidden_dim : dimensionality of hidden state
        word_dim : dimensionality of word vectors
        
        """
        self.hidden_dim, self.word_dim = hidden_dim, word_dim

        # Random weight initialization
        Wh = np.random.uniform(-1, 1, size=[hidden_dim, hidden_dim]) # hidden to hidden function
        Wx = np.random.uniform(-1, 1, size=[word_dim, hidden_dim]) # input to hidden function
        
        # Theano variables
        self.Wh = theano.shared(value=Wh, name='Wh') # hidden-hidden function
        self.Wx = theano.shared(value=Wx, name='Wx') # input-hidden function

        self.params = [self.Wh, self.Wx]

        initial_h = np.zeros(hidden_dim)
        self.initial_vals = [dict(initial=initial_h)]

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

class LSTMCell(object):

    def __init__(self, hidden_dim, word_dim, memory_dim):
        """Initialize learnable weights of a LSTM cell
        
        Parameters
        ----------
        H : dimensionality of hidden state
        word_dim : dimensionality of word vectors
        memory_dim : dimensionality of memory vectors
        
        """
        dim = word_dim + hidden_dim # concatenate input and hidden vector

        # Random weight initialization
        F = np.random.uniform(-1, 1, size=[dim, memory_dim]) # forget gate
        I = np.random.uniform(-1, 1, size=[dim, memory_dim]) # input gate
        H = np.random.uniform(-1, 1, size=[dim, memory_dim]) # candidate h
        O = np.random.uniform(-1, 1, size=[dim, memory_dim]) # output gate
        
        # Theano variables
        self.F = theano.shared(value=F, name='F')
        self.I = theano.shared(value=I, name='I')
        self.H = theano.shared(value=H, name='H')
        self.O = theano.shared(value=O, name='O')

        self.params = [self.F, self.I, self.H, self.O]

        # Initialize values
        initial_h, initial_c = np.zeros(hidden_dim), np.zeros(memory_dim)
        initials = [initial_h, initial_c]
        self.initial_vals = [dict(initial=initial) for initial in initials]

    def step(self, x, c_prev, h_prev, *params):
        """Defines one step of a LSTM unit

        Parameters
        ----------
        x : element of original sequence passed by scan()
        c_prev : previous memory
        h_prev : previous hidden state
        params : parameters of the LSTM network

        Take the input at the current time step, along with the previous hidden
        state and combine them to produce a new hidden state.

        """
        F, I, H, O = params

        input = T.concatenate([x, h_prev])

        # Forget component
        forget_gate = T.nnet.sigmoid(T.dot(input, F)) # forget gate
        c_forget = forget_gate * c_prev

        # Input component
        input_gate = T.nnet.sigmoid(T.dot(input, I)) # input gate
        candidate_h = T.tanh(T.dot(input, H))
        c_input = input_gate * candidate_h

        c = c_forget + c_input # combine forgotten memory with new input

        # Output component
        output_gate = T.nnet.sigmoid(T.dot(input, O))
        hidden = output_gate * T.tanh(c)

        return c, hidden
