import numpy as np

import theano
import theano.tensor as T

class Decoder:

    def decode(self, hidden):
        raise NotImplementedError

class SoftmaxDecoder(Decoder):

    def __init__(self, H, num_classes):
        """Initialize parameters

        Parameters
        ----------
        H : dimensionality of hidden vector
        num_classes : number of target classes

        """
        self.num_classes = num_classes

        # Softmax weights
        Ws = np.random.uniform(-1, 1, size=[H, num_classes])
        self.Ws = theano.shared(value=Ws, name='Ws')

        self.params = [self.Ws]

    def decode(self, hidden):
        [H] = hidden.shape

        scores = T.dot(hidden, self.Ws) # compute class scores with encoded abstract
        probs = T.nnet.softmax(scores)

        return probs
