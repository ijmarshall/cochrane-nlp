import theano
import theano.tensor as T

from cell import LSTMCell

class FixedVectorEncoder:
    """Encoder which encodes entire sequence into a fixed-length vector"""

    def __init__(self, cell, bptt=4):
        self.cell = cell
        self.bptt = bptt

        self.params = cell.params

    def encode(self, x):
        if type(self.cell) == LSTMCell:
            [memories, hiddens], updates = theano.scan(
                    self.cell.step, # function to apply at each step in the sequence
                    sequences=x, # sequence to scan() along
                    outputs_info=self.cell.initial_vals,
                    non_sequences=self.cell.params, # tells scan() to pass in Whh and Wxh at each step
                    truncate_gradient=self.bptt, # how far to propagage gradients back in time
                    strict=True) # tells scan() to throw an error if we're accessing non-local variables in the body of the function
        else:
            hiddens, updates = theano.scan(
                    self.cell.step, # function to apply at each step in the sequence
                    sequences=x, # sequence to scan() along
                    outputs_info=self.cell.initial_vals,
                    non_sequences=self.cell.params, # tells scan() to pass in Whh and Wxh at each step
                    truncate_gradient=self.bptt, # how far to propagage gradients back in time
                    strict=True) # tells scan() to throw an error if we're accessing non-local variables in the body of the function

        hidden = hiddens[-1]

        return hidden
