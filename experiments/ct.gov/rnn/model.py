import theano
import theano.tensor as T

class RNN:
    def __init__(self, embeddings, H, num_classes, encoder, decoder):
        """Initialize parameters and build theano graph
        
        Parameters
        ----------
        embeddings : V x word_dim 2darray of word embeddings
        H : hidden layer size
        num_classes : number of classes to predict
        encoder : instance of Encoder
        decoder : instance of Decoder
        
        """
        self.embeddings = theano.shared(value=embeddings, name='embeddings') # word embedding matrix
        self.H, self.num_classes = H, num_classes
        self.encoder, self.decoder = encoder, decoder

        self.params = encoder.params + decoder.params #+ [self.embeddings]
        
        self._theano_build() # build the theano graph

    def _theano_build(self):
        """Build theano computational graph
        
        Also export useful functions on the graph like sgd and prediction
        
        """
        # Inputs
        x_idxs = T.ivector('x_idxs') # indices into embedding matrix for current abstract
        y = T.iscalar('y') # class label number
        lr = T.scalar('learning_rate')

        # Model
        # 
        # Encode abstract and decode it into a class prediction
        x = self.embeddings[x_idxs] # extract abstract words from embedding matrix
        hidden = self.encoder.encode(x)
        probs = self.decoder.decode(hidden) # get class probabilities
        prediction = T.argmax(probs, axis=1)
        loss = -T.log(probs[0, y])

        # Compute gradients for sgd
        grads = [T.grad(loss, wrt=param) for param in self.params]

        self.sgd_step = theano.function(inputs=[x_idxs, y, lr],
                                        outputs=loss,
                                        updates=[(param, param - lr*grad) for param, grad in zip(self.params, grads)])

        self.predict = theano.function(inputs=[x_idxs], outputs=prediction)
