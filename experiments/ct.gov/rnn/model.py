import theano
import theano.tensor as T

class RNN:
    def __init__(self, embeddings, encoder, decoder):
        """Initialize parameters and build theano graph
        
        Parameters
        ----------
        embeddings : V x word_dim 2darray of word embeddings
        encoder : method of encoding abstract
        decoder : method of decoding abstract
        
        """
        self.embeddings = theano.shared(value=embeddings, name='embeddings') # word embedding matrix
        self.encoder, self.decoder = encoder, decoder

        self.params = encoder.params + decoder.params #+ [self.embeddings]
        
        print 'Building theano graph...'
        self._theano_build()
        print 'Done!'

    def _theano_build(self):
        """Build theano computational graph
        
        Also export useful functions on the graph like sgd and prediction
        
        """
        # Inputs
        x_idxs = T.ivector('x_idxs') # indices into embedding matrix for current abstract
        y = T.iscalar('y') # class label number
        lr = T.scalar('learning_rate')
        reg = T.scalar('regularization')

        # Model
        # 
        # Encode abstract and decode it into a class prediction
        x = self.embeddings[x_idxs] # extract abstract words from embedding matrix
        hidden = self.encoder.encode(x)
        probs = self.decoder.decode(hidden) # get class probabilities
        prediction = T.argmax(probs, axis=1)
        loss = -T.log(probs[0, y]) + reg*self.l2_loss()
        # Compute gradients for sgd
        grads = [T.grad(loss, wrt=param) for param in self.params]

        self.do_sgd = theano.function(inputs=[x_idxs, y, lr, reg],
                                      outputs=loss,
                                      updates=[(param, param - lr*grad) for param, grad in zip(self.params, grads)],
                                      on_unused_input='warn')

        self.predict = theano.function(inputs=[x_idxs], outputs=prediction, on_unused_input='warn')

        self.loss_prediction = theano.function(inputs=[x_idxs, y, reg], outputs=[loss, prediction], on_unused_input='warn')

    def l2_loss(self):
        """Compute L2 regularization loss for parameter settings"""

        pows = [T.pow(param, 2) for param in self.params]
        sums = [T.sum(pow) for pow in pows]

        sum = T.sum(sums)

        return sum
