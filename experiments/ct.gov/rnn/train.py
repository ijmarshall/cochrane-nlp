import numpy as np

from rnn.model import RNN

from rnn.cell import VanillaCell, LSTMCell
from rnn.encoder import FixedVectorEncoder
from rnn.decoder import SoftmaxDecoder

class RNNTrainer:
    def __init__(self, embeddings, hidden_dim=16, num_classes=2,
                 cell_type='vanilla', encoder_type='fixed', decoder_type='softmax'):
        """Initialize parameters of the rnn trainer
        
        Parameters
        ----------
        embeddings : embedding matrix of all words in corpus
        hidden_dim : size of hidden layers
        num_classes : number of classes being predicted
        num_epochs : epochs to train for
        cell_type : {'vanilla', 'lstm'}
        encoder_type : {'fixed-vector'}
        decoder_type : {'softmax'}
        
        """
        (_, word_dim) = embeddings.shape
        
        embeddings = embeddings
        hidden_dim = hidden_dim
        
        if cell_type == 'vanilla':
            cell = VanillaCell(hidden_dim, word_dim)
        elif cell_type == 'lstm':
            cell = LSTMCell(hidden_dim, word_dim, memory_dim=hidden_dim)
            
        encoder = FixedVectorEncoder(cell)
        decoder = SoftmaxDecoder(hidden_dim, num_classes)
        
        self.rnn = RNN(embeddings, encoder, decoder)
        
    def epoch_trainer(self, X, ys, num_epochs=1):
        """"""
        
        for _ in range(num_epochs):
            for x_idxs, y in zip(X, ys):
                yield x_idxs, y
        
    def train(self, X_train, ys_train, X_val, ys_val, learning_rate=.005, schedule_type='epoch', val_every=1e100):
        """Train for specified number of epochs
        
        X_train : list of training examples
        ys_train : training labels
        X_val : list of validation examples
        ys_val : validation labels
        schedule : generator that yields training examples
        learning_rate : learning rate to use in sgd
        
        """
        if schedule_type == 'epoch':
            schedule = self.epoch_trainer(X_train, ys_train, num_epochs=5)
        
        self.val_loss = [] # maintain val_loss state after this training is done
        for i, (x, y) in enumerate(schedule):
            self.rnn.do_sgd(x, y, learning_rate)
            
            if not i % val_every:
                val_schedule = self.epoch_trainer(X_val, ys_val)
                val_loss = np.mean([self.rnn.compute_loss(x, y) for (x, y) in val_schedule])
                
                print 'Validation Loss: {}'.format(val_loss)
                
                self.val_loss.append(val_loss)
