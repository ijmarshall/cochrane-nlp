import numpy as np

from rnn.model import RNN

from rnn.cell import VanillaCell, LSTMCell
from rnn.encoder import FixedVectorEncoder
from rnn.decoder import SoftmaxDecoder

class LMTrainer:
    def __init__(self, embeddings, hidden_dim=64 cell_type='vanilla'):
        """Initialize parameters of the rnn trainer
        
        Parameters
        ----------
        embeddings : embedding matrix of all words in corpus
        hidden_dim : size of hidden layers
        cell_type : {'vanilla', 'lstm'}
        
        """
        (V, word_dim) = embeddings.shape
        
        embeddings = embeddings
        hidden_dim = hidden_dim
        
        if cell_type == 'vanilla':
            cell = VanillaCell(hidden_dim, word_dim)
        elif cell_type == 'lstm':
            cell = LSTMCell(hidden_dim, word_dim, memory_dim=hidden_dim)
            
        self.model = RNNLM(embeddings, encoder, decoder)
        
    def epoch_trainer(self, X, ys, num_epochs=1):
        """"""
        
        for _ in range(num_epochs):
            for x_idxs, y in zip(X, ys):
                yield x_idxs, y
        
    def train(self, X_train, ys_train, X_val, ys_val,
            learning_rate=.1, reg=.001, num_epochs=1, val_every=1e100,
            class_weights='auto'):
        """Train for specified number of epochs
        
        X_train : list of training examples
        ys_train : training labels
        X_val : list of validation examples
        ys_val : validation labels
        learning_rate : learning rate to use in sgd
        reg : regularization term
        schedule_type : training schedule {'epoch'}
        val_every : number of iterations to go before reporting validation loss
        class_weights : correct for class imbalance?
        
        """
        schedule = self.epoch_trainer(X_train, ys_train, num_epochs=num_epochs)

        # Correct for class imbalance?
        class_counts = ys_train.value_counts()
        if class_weights == 'auto':
            class_weights_ = np.reciprocal(class_counts / class_counts.sum())
        else:
            class_weights_ = np.ones_like(class_counts)
        
        self.val_losses, self.val_accuracies, self.grad_magnitudes = [], [], []
        for i, (x, y) in enumerate(schedule):
            grad_magnitudes = self.rnn.do_sgd(x, y, learning_rate, reg, class_weights_)
            self.grad_magnitudes.append([float(grad_magnitude) for grad_magnitude in grad_magnitudes])
            
            if not i % val_every:
                val_schedule = self.epoch_trainer(X_val, ys_val)
                tuples = [self.rnn.loss_prediction(x, y, reg, class_weights_) for (x, y) in val_schedule]

                losses, predictions, penalty, reg_losses, probs, hidden, x = zip(*tuples)
                val_loss, predictions = np.mean(losses), map(float, predictions)
                val_accuracy = np.mean(np.array(predictions)==ys_val)
                reg_loss = np.mean(reg_losses)
                
                print 'Loss, Accuracy = ({}, {})'.format(val_loss, val_accuracy)
                # print 'Penalty = {}'.format(penalty)
                # print 'Reg Losses = {}'.format(reg_losses)
                # print 'Probs = {}'.format(probs)
                # print 'Hidden = {}'.format(hidden.shape)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

class Seq2SeqTrainer:
    def __init__(self, embeddings, hidden_dim=64, num_classes=2,
                 cell_type='vanilla', encoder_type='fixed',
                 decoder_type='softmax', bptt=-1):
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
            
        encoder = FixedVectorEncoder(cell, bptt)
        decoder = SoftmaxDecoder(hidden_dim, num_classes)
        
        self.rnn = RNN(embeddings, encoder, decoder)
        
    def epoch_trainer(self, X, ys, num_epochs=1):
        """"""
        
        for _ in range(num_epochs):
            for x_idxs, y in zip(X, ys):
                yield x_idxs, y
        
    def train(self, X_train, ys_train, X_val, ys_val,
            learning_rate=.1, reg=.001, num_epochs=1, val_every=1e100,
            class_weights='auto'):
        """Train for specified number of epochs
        
        X_train : list of training examples
        ys_train : training labels
        X_val : list of validation examples
        ys_val : validation labels
        learning_rate : learning rate to use in sgd
        reg : regularization term
        schedule_type : training schedule {'epoch'}
        val_every : number of iterations to go before reporting validation loss
        class_weights : correct for class imbalance?
        
        """
        schedule = self.epoch_trainer(X_train, ys_train, num_epochs=num_epochs)

        # Correct for class imbalance?
        class_counts = ys_train.value_counts()
        if class_weights == 'auto':
            class_weights_ = np.reciprocal(class_counts / class_counts.sum())
        else:
            class_weights_ = np.ones_like(class_counts)
        
        self.val_losses, self.val_accuracies, self.grad_magnitudes = [], [], []
        for i, (x, y) in enumerate(schedule):
            grad_magnitudes = self.rnn.do_sgd(x, y, learning_rate, reg, class_weights_)
            self.grad_magnitudes.append([float(grad_magnitude) for grad_magnitude in grad_magnitudes])
            
            if not i % val_every:
                val_schedule = self.epoch_trainer(X_val, ys_val)
                tuples = [self.rnn.loss_prediction(x, y, reg, class_weights_) for (x, y) in val_schedule]

                losses, predictions, penalty, reg_losses, probs, hidden, x = zip(*tuples)
                val_loss, predictions = np.mean(losses), map(float, predictions)
                val_accuracy = np.mean(np.array(predictions)==ys_val)
                reg_loss = np.mean(reg_losses)
                
                print 'Loss, Accuracy = ({}, {})'.format(val_loss, val_accuracy)
                # print 'Penalty = {}'.format(penalty)
                # print 'Reg Losses = {}'.format(reg_losses)
                # print 'Probs = {}'.format(probs)
                # print 'Hidden = {}'.format(hidden.shape)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
