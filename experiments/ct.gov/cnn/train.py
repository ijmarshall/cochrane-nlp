# coding: utf-8

# # Multitask Learning
# 
# Use a single shared representation to predict gender and phase 2

# ### Load Embeddings and Abstracts

# In[1]:

import os
import sys

from collections import OrderedDict

import plac
import pickle

import numpy as np

from sklearn.cross_validation import KFold

import keras

from keras.models import Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.layer_utils import model_summary
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

from support import classinfo_generator, produce_labels, ValidationCallback, repeat_labels

class Model:
    def __init__(self, init, init_exp_group, init_exp_id, lr_multipliers,
            round_robin, num_labels):
        self.pretrain = init
        self.init_exp_group = init_exp_group
        self.init_exp_id = init_exp_id
        self.shared_multiplier, self.softmax_multiplier = lr_multipliers

        # Use same lr multiplier for both weights and biases
        self.shared_multiplier = [self.shared_multiplier]*2

        self.round_robin = round_robin
        self.num_labels = num_labels

    def load_embeddings(self, embeddings_file, word_vectors):
        """Load word embeddings and abstracts
        
        embeddings_info dict
        --------------------
        abstracts: full-text abstracts
        abstracts_padded: abstracts indexed and padded
        embeddings: embedding matrix
        word_dim: dimension word embeddings
        word2idx: dictionary from word to embedding index
        idx2word: dictionary from embedding index to word
        maxlen: size of each padded abstract
        vocab_size: number of words in the vocabulary
            
        """
        embeddings_info = pickle.load(open('pickle/{}'.format(embeddings_file), 'rb'))

        self.abstracts = embeddings_info['abstracts']

        self.abstracts_padded = embeddings_info['abstracts_padded']
        self.embeddings = embeddings_info['embeddings'][word_vectors]
        if self.round_robin: # repeat embeddings
            self.abstracts_repeated = np.tile(self.abstracts_padded, [self.num_labels, 1])

        self.word_dim = embeddings_info['word_dim']
        self.word2idx, idx2word = embeddings_info['word2idx'], embeddings_info['idx2word']
        self.maxlen = embeddings_info['maxlen']
        self.vocab_size = embeddings_info['vocab_size']

    def load_labels(self, labels_file, label_names):
        """Load labels for dataset

        Mainly configure class names and validation data

        """
        self.df = pickle.load(open('pickle/{}_labels.p'.format(labels_file), 'rb'))
        self.bdf = pickle.load(open('pickle/{}_binarized.p'.format(labels_file), 'rb'))

        # Cut down labels to only the ones we're predicting on
        df, bdf = self.df[label_names], self.bdf[label_names]

        # Get class names and sizes
        class_info = list(classinfo_generator(df))
        class_names, self.class_sizes = zip(*class_info)
        # class_names = {label: classes for label, classes in zip(label_names, class_names)}

        self.ys = np.array(bdf).T # turn labels into numpy array
        if self.round_robin: # repeat labels
            self.ys_repeated = np.hstack(list(repeat_labels(self.ys)))

        self.label_names = df.columns.tolist()

    def do_train_val_split(self, num_train):
        """Split data up into separate train and validation sets

        Parameters
        ----------
        num_train : number of examples to train on

        Pass a big number for num_train to include the entire training set.

        """
        fold = KFold(len(self.abstracts_padded),
                     n_folds=5,
                     shuffle=True,
                     random_state=0) # for reproducibility!

        p = iter(fold)
        train_idxs, val_idxs = next(p)

        if self.round_robin: # repeat training idxs
            train_idxs = [self.ys.shape[1]*i + train_idx for train_idx in train_idxs for i in range(self.num_labels)]

        if len(self.label_names) == 1:
            # Take special care to move distinct class examples to the front of
            # the line!
            #
            df = self.df.loc[train_idxs] # only consider train examples
            label = self.label_names[0]
            classes = df[label].cat.categories # get class names

            # Get first occurring indexes of each class and prepend it to the
            # front. Make sure to not double count!

            indexes = [df[df[label] == class_].iloc[0].name for class_ in classes]
            filtered_train_idxs = [train_idx for train_idx in train_idxs if train_idx not in indexes]
            train_idxs = np.array(indexes + filtered_train_idxs)

        train_idxs = train_idxs[:num_train] # cut down the number of examples to train on!
        self.num_train, self.num_val = len(train_idxs), len(val_idxs)

        # Extract training data to pass to keras fit()
        #
        # Take special care to repeat the abstracts if we're doing round robin
        ys = self.ys_repeated if self.round_robin else self.ys
        abstracts = self.abstracts_repeated if self.round_robin else self.abstracts_padded

        self.train_data = OrderedDict(produce_labels(self.label_names, ys[:, train_idxs], self.class_sizes))
        self.train_data.update({'input': abstracts[train_idxs]})

        # Extract validation data to validate over
        #
        # Never repeat abstracts nor labels here!
        self.val_data = OrderedDict(produce_labels(self.label_names, self.ys[:, val_idxs], self.class_sizes))
        self.val_data.update({'input': self.abstracts_padded[val_idxs]})

    def add_representation(self, input, filter_lens, nb_filter, reg,
            name, dropouts, hidden_dim, dropout_prob):
        """Add a representation for a task

        Parameters
        ----------
        input : name of input node
        filter_lens : lengths of filters to use for convolution
        name : name of final hidden vector representation
        
        Pipeline
        --------
        embedding
        convolutions (multiple filters)
        pool
        flatten
        dense
        dropout
        
        """
        convs, pools, flats = {}, {}, {}
        for filter_len in filter_lens:
            convs[filter_len] = 'conv_{}_{}'.format(filter_len, name)
            pools[filter_len] = 'pool_{}_{}'.format(filter_len, name)
            flats[filter_len] = 'flat_{}_{}'.format(filter_len, name)

        # Add convolution -> max_pool -> flatten for each filter length
        convs_list = []
        for filter_len in filter_lens:
            self.model.add_node(Convolution1D(nb_filter=nb_filter,
                                              filter_length=filter_len,
                                              activation='relu',
                                              W_regularizer=l2(reg)),
                                name=convs[filter_len],
                                input=input)

            self.model.add_node(MaxPooling1D(pool_length=self.maxlen-(filter_len-1)),
                           name=pools[filter_len],
                           input=convs[filter_len])
            self.model.add_node(Flatten(), name=flats[filter_len], input=pools[filter_len])

            convs_list.append(flats[filter_len])

        # Hack to merge together activation maps
        self.model.add_node(Dropout(1e-100), name=dropouts[name], inputs=convs_list)

    def build_model(self, nb_filter, filter_lens, hidden_dim,
            dropout_prob, dropout_emb, task_specific, reg, task_reg, resid_reg, resid_regs,
            backprop_emb, word2vec_init, exp_desc, exp_group, exp_id):
        """Build keras model

        Start with declaring model names and have graph construction mirror it
        as closely as possible.

        """
        dropouts = {}

        # Define layer names ahead of model construction!

        input = 'input'
        embedding = 'embedding'
        dropouts[embedding] = embedding + '_'

        # shared_rep = 'shared_rep'
        # dropouts[shared_rep] = shared_rep + '_'

        individual_reps = {}
        for label in self.label_names:
            individual_rep = '{}_indiv'.format(label)

            individual_reps[label] = individual_rep
            dropouts[individual_rep] = individual_rep + '_'

        if self.num_labels > 1:
            #
            # Residual layers
            #
            rests = {label: 'except_{}'.format(label) for label in self.label_names}
            primes = {label: '{}_prime'.format(label) for label in self.label_names}
            for label in self.label_names:
                dropouts[primes[label]] = primes[label] + '_'

        denses = {label: '{}_dense'.format(label) for label in self.label_names}

        probs = {label: '{}_probs'.format(label) for label in self.label_names}

        outputs = self.label_names

        # Argument parsing
        if not resid_regs:
            resid_regs = [resid_reg]*self.num_labels
            
        if self.pretrain:
            # Load architecture up to the shared layer and weights

            model_path = 'models/{}/{}-base.json'.format(self.init_exp_group, self.init_exp_id)
            print >> sys.stderr, 'Loading model from {}...'.format(model_path)

            self.model = model_from_json(open(model_path).read())
            model = self.model
            self.model.load_weights('weights/{}/{}-val.h5'.format(self.init_exp_group,
                                                                  self.init_exp_id))

            # Learning rate multipliers for convs and shared representation - hardcode at 3 filters for now!
            for i in range(1, 4):
                model.nodes['conv_{}_shared_rep'.format(i)].set_lr_multipliers(*self.shared_multiplier)

            model.nodes[shared_rep].set_lr_multipliers(*self.shared_multiplier)

        else:
            # Build model from scratch

            self.model = Graph()
            model = self.model

            model.add_input(name=input,
                            input_shape=[self.maxlen],
                            dtype='int') # dtype='int' is 100% necessary for some reason!

            init_embeddings = [self.embeddings] if word2vec_init else None
            model.add_node(Embedding(input_dim=self.vocab_size, output_dim=self.word_dim,
                                    weights=init_embeddings,
                                    input_length=self.maxlen,
                                    trainable=backprop_emb),
                           name=embedding,
                           input=input)

            model.add_node(Dropout(dropout_emb), name=dropouts[embedding], input=embedding)

            # # Shared representation
            # self.add_representation(input=dropouts[embedding],
            #                         filter_lens=filter_lens,
            #                         nb_filter=nb_filter,
            #                         reg=reg,
            #                         name=shared_rep,
            #                         dropouts=dropouts,
            #                         hidden_dim=hidden_dim,
            #                         dropout_prob=dropout_prob)
            #
            # # Save the model up to this point in case we want to do pretraining
            # # in the future!
            # json_string = model.to_json()
            # open('models/{}/{}-base.json'.format(exp_group, exp_id), 'w').write(json_string)

            # # Pickle the experiment description so it can be easily loaded back in!
            # pickle.dump(exp_desc, open('params/{}/{}.p'.format(exp_group, exp_id), 'wb'))

        #
        # Take notice!
        #
        # It is at *this* point that both the pretrained and newly
        # contructed models are on equal footing! In both cases, we've
        # constructed the model up to the shared representation and we need to
        # add on the task specific portion(s)!
        #

        for label_name in self.label_names:
            self.add_representation(input=dropouts[embedding],
                                    filter_lens=filter_lens,
                                    nb_filter=nb_filter,
                                    reg=task_reg,
                                    name=individual_reps[label_name],
                                    dropouts=dropouts,
                                    hidden_dim=hidden_dim,
                                    dropout_prob=dropout_prob)

        if self.num_labels > 1: # residual units
            for label, num_classes, resid_reg in zip(self.label_names, self.class_sizes, resid_regs):
                rest = [dropouts[individual_reps[lbl]] for lbl in self.label_names if not lbl == label]

                model.add_node(Dense(output_dim=nb_filter*len(filter_lens),
                                    activation='relu',
                                    W_regularizer=l2(resid_reg)), #                                        |
                            name=rests[label], # everything *except* the current representation as input v
                            input=rest[0] if len(rest) == 1 else None,
                            inputs=rest if len(rest) > 1 else None)

                model.add_node(Dropout(1e-100),
                            name=primes[label],
                            inputs=[dropouts[individual_reps[label]], rests[label]], merge_mode='sum')

                model.add_node(Dense(output_dim=hidden_dim,
                                    activation='relu',
                                    W_regularizer=l2(reg)),
                            name=denses[label],
                            input=primes[label])

                model.add_node(Dense(output_dim=num_classes,
                                    activation='softmax',
                                    W_regularizer=l2(reg),
                                    W_learning_rate_multiplier=self.softmax_multiplier,
                                    b_learning_rate_multiplier=self.softmax_multiplier),
                            name=probs[label],
                            input=denses[label])

                model.add_output(name=label, input=probs[label]) # separate output for each label
        else:

            #
            # Just a dense layer
            #

            for label, num_classes in zip(self.label_names, self.class_sizes):
                model.add_node(Dense(output_dim=hidden_dim,
                                     activation='relu',
                                     W_regularizer=l2(reg)),
                               name=denses[label],
                               input=dropouts[individual_reps[label]])

        # Softmaxes and outputs
        #
        for label, num_classes in zip(self.label_names, self.class_sizes):
            model.add_node(Dense(output_dim=num_classes,
                                 activation='softmax',
                                 W_regularizer=l2(reg),
                                 W_learning_rate_multiplier=self.softmax_multiplier,
                                 b_learning_rate_multiplier=self.softmax_multiplier),
                            name=probs[label],
                            input=denses[label])

            model.add_output(name=label, input=probs[label]) # separate output for each label

        model.compile(optimizer='adam',
                      loss={label: 'categorical_crossentropy' for label in self.label_names}) # CE for all the targets

                                                                              #
        ### END GRAPH CONSTRUCTION ############################################

        print exp_desc
        model_summary(model)

        # Write architecture to disk
        json_string = model.to_json()
        open('models/{}/{}.json'.format(exp_group, exp_id), 'w').write(json_string)

        self.model = model

    def train(self, nb_epoch, batch_size, val_every, val_weights, f1_weights,
            class_weight, save_weights, probs_loc):
        """Train the model for a fixed number of epochs

        Set up callbacks first.

        """
        val_callback = ValidationCallback(self.val_data, batch_size,
                self.num_train, val_every, val_weights, f1_weights,
                save_weights, probs_loc)

        if class_weight:
            class_weights_fname = 'composite_weights.p'

            # Load class weights and filter down to only labels we are considering
            all_class_weights = pickle.load(open('pickle/{}'.format(class_weights_fname, 'rb')))
            class_weights = {label: weights for label, weights in all_class_weights.items() if label in self.label_names}
        else:
            class_weights = {} # no weighting

        history = self.model.fit(self.train_data, batch_size=batch_size,
                                 nb_epoch=nb_epoch, verbose=2,
                                 callbacks=[val_callback],
                                 class_weight=class_weights)


@plac.annotations(
        nb_epoch=('number of epochs', 'option', None, int),
        labels=('labels to predict', 'option'),
        task_specific=('whether to include an addition task-specific hidden layer', 'option', None, str),
        nb_filter=('number of filters', 'option', None, int),
        filter_lens=('length of filters', 'option', None, str),
        hidden_dim=('size of hidden state', 'option', None, int),
        dropout_prob=('dropout probability', 'option', None, float),
        dropout_emb=('perform dropout after the embedding layer', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        task_reg=('l2 regularization constant for task-specific representation', 'option', None, float),
        backprop_emb=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        val_every=('number of times to compute validation per epoch', 'option', None, int),
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        class_weight=('enfore class balance through loss scaling', 'option', None, str),
        word2vec_init=('initialize embeddings with word2vec', 'option', None, str),
        use_pretrained=('experiment ID and group to init from', 'option', None, str),
        num_train=('number of examples to train on', 'option', None, int),
        lr_multipliers=('learning rate multipliers for shared representation and softmax layer', 'option', None, str),
        learning_curve_id=('id of the learning curve (for visualization!)', 'option', None, int),
        save_weights=('whether to save weights during training', 'option', None, str),
        word_vectors=('what kind of word vectors to initialize with', 'option', None, str),
        round_robin=('whether to do multitask learning in a round robin fashion', 'option', None, str),
        resid_reg=('how much to regularize the residual weights', 'option', None, float),
        resid_regs=('how residual weights', 'option', None, str),
        embeddings_file=('name of embeddings file to load', 'option', None, str),
        labels_file=('name of labels file to load', 'option', None, str),
)
def main(nb_epoch=5, labels='allocation,masking', task_specific='False',
        nb_filter=729, filter_lens='1,2,3', hidden_dim=1024, dropout_prob=.5, dropout_emb='True',
        reg=0, task_reg=0, backprop_emb='False', batch_size=128, val_every=1, exp_group='', exp_id='',
        class_weight='False', word2vec_init='True', use_pretrained='None', num_train=100000,
        lr_multipliers='.0001,1', learning_curve_id=0, save_weights='True', word_vectors='pubmed',
        round_robin='False', resid_reg=0., resid_regs='', embeddings_file='embeddings_info.p',
        labels_file='composite'):
    """Training process

    1. Load embeddings and labels
    2. Build the keras model and load weights files
    3. Train

    """
    # Parse list parameters into lists!
    labels = labels.split(',')
    filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]
    use_pretrained = '' if use_pretrained == 'None' else use_pretrained
    pretrained_group, pretrained_id = use_pretrained.split(',') if use_pretrained else (None, None)
    lr_multipliers = [float(lr_multiplier) for lr_multiplier in lr_multipliers.split(',')]
    if resid_regs: resid_regs = [float(rr) for rr in resid_regs.split(',')]

    # Build exp info string for visualization code...
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    exp_desc = '+'.join('='.join(arg_pair) for arg_pair in zip(pnames, pvalues))
    if use_pretrained:
        try: # Enrich experiment description with base model description
            pretrained_exp_desc_loc = 'params/{}/{}.p'.format(pretrained_group, pretrained_id)
            pretrained_exp_desc = pickle.load(open(pretrained_exp_desc_loc, 'rb'))
            exp_desc = '+'.join([exp_desc, pretrained_exp_desc])
        except IOError:
            print >> sys.stderr, 'No params file there yet!'

    # Convert boolean strings to actual booleans
    task_specific = True if task_specific == 'True' else False
    backprop_emb = True if backprop_emb == 'True' else False
    class_weight = True if class_weight == 'True' else False
    dropout_emb = dropout_prob if dropout_emb == 'True' else 1e-100
    word2vec_init = True if word2vec_init == 'True' else False
    save_weights = True if save_weights == 'True' else False
    round_robin = True if round_robin == 'True' else False

    # Make it so there are only nb_filter total - NOT nb_filter*len(filter_lens)
    nb_filter /= len(filter_lens)

    m = Model(use_pretrained, pretrained_group, pretrained_id, lr_multipliers,
            round_robin, num_labels=len(labels))

    m.load_embeddings(embeddings_file, word_vectors)
    m.load_labels(labels_file, labels)
    m.do_train_val_split(num_train)
    m.build_model(nb_filter, filter_lens, hidden_dim, dropout_prob, dropout_emb,
                  task_specific, reg, task_reg, resid_reg, resid_regs, backprop_emb, word2vec_init, exp_desc, exp_group, exp_id)

    # Weights
    weights_str = 'weights/{}/{}-{}.h5'
    val_weights = weights_str.format(exp_group, exp_id, 'val')
    f1_weights = weights_str.format(exp_group, exp_id, 'f1')
    probs_loc = 'probs/{}/{}.p'.format(exp_group, exp_id)

    # Only load weights if we are not using pretraining (i.e. we're picking up where we left off)
    if not use_pretrained and os.path.isfile(val_weights):
        print >> sys.stderr, 'Loading weights from {}!'.format(val_weights)
        m.model.load_weights(val_weights)

    m.train(nb_epoch, batch_size, val_every, val_weights, f1_weights,
            class_weight, save_weights, probs_loc)


if __name__ == '__main__':
    plac.call(main)
