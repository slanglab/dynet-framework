import numpy as np
import _dynet as dy

class LanguageModelBase:
    def to_sequence(self, decoding, out_vocab):
        decoding = [ out_vocab[x] for x in decoding ]
        return decoding

    def one_batch(self, X_batch, y_batch, X_masks, y_masks, eos=133, training=True):
        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        X_masks = zip(*X_masks)
        y_batch = zip(*y_batch)
        y_masks = zip(*y_masks)

        if training:
            decoding = self.backward_batch(X_batch, y_batch, X_masks)
        else:
            decoding = self.forward_batch(X_batch, len(y_batch), X_masks, eos)

        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * self.pickneglogsoftmax_batch(x, y))    #calls subclass function...
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        return batch_loss, decoding

class LSTMLanguageModel(LanguageModelBase):
    def __init__(self, collection, vocab_size, out_vocab_size, input_embedding_dim=512, \
            lstm_layers=3, lstm_hidden_dim=1024, input_dropout=0.5, recurrent_dropout=0.):
        self.collection = collection
        self.params = {}

        self.lstm = dy.VanillaLSTMBuilder(lstm_layers, input_embedding_dim, lstm_hidden_dim, collection)

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, input_embedding_dim))
        self.softmax = dy.StandardSoftmaxBuilder(lstm_hidden_dim, out_vocab_size, collection)

        self.layers = lstm_layers
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout

    def get_params(self):
        W_emb = self.params['W_emb']
        softmax = self.softmax
        return W_emb, softmax

    def backward_batch(self, X_batch, y_batch, X_masks):
        W_emb, softmax = self.get_params()
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        self.lstm.set_dropouts(self.input_dropout, self.recurrent_dropout)
        s0 = self.lstm.initial_state()

        states = s0.transduce(X)
        states = [ dy.dropout(h_i, self.input_dropout) for h_i in states ]

        return states

    def forward_batch(self, X_batch, maxlen, X_masks, eos):
        W_emb, softmax = self.get_params()
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        self.lstm.disable_dropout()
        s0 = self.lstm.initial_state()
        states = s0.transduce(X)

        return states

    def pickneglogsoftmax_batch(self, x, y):
        W_emb, softmax = self.get_params()
        return softmax.neg_log_softmax_batch(x, y)

    def sample_one(self, maxlen, eos):
        W_emb, softmax = self.get_params()
        self.lstm.set_dropouts(0, 0)
        s0 = self.lstm.initial_state()
        state = s0.add_input(W_emb[eos])

        samples = []
        for i in range(0, maxlen):
            #probability dist
            h_i = state.h()[-1]
            choice = softmax.sample(h_i)

            samples.append(choice)
            state = state.add_input(W_emb[choice])

        return samples
