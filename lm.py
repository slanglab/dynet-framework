import numpy as np
import _dynet as dy

class LanguageModelBase:
    def to_sequence_batch(self, decoding, out_vocab):
        batch_size = decoding[0].dim()[1]
        decoding = [ dy.softmax(x) for x in decoding ]
        decoding = [ dy.reshape(x, (len(out_vocab), batch_size), batch_size=1) for x in decoding ]
        decoding = [ np.argmax(x.value(), axis=0) for x in decoding ]
        decoding = [  [ x[i] for x in decoding ] for i in range(0, batch_size) ]
        return [ [ out_vocab[y] for y in x ] for x in decoding ]

    def one_batch(self, X_batch, y_batch, X_masks, y_masks, eos=133, training=True):
        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        X_masks = zip(*X_masks)
        y_batch = zip(*y_batch)
        y_masks = zip(*y_masks)

        if training:
            decoding = self.backward_batch(X_batch, y_batch, X_masks)

            batch_loss = []
            for x, y, mask in zip(decoding, y_batch, y_masks):
                mask_expr = dy.inputVector(mask)
                mask = dy.reshape(mask_expr, (1,), batch_size)
                batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
            batch_loss = dy.esum(batch_loss)
            batch_loss = dy.sum_batches(batch_loss)

        else:
            decoding = self.forward_batch(X_batch, len(y_batch), X_masks, eos)

        #needs to be averaged...
        return batch_loss, decoding

class LSTMLanguageModel(LanguageModelBase):
    def __init__(self, collection, vocab_size, out_vocab_size, input_embedding_dim=256, output_embedding_dim=128, \
            lstm_layers=4, lstm_hidden_dim=256, input_dropout=0.3, recurrent_dropout=0.3):
        self.collection = collection
        self.params = {}

        self.lstm = dy.VanillaLSTMBuilder(lstm_layers, input_embedding_dim, lstm_hidden_dim, collection)

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, input_embedding_dim))
        self.params['R'] = collection.add_parameters((out_vocab_size, lstm_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.layers = lstm_layers
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout

    def get_params(self):
        W_emb = self.params['W_emb']
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])
        return W_emb, R, b

    def backward_batch(self, X_batch, y_batch, X_masks):
        W_emb, R, b = self.get_params()
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        self.lstm.set_dropouts(self.input_dropout, self.recurrent_dropout)
        s0 = self.lstm.initial_state()
        states = s0.transduce(X)
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in states ]

        return decoding

    def forward_batch(self, X_batch, maxlen, X_masks, eos):
        W_emb, R, b = self.get_params()
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        self.lstm.set_dropouts(0, 0)
        s0 = self.lstm.initial_state()
        states = s0.transduce(X)
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in states ]

        return decoding

    def sample_batch(self, batch_size, maxlen, eos):
        W_emb, R, b = self.get_params()
        X = dy.lookup_batch(W_emb, [ eos for i in xrange(0, batch_size) ])

        self.lstm.set_dropouts(0, 0)
        s0 = self.lstm.initial_state()

        return samples
