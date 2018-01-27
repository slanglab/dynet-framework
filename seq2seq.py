import numpy as np
import _dynet as dy

class Seq2SeqBase:
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
            decoding = self.backward_batch(X_batch, y_batch, X_masks, eos)
        else:
            decoding = self.forward_batch(X_batch, len(y_batch), X_masks, eos)

        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        #needs to be averaged...
        return batch_loss, decoding

class Seq2SeqVanilla(Seq2SeqBase):
    def __init__(self, collection, vocab_size, out_vocab_size, input_embedding_dim=512, output_embedding_dim=32, \
            encoder_layers=2, decoder_layers=2, encoder_hidden_dim=512, decoder_hidden_dim=512, \
            encoder_dropout=0.5, decoder_dropout=0.5):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, input_embedding_dim))
        self.params['Wout_emb'] = collection.add_lookup_parameters((out_vocab_size, output_embedding_dim))

        self.encoder = dy.VanillaLSTMBuilder(encoder_layers, input_embedding_dim, encoder_hidden_dim, collection)
        self.decoder = dy.VanillaLSTMBuilder(decoder_layers, output_embedding_dim, decoder_hidden_dim, collection)

        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.layers = encoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

    def get_params(self):
        W_emb = self.params['W_emb']
        Wout_emb = self.params['Wout_emb']
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])
        return W_emb, Wout_emb, R, b

    def encode(self, W_emb, X_batch, X_masks):
        Xs = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        s0 = self.encoder.initial_state()
        state = s0

        h_t = None
        for X, mask in zip(Xs, X_masks):
           curr = state.add_input(X)
           h_i = curr.s()

           mask_expr = dy.inputVector(mask)
           mask = dy.reshape(mask_expr, (1,), len(mask))

           if h_t != None:
               h_t = [ h * mask + h * (1. - mask) for h in h_t ]
           else:
               h_t = h_i

        return h_t

    def backward_batch(self, X_batch, y_batch, X_masks, eos):
        W_emb, Wout_emb, R, b = self.get_params()

        self.encoder.set_dropouts(self.encoder_dropout, 0)
        self.decoder.set_dropouts(self.decoder_dropout, 0)

        final = self.encode(W_emb, X_batch, X_masks)
        s0 = self.decoder.initial_state(vecs=final)

        #teacher forcing
        eoses = dy.lookup_batch(Wout_emb, [ eos ] * len(X_batch[0]))
        tf = [ eoses ] + [ dy.lookup_batch(Wout_emb, y) for y in y_batch ]

        history = s0.transduce(tf)         #transduce lower layers
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in history ]

        return decoding

    def forward_batch(self, X_batch, maxlen, X_masks, eos):
        W_emb, Wout_emb, R, b = self.get_params()

        self.encoder.set_dropouts(0, 0)
        self.decoder.set_dropouts(0, 0)

        final = self.encode(W_emb, X_batch, X_masks)
        s0 = self.decoder.initial_state(vecs=final)

        #eos to start the sequence
        eoses = dy.lookup_batch(Wout_emb, [ eos ] * len(X_batch[0]))
        state = s0.add_input(eoses)

        decoding = []
        for i in range(0, maxlen):
            h_i = state.h()[-1]

            #probability dist
            decoding.append(dy.affine_transform([b, R, h_i]))

            #beam size 1
            probs = dy.softmax(decoding[-1])
            dim = probs.dim()
            flatten = dy.reshape(probs, (dim[0][0], dim[1]), batch_size=1)
            beam = np.argmax(flatten.value(), axis=0)
            state = state.add_input(dy.lookup_batch(Wout_emb, beam))
        return decoding

class Seq2SeqAttention(Seq2SeqBase):
    def __init__(self, collection, vocab_size, out_vocab_size, input_embedding_dim=512, output_embedding_dim=32, \
            encoder_layers=2, decoder_layers=2, encoder_hidden_dim=512, decoder_hidden_dim=512, \
            encoder_dropout=0.5, decoder_dropout=0.5, attention_dropout=0.5):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, input_embedding_dim))
        self.params['Wout_emb'] = collection.add_lookup_parameters((out_vocab_size, output_embedding_dim))

        self.encoder = dy.VanillaLSTMBuilder(encoder_layers, input_embedding_dim, encoder_hidden_dim, collection)
        self.decoder = [ dy.VanillaLSTMBuilder(decoder_layers-1, output_embedding_dim, decoder_hidden_dim, collection), \
                dy.VanillaLSTMBuilder(1, encoder_hidden_dim+decoder_hidden_dim, decoder_hidden_dim, collection) ]

        self.params['W_1'] = collection.add_parameters((decoder_hidden_dim, encoder_hidden_dim)) 
        self.params['W_2'] = collection.add_parameters((decoder_hidden_dim, decoder_hidden_dim)) 
        self.params['vT'] = collection.add_parameters((1, decoder_hidden_dim)) 

        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.layers = encoder_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.attention_dropout = attention_dropout

    def get_params(self):
        W_emb = self.params['W_emb']
        Wout_emb = self.params['Wout_emb']
        W_1 = dy.parameter(self.params['W_1'])
        W_2 = dy.parameter(self.params['W_2'])
        vT = dy.parameter(self.params['vT'])
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])
        return W_emb, Wout_emb, W_1, W_2, vT, R, b

    def encode(self, W_emb, X_batch, X_masks):
        X = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]

        s0 = self.encoder.initial_state()
        states = s0.add_inputs(X)
        encoding = [ state.h()[-1] for state in states ]

        final = states[-1].s()
        layers = self.layers
        vecs_low = final[:layers-1] + final[layers:-1]
        vecs_high = final[layers-1:layers] + final[-1:]

        for i, mask in enumerate(X_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), len(mask))
            encoding[i] = encoding[i] * mask
        encoding = dy.concatenate_cols(encoding)

        return encoding, vecs_low, vecs_high

    def attend(self, W_2, vT, state_high, xs, encoding, training=True):
        y = W_2 * state_high.h()[-1]
        if training:
            y = dy.dropout(y, self.attention_dropout)

        u = vT * dy.tanh(dy.colwise_add(xs, y))
        u = dy.reshape(u, (u.dim()[0][1],))
        a_t = dy.softmax(u)
        d_t = encoding * a_t
        return d_t, a_t

    def backward_batch(self, X_batch, y_batch, X_masks, eos):
        W_emb, Wout_emb, W_1, W_2, vT, R, b = self.get_params()

        self.encoder.set_dropout(self.encoder_dropout)
        self.decoder[0].set_dropout(self.decoder_dropout)
        self.decoder[1].set_dropout(self.decoder_dropout)

        encoding, vecs_low, vecs_high = self.encode(W_emb, X_batch, X_masks)

        s0_low = self.decoder[0].initial_state(vecs=vecs_low)
        s0_high = self.decoder[1].initial_state(vecs=vecs_high)

        xs = W_1 * encoding
        xs = dy.dropout(xs, self.attention_dropout)

        #teacher forcing
        eoses = dy.lookup_batch(Wout_emb, [ eos ] * len(X_batch[0]))
        tf = [ eoses ] + [ dy.lookup_batch(Wout_emb, y) for y in y_batch ]
        histories = s0_low.transduce(tf)         #transduce lower layers

        decoding = []
        state_high = s0_high
        for history in histories:
            d_t, a_t = self.attend(W_2, vT, state_high, xs, encoding, training=True)

            #input
            inp = dy.concatenate([history, d_t])
            state_high = state_high.add_input(inp)
            h_i = state_high.h()[-1]

            #logits
            decoding.append(dy.affine_transform([b, R, h_i]))
        return decoding

    def forward_batch(self, X_batch, maxlen, X_masks, eos):
        W_emb, Wout_emb, W_1, W_2, vT, R, b = self.get_params()

        self.encoder.set_dropouts(0, 0)
        self.decoder[0].set_dropouts(0, 0)
        self.decoder[1].set_dropouts(0, 0)

        encoding, vecs_low, vecs_high = self.encode(W_emb, X_batch, X_masks)

        s0_low = self.decoder[0].initial_state(vecs=vecs_low)
        s0_high = self.decoder[1].initial_state(vecs=vecs_high)

        xs = W_1 * encoding

        #eos to start the sequence
        eoses = dy.lookup_batch(Wout_emb, [ eos ] * len(X_batch[0]))
        state_low = s0_low.add_input(eoses)

        decoding = []
        state_high = s0_high
        for i in range(0, maxlen):
            d_t, a_t = self.attend(W_2, vT, state_high, xs, encoding, training=False)

            #input
            history = state_low.h()[-1]
            inp = dy.concatenate([history, d_t])

            state_high = state_high.add_input(inp)
            h_i = state_high.h()[-1]

            #probability dist
            decoding.append(dy.affine_transform([b, R, h_i]))

            #beam size 1
            probs = dy.softmax(decoding[-1])
            dim = probs.dim()
            flatten = dy.reshape(probs, (dim[0][0], dim[1]), batch_size=1)
            beam = np.argmax(flatten.value(), axis=0)
            state_low = state_low.add_input(dy.lookup_batch(Wout_emb, beam))
        return decoding
