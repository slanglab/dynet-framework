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

    def kl_divergence(self, mu, log_sigma):
        return 0.5 * dy.sum_elems(dy.exp(log_sigma) + dy.square(mu) - 1. - log_sigma)

    def one_batch(self, X_batch, y_batch, X_masks, y_masks, eos=133, training=True):
        batch_size = len(X_batch)
        X_batch = zip(*X_batch)
        X_masks = zip(*X_masks)
        y_batch = zip(*y_batch)
        y_masks = zip(*y_masks)

        if training:
            decoding, mus, log_sigmas = self.backward_batch(X_batch, y_batch, X_masks, eos)

            #calculate KL divergence
            kl_loss = []
            for mu, log_sigma in zip(mus, log_sigmas):
                kl_loss.append(self.kl_divergence(mu, log_sigma))
            kl_loss = dy.esum(kl_loss)
            kl_loss = dy.sum_batches(kl_loss)
        else:
            decoding, mus, log_sigmas = self.forward_batch(X_batch, len(y_batch), X_masks, eos)
            kl_loss = 0.

        batch_loss = []
        for x, y, mask in zip(decoding, y_batch, y_masks):
            mask_expr = dy.inputVector(mask)
            mask = dy.reshape(mask_expr, (1,), batch_size)
            batch_loss.append(mask * dy.pickneglogsoftmax_batch(x, y))
        batch_loss = dy.esum(batch_loss)
        batch_loss = dy.sum_batches(batch_loss)

        #needs to be averaged...
        return batch_loss, kl_loss, decoding

class VanillaVAE(Seq2SeqBase):
    def __init__(self, collection, vocab_size, out_vocab_size, input_embedding_dim=512, output_embedding_dim=32, \
            encoder_layers=2, decoder_layers=2, encoder_hidden_dim=1024, decoder_hidden_dim=1024, latent_dim=1024, \
            encoder_dropout=0.5, decoder_dropout=0.5):
        self.collection = collection
        self.params = {}

        self.params['W_emb'] = collection.add_lookup_parameters((vocab_size, input_embedding_dim))
        self.params['Wout_emb'] = collection.add_lookup_parameters((out_vocab_size, output_embedding_dim))

        self.encoder = dy.VanillaLSTMBuilder(encoder_layers, output_embedding_dim, encoder_hidden_dim, collection)
        self.decoder = dy.VanillaLSTMBuilder(decoder_layers, input_embedding_dim, decoder_hidden_dim, collection)

        self.params['R'] = collection.add_parameters((out_vocab_size, decoder_hidden_dim)) 
        self.params['b'] = collection.add_parameters((out_vocab_size,)) 

        self.params['W_mu'] = collection.add_parameters((latent_dim, encoder_hidden_dim)) 
        self.params['W_logsigma'] = collection.add_parameters((latent_dim, encoder_hidden_dim)) 
        self.params['W_z_h'] = collection.add_parameters((decoder_hidden_dim, latent_dim))

        self.latent_dim = latent_dim
        self.layers = encoder_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

    def get_params(self):
        W_emb = self.params['W_emb']
        Wout_emb = self.params['Wout_emb']
        R = dy.parameter(self.params['R'])
        b = dy.parameter(self.params['b'])
        W_mu = dy.parameter(self.params['W_mu'])
        W_logsigma = dy.parameter(self.params['W_logsigma'])
        W_z_h = dy.parameter(self.params['W_z_h'])
        return W_emb, Wout_emb, R, b, W_mu, W_logsigma, W_z_h

    def encode(self, W_emb, X_batch, X_masks):
        Xs = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]
        state = self.encoder.initial_state()

        h_t = None
        for X, mask in zip(Xs, X_masks):
           state = state.add_input(X)
           h_i = state.s()

           mask_expr = dy.inputVector(mask)
           mask = dy.reshape(mask_expr, (1,), len(mask))

           if h_t != None:
               h_t = [ c * mask + h * (1. - mask) for c, h in zip(h_i, h_t) ]
           else:
               h_t = h_i

        return h_t

    #y_batch is prior, x_batch is posterior
    #y_batch is parse tree, x_batch is sentence
    def backward_batch(self, X_batch, y_batch, X_masks, eos, dropout=True):
        W_emb, Wout_emb, R, b, W_mu, W_logsigma, W_z_h = self.get_params()

        if dropout:
            self.encoder.set_dropouts(self.encoder_dropout, 0)
            self.decoder.set_dropouts(self.decoder_dropout, 0)
        else:
            self.encoder.set_dropouts(0, 0)
            self.decoder.set_dropouts(0, 0)

        h_ts = self.encode(Wout_emb, y_batch, X_masks)

        #sample z
        final = []
        mus, log_sigmas = [], []
        for i, h_t in enumerate(h_ts):
            if not i % 2 == 0:
                mu = W_mu * h_t
                log_sigma = dy.elu(W_logsigma * h_t) + 1.
                z = mu + dy.cmult(dy.random_normal(self.latent_dim), log_sigma)

                mus.append(mu)
                log_sigmas.append(log_sigma)
                final.append(W_z_h * z)
            else:
                #memory cells for lstm
                final.append(dy.zeros(self.decoder_hidden_dim))

        s0 = self.decoder.initial_state(vecs=final)

        #language modelling
        Xs = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]
        history = s0.transduce(Xs)         #transduce lower layers
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in history ]

        return decoding, mus, log_sigmas

    def forward_batch(self, X_batch, maxlen, X_masks, eos):
        W_emb, Wout_emb, R, b, W_mu, W_logsigma, W_z_h = self.get_params()

        final = []
        for i in range(0, 2*self.layers):
            if not i % 2 == 0:
                h_t = W_z_h * dy.random_normal(self.latent_dim)
                final.append(h_t)
            else:
                #memory cells for lstm
                final.append(dy.zeros(self.decoder_hidden_dim))
        s0 = self.decoder.initial_state(vecs=final)

        #language modelling
        Xs = [ dy.lookup_batch(W_emb, tok_batch) for tok_batch in X_batch ]
        history = s0.transduce(Xs)         #transduce lower layers
        decoding = [ dy.affine_transform([b, R, h_i]) for h_i in history ]

        return decoding, [], []
