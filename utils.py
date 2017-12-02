import os

def ptb(section, padding='<EOS>', column=0):
    with open(section, 'rt') as fh:
        data = [ i.split('\t')[column] for i in fh ]
    data = [ ex.strip().split(' ') for ex in data ]
    data = [ ex + [padding] for ex in data ]
    vocab = set([ word for sent in data for word in sent ])
    return vocab, data

def read_vocab(vocab='vocab', directory='data/'):
    with open(vocab, 'rt') as fh:
        vocab = [ i.strip().split('\t')[0] for i in fh ]
    return vocab

def text_to_sequence(texts, vocab):
    word_to_n = { word : i for i, word in enumerate(vocab, 0) }
    n_to_word = { i : word for word, i in word_to_n.items() }
    sequences = []
    for sent in texts:
        sequences.append([ word_to_n[word] for word in sent ])
    return sequences, word_to_n, n_to_word

remove_tokens = \
        lambda x, token='<TOK>': [ [ tok for tok in ex if tok != token ] for ex in x ]

def sort_by_len(X, y, order=-1):
    data = list(zip(X, y))
    data.sort(key=lambda x: order*len(x[1]))
    return [ i[0] for i in data ], [ i[1] for i in data ]

def batch(X, batch_size, mask=0., masking='post'):
    ex, masks = [], []
    for i in xrange(0, len(X), batch_size):
        X_ = X[i:i + batch_size]
        X_len = max([ len(x) for x in X_ ])
        X_padding = [ X_len - len(x) for x in X_ ]

        if masking == 'post':
            X_padded = [ x + [mask] * mask_len for x, mask_len  in zip(X_, X_padding) ]
            X_mask = [ [1]*len(x)  + [0]*mask_len for x, mask_len  in zip(X_, X_padding) ]
        elif masking == 'pre':
            X_padded = [ ([mask] * mask_len) + x for x, mask_len  in zip(X_, X_padding) ]
            X_mask = [ [0]*mask_len + [1]*len(x) for x, mask_len  in zip(X_, X_padding) ]

        ex.append(X_padded)
        masks.append(X_mask)
    return ex, masks

def load_vocab(in_fn, out_fn, in_toks=['<unk>', '<EOS>', '<mask>'], out_toks=['<EOS>', '<mask>']):
    in_vocab = read_vocab(vocab=in_fn)
    out_vocab = read_vocab(vocab=out_fn)
    in_vocab += in_toks
    out_vocab += out_toks
    return in_vocab, out_vocab

def load_raw(section='wsj_24', batch_size=128):
    _, X_valid = ptb(section=section, column=0)
    _, y_valid = ptb(section=section, column=1)
    X_valid, y_valid = remove_tokens(X_valid), remove_tokens(y_valid)
    X_valid, y_valid = sort_by_len(X_valid, y_valid)
    X_valid_raw, _ = batch(X_valid, batch_size=batch_size, mask='<mask>', masking='pre') 
    y_valid_raw, _ = batch(y_valid, batch_size=batch_size, mask='<mask>', masking='post')
    return X_valid_raw, y_valid_raw

def load(in_vocab, out_vocab, section='wsj_2-21', batch_size=128, cutoff=0):
    _, X_train = ptb(section=section, column=0)
    _, y_train = ptb(section=section, column=1)
    X_train, y_train = remove_tokens(X_train), remove_tokens(y_train)       #TODO makes y sequences shorter
    X_train, y_train = sort_by_len(X_train, y_train)
    if cutoff != 0:
        X_train, y_train = X_train[cutoff:], y_train[cutoff:]             #cutoff longest examples
        #X_train, y_train = X_train[:-cutoff], y_train[:-cutoff]
    X_train_seq, word_to_n, n_to_word = text_to_sequence(X_train, in_vocab)
    y_train_seq, _, _ = text_to_sequence(y_train, out_vocab)
    X_train_seq, X_train_masks = batch(X_train_seq, batch_size=batch_size, mask=len(in_vocab)-1, masking='pre')
    y_train_seq, y_train_masks = batch(y_train_seq, batch_size=batch_size, mask=len(out_vocab)-1, masking='post')
    return X_train_seq, y_train_seq, X_train_masks, y_train_masks
