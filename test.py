import os
import random

import utils

def validate(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw, \
        dy, seq2seq, out_vocab, run='/runs/baseline', valid_fn='validation'):
    val_loss = 0.
    correct_toks = 0.
    total_toks = 0.
    correct_val_seqs = 0.
    total_val_seqs = 0.

    validation = open(os.path.join(run, valid_fn), 'wt')
    for X_batch, y_batch, X_masks, y_masks, X_batch_raw, y_batch_raw in \
            zip(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw):
        dy.renew_cg()
        batch_loss, decoding = seq2seq.one_batch( \
                X_batch, y_batch, X_masks, y_masks, training=False)
        val_loss += batch_loss.value()

        y_pred = seq2seq.to_sequence_batch(decoding, out_vocab)
        for X_raw, y_, y in zip(X_batch_raw, y_batch_raw, y_pred):
            validation.write('%s\t%s\t%s\n' % \
                    (' '.join(X_raw), ' '.join(y_), ' '.join(y)))
            correct_val_seqs += 1 if all([ tok_ == tok or tok_ == '<mask>' \
                    for tok_, tok in zip(y_, y) ]) else 0
            total_val_seqs += 1
            count = [ tok_ == tok for tok_, tok in zip(y_, y) if tok_ != '<mask>' ]
            correct_toks += count.count(True)
            total_toks += len(count)
    seq_accuracy = correct_val_seqs/total_val_seqs
    tok_accuracy = correct_toks/total_toks

    validation.close()

    return val_loss, seq_accuracy, tok_accuracy

#tests on section wsj_23
#TODO implement argparse
if __name__ == '__main__':
    import _dynet as dy
    dy_params = dy.DynetParams()
    dy_params.set_random_seed(random.randint(0, 1000))
    dy_params.set_autobatch(True)
    dy_params.set_requested_gpus(1)
    dy_params.set_mem(20480)
    dy_params.init()

    RUN = 'runs/att_baseline'

    print('Reading vocab...')
    in_vocab, out_vocab = utils.load_vocab()
    print('Done.')

    print('Reading test data...')
    VALID_BATCH_SIZE=32
    X_valid, y_valid, X_valid_masks, y_valid_masks = \
            utils.load(in_vocab, out_vocab, section='wsj_24', batch_size=VALID_BATCH_SIZE)
    X_valid_raw, y_valid_raw = utils.load_raw(section='wsj_24', batch_size=VALID_BATCH_SIZE)
    print('Done.')

    print('Building model...')
    collection = dy.ParameterCollection()
    seq2seq = Seq2SeqAttention(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    checkpoint = os.path.join(RUN, 'baseline.model')
    print('Loading model from %s.' % checkpoint)

    print('Loading model...')
    collection.populate(checkpoint)
    print('Done.')

    print('Testing...')
    val_loss, accuracy, tok_accuracy = validate(X_valid, y_valid, X_valid_masks, y_valid_masks, \
            X_valid_raw, y_valid_raw, dy, seq2seq, out_vocab, run=RUN, valid_fn='test')
    print('Done. Sequence-level accuracy: %f. Token-level accuracy: %f' % (accuracy, tok_accuracy))

    #TODO generates reports from evalb automatically
