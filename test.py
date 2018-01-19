from __future__ import print_function

import os
import sys
import time
import random
import argparse

import _dynet as dy
import numpy as np

import utils

def get_val_metric(val_metric, imports):
    if val_metric in [ 'perplexity', 'accuracy' ]:
        imports = __import__('test')
        validation = getattr(imports, val_metric)
    else:
        imports = __import__(imports)
        validation = getattr(imports, val_metric)
    return validation

def accuracy(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw, \
        dy, seq2seq, out_vocab, run='/runs/baseline', valid_fn='validation'):
    val_loss = 0.
    correct_toks = 0.
    total_toks = 0.
    correct_seqs = 0.
    total_seqs = 0.

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
            correct_seqs += 1 if all([ tok_ == tok or tok_ == '<mask>' \
                    for tok_, tok in zip(y_, y) ]) else 0
            total_seqs += 1
            count = [ tok_ == tok for tok_, tok in zip(y_, y) if tok_ != '<mask>' ]
            correct_toks += count.count(True)
            total_toks += len(count)
    seq_accuracy = correct_seqs/total_seqs
    tok_accuracy = correct_toks/total_toks
    validation.close()

    metrics = [ ('Validation loss: %f.', val_loss), \
            ('Token-level accuracy: %f.', tok_accuracy), \
            ('Sequence-level accuracy: %f.', seq_accuracy) ]

    return val_loss, seq_accuracy, metrics

def perplexity(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw, \
        dy, lm, out_vocab, validate_samples=True, run='/runs/baseline', valid_fn='validation'):
    val_loss = 0.

    validation = open(os.path.join(run, valid_fn), 'wt')
    for X_batch, y_batch, X_masks, y_masks, X_batch_raw, y_batch_raw in \
            zip(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw):
        dy.renew_cg()
        batch_loss, decoding = lm.one_batch( \
                X_batch, y_batch, X_masks, y_masks, training=False)
        val_loss += batch_loss.value()

    M = sum([ sum([ sum(seq) for seq in batch ]) for batch in X_valid_masks ])
    avg_tok_loss = val_loss / M
    perplexity = np.exp(val_loss / M)

    #validate some samples from lm
    if validate_samples:
        samples = []
        for i in range(0, 32):
            sample = lm.sample_one(20, X_batch[0][0])
            samples.append(lm.to_sequence(sample, out_vocab))

        for sample in samples:
            validation.write('%s\n' % ' '.join(sample))
        validation.close()

    metrics = [ ('Validation loss: %f.', val_loss), \
            ('Average Token loss: %f.', avg_tok_loss), \
            ('Perplexity: %f.', perplexity) ]
    return val_loss, -perplexity, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General seq2seq and language \
            modelling framework (testing) for Dynet written by Johnny Wei - jwei@umass.edu')

    parser.add_argument('--run', type=str, default='runs/experiment', 
            help='Experiment directory.')
    parser.add_argument('--model', type=str, default='Seq2SeqVanilla', 
            help='Model to train.')
    parser.add_argument('--test', type=str, default='data/wsj_24', 
            help='Test set.')
    parser.add_argument('--in_vocab', type=str, default='data/in_vocab', 
            help='Input vocabulary.')
    parser.add_argument('--out_vocab', type=str, default='data/out_vocab', 
            help='Ouput vocabulary.')

    parser.add_argument('--format', type=str, default='parse',
            help='Format of input data.')
    parser.add_argument('--val_metric', type=str, default='evalb',
            help='Metric to use for validation.')
    parser.add_argument('--batch_size', type=int, default=128, 
            help='Training batch size.')

    parser.add_argument('--mem', type=int, default=22528,
            help='Memory to allocate (default=22GB).')
    parser.add_argument('--dy_seed', type=int, default=0,
            help='Random seed for dynet.')
    parser.add_argument('--gpus', type=int, default=1,
            help='GPUs to allocate to dynet.')
    parser.add_argument('--autobatch', type=bool, default=False,
            help='Autobatching for dynet')

    parser.add_argument('--seed', type=int, default=0,
            help='Seed for python random.')
    parser.add_argument('--imports', type=str, default='seq2seq',
            help='File to look for model classes in (import seq2seq).')
    parser.add_argument('--populate', type=str, required=True,
            help='Load a pretrained model.')
    args = parser.parse_args()

    random.seed(args.seed)
    dy_params = dy.DynetParams()
    dy_params.set_random_seed(args.dy_seed)
    dy_params.set_autobatch(args.autobatch)
    dy_params.set_requested_gpus(args.gpus)
    dy_params.set_mem(args.mem)
    dy_params.init()

    print('Reading vocab...')
    in_vocab, out_vocab = utils.load_vocab(args.in_vocab, args.out_vocab)
    eos = out_vocab.index('<EOS>')
    print('Done.')

    print('Reading test data...')
    X_valid_raw, y_valid_raw = utils.load_raw( \
            section=args.test, batch_size=args.batch_size,
            imports=args.imports, format=args.format)
    X_valid, y_valid, X_valid_masks, y_valid_masks = utils.load( \
            in_vocab, out_vocab, section=args.test, batch_size=args.batch_size,
            imports=args.imports, format=args.format)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d examples.' % len(X_train))

    print('Input vocabulary sample...')
    print(', '.join(in_vocab[:10]))
    print('Output vocabulary sample...')
    print(', '.join(out_vocab[:10]))

    print('Building model...')
    collection = dy.ParameterCollection()
    imports = __import__(args.imports)
    Model = getattr(imports, args.model)
    seq2seq = Model(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    print('Loading model...')
    collection.populate(checkpoint)
    print('Done.')
 
    print('Testing...')
    val_loss, accuracy, metrics = validate( \
            X_valid, y_valid, X_valid_masks, y_valid_masks, \
            X_valid_raw, y_valid_raw, dy, seq2seq, out_vocab, \
            run=args.run, valid_fn=args.validation)
    print('Done. ' + ' '.join([ i[0] for i in metrics ] % metrics[0][1] \
            if len(metrics) == 1 else [ i[0] % i[1] for i in metrics ]))


