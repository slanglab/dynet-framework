from __future__ import print_function

import os
import sys
import time
import random
import argparse

import _dynet as dy
import numpy as np

import utils

def sample(X_valid, y_valid, X_valid_masks, y_valid_masks, X_valid_raw, y_valid_raw, \
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
        for i in range(0, 100000):
            sample = lm.sample_one(30, X_batch[0][0])
            sample = lm.to_sequence(sample, out_vocab)
            validation.write('%s\n' % ' '.join(sample))
        validation.close()

    metrics = [ ('Validation loss: %f.', val_loss), \
            ('Average Token loss: %f.', avg_tok_loss), \
            ('Perplexity: %f.', perplexity) ]
    return val_loss, -perplexity, metrics
