from __future__ import print_function

import os
import sys
import time
import random
import argparse

import _dynet as dy
import numpy as np

import utils, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='General seq2seq and language \
            modelling framework for Dynet written by Johnny Wei - jwei@umass.edu')

    parser.add_argument('--run', type=str, default='runs/experiment', 
            help='Experiment directory.')
    parser.add_argument('--model', type=str, default='Seq2SeqVanilla', 
            help='Model to train.')
    parser.add_argument('--train', type=str, default='data/wsj_2-21', 
            help='Training set.')
    parser.add_argument('--cutoff', type=int, default=1048,
            help='Cutoff N longest training examples (default for ptb).')
    parser.add_argument('--dev', type=str, default='data/wsj_24', 
            help='Validation set.')
    parser.add_argument('--format', type=str, default='parse',
            help='Format of input data.')
    parser.add_argument('--in_vocab', type=str, default='data/in_vocab', 
            help='Input vocabulary.')
    parser.add_argument('--out_vocab', type=str, default='data/out_vocab', 
            help='Ouput vocabulary.')

    parser.add_argument('--validation', type=str, default='validation', 
            help='Name of validation results.')
    parser.add_argument('--val_metric', type=str, default='evalb',
            help='Metric to use for validation.')
    parser.add_argument('--log', type=str, default='log',
            help='Metric to use for validation.')
    parser.add_argument('--batch_size', type=int, default=128, 
            help='Training batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, 
            help='Validation batch size.')

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
    parser.add_argument('--populate', type=str,
            help='Load a pretrained model.')
    parser.add_argument('--checkpoint', type=str, default='seq2seq.model',
            help='Name for checkpoints.')

    parser.add_argument('--epochs', type=int, default=200, 
            help='Epochs.')
    parser.add_argument('--trainer', type=str, default='adam', 
            help='Optimizer to use (adam and sgd only).')
    parser.add_argument('--lr', type=float, default=0.001, 
            help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=1.0, 
            help='Learning rate decay per epoch.')
    parser.add_argument('--clip', type=float, default=5.0, 
            help='Gradient clipping threshold.')
    parser.add_argument('--patience', type=int, default=3, 
            help='Epochs to half learning rate if no improvement.')
    parser.add_argument('--monitor', type=str, default='train_loss', 
            help='Quantity to monitor for learning rate halving.')
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

    print('Reading train/valid data...')
    X_train, y_train, X_train_masks, y_train_masks = utils.load( \
            in_vocab, out_vocab, section=args.train, 
            batch_size=args.batch_size, imports=args.imports,
            format=args.format, cutoff=args.cutoff)

    X_valid_raw, y_valid_raw = utils.load_raw( \
            section=args.dev, batch_size=args.val_batch_size,
            imports=args.imports, format=args.format)
    X_valid, y_valid, X_valid_masks, y_valid_masks = utils.load( \
            in_vocab, out_vocab, section=args.dev, batch_size=args.val_batch_size,
            imports=args.imports, format=args.format)
    print('Done.')

    print('Contains %d unique words.' % len(in_vocab))
    print('Read in %d batches containing %d examples.' % (len(X_train), sum(len(X_batch) for X_batch in X_train)))

    print('Input vocabulary sample...')
    print(', '.join(in_vocab[:10]))
    print('Output vocabulary sample...')
    print(', '.join(out_vocab[:10]))

    print('Checkpointing models on validation accuracy...')
    validate = test.get_val_metric(args.val_metric, args.imports)
    highest_val_accuracy = float('-infinity')

    print('Halving learning rate on metric (%s) with patience.' % args.val_metric)
    monitor = float('-infinity')

    print('Building model...')
    collection = dy.ParameterCollection()
    imports = __import__(args.imports)
    Model = getattr(imports, args.model)
    seq2seq = Model(collection, len(in_vocab), len(out_vocab))
    print('Done.')

    log = open(os.path.join(args.run, args.log), 'wta')
    print('Training logs will be written to %s.' % os.path.join(args.run, args.log))

    checkpoint = os.path.join(args.run, args.checkpoint)
    print('Checkpoints will be written to %s.' % checkpoint)

    if args.populate:
        print('Loading model...')
        collection.populate(checkpoint)
        print('Done.')
    else:
        print('Initialized new model.')

    print('Training model...')
    if args.trainer == 'adam':
        trainer = dy.AdamTrainer(collection, alpha=args.lr)
    elif args.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(collection, learning_rate=args.lr)
    trainer.set_clip_threshold(args.clip)

    patience = 1
    total_seqs = sum([ len(X_batch) for X_batch in X_train ])
    Ms = [ sum([ sum(seq) for seq in batch ]) for batch in X_train_masks ]

    for epoch in range(1, args.epochs+1):
        seqs, toks, loss = 0, 0, 0.
        start = time.time()

        #length sorted batches, train batches in random order
        indexes = range(0, len(X_train))
        random.shuffle(indexes)

        for i, index in enumerate(indexes, 1):
            X_batch, y_batch, X_masks, y_masks, M = X_train[index], y_train[index], \
                    X_train_masks[index], y_train_masks[index], Ms[index]

            dy.renew_cg()
            batch_loss, _ = seq2seq.one_batch(X_batch, y_batch, X_masks, y_masks, eos=eos)
            normalized_batch_loss = batch_loss / len(X_batch)

            normalized_batch_loss.backward()
            trainer.update()

            loss += batch_loss.value()
            seqs += len(X_batch)
            toks += M
            avg_seq_loss = loss / seqs
            avg_tok_loss = loss / toks
            elapsed = time.time() - start

            print(('Epoch %d. Time elapsed: %ds, %d/%d. Total Loss: %.4f. ' + \
                    'Average sequence loss: %.4f. Average Token Loss: %.4f.\r') % \
                    (epoch, elapsed, seqs, total_seqs, loss, avg_seq_loss, avg_tok_loss), \
                end='')

        print()
        print('Done. Total loss: %f' % loss)
        trainer.status()
        print()

        #decay learning rate
        trainer.learning_rate *= args.lr_decay

        print('Validating...')
        val_loss, accuracy, metrics = validate( \
                X_valid, y_valid, X_valid_masks, y_valid_masks, \
                X_valid_raw, y_valid_raw, dy, seq2seq, out_vocab, \
                run=args.run, valid_fn=args.validation)
        print('Done. ' + ' '.join([ i[0] for i in metrics ] % metrics[0][1] \
                if len(metrics) == 1 else [ i[0] % i[1] for i in metrics ]))

        print('Logging training status...')
        log_quantities = [ epoch, trainer.learning_rate, loss ]
        log_quantities.extend(i[1] for i in metrics)
        log_quantities = [ str(q) for q in log_quantities ]
        log.write('%s\n' % '\t'.join(log_quantities))
        log.flush()

        #checkpointing
        if accuracy > highest_val_accuracy:
            print('Highest accuracy yet. Saving model...')
            highest_val_accuracy = accuracy
            collection.save(checkpoint)

        if args.monitor == 'train_loss':
            quantity = -loss
        elif args.monitor == 'dev_loss':
            quantity = -val_loss
        elif args.monitor == 'val_metric':
            quantity = accuracy
        elif args.monitor == 'none':
            quantity = 0 if monitor < 0 else monitor + 1
        else:
            print('Not implemented.')
            quantity = 0

        #patience for learning rate halving - adam
        if monitor < quantity:
            print('Monitored quantity improved.')
        else:
            if patience >= args.patience:
                print('No improvement. Halving learning rate.')
                trainer.learning_rate *= 0.5
                patience = 1
            else:
                print('Patience at %d' % patience)
                patience += 1
        monitor = quantity

        print('Done.')
    print('Done.')
