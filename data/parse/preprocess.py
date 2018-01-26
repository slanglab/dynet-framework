import os
import re

def process(raw):
    raw = raw[2:-2]                             #strip outer parenthesis
    raw = re.sub(r'^[\s\)]+\)', '<TOK>)', raw)       #normalize words
    raw = raw.replace(')', ' )')                #tokenize with space
    raw = raw.replace('@', '')                  #normalize binarized nodes
    
    #label closing brackets
    raw = raw.split(' ')
    raw = label_closing_brackets(raw)
    raw = ' '.join(raw)

    return raw

def label_closing_brackets(lin):
    stack = []
    for i, tok in enumerate(lin):
        if tok.startswith('('):
            stack.append(tok)

        if tok == ')':
            lin[i] = tok + stack.pop()[1:]
    return lin

#TODO options to include:
#whether to include <TOK>
#labeled or unlabeled
seq2seqroot = os.environ['SEQ2SEQROOT']
fns = [ 'train', 'test', 'valid' ]

#simple tree preprocessing
toks = set()
for fn in fns:
    lm = list(open(os.path.join(seq2seqroot, 'data/lm/', fn), 'rt'))
    parse = list(open(os.path.join(seq2seqroot, 'data/parse/', fn), 'rt'))

    out = open(os.path.join(seq2seqroot, 'data/parse', '%s.seqs' % fn), 'wt')
    for sent, raw in zip(lm, parse):
        sent, raw = sent.strip(), raw.strip()

        seq = process(raw)
        out.write('%s\t%s\n' % (sent, seq))

        #vocab
        toks.update(seq.strip().split(' '))

#write vocabulary file
out_vocab = open(os.path.join(seq2seqroot, 'data/parse/out_vocab'), 'wt')
[ out_vocab.write('%s\n' % tok) for tok in toks ]
