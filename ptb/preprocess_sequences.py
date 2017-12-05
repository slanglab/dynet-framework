import re
from nltk.tree import *

def simple_normalize(word):
    word = re.sub(r'[$,-/]', '', word)
    word = word.strip()
    try:
        if float(word):
            return 'N'
    except:
        pass
    return word

def label_closing_brackets(lin):
    stack = []
    lin = lin.split(' ')
    for i, tok in enumerate(lin):
        if tok.startswith('('):
            stack.append(tok)

        if tok == ')':
            lin[i] = tok + stack.pop()[1:]

    return ' '.join(lin)

def norm_label(label):
    if '+' in label:
        label = label[:label.index('+')]

    if '~' in label:
        label = label[:label.index('~')]

    if label == '$':
        return label

    if label == '-RRB-' or label == '-LRB-':
        return label

    punct = ['``', "''" , '.', ',', ':', ':pound:']
    if label in punct:
        return 'PUNCT'
    
    if label == '$':
        return label

    chars = ['$', '|', '-', '=', '~']
    for char in chars:
        if char in label:
            label = label[:label.index(char)]

    return label

def simple_linearize(tree, normalize=simple_normalize, label=True, token=True, margin=2**8):
    if not label:
        tree.set_label('')
        for subtree in tree.subtrees():
            subtree.set_label('')
    else:
        label = tree.label()
        tree.set_label(norm_label(label))

        for subtree in tree.subtrees():
            label = norm_label(subtree.label())
            subtree.set_label(label)

    for subtree in tree.subtrees(filter=lambda x: x.height() == 2):
        leaf = normalize(subtree[0])
        
        if token:
            subtree[0] = '<TOK>'
            continue

    #output as string
    lin = tree.pformat(margin=margin, nodesep='', parens=['(', ' )'])
    lin = re.sub(r'\s+', ' ', lin)
    lin = label_closing_brackets(lin)
    return lin

def vocabularize(toks, normalize=simple_normalize, threshold=10000, symbols=2):
    vocab = {}
    for tok in toks:
        word = normalize(tok)
        counter = vocab.get(word, 0)
        vocab[word] = counter + 1
    
    #filter out vocab
    vocab = list(vocab.items())
    vocab.sort(key=lambda x: -x[1])
    vocab = vocab[:threshold-symbols]
    vocab = [ i[0] for i in vocab ]

    return vocab

if __name__ == '__main__':
    a_sentences = 'processed/wsj_23.sent'
    a_parses = 'processed/wsj_23.parse'
    out = 'processed/wsj_23'
    out_vocab = 'processed/out_vocab'
    in_vocab = 'processed/in_vocab'
    linearize = simple_linearize

    print('Loading data...')
    sentences = [ i.strip().split(' ') for i in open(a_sentences) ]
    #sentences = [ [ tok for tok in sentence if tok != '' ] for sentence in sentences ]
    parses = [ linearize(Tree.fromstring(i)) for i in open(a_parses) ]
    print('done.')

    print('Generating input vocabulary...')
    vocab = vocabularize(sum(sentences, []))
    h = open(in_vocab, 'wt')
    for word in vocab:
        h.write('%s\n' % word)
    h.close()
    vocab = set(vocab)
    print('done.')

    print('Generating output vocabulary...')
    toks = vocabularize(sum([ i.split(' ') for i in parses], []))
    h = open(out_vocab, 'wt')
    for word in toks:
        h.write('%s\n' % word)
    h.close()
    print('done.')

    print('Generating training file...')
    h = open(out, 'wt')
    for sentence, parse in zip(sentences, parses):
        sentence = [ '<unk>' if tok not in vocab else tok for tok in sentence ]
        sentence = ' '.join(sentence).strip()
        h.write('%s\t%s\n' % (sentence, parse))
    print('done.')

    sentences = [ sentence for sentence in sentences if '\t' in sentence ]
    parses = [ parse for parse in parses if ' ( ' in parse ]
    print(sentences)
    print(parses)
