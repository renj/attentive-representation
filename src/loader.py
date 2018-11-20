import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import read_pre_training
import numpy as np
import sys
sys.path.append('/home/rjzhen/anaconda2/lib/python2.7/site-packages/torchtext-0.1.1-py2.7.egg')
import torchtext
from torchtext import datasets
from torchtext import data
from nltk import word_tokenize

CAP_DIM = 4


'''
def read_data(data_set, batch_size):
    if len(data_set) == 1:
        text_field = data.Field(lower=True)
        if 'sst1' in data_set:
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            label_field = data.Field()
            train_iter, test_iter, _ = conll(text_field, label_field, batch_size)
            dev_iter = test_iter
        elif data_set[0].startswith('product'):
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter = product(text_field, label_field, batch_size, data_set=data_set[0])
        else:
            print('dataset not found')
        c = len(label_field.vocab.itos)
        train_data, dev_data, test_data = train_iter, dev_iter, test_iter
        label_field.build_vocab(test_data, dev_data, train_data)
        text_field.build_vocab(test_data, dev_data, train_data)
    else:
        train_data = []
        dev_data = []
        test_data = []
        all_data = []

        text_field = data.Field(lower=True)
        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                label_field = data.Field()
                train_iter, dev_iter, both_data = conll(text_field, label_field, batch_size)
                test_iter = dev_iter
            else:
                label_field = data.Field(sequential=False)
                train_iter, dev_iter, test_iter, both_data = product(text_field, label_field, batch_size, data_set=data_set[i])
            label_fields.append(label_field)
            train_data.append(train_iter)
            dev_data.append(dev_iter)
            test_data.append(test_iter)
            all_data += both_data
        #TODO: Support different number of data
        #text_field.build_vocab(all_data[0], all_data[1], all_data[2], all_data[3])

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)

    return train_data, dev_data, test_data, c, vocab_size, text_field.vocab.vectors
'''


'''
def call_args(func, args):
    eval_str = func.__name__ + '(' + str(args)[1:-1] + ')'
    print(eval_str)
    eval(eval_str)
'''


def call_args(text_field, func, args):
    eval_str = func + '('
    for i in range(len(args)):
        eval_str += 'args[%d],' % i
    eval_str += ')'
    print(eval_str)
    eval(eval_str)


def read_adv_data(data_set, batch_size):
    train_datas = []
    all_datas = []

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    label_fields = []
    for i in range(len(data_set)):
        if data_set[i] == 'conll':
            label_field = data.Field()
            train_data, dev_data, both_data = conll(text_field, label_field, batch_size)
            test_data = dev_data
        else:
            if 'topic' in data_set[i]:
                label_field = data.Field(sequential=False)
            train_data, dev_data, test_data = product(text_field, label_field, batch_size, data_set=data_set[i])
        if len(label_fields) == 0 or 'topic' in data_set[i] or data_set[i] == 'conll':
            label_field.build_vocab(test_data, dev_data, train_data)
        label_fields.append(label_field)
        train_datas.append(train_data)
        all_datas += (train_data, dev_data, test_data)
    call_args(text_field, 'text_field.build_vocab', all_datas)

    train_iters = []
    for train_data in train_datas:
        train_iter = data.BucketIterator.splits(train_data, batch_sizes=(batch_size), device=-1, repeat=False)
        train_iters.append(train_iter)

    return train_iters


def read_data(data_set, batch_size):
    if type(data_set) is not list:
        data_set = data_set.split(',')
    if len(data_set) == 1:
        text_field = data.Field(lower=True)
        if 'sst1' in data_set:
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            label_field = data.Field()
            train_data, test_data = conll(text_field, label_field, batch_size)
            dev_data = test_data
        elif data_set[0].startswith('product'):
            label_field = data.Field(sequential=False)
            train_data, dev_data, test_data = product(text_field, label_field, batch_size, data_set=data_set[0])
        elif 'imdb' in data_set:
            label_field = data.Field(sequential=False)
            train_data, dev_data, test_data = imdb(text_field, label_field, batch_size)
        elif 'yelp' in data_set:
            label_field = data.Field(sequential=False)
            train_data, dev_data, test_data = yelp(text_field, label_field, batch_size)
        else:
            print('dataset not found')
        label_field.build_vocab(test_data, dev_data, train_data)
        text_field.build_vocab(test_data, dev_data, train_data)
        train_iters, dev_iters, test_iters = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, batch_size, batch_size), device=-1, repeat=False)
        label_fields = [label_field]
        c = len(label_field.vocab.itos)
    else:
        train_datas = []
        dev_datas = []
        test_datas = []
        all_datas = []

        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                label_field = data.Field()
                train_data, dev_data, both_data = conll(text_field, label_field, batch_size)
                test_data = dev_data
            elif data_set[i] == 'pos':
                label_field = data.Field()
                train_data, dev_data, both_data = pos(text_field, label_field, batch_size)
                test_data = dev_data
            else:
                if 'topic' in data_set[i]:
                    label_field = data.Field(sequential=False)
                train_data, dev_data, test_data = product(text_field, label_field, batch_size, data_set=data_set[i])
            if len(label_fields) == 0 or 'topic' in data_set[i] or data_set[i] == 'conll' or data_set[i] == 'pos':
                label_field.build_vocab(test_data, dev_data, train_data)
            label_fields.append(label_field)
            train_datas.append(train_data)
            dev_datas.append(dev_data)
            test_datas.append(test_data)
            all_datas += (train_data, dev_data, test_data)
        call_args(text_field, 'text_field.build_vocab', all_datas)

        train_iters, dev_iters, test_iters = [], [], []
        for train_data, dev_data, test_data in zip(train_datas, dev_datas, test_datas):
            train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, 30, 30), device=-1, repeat=False)
            train_iters.append(train_iter)
            dev_iters.append(dev_iter)
            test_iters.append(test_iter)

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    #vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)
    #text_field.vocab.load_vectors('../embed', wv_type='glove.6B', wv_dim=100)

    #return train_iters, dev_iters, test_iters, c, vocab_size, text_field.vocab.vectors
    return train_iters, dev_iters, test_iters, c, text_field, label_fields


def read_data_for_fields(data_set, batch_size):
    if len(data_set) == 1:
        text_field = data.Field(lower=True)
        if 'sst1' in data_set:
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            label_field = data.Field()
            train_data, test_data = conll(text_field, label_field, batch_size)
            dev_data = test_data
        elif data_set[0].startswith('product'):
            label_field = data.Field(sequential=False)
            train_data, dev_data, test_data = product(text_field, label_field, batch_size, data_set=data_set[0])
        else:
            print('dataset not found')
        label_field.build_vocab(test_data, dev_data, train_data)
        text_field.build_vocab(test_data, dev_data, train_data)
        train_iters, dev_iters, test_iters = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, 200, 200), device=-1, repeat=False)

        c = len(label_field.vocab.itos)
    else:
        train_datas = []
        dev_datas = []
        test_datas = []
        all_datas = []

        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)
        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                label_field = data.Field()
                train_data, dev_data, both_data = conll(text_field, label_field, batch_size)
                test_data = dev_data
            else:
                if 'topic' in data_set[i]:
                    label_field = data.Field(sequential=False)
                train_data, dev_data, test_data = product(text_field, label_field, batch_size, data_set=data_set[i])
            if len(label_fields) == 0 or 'topic' in data_set[i] or data_set[i] == 'conll':
                label_field.build_vocab(test_data, dev_data, train_data)
            label_fields.append(label_field)
            train_datas.append(train_data)
            dev_datas.append(dev_data)
            test_datas.append(test_data)
            all_datas += (train_data, dev_data, test_data)
        call_args(text_field, 'text_field.build_vocab', all_datas)

        train_iters, dev_iters, test_iters = [], [], []
        for train_data, dev_data, test_data in zip(train_datas, dev_datas, test_datas):
            train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, 200, 200), device=-1, repeat=False)
            train_iters.append(train_iter)
            dev_iters.append(dev_iter)
            test_iters.append(test_iter)

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)

    return train_iters, dev_iters, test_iters, c, vocab_size, text_field.vocab.vectors, text_field, label_fields

'''
def read_data_for_fields(data_set, batch_size):
    if len(data_set) == 1:
        text_field = data.Field(lower=True)
        if 'sst1' in data_set:
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            label_field = data.Field()
            train_iter, test_iter, _ = conll(text_field, label_field, batch_size)
            dev_iter = test_iter
        elif data_set[0].startswith('product'):
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = product(text_field, label_field, batch_size, data_set=data_set[0])
        else:
            print('dataset not found')
        c = len(label_field.vocab.itos)
        train_data, dev_data, test_data = train_iter, dev_iter, test_iter
    else:
        train_data = []
        dev_data = []
        test_data = []
        all_data = []

        text_field = data.Field(lower=True)
        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                label_field = data.Field()
                train_iter, dev_iter, both_data = conll(text_field, label_field, batch_size)
                test_iter = dev_iter
            else:
                label_field = data.Field(sequential=False)
                train_iter, dev_iter, test_iter, both_data = product(text_field, label_field, batch_size, data_set=data_set[i])
            label_fields.append(label_field)
            train_data.append(train_iter)
            dev_data.append(dev_iter)
            test_data.append(test_iter)
            all_data += both_data
        #TODO: Support different number of data
        text_field.build_vocab(all_data[0], all_data[1], all_data[2], all_data[3])

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)

    return train_data, dev_data, test_data, c, vocab_size, text_field.vocab.vectors, text_field, label_field
'''


def read_data_with_fields(data_set, batch_size, text_field, label_field):
    if len(data_set) == 1:
        if 'sst1' in data_set:
            #label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            #label_field = data.Field()
            train_iter, test_iter, _ = conll(text_field, label_field, batch_size)
            dev_iter = test_iter
        elif data_set[0].startswith('product'):
            #label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = product(text_field, label_field, batch_size, data_set=data_set[0])
        else:
            print('dataset not found')
        c = len(label_field.vocab.itos)
        train_data, dev_data, test_data = train_iter, dev_iter, test_iter
    else:
        train_data = []
        dev_data = []
        test_data = []
        all_data = []

        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                #label_field = data.Field()
                train_iter, dev_iter, both_data = conll(text_field, label_field, batch_size)
                test_iter = dev_iter
            else:
                #label_field = data.Field(sequential=False)
                train_iter, dev_iter, test_iter, both_data = product(text_field, label_field, batch_size, data_set=data_set[i], build_vocab=False)
            label_fields.append(label_field)
            train_data.append(train_iter)
            dev_data.append(dev_iter)
            test_data.append(test_iter)
            all_data += both_data
        #TODO: Support different number of data
        #text_field.build_vocab(all_data[0], all_data[1], all_data[2], all_data[3])

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)

    return train_data, dev_data, test_data, c, vocab_size, text_field.vocab.vectors


def read_data_all(data_set, batch_size):
    if len(data_set) == 1:
        text_field = data.Field(lower=True)
        if 'sst1' in data_set:
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
        elif 'conll' in data_set:
            label_field = data.Field()
            train_iter, dev_iter, _ = conll(text_field, label_field, batch_size)
        elif data_set[0].startswith('product'):
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, _ = product(text_field, label_field, batch_size, data_set=data_set[0])
        else:
            print('dataset not found')
        c = len(label_field.vocab.itos)
        train_data, dev_data = train_iter, dev_iter
        label_fields = label_field
    else:
        train_data = []
        dev_data = []
        all_data = []

        text_field = data.Field(lower=True)
        label_fields = []
        for i in range(len(data_set)):
            if data_set[i] == 'conll':
                label_field = data.Field()
                train_iter, dev_iter, both_data = conll(text_field, label_field, batch_size)
            else:
                label_field = data.Field(sequential=False)
                train_iter, dev_iter, both_data = product(text_field, label_field, batch_size, data_set=data_set[i])
            label_fields.append(label_field)
            train_data.append(train_iter)
            dev_data.append(dev_iter)
            all_data += both_data
        #TODO: Support different number of data
        #text_field.build_vocab(all_data[0], all_data[1], all_data[2], all_data[3])

        c = []
        for lf in label_fields:
            c.append(len(lf.vocab.itos))

    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)

    return train_data, dev_data, c, vocab_size, text_field.vocab.vectors, text_field, label_fields


def pos(text_field, label_field, batch_size, **kargs):
    train_data = conll_dataset('../data/conll2000.train.txt', text_field, label_field, pos=True)
    test_data = conll_dataset('../data/conll2000.test.txt', text_field, label_field, pos=True)

    #text_field.build_vocab(test_data, train_data)
    #label_field.build_vocab(test_data, train_data)
    #train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_sizes=(batch_size, len(test_data)), device=-1, repeat=False)
         
    return train_data, test_data, [train_data, test_data]


def read_CONLL(path, zeros=True, lower=True, pos=False):
    sentences = []
    sentence = []
    idx = 0
    for line in codecs.open(path, 'r', 'utf8'):
        idx += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            if len(word) < 2:
                print(idx, line)
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    if pos:
        tags = [[w[-2] for w in s] for s in sentences]
    else:
        tags = [[w[-1] for w in s] for s in sentences]
    #tags = [[w[1] for w in s] for s in sentences]
    return words, tags


def conll_dataset(path, text_field, label_field, pos=False):
    fields = [('text', text_field), ('label', label_field)]
    texts, labels = read_CONLL(path, 1, 1, pos)
    examples = []
    for text, label in zip(texts, labels):
        _data = [text, label]
        _fields = [('text', text_field), ('label', label_field)]
        a = data.Example()
        examples.append(a.fromlist(_data, _fields))
    ret_data = data.Dataset(examples, fields)
    ret_data.sort_key = lambda x: -1 * len(x.text)
    return ret_data


def conll(text_field, label_field, batch_size, **kargs):
    train_data = conll_dataset('../data/conll2000.train.txt', text_field, label_field)
    test_data = conll_dataset('../data/conll2000.test.txt', text_field, label_field)

    #text_field.build_vocab(test_data, train_data)
    #label_field.build_vocab(test_data, train_data)
    #train_iter, test_iter = data.BucketIterator.splits((train_data, test_data), batch_sizes=(batch_size, len(test_data)), device=-1, repeat=False)
         
    return train_data, test_data, [train_data, test_data]


def read_imdb(path, sent, zeros=True, lower=True):
    path = os.path.join(path, sent)
    print(path)
    files = os.listdir(path)
    texts = []
    for f in files:
        _text = []
        p = os.path.join(path, f)
        for line in codecs.open(p, 'r', 'utf8'):
            #idx += 1
            #text = line.split(' ')
            line = line.replace("<br />", "\n")
            text = word_tokenize(line)
            if lower:
                text = [w.lower() for w in text]
            _text += text
        texts.append(_text)
    return texts


def imdb_dataset(path, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    texts = read_imdb(path, 'pos', 1, 1)
    labels = [1] * len(texts)
    _texts = read_imdb(path, 'neg', 1, 1)
    labels += [0] * len(_texts)
    texts += _texts

    examples = []
    for text, label in zip(texts, labels):
        _data = [text, label]
        _fields = [('text', text_field), ('label', label_field)]
        a = data.Example()
        examples.append(a.fromlist(_data, _fields))
    ret_data = data.Dataset(examples, fields)
    ret_data.sort_key = lambda x: -1 * len(x.text)
    return ret_data


def imdb(text_field, label_field, batch_size, **kargs):
    train_data = imdb_dataset('/home/rjzhen/Project/dnlp/data/aclImdb/train', text_field, label_field)
    test_data = imdb_dataset('/home/rjzhen/Project/dnlp/data/aclImdb/test', text_field, label_field)
    return train_data, test_data, test_data

def read_product(path, zeros=True, lower=True):
    # TODO: ():,?!, &quot;
    texts = []
    labels = []
    idx = 0
    print(path)
    for line in codecs.open(path, 'r', 'utf8'):
        idx += 1
        label, text = line.split('\t')
        labels.append(label)
        text = text.split(' ')
        if lower:
            text = [w.lower() for w in text]
        texts.append(text)

    return texts, labels

def product_dataset(path, text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    texts, labels = read_product(path, 1, 1)
    examples = []
    for text, label in zip(texts, labels):
        _data = [text, label]
        _fields = [('text', text_field), ('label', label_field)]
        a = data.Example()
        examples.append(a.fromlist(_data, _fields))
    ret_data = data.Dataset(examples, fields)
    ret_data.sort_key = lambda x: -1 * len(x.text)
    return ret_data


def product(text_field, label_field, batch_size, **kargs):
    #train_data = product_dataset('../data/' + kargs['data_set'] + '.token.train', text_field, label_field)
    #test_data = product_dataset('../data/' + kargs['data_set'] + '.token.test', text_field, label_field)
    train_data = product_dataset('../data/' + kargs['data_set'] + '.task.train', text_field, label_field)
    dev_data = product_dataset('../data/' + kargs['data_set'] + '.task.dev', text_field, label_field)
    test_data = product_dataset('../data/' + kargs['data_set'] + '.task.test', text_field, label_field)
    return train_data, dev_data, test_data
    '''
    if 'build_vocab' not in kargs:
        text_field.build_vocab(test_data, dev_data, train_data)
        label_field.build_vocab(test_data, dev_data, train_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train_data, dev_data, test_data), batch_sizes=(batch_size, 200, 200), device=-1, repeat=False)
    return train_iter, dev_iter, test_iter, [train_data, dev_data, test_data]
    '''

def yelp(text_field, label_field, batch_size, **kargs):
    #train_data = product_dataset('../data/' + kargs['data_set'] + '.token.train', text_field, label_field)
    #test_data = product_dataset('../data/' + kargs['data_set'] + '.token.test', text_field, label_field)
    train_data = product_dataset('../data/yelp.train', text_field, label_field)
    dev_data = product_dataset('../data/yelp.dev', text_field, label_field)
    test_data = product_dataset('../data/yelp.test', text_field, label_field)
    return train_data, dev_data, test_data

def sst(text_field, label_field, batch_size, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(batch_size,
                                                     len(dev_data),
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter, [train_data, dev_data, test_data]


def read_SST(path, zeros, lower):
    text_field = torchtext.data.Field(lower=True)
    label_field = torchtext.data.Field(sequential=False)
    train_data, dev_data, test_data = torchtext.datasets.SST.splits(text_field, label_field, fine_grained=True)
    if path.endswith('TRAIN'):
        sentences = train_data.examples
    elif path.endswith('DEV'):
        sentences = dev_data.examples

    if path.startswith('SST1'):
        words = [[x.lower() if lower else x for x in s.text] for s in sentences]
        tags = [[s.label] for s in sentences]
    elif path.startswith('SST2'):
        words = []
        tags = []
        for s in sentences:
            if s.label == 'neutral':
                continue

            if 'negative' in s.label:
                tags.append(list(['negative']))
            else:
                tags.append(list(['positive']))
            words.append([x.lower() if lower else x for x in s.text])
    #print(len(words))
    return words, tags


def load_sentences(path, zeros, lower):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    if path.startswith('SST'):
        sentences, tags = read_SST(path, zeros, lower)
    elif os.path.isfile(path):
        sentences, tags = read_CONLL(path, zeros, lower)
    else:
        print('invalie path: %s'%(path))
        assert(True)
    return sentences, tags

def check_tag_chunking(sentences):
    """
    Check the input format is chunking or not 
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        for j, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            #if len(split) != 2 or split[0] not in ['I', 'B'] \
            #            or split[1] not in ['NP', 'VP', 'PP', 'SBAR', 'ADVP','ADJP']:
            #    print(split)
            #    raise Exception('Unknown tagging scheme!')



def word_mapping(words, vocabulary_size, pre_train = None):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    dico = create_dico(words)
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    print("Found %i unique words (%i in total)" % 
        (len(dico), sum(len(x) for x in words))
    )

    if pre_train:
        emb_dictionary = read_pre_training(pre_train)
        for word in dico.keys():
              if word not in emb_dictionary:
                  dico[word]=0
                  
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    return dico, word_to_id, id_to_word


def tag_mapping(tags):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % (len(dico)))
    return dico, tag_to_id, id_to_tag

def sst_tag_mapping(data):
    pass

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def load_dataset(sentences, tags, word_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for i in range(len(sentences)):
        str_words = sentences[i]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>'] for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tag_ids = [tag_to_id[t] for t in tags[i]]
        data.append({
            'str_words': str_words,
            'words': words,
            'caps': caps,
            'tags': tag_ids,
            'pos': tag_ids,
        })
    return data


def prepare_dic(param):
    lower = param['lower']
    zeros = param['zeros']
    train_path = param['train']
    dev_path = param['dev']
    vocabulary_size = param['vocab_size']

    # Load sentences
    train_sentences = []
    train_tags = []
    for tp in train_path:
        sents, tags = load_sentences(tp, zeros, lower)
        train_sentences += sents
        train_tags += tags

    # Use selected tagging scheme
    check_tag_chunking(train_sentences)

    #if param['pre_emb']:
    dev_sentences = []
    dev_tags = []
    for tp in dev_path:
        sents, tags = load_sentences(tp, zeros, lower)
        dev_sentences += sents
        dev_tags += tags
    sentences = train_sentences + dev_sentences
    dico_words, word_to_id, id_to_word = word_mapping(sentences, vocabulary_size, param['pre_emb'])

    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_tags + dev_tags)

    dictionaries = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'tag_to_id': tag_to_id,
        'id_to_tag': id_to_tag,
    }

    return dictionaries


def prepare_dataset(param, path, dictionaries):
    # Data param
    lower = param['lower']
    zeros = param['zeros']

    # Load sentences
    sentences, tags = load_sentences(path, zeros, lower)
    #print sentences
    #print tags
    dataset = load_dataset(
        sentences, tags, dictionaries['word_to_id'], dictionaries['tag_to_id'], lower)
    print("%i sentences in %s ."%(len(dataset), path))
    return dataset

def get_embed(dictionary, pre_train, embedding_dim):
    emb_dictionary = read_pre_training(pre_train)
    dic_size = len(dictionary)
    initial_matrix = np.random.random(size=(dic_size, embedding_dim))
    for word, idx in dictionary.items(): 
        if word != '<UNK>':
            initial_matrix[idx] = emb_dictionary[word]
    return initial_matrix
