from __future__ import unicode_literals, print_function, division
#import optparse
import os
from collections import OrderedDict
import loader
#import LstmCrfModel
import torch
import utils
from sacred import Experiment
from sacred.observers import MongoObserver
import time
import logging
import Model
import sys
import pickle
from torchtext import data
from torchtext import datasets
import ast
reload(sys)
sys.setdefaultencoding('utf8')


def init():
    log_id = str(int(time.time() * 10) % (60 * 60 * 24 * 365 * 10)) + str(os.getpid())
    global logger
    logger = logging.getLogger(str(log_id))

    logger.setLevel(logging.DEBUG)

    # write to file
    fh = logging.FileHandler('ex.log')
    fh.setLevel(logging.DEBUG)

    # write to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Handler format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n\t%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    global ex
    ex = Experiment('DNLP')
    ex.logger = logger
    return ex, logger


init()


@ex.config
def cfg():
    ex_name = 'test'
    #data_set = 'sst1','conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'sst1'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'sst1,conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'product/books'
    #data_set = 'product/dvd'
    #data_set = 'product/electronics'
    #data_set = 'product/kitchen_houseware'
    #data_set = 'product/music'
    #data_set = 'product/video'
    data_set = 'product/music,product/video'
    pre_emb = '../embed/glove.6B.100d.txt'
    #use_model = 'SingleCNN'
    #use_model = 'NewSelfAttn'
    #use_model = 'SelfAttnTagger'
    #use_model = 'ShareLSTM2'
    use_model = 'JKSelfAttn'
    #use_model = 'NewSelfAttn'
    vocab_size = 19540
    n_epochs = 25 # number of epochs over the training set
    lr = 0.001
    use_cuda = True
    batch_size = 32
    lstm_dim = 200
    embed_dim = 200
    da = 100
    r = 30
    bilstm = True
    optim = 'adam'
    device = 1
    dropout = 0.5
    penalty = 1.0
    l2_reg = 0.0001


@ex.automain
def main(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, use_cuda, n_epochs, batch_size, lstm_dim, da, r, bilstm, optim, device, _log, _run, l2_reg, penalty = 0, dropout=0):
    _run.info['ex_name'] = ex_name

    if type(data_set) is not list:
        data_set = data_set.split(',')
    if len(data_set) == 1:
        if 'sst1' in data_set:
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, _ = loader.sst(text_field, label_field, batch_size, device=-1, repeat=False)
            # TODO: ['<unk>', 'negative', 'positive', 'neutral', 'very positive', 'very negative']
            tagset_size = len(label_field.vocab.itos)
            text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)
            vocab_size = text_field.vocab.vectors.size(0)
        elif 'conll' in data_set:
            text_field = data.Field(lower=True)
            label_field = data.Field()
            train_iter, dev_iter, _ = loader.conll(text_field, label_field, batch_size)
            vocab_size = len(text_field.vocab.itos)
            tagset_size = len(label_field.vocab.itos)
            text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)
        elif data_set[0].startswith('product'):
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, _ = loader.product(text_field, label_field, batch_size, data_set=data_set[0])
            vocab_size = len(text_field.vocab.itos)
            tagset_size = len(label_field.vocab.itos)
            text_field.vocab.load_vectors('../embed', wv_type='glove.6B2', wv_dim=200)
        else:
            print('dataset not found')
    else:
        train_data = []
        dev_data = []
        all_data = []

        text_field = data.Field(lower=True)
        sst_label_field = data.Field(sequential=False)
        train_iter, dev_iter, test_iter, sst_data = loader.sst(text_field,
                            sst_label_field, batch_size, device=-1, repeat=False)
        train_data.append(train_iter)
        dev_data.append(dev_iter)
        all_data += sst_data

        con_label_field = data.Field()
        train_iter, dev_iter, con_data = loader.conll(text_field, con_label_field, batch_size)

        train_data.append(train_iter)
        dev_data.append(dev_iter)
        all_data += con_data
        text_field.build_vocab(all_data[0], all_data[1], all_data[2], all_data[3], all_data[4])
        text_field.vocab.load_vectors('../embed', wv_type='glove.6B', wv_dim=100)
        vocab_size = len(text_field.vocab.itos)
        c0 = len(sst_label_field.vocab.itos)
        c1 = len(con_label_field.vocab.itos)

    model_param = OrderedDict()
    model_param['vocab_size'] = vocab_size
    model_param['embed_dim'] = embed_dim
    model_param['lstm_dim'] = lstm_dim
    model_param['da'] = da
    if len(data_set) == 1:
        model_param['tagset_size'] = tagset_size
    else:
        model_param['c0'] = c0
        model_param['c1'] = c1
    model_param['batch_size'] = batch_size
    model_param['r'] = r
    model_param['bilstm'] = bilstm
    model_param['device'] = device
    model_param['dropout'] = dropout
    model_param['penalty'] = penalty

    if use_cuda and torch.cuda.is_available():
        model_param['use_cuda'] = True
    else:
        use_cuda = False
        model_param['use_cuda'] = False
    _run.info['model_param'] = model_param

    if use_model == 'SingleCNN':
        model = Model.SingleCNN(model_param)
    elif use_model == 'SingleLSTM':
        model = Model.SingleLSTM(model_param)
    elif use_model == 'ShareCNN':
        model = Model.ShareCNN(model_param)
    elif use_model == 'ShareLSTM':
        model = Model.ShareLSTM(model_param)
    elif use_model == 'SingleSelfAttn':
        model = Model.SingleSelfAttn(model_param)
    elif use_model == 'SingleSelfAttn2':
        model = Model.SingleSelfAttn2(model_param)
    elif use_model == 'SelfAttnTagger':
        model = Model.SelfAttnTagger(model_param)
    elif use_model == 'LSTMTagger':
        model = Model.LSTMTagger(model_param)
    elif use_model == 'NewSelfAttn':
        model = Model.NewSelfAttn(model_param)
    elif use_model == 'ShareLSTM':
        model = Model.ShareLSTM(model_param)
    elif use_model == 'ShareLSTM2':
        model = Model.ShareLSTM2(model_param)
    elif use_model == 'JointLSTM':
        model = Model.JointLSTM(model_param)
    elif use_model == 'JKSelfAttn':
        model = Model.JKSelfAttn(model_param)
    else:
        print('Wrong model')
        exit()
    print(model)
    _run.info['model'] = str(model)

    if model_param['use_cuda']:
        model.cuda(device)

    if pre_emb:
        print("Initialize the word-embed layer")
        model.init_embed(text_field.vocab.vectors)

    division = 100
    if len(data_set) == 1:
        utils.train(n_epochs, division, train_iter, dev_iter, model, lr, optim, use_cuda, l2_reg, _run)
    else:
        utils.multi_train(n_epochs, division, train_data, dev_data, model, lr, optim, use_cuda, l2_reg, _run)
