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
    ex_name = 'paper'
    #data_set = 'sst1','conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'sst1'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
    #data_set = 'product/books'
    #data_set = 'product/dvd'
    #data_set = 'product/electronics'
    #data_set = 'product/kitchen'
    #data_set = 'product/music'
    #data_set = 'product/video'
    #data_set = 'product/books,product/music,product/video,product/kitchen,product/electronics'
    #data_set = 'product/books,product/music,product/video,product/kitchen,product/electronics,product/dvd,conll'
    #data_set = 'product/class1'
    #data_set = 'product/books,product/music,product/dvd,product/class1,product/class2'
    #data_set = 'product/books,product/music,product/dvd,product/class1,product/class2'
    #data_set = 'product/books,product/music,product/dvd,product/class1,product/class2,conll'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,conll'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,pos'
    #data_set = 'product/sentiment,product/topic'
    #data_set = 'product/sentiment'
    #data_set = 'imdb'
    #data_set = 'product/books,product/music'
    data_set = 'product/books'
    #data_set = 'product/mr'
    #data_set = 'product/apparel'
    #data_set = 'product/sports'
    #data_set = 'product/toys'
    #data_set = 'product/sentiment,pos'
    #data_set = 'product/sentiment,conll'
    #data_set = 'product/books,product/music'
    #data_set = 'product/topic2'
    #data_set = 'product/sentiment'
    #data_set = 'product/mr'
    #data_set = 'product/class2,product/class1'
    #data_set = 'product/class2,product/class1,conll'
    #data_set = 'product/books,conll'
    pre_emb = '../embed/glove.6B.100d.txt'
    #use_model = 'SingleCNN'
    #use_model = 'NewSelfAttn'
    #use_model = 'SelfAttnTagger'
    #use_model = 'ShareLSTM2'
    #use_model = 'SingleAttn'
    #use_model = 'ShareAllAttn'
    #use_model = 'LSTMTagger'
    #use_model = 'AttnTagger'
    #use_model = 'MultiAllAttn'
    #use_model = 'SelfAttn'
    #use_model = 'MultiSelfAttn'
    #use_model = 'MultiSingleAttn'
    #use_model = 'ShareTwoLSTM'
    #use_model = 'CatTwoLSTM'
    #use_model = 'TreeAttn'
    #use_model = 'MaxAttn'
    #use_model = 'SingleAttn'
    #use_model = 'RelateNet'
    #use_model = 'RelateAttn'
    #use_model = 'AddTwoLSTM'
    #use_model = 'FSSingleAttn'
    #use_model = 'ASPCatAttn'
    #use_model = 'ShareTwoLSTM'
    #use_model = 'JKSelfAttn'
    #use_model = 'AttnOverAttn'
    #use_model = 'ShareSingleAttn'
    #use_model = 'MultiSingleAttn'
    #use_model = 'SPLSTM'
    #use_model = 'SecOrdAttn'
    #use_model = 'AttnOverAttn'
    #use_model = 'DocAttn'
    #use_model = 'DocThetaAttn'
    #use_model = 'SingleAttn'
    #use_model = 'DocMultAttn'
    #use_model = 'SentThetaAttn'
    #use_model = 'SingleAttn'
    #use_model = 'RelateAttn'
    #use_model = 'BoW'
    #use_model = 'CNN'
    #use_model = 'SingleAttnBoW'
    #use_model = 'DualAttn'
    #use_model = 'NewMemAttn'
    #use_model = 'MemAttn'
    #use_model = 'MemAttnOld'
    vocab_size = 19540
    n_epochs = 15 # number of epochs over the training set
    lr = 0.001
    #lr = 0.0001
    use_cuda = True
    batch_size = 32
    lstm_dim = 200
    embed_dim = 200
    da = 100
    r = 30
    optim = 'adam'
    dropout = 0.5
    #p_lambda = 0.01
    #p_gamma = 0.001
    l2_reg = 1e-5
    division = 25
    seed = 283953214
    fine_tune = None
    #adversarial = True
    adversarial = False
    #p_lambda = 0.0
    #p_gamma = 0.0

    '''PRE-TRAIN'''
    use_model = 'MemAttn'
    #use_model = 'SingleAttn'
    data_set = 'product/mrimdb'
    #data_set = 'product/books'
    #data_set = 'product/sentiment'
    #data_set = 'imdb'
    #data_set = 'yelp'

    device = 2
    #transfer_params = True
    #transfer_params = False
    #transfer = '../model/new/Model.MemAttn_books_pause_5_1_83.75.torch'
    #transfer = '../model/new/Model.MemAttn_imdb_pause_5_1_91.45.torch'
    #transfer = '../model/new/Model.MemAttn_mrimdb_save_perf_last_h_q_D_5_1_87.50.torch'
    transfer = '../model/new/Model.MemAttn_mrimdb_save_perf_q_P_q_D_pause_last_h_isolate_P_H_h_5_1_86.88.torch'
    #transfer = '../model/new/Model.MemAttn_mrimdb_save_perf_pause_isolate_P_cpx_Hh_bpause_even_5_1_88.00.torch'
    #bilstm = False
    bilstm = True

    #search_n = 1
    save_model = True

    #epsilon = 0.1 # for RL
    #epsilon = 0.0 # for RL
    #p_lambda = 0.001 # 
    #p_lambda2 = 0.001 # 
    p_lambda = 0.01 # 
    #p_lambda = 0.0 # 
    p_lambda2 = 0.01 # 
    p_gamma = 0.8
    #a_pen = 0.001

    dropout = 0.2
    #rl_batch = 16
    #clip = 0.25
    #clip = 0.05
    nlayers = 5
    rl_batch = 1
    beta = 0.0
    #params = ['test_rl'] # fix_H, rl, pos, pause, isolate_P, last_P, last_h, H_h, test_rl, discount, bpause_even, brl_even, rl_torch, pause_torch
    #params = ['pause', 'rl'] # fix_H, rl, pos, pause
    #params = ['pause', 'bpause_even', 'q_D', 'isolate_P', 'H_h', 'rl', 'brl_even', 'rl_torch'] # fix_H, rl, pos, pause, q_D
    #params = ['pause', 'bpause_even', 'q_D', 'isolate_P', 'q_P', 'last_h', 'rl', 'brl_even', 'rl_torch'] # fix_H, rl, pos, pause, q_D
    #params = ['pause', 'bpause_even', 'isolate_P', 'H_h'] # fix_H, rl, pos, pause, q_D
    #params = ['pause', 'bpause_even', 'q_D', 'q_P', 'H_h', 'last_h', 'isolate_P'] # fix_H, rl, pos, pause, q_D
    '''
    Parameter Notes
    pause:  RL to choose when to pause OR use last state
    layer_ce: add cross entropy loss at every layer
    q_D:    use q to classify OR use h to classify
    q_P:    use q to 
    last_h: process last h into the GRU OR sum of hs
    hinge:  add loss to make upper layer has higher accuracy
    aoa:    use attention to fuse the information of qs
    H_h:    add sum of H to current h or q for RL pause
    isolate_P: detach H from Pause
    all_ce: add cross entropy loss at every layer
    save_perf: save maximum of perfect acc
    cpx_Hh: complex H+h
    discount: use p_gamma to discount
    diff_Ds: use different D for every layer
    max_ce: cross entropy on maximum D
    '''
    # MemoryNet
    #params = []
    # MemoryNet q_D
    #params = ['q_D', 'last_h'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h', 'hinge'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h', 'save_perf'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h', 'all_ce'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h', 'pause', 'isolate_P'] # fix_H, rl, pos, pause, q_D
    #params = ['pause', 'bpause_even', 'isolate_P', 'cpx_Hh', 'diff_Ds', 'max_ce'] # fix_H, rl, pos, pause, q_D
    params = ['pause', 'bpause_even', 'q_D', 'last_h', 'cpx_Hh'] # fix_H, rl, pos, pause, q_D
    #params = ['isolate_P', 'cpx_Hh', 'diff_Ds', 'max_ce'] # fix_H, rl, pos, pause, q_D
    #params = ['q_D', 'last_h', 'aoa'] # fix_H, rl, pos, pause, q_D
    #params = ['pause', 'bpause_even', 'q_D', 'isolate_P', 'H_h'] # fix_H, rl, pos, pause, q_D
    #transfer = '../model/new/Model.MemAttn_books_pause_5_1_85.00.torch'
    #transfer = '../model/new/Model.MemAttn_imdb_pause_5_1_91.74.torch'
    #params = ['rl', 'brl_even', 'q_D', 'last_h', 'rl_torch'] # fix_H, rl, pos, pause
    #params = ['pause', 'bpause_even'] #window, pos, tgf, unfix, sent, rl
    #params = ['pause', 'rl'] #window, pos, tgf, unfix, sent, rl
    #params = ['pause'] #window, pos, tgf, unfix, sent, rl
    #params = ['rl', 'unfix'] #window, pos, tgf, unfix, sent, rl
    #params = ['unfix', 'rl'] #window, pos, tgf, unfix, sent, rl
    #window_size = 10
    #params = ['pos', 'tgf', 'unfix'] #window, pos, tgf, unfix
    #params.sort()
    #print(params)
    #penalty = -0.1
    #fine_tune = 'electronics_kitchen_video_music_books_0.0_82.95.torch'
    #fine_tune = 'conll_allattn/conll_dvd_electronics_kitchen_video_music_books_0.0_87.18.torch'
    #fine_tune = 'conll_selfattn/conll_dvd_electronics_kitchen_video_music_books_0.0_87.52.torch'

def copy_param(a, b):
    for k in a._parameters.keys():
        b._parameters[k] = a._parameters[k]

@ex.automain
def main(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, use_cuda, n_epochs, batch_size, lstm_dim, da, r, bilstm, optim, device, _log, _run, l2_reg, division, fine_tune=None, seed=283953214, p_lambda=0.0, dropout=0, adversarial=False, save_model=False, transfer_params=False, transfer=None, penalty=0.0, params=[], window_size=0, epsilon=0.0, search_n=0, nlayers=1, p_lambda2=0.0, rl_batch=1, p_gamma=1.0, a_pen=0.0, clip=0.0, beta = 0.0):
    _run.info['ex_name'] = ex_name
    seed = 12345678

    torch.manual_seed(seed)
    data_set = data_set.split(',')

    #train_data, dev_data, test_data, c, vocab_size, word_vector = loader.read_data(data_set, batch_size)
    import dill
    import pickle
    import os
    if len(data_set) == 0 and data_set[0] == 'imdb':
        print('IMDB!')
        file_path = '../data/aclImdb/data_%d.pickle' % batch_size
        if os.path.exists(file_path):
            print('reading IMDB pickle')
            with open(file_path, 'r') as f:
                train_data, dev_data, test_data, c, text_field, label_fields = pickle.load(f)
        else:
            train_data, dev_data, test_data, c, text_field, label_fields = loader.read_data(data_set, batch_size)
            print('writing IMDB pickle')
            with open(file_path, 'w') as f:
                pickle.dump((train_data, dev_data, test_data, c, text_field, label_fields), f)
    else:
        train_data, dev_data, test_data, c, text_field, label_fields = loader.read_data(data_set, batch_size)
        if len(data_set) == 1 and data_set[0] == 'product/mrimdb':
            _train_data, _dev_data, mr_test_data = loader.product(text_field, label_fields[0], batch_size, data_set='product/mr')
            _train_iters, _dev_iters, mr_test_data = data.BucketIterator.splits((_train_data, _dev_data, mr_test_data), batch_sizes=(batch_size, batch_size, batch_size), device=-1, repeat=False)
            _train_data, _dev_data, imdb_test_data = loader.product(text_field, label_fields[0], batch_size, data_set='product/imdb')
            _train_iters, _dev_iters, imdb_test_data = data.BucketIterator.splits((_train_data, _dev_data, imdb_test_data), batch_sizes=(batch_size, batch_size, batch_size), device=-1, repeat=False)
        else:
            mr_test_data = None
            imdb_test_data = None

    #exit()
    lf = label_fields[-1]
    print(lf.vocab.itos)
    #exit(0)
    vocab_size = len(text_field.vocab.itos)
    word_vector = text_field.vocab.vectors

    if adversarial and len(data_set) >= 15:
        adv_train_data, _dev_data, _test_data, _c, _text_field, _label_fields = loader.read_data(['product/topic'], batch_size)


    model_param = OrderedDict()
    model_param['vocab_size'] = vocab_size
    model_param['embed_dim'] = embed_dim
    model_param['lstm_dim'] = lstm_dim
    model_param['da'] = da
    if len(data_set) == 1:
        model_param['c'] = c
    else:
        model_param['cs'] = c
    model_param['batch_size'] = batch_size
    model_param['r'] = r
    model_param['bilstm'] = bilstm
    model_param['device'] = device
    model_param['dropout'] = dropout
    model_param['params'] = params
    model_param['window_size'] = window_size
    model_param['epsilon'] = epsilon
    model_param['p_lambda'] = p_lambda
    model_param['p_lambda2'] = p_lambda2
    model_param['p_gamma'] = p_gamma
    model_param['search_n'] = search_n
    model_param['nlayers'] = nlayers
    model_param['rl_batch'] = rl_batch
    model_param['a_pen'] = a_pen
    model_param['clip'] = clip
    model_param['beta'] = beta

    if use_cuda and torch.cuda.is_available():
        model_param['use_cuda'] = True
    else:
        use_cuda = False
        model_param['use_cuda'] = False
    _run.info['model_param'] = model_param


    model = eval('Model.' + use_model)(model_param)
    print(model)
    _run.info['model'] = str(model)

    '''
    params = list(model.parameters())
    ans = 0
    for p in params:
        s = p.size()
        if len(s) == 1:
            ans += s[0]
        else:
            if s[0] >= 24739:
                continue
            ans += s[0] * s[1]
    #print(params)
    print(ans)
    exit()
    '''

    if model_param['use_cuda']:
        model.cuda(device)

    print("Initialize the word-embed layer")
    model.init_embed(word_vector)

    lock_params = False
    #if transfer_params:
    if transfer is not None:
        model_from = torch.load(transfer)
        if 'params' in dir(model_from):
            print("model_from's params:", model_from.params)
        copy_param(model_from.embed, model.embed)
        model.embed.cuda(model.device)
        copy_param(model_from.lstm, model.lstm)
        model.lstm.cuda(model.device)
        copy_param(model_from.attn, model.attn)
        model.attn.cuda(model.device)
        copy_param(model_from.gru, model.gru)
        model.gru.cuda(model.device)
        copy_param(model_from.mlp1, model.mlp1)
        model.mlp1.cuda(model.device)
        #copy_param(model_from.P, model.P)
        #model.P.cuda(model.device)
        lock_params = False
         
        '''
        #model_from = torch.load('../model/copy/Model.MaxAttn_sentiment_86.77_copy.torch')
        model_from = torch.load('../model/LM_imdb_151.torch')
        #model_from = torch.load('../model/copy/Model.MaxAttn_sentiment_0.50_0.50_86.80.torch')
        #model_from = torch.load('../model/copy/Model.MaxAttn_sentiment_0.00_1.00_87.17.torch')
        #model_from = torch.load('../model/copy/Model.MaxAttn_sentiment_0.50_1.00_86.58.torch')
        copy_param(model_from.encoder, model.embed)
        model.embed.cuda(model.device)
        copy_param(model_from.rnn, model.lstm)
        model.lstm.cuda(model.device)
        transfer_params = False
        '''
  
        '''MaxAttn
        copy_param(model_from.embed, model.embed)
        model.embed.cuda(model.device)
        copy_param(model_from.lstm, model.lstm)
        model.lstm.cuda(model.device)
        copy_param(model_from.attn, model.attn)
        model.attn.cuda(model.device)
        copy_param(model_from.mlp1, model.mlp1)
        model.mlp1.cuda(model.device)
        if 'unfix' in params:
            transfer_params = False
        '''
  
        """
        '''RelateNet'''
        model_from = torch.load('../model/Model.MultiSingleAttn_topic_sentiment__82.79.torch')
        copy_param(model_from.embed, model.embed)
        model.embed.cuda(model.device)
        copy_param(model_from.lstm, model.lstm)
        model.lstm.cuda(model.device)
        copy_param(model_from.attn[0], model.attn1[0])
        model.attn1[0].cuda(model.device)
        copy_param(model_from.attn[1], model.attn1[1])
        model.attn1[1].cuda(model.device)
        if 'unfix' in params:
            transfer_params = False
        """

        '''
        copy_param(model_from.embed, model.embed)
        model.embed.cuda(model.device)
        copy_param(model_from.lstm, model.lstm)
        model.lstm.cuda(model.device)
        copy_param(model_from.attn[0], model.attn[0])
        model.attn[0].cuda(model.device)
        copy_param(model_from.attn[1], model.attn[1])
        model.attn[1].cuda(model.device)
        #transfer_params = False
        '''

        #copy_param(model_from.attn[1], model.attn2)
        #model.attn2.cuda(model.device)

    stop = set([text_field.vocab.stoi[w] for w in ['.', '!', '?']])

    if fine_tune is not None:
        model = torch.load('../model/' + fine_tune)
        model.cuda(model.device)
        #utils.fine_tune(n_epochs, division, train_data, dev_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, _run)
        #exit()
        utils.fine_tune(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=True, penalty=penalty, params=params, stop=stop)
        exit()


    if len(data_set) == 1:
        if str(model.__class__)[8:-2] == 'Model.MemAttn':
            utils.mem_train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, mr_test_data, imdb_test_data, save_model, stop, p_gamma, _run, lock_params=lock_params)
        else:
            utils.train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, save_model, stop, _run, lock_params=lock_params)
    else:
        if adversarial and len(data_set) >= 15:
            utils.multi_train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, adv_train_iters=adv_train_data, adv_label_field=_label_fields[0], params=params)
        else:
            utils.multi_train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=lock_params, penalty=penalty, params=params, stop=stop)
