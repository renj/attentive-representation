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
    #n_epochs = 25 # number of epochs over the training set
    n_epochs = 3 # number of epochs over the training set
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
    #use_model = 'MultiSingleAttn'
    #data_set = 'imdb'
    #data_set = 'product/books,product/elec'
    #data_set = 'product/sentiment'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr'

    #lock_params = False
    #lock_params 
    n_epochs = 25 # number of epochs over the training set
    #n_epochs = 3 # number of epochs over the training set
    device = 3
    #device = 3
    lstm_dim = 200
    division = 25
    #division = 5
    #rec_acc = True
    #params = ['end_ten'] # top_six, end_ten
    #params = ['top_six'] # top_six, end_ten
    #params = ['mean'] # top_six, end_ten
    #use_model = 'MultiDynAttn'
    #use_model = 'StackLSTM'
    use_model = 'ASPCatAttn'
    #use_model = 'SingleLSTM'
    #use_model = 'MultiSingleAttn'
    #use_model = 'ShareLastAttn'
    #use_model = 'SPLSTM'
    #use_model = 'SingleAttn'
    #use_model = 'SingleLSTM'
    #data_set = 'yelp'
    #data_set = 'product/books'
    #data_set = 'product/books,product/books'
    #data_set = 'product/books,product/books'
    #fine_tune = 'Model.StackLSTM_books_books_72.88.torch'
    #fine_tune = 'Model.SPLSTM_books_books_74.12.torch'
    # 2
    #data_set = 'product/books,product/elec'
    #data_set = 'product/books,product/elec,product/topic2'
    #fine_tune = 'Model.StackLSTM_elec_books_72.62.torch'
    #fine_tune = 'Model.SPLSTM_elec_books_76.00.torch'
    ## 4
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/topic4'
    #fine_tune = 'Model.StackLSTM_kitchen_dvd_elec_books_81.06.torch'
    #fine_tune = 'Model.SPLSTM_kitchen_dvd_elec_books_80.50.torch'
    ## 8
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music'
    #fine_tune = 'Model.StackLSTM_music_health_camera_apparel_kitchen_dvd_elec_books_82.59.torch'
    #fine_tune = 'Model.SPLSTM_music_health_camera_apparel_kitchen_dvd_elec_books_84.31.torch'
    data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,product/topic'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,product/topic,conll'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,conll'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,pos'
    #data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr,product/topic,pos'
    #data_set = 'product/sentiment,product/topic'
    #data_set = 'product/sentiment,product/topic'
    #data_set = 'product/sentiment,product/topic,pos'
    #data_set = 'product/sentiment,product/topic,conll'
    #fine_tune = 'new/Model.AttnOverAttn_topic_sentiment_81.66.torch'
    #fine_tune = 'new/Model.AttnOverAttn_topic_sentiment_84.97.torch'
    #fine_tune = 'new/Model.AttnOverAttn_topic_sentiment_87.22.torch'
    #fine_tune = 'new/Model.AttnOverAttn_pos_topic_sentiment_85.61.torch'
    #fine_tune = 'new/Model.AttnOverAttn_conll_topic_sentiment_87.52.torch'

    #fine_tune = 'new/Model.StackLSTM_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.55.torch'
    #fine_tune = 'new/Model.StackLSTM_end_ten_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_83.12.torch'
    #fine_tune = 'new/Model.SPLSTM_end_ten_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_83.25.torch'

    #fine_tune = 'new/Model.SPLSTM_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.39.torch'

    #fine_tune = 'new/Model.ShareLastAttn_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.30.torch'
    #fine_tune = 'new/Model.MultiDynAttn_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.48.torch'

    #fine_tune = 'new/Model.AttnOverAttn_pos_topic_sentiment_87.11.torch'
    #fine_tune = 'new/Model.MultiDynAttn_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_84.70.torch'
    #fine_tune = 'new/Model.MultiDynAttn_conll_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_86.97.torch'
    #fine_tune = 'new/Model.MultiDynAttn_pos_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.50.torch'

    #fine_tune = 'new/Model.MultiSingleAttn_end_ten_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_84.50.torch'
    #fine_tune = 'new/Model.MultiDynAttn_end_ten_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.88.torch'

    #fine_tune = 'new/Model.MultiSingleAttn_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_86.72.torch'
    #fine_tune = 'new/Model.MultiSingleAttn_conll_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.61.torch'
    #fine_tune = 'new/Model.MultiSingleAttn_pos_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.57.torch'
    #fine_tune = 'new/Model.MultiSingleAttn_pos_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.57.torch'


    bilstm = False
    save_model = True
    transfer_params = False

    #search_n = 1
    #epsilon = 0.1 # for RL
    #epsilon = 0.0 # for RL
    #p_lambda = 0.01 # 
    #p_lambda2 = 0.01 # 
    #p_gamma = 1.0
    #a_pen = 0.001
    #rl_batch = 16
    #nlayers = 5
    #rl_batch = 4
    #params = ['test_rl'] # fix_H, rl, pos, pause, isolate_P, last_P, last_h, H_h, test_rl, discount
    #params = ['pause', 'rl'] # fix_H, rl, pos, pause
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
def main(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, use_cuda, n_epochs, batch_size, lstm_dim, da, r, bilstm, optim, device, _log, _run, l2_reg, division, fine_tune=None, seed=283953214, p_lambda=0.0, dropout=0, adversarial=False, save_model=False, transfer_params=False, penalty=0.0, params=[], window_size=0, epsilon=0.0, search_n=0, nlayers=1, p_lambda2=0.0, rl_batch=1, p_gamma=1.0, a_pen=0.0, rec_acc=False, lock_params=False):
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
    model_param['data_set'] = data_set

    if use_cuda and torch.cuda.is_available():
        model_param['use_cuda'] = True
    else:
        use_cuda = False
        model_param['use_cuda'] = False
    _run.info['model_param'] = model_param


    model = eval('Model.' + use_model)(model_param)
    print(model)
    _run.info['model'] = str(model)

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

    if model_param['use_cuda']:
        model.cuda(device)

    print("Initialize the word-embed layer")
    model.init_embed(word_vector)

    if transfer_params:
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
        #lock_params = True
        #lock_params = False
  
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

    print('fine_tine', fine_tune)
    if fine_tune is not None:
        model = torch.load('../model/' + fine_tune, map_location=lambda storage, loc: storage)
        model.cuda(device)
        model.device = device
        #utils.fine_tune(n_epochs, division, train_data, dev_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, _run)
        #exit()
        #utils.fine_tune(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=lock_params, penalty=penalty, params=params, stop=stop)
        #exit()


    if len(data_set) == 1:
        utils.train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, save_model, stop, _run)
    else:
        if adversarial and len(data_set) >= 15:
            utils.multi_train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, adv_train_iters=adv_train_data, adv_label_field=_label_fields[0], params=params)
        else:
            utils.multi_train(n_epochs, division, train_data, dev_data, test_data, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=False, penalty=penalty, params=params, stop=stop, rec_acc=rec_acc)
