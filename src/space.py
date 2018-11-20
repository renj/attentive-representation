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

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from loader import CAP_DIM
from torch.autograd import Variable
from imp import reload

ex_name = 'test'
#data_set = 'sst1','conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
#data_set = 'sst1'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
#data_set = 'conll'  # 'sst1', 'sst2', 'conll', ['sst1', 'sst2']
#data_set = 'product/books'
#data_set = 'product/dvd'
#data_set = 'product/electronics'
#data_set = 'product/kitchen'
#data_set = 'product/music'
#data_set = 'product/video'
#data_set = 'product/books,product/music,product/video,product/kitchen,product/electronics,product/dvd'
#data_set = 'product/books,product/music,product/video,product/kitchen,product/electronics,product/dvd,conll'
data_set = 'product/books,product/elec,product/dvd,product/kitchen,product/apparel,product/camera,product/health,product/music,product/toys,product/video,product/baby,product/mag,product/soft,product/sports,product/imdb,product/mr'
#data_set = 'product/sentiment,product/books'
#data_set = 'product/books,product/music,conll'
pre_emb = '../embed/glove.6B.100d.txt'
#use_model = 'SingleCNN'
#use_model = 'NewSelfAttn'
#use_model = 'SelfAttnTagger'
#use_model = 'ShareLSTM2'
#use_model = 'ShareSelfAttn'
#use_model = 'LSTMTagger'
use_model = 'AttnTagger'
#use_model = 'JKSelfAttn'
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
penalty = 0.0
l2_reg = 0.0001
division = 25
seed = 283953214
fine_tune = None

if type(data_set) is not list:
    data_set = data_set.split(',')
print(data_set)
#train_data, dev_data, test_data, c, text_field, label_fields = loader.read_data(data_set, batch_size)

#model_path = '../model/conll_dvd_electronics_kitchen_video_music_books_0.0_82.74.torch'
#model_path = '../model/Model.SingleAttn_sentiment_88.03.torch'
#model_path = '../model/old/Model.ASPCatAttn_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_0.0001_0.0001_87.11.torch'
#model_path = '../model/Old_Model.AttnOverAttn_sentiment_87.73.torch'
model_path = '../model/Model.SPLSTM_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.39.torch'
#model_path = '../model/Model.ShareSingleAttn_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_86.08.torch'
#model_path = '../model/Model.SingleAttn_sentiment_88.33.torch'
#model_path = '../model/old/Model.MultiSingleAttn_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_0.000_0.000_82.55.torch'
#model_path = '../model/Model.AttnOverAttn_topic_sentiment_82.00.torch'
#model_path = '../model/Model.MultiDynAttn_topic_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_87.48.torch'
#model_path = '../model/Model.StackLSTM_mr_imdb_sports_soft_mag_baby_video_toys_music_health_camera_apparel_kitchen_dvd_elec_books_85.55.torch'
model = torch.load(model_path)

params = list(model.parameters())
ans = 0
for p in params:
    s = p.size()
    if len(s) == 1:
        ans += s[0]
    else:
        if s[0] >= 94525:
            continue
        ans += s[0] * s[1]
print(ans)

'''
utils.evaluate(model, test_data, idx=1, use_cuda=True, data_set=data_set)
accs = []
if len(data_set) > 1:
    for i in range(len(test_data)):
        #acc = utils.evaluate(model, test_data, idx=i, use_cuda=True, data_set=data_set)
        acc = utils.evaluate(model, test_data[i], use_cuda=True, data_set=data_set)
        accs.append(acc)
else:
    utils.evaluate(model, test_data, True)
for acc in accs:
    print(acc)
'''