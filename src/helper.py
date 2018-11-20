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
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib import lines
from matplotlib import image
import matplotlib.cm as matcm
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from loader import CAP_DIM
from torch.autograd import Variable
from imp import reload
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
#from itertools import tee, izip


'''
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)
'''

def get_text(ids, weights, text_field):
    text = []
    ws = []
    if type(ids[0]) is Variable:
        ids = ids.data
    for w, a in zip(ids, weights):
        #cnt += 1
        if w == 1:
            #print(cnt)
            break
        text.append(text_field.vocab.itos[w].strip())
        ws.append(a.data[0])
    #print(ws)
    w_max = max(ws)
    w_min = min(ws)
    for i in range(len(ws)):
        ws[i] = (ws[i] - w_min) / (w_max - w_min)
    return text, ws


def draw_text(text, ws, title=None):
    #setfont = lambda x: r'$\mathtt{%s}$' % x
    f = plt.figure(figsize=(4, 3.5))
    ax=f.add_subplot(111)
    r = f.canvas.get_renderer()

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    x = .0
    y = .0
    _width, _height = 0, 0
    lines = 1
    for t, w in zip(text, ws):
        txt = plt.text(x, y, t, size=20,
                 bbox=dict(facecolor='red', alpha=w, edgecolor='none'))
        tbox = txt.get_window_extent(renderer=r)
        #tbox = txt.get_bbox_patch()
        dbox = tbox.transformed(ax.transData.inverted())
        width = dbox.x1-dbox.x0
        height = dbox.y1-dbox.y0
        x = x + width + 0.028
        #x = _x + width * 2
        y = y
        #ax.add_patch(Rectangle((x - width/2, y), width, height))
        #_x, _y, _width, _height = x, y, width, height
        #print(width, height)
        #print(x, y)
        #x += width
        if x > 3:
            lines += 1
            y -= height/5
            x = 0
    f.set_size_inches(4, 0.7 * lines)
    plt.ylim(y - height/5, 0)
    if title is not None:
        plt.savefig(title + ".pdf", bbox_inches='tight')

def text2feature(text, dic):
    ids = []
    if type(text) is str:
        text = text.split(' ')
    for t in text:
        ids.append(dic[t])
    feature = torch.Tensor(ids).long()
    feature = Variable(feature.view(1,-1))
    return ids, feature

'''
def draw_text(text, ws):
    setfont = lambda x: r'$\mathtt{%s}$' % x
    plt.ylim(-5, 0)
    plt.xlim(0, 0.7)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    slen = 0
    hight = 0
    for t, w in zip(text, ws):
        #plt.text(slen * 0.015, hight, setfont(t),
        plt.text(slen * 0.015, hight, t,
                 bbox=dict(facecolor='red', alpha=w, edgecolor='none'))
        slen += len(t) + 1
        if slen > 80:
            hight -= 0.5
            slen = 0
    plt.show()
    #plt.close()
    #print(hight)
'''

def get(self, words, idx=None, is_tagger=False, log=False):
    embeds = self.embed(words)
    batch = embeds.size(0)
    slen = embeds.size(1)
    if self.debug:
        print('embeds', embeds.size())
    #embeds = torch.transpose(embeds, 0, 1)

    lens = []
    for line in words.data:
        length = 0
        for l in range(len(line) - 1, -1, -1):
            if line[l] != 1:
                length = l + 1
                break
        if length == 0:
            print('Error', line)
        lens.append(length)

    packed = pack_padded_sequence(embeds, lens, batch_first=True)
    H, _ = self.lstm(packed)
    H, self.hidden = pad_packed_sequence(H, batch_first=True)

    #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

    if self.debug:
        #print('H', H.size())
        print('u', self.u)
        print('batch', self.batch)
        print('slen', slen)
    H = H.contiguous().view(batch * slen, self.u)
    #H = H.view(batch * slen, self.u)
    if self.debug:
        print('H', H.size())

    if is_tagger:
        tag_space = self.tagger(H)
        return tag_space

    if idx is None:
        A = self.attn(H).view(batch, slen, 1)
    else:
        A = self.attn[idx](H).view(batch, slen, 1)
    A = torch.transpose(A, 1, 2)
    A = A.contiguous().view(-1, slen)
    A = F.softmax(A)
    A = A.view(batch, 1, slen)
    H = H.view(batch, slen, self.u)

    if self.debug:
        print('A', A.size())
        print('H', H.size())

    M = torch.bmm(A, H)

    M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
    if self.debug:
        print('M', M.size())

    if idx is None:
        return A, H, M
    else:
        mlp2 = self.mlp2[idx](M)
    if self.debug:
        print('mlp1', mlp2.size())

    #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
    #out = F.softmax(mlp1)
    out = self.dropout(mlp2)
    self.debug = False

    return A, H, M, mlp2, out

def get_da(self, words, idx, is_tagger=False, log=False, penalty=False):
    embeds = self.embed(words)
    batch = embeds.size(0)
    slen = embeds.size(1)
    if self.debug:
        print('embeds', embeds.size())
    #embeds = torch.transpose(embeds, 0, 1)

    lens = []
    for line in words.data:
        length = 0
        for l in range(len(line) - 1, -1, -1):
            if line[l] != 1:
                length = l + 1
                break
        if length == 0:
            print('Error', line)
        lens.append(length)

    packed = pack_padded_sequence(embeds, lens, batch_first=True)
    H, _ = self.lstm(packed)
    H, self.hidden = pad_packed_sequence(H, batch_first=True)

    #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

    if self.debug:
        #print('H', H.size())
        print('u', self.u)
        print('batch', self.batch)
        print('slen', slen)
    H = H.contiguous().view(batch * slen, self.u)
    #H = H.view(batch * slen, self.u)
    if self.debug:
        print('H', H.size())

    if is_tagger:
        tag_space = self.tagger(H)
        return tag_space
    '''
    if idx >= len(self.mlp1):
        tag_space = self.tagger(H)
        return tag_space
    '''
    '''
    Attn = Variable(torch.zeros(batch * slen, self.n))
    if self.use_cuda:
        Attn = Attn.cuda(self.device)
    for i in range(self.n):
        if i == idx:
            self.attn[i].weight.requires_grad = True
        else:
            self.attn[i].weight.requires_grad = False
        Attn[:, i] = self.attn[i](H)

    if self.debug:
        print('Attn', Attn.size())
    '''
    '''
    if log:
        print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))
    '''

    A = self.attn2(H).view(batch, slen, 1)
    A = torch.transpose(A, 1, 2)
    A = A.contiguous().view(-1, slen)
    A = F.softmax(A)
    A = A.view(batch, 1, slen)
    H = H.view(batch, slen, self.u)

    M = torch.bmm(A, H).squeeze().view(1, -1)

    if self.debug:
        print('A', A.size())
        print('H', H.size())
        print('M', M.size())

    if idx == 1:
        mlp2 = self.mlp2(M)
        if self.debug:
            print('mlp2', mlp2.size())
        out = self.dropout(mlp2)
        self.debug = False
        return A, out

    '''
    M = torch.mean(M, 1)
    M = M.squeeze()
    if self.debug:
        print('M', M.size())
    '''
    #M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
    if self.debug:
        print('M', M.size())
    '''
    mlp1 = self.mlp1[idx](M)
    if self.debug:
        print('mlp1', mlp1.size())
    if log and self.use_cuda:
        print('mlp1', self.mlp1[idx].weight.data.cpu().numpy())
    mlp1 = mlp1.view(batch, self.u)
    '''
    q = self.attn2q(M).view(batch, self.u, 1)
    A = torch.bmm(H, q).view(batch, slen, 1)
    A = torch.transpose(A, 1, 2)
    A = A.contiguous().view(-1, slen)
    A = F.softmax(A)
    A = A.view(batch, 1, slen)
    H = H.view(batch, slen, self.u)
    M = torch.bmm(A, H).squeeze().view(1, -1)
    if self.debug:
        print('M', M.size())

    mlp1 = self.mlp1(M)
    if self.debug:
        print('mlp1', mlp1.size())

    #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
    #out = F.softmax(mlp1)
    out = self.dropout(mlp1)
    self.debug = False

    if penalty is True:
        pen = 0
        a = M.view(batch, self.u, self.n)[:, :, idx]
        b = mlp1
        target = Variable(torch.zeros(batch, self.u))
        if self.use_cuda:
            target = target.cuda(self.device)
        l1_crit = nn.L1Loss()
        pen += l1_crit(a - b, target)
        '''
        for i in range(self.n):
            if i == idx:
                continue
            a = M.view(batch, self.u, self.n)[:, :, i]
            b = mlp1
            a = a.unsqueeze(2)
            b = b.unsqueeze(1)
            bmm = torch.bmm(a, b)
            diag = torch.diag(torch.ones(self.u))
            if self.use_cuda:
                diag = diag.cuda(self.device)
            for j in range(batch):
                pen += torch.norm(bmm[j].data - diag)
        '''
        return out, pen

    return A, out

'''
def get(self, words, idx, is_tagger=False, log=False):
    embeds = self.embed(words)
    batch = embeds.size(0)
    slen = embeds.size(1)
    if self.debug:
        print('embeds', embeds.size())
    #embeds = torch.transpose(embeds, 0, 1)

    lens = []
    for line in words.data:
        length = 0
        for l in range(len(line) - 1, -1, -1):
            if line[l] != 1:
                length = l + 1
                break
        if length == 0:
            print('Error', line)
        lens.append(length)

    packed = pack_padded_sequence(embeds, lens, batch_first=True)
    H, _ = self.lstm(packed)
    H, self.hidden = pad_packed_sequence(H, batch_first=True)


    if self.debug:
        #print('H', H.size())
        print('u', self.u)
        print('batch', self.batch)
        print('slen', slen)
    H = H.contiguous().view(batch * slen, self.u)
    #H = H.view(batch * slen, self.u)
    if self.debug:
        print('H', H.size())

    if is_tagger:
        tag_space = self.tagger(H)
        return tag_space

    Attn1 = F.tanh(self.attn1[idx](H))
    if self.debug:
        print('Attn1', Attn1.size())


    Attn2 = self.attn2[idx](Attn1)
    if self.debug:
        print('Attn2', Attn2.size())



    A = Attn2.view(batch, slen, self.r)
    A = torch.transpose(A, 1, 2)
    A = A.contiguous().view(-1, slen)
    A = F.softmax(A)
    A = A.view(batch, self.r, slen)
    H = H.view(batch, slen, self.u)

    if self.debug:
        print('A', A.size())
        print('H', H.size())

    M = torch.bmm(A, H)
    if self.debug:
        print('M', M.size())

    M = torch.mean(M, 1)
    M = M.squeeze()
    if self.debug:
        print('M', M.size())
    mlp = self.mlp[idx](M)
    if self.debug:
        print('mlp1', mlp.size())

    #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
    #out = F.softmax(mlp1)
    out = self.dropout(mlp)
    self.debug = False
    return Attn2, H, A, M, mlp, out
'''


def get_one(self, words, log=False):
    embeds = self.embed(words)
    batch = embeds.size(0)
    slen = embeds.size(1)
    if self.debug:
        print('embeds', embeds.size())
    #embeds = torch.transpose(embeds, 0, 1)

    lens = []
    for line in words.data:
        length = 0
        for l in range(len(line) - 1, -1, -1):
            if line[l] != 1:
                length = l + 1
                break
        if length == 0:
            print('Error', line)
        lens.append(length)

    packed = pack_padded_sequence(embeds, lens, batch_first=True)
    H, _ = self.lstm(packed)
    H, self.hidden = pad_packed_sequence(H, batch_first=True)

    if self.debug:
        #print('H', H.size())
        print('u', self.u)
        print('batch', self.batch)
        print('slen', slen)
    H = H.contiguous().view(batch * slen, self.u)
    #H = H.view(batch * slen, self.u)
    if self.debug:
        print('H', H.size())

    #Attn1 = F.tanh(self.attn1(H))
    Attn1 = self.attn1(H)
    if self.debug:
        print('Attn1', Attn1.size())
    if log:
        print('Attn1 max: %f, min: %f' % (Attn1.max().data[0], Attn1.min().data[0]))

    Attn2 = self.attn2(Attn1)
    if self.debug:
        print('Attn2', Attn2.size())
    if log:
        print('Attn2 max: %f, min: %f' % (Attn2.max().data[0], Attn2.min().data[0]))

    '''old
    A = F.softmax(Attn2)
    if log:
        print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
    A = A.view(batch, slen, self.r)
    A = torch.transpose(A, 1, 2)
    '''
    #New start
    A = Attn2.view(batch, slen, self.r)
    A = torch.transpose(A, 1, 2)
    A = A.contiguous().view(-1, slen)
    A = F.softmax(A)
    A = A.view(batch, self.r, slen)
    #New end

    H = H.view(batch, slen, self.u)

    if self.debug:
        print('A', A.size())
        print('H', H.size())

    M = torch.bmm(A, H)
    if self.debug:
        print('M', M.size())
        #print(M)

    M = torch.mean(M, 1)
    M = M.squeeze()
    if self.debug:
        print('M', M.size())
    mlp1 = self.mlp1(M)
    if self.debug:
        print('mlp1', mlp1.size())

    #out = F.softmax(mlp1)
    out = self.dropout(mlp1)
    self.debug = False
    return Attn2, H, A, M, mlp1, out