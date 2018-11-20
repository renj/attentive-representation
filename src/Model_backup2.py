# -*- coding: utf8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
# import numpy as np
from loader import CAP_DIM
from torch.autograd import Variable
import math


'''
def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_normal(tensor, gain=1):
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def xavier_uniform(tensor, gain=1):
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)
'''


class Base(nn.Module):
    def __init__(self, param):
        super(Base, self).__init__()
        #self.device = 0

    def init_embed(self, init_matrix):
        if self.use_cuda:
            self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix).cuda(self.device))
            self.embed_weight = self.embed.weight
        else:
            self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix))

    def predict(self, words):
        #TODO: caps 
        output = self.forward(words)
        _, tag_id = torch.max(output, dim=0)
        return tag_id

    def get_loss(self, tags, words):
        logit = self.forward(words)
        loss = F.cross_entropy(logit, tags)
        #loss = F.cross_entropy(self, tags)
        return loss


class JointLSTM(Base):
    def __init__(self, param):
        super(JointLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c0 = param['c0']
        self.c1 = param['c1']
        self.device = param['device']
        self.batch = param['batch_size']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm1 = nn.LSTM(self.e, self.u, bidirectional=self.bilstm).cuda(self.device)
            self.lstm2 = nn.LSTM(self.e, self.u, bidirectional=self.bilstm).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u * 2, self.r, bias=False).cuda(self.device)
            #self.attn2 = nn.Linear(self.u, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u * 2, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c0).cuda(self.device)
            self.tagger = nn.Linear(self.u, self.c1).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False)
            self.attn2 = nn.Linear(self.u, self.r, bias=False)
            self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c0)
            self.tagger = nn.Linear(self.r, self.c1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, idx, words):
        #self.embed.weight = self.embed_weight
        embeds = self.embed(words)
        batch = embeds.size(1)
        slen = embeds.size(0)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        if idx == 0:
            H, self.hidden = self.lstm1(embeds)
            H2, self.hidden = self.lstm2(embeds)
            H = torch.cat((H, H2), 2)
            if self.debug:
                print('H', H.size())
                print('u', self.u)
                print('batch', self.batch)
                print('slen', slen)
            H = H.view(batch * slen, self.u*2)
            if self.debug:
                print('H', H.size())

            A = self.attn1(H)
            if self.debug:
                print('A', A.size())

            A = A.view(slen, batch, self.r)
            H = H.view(slen, batch, self.u * 2)
            H = torch.transpose(H, 0, 1)
            A = torch.transpose(A, 0, 1)
            A = torch.transpose(A, 1, 2)
            if self.debug:
                print('A', A.size())
                print('H', H.size())

            M = torch.bmm(A, H)
            if self.debug:
                print('M', M.size())

            M = M.view(batch * self.r, -1)
            mlp1 = F.relu(self.mlp1(M))
            mlp1 = mlp1.view(batch, -1)
            if self.debug:
                print('mlp1', mlp1.size())

            mlp2 = F.relu(self.mlp2(mlp1))
            if self.debug:
                print('mlp2', mlp2.size())

            out = F.softmax(mlp2)
        else:
            H, self.hidden = self.lstm2(embeds)
            if self.debug:
                print('H', H.size())
                print('u', self.u)
                print('batch', self.batch)
                print('slen', slen)
            H = H.view(batch * slen, self.u)
            if self.debug:
                print('H', H.size())

            #A = self.attn2(H)
            A = H
            if self.debug:
                print('A', A.size())
            out = self.tagger(A)

        self.debug = False
        return out


class ShareLSTM2(Base):
    def __init__(self, param):
        super(ShareLSTM2, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c0 = param['c0']
        self.c1 = param['c1']
        self.device = param['device']
        self.batch = param['batch_size']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.u, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c0).cuda(self.device)
            self.tagger = nn.Linear(self.r, self.c1).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False)
            self.attn2 = nn.Linear(self.u, self.r, bias=False)
            self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c0)
            self.tagger = nn.Linear(self.r, self.c1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, idx, words):
        embeds = self.embed(words)
        batch = embeds.size(1)
        slen = embeds.size(0)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)
        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        if idx == 0:
            A = self.attn1(H)
            if self.debug:
                print('A', A.size())

            A = A.view(slen, batch, self.r)
            H = H.view(slen, batch, self.u)
            H = torch.transpose(H, 0, 1)
            A = torch.transpose(A, 0, 1)
            A = torch.transpose(A, 1, 2)
            if self.debug:
                print('A', A.size())
                print('H', H.size())

            M = torch.bmm(A, H)
            if self.debug:
                print('M', M.size())

            M = M.view(batch * self.r, -1)
            mlp1 = F.relu(self.mlp1(M))
            mlp1 = mlp1.view(batch, -1)
            if self.debug:
                print('mlp1', mlp1.size())

            mlp2 = F.relu(self.mlp2(mlp1))
            if self.debug:
                print('mlp2', mlp2.size())

            out = F.softmax(mlp2)
        else:
            A = self.attn2(H)
            if self.debug:
                print('A', A.size())
            out = self.tagger(A)

        self.debug = False
        return out



class ShareLSTM(Base):
    def __init__(self, param):
        super(ShareLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c0 = param['c0']
        self.c1 = param['c1']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c0).cuda(self.device)
            self.tagger = nn.Linear(self.r, self.c1).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False)
            self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c0)
            self.tagger = nn.Linear(self.r, self.c1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, idx, words):
        embeds = self.embed(words)
        batch = embeds.size(1)
        slen = embeds.size(0)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)
        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        A = self.attn1(H)
        if self.debug:
            print('A', A.size())

        if idx == 0:
            A = A.view(slen, batch, self.r)
            H = H.view(slen, batch, self.u)
            H = torch.transpose(H, 0, 1)
            A = torch.transpose(A, 0, 1)
            A = torch.transpose(A, 1, 2)
            if self.debug:
                print('A', A.size())
                print('H', H.size())

            M = torch.bmm(A, H)
            if self.debug:
                print('M', M.size())

            M = M.view(batch * self.r, -1)
            mlp1 = F.relu(self.mlp1(M))
            mlp1 = mlp1.view(batch, -1)
            if self.debug:
                print('mlp1', mlp1.size())

            mlp2 = F.relu(self.mlp2(mlp1))
            if self.debug:
                print('mlp2', mlp2.size())

            out = F.softmax(mlp2)
        else:
            out = self.tagger(A)

        self.debug = False
        return out


class LSTMTagger(Base):
    def __init__(self, param):
        super(LSTMTagger, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['tagset_size']
        self.batch = param['batch_size']
        self.debug = True
        self.device = param['device']

        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            # TODO 这里如果使用first_batch会对性能优很大损耗，这是为什么呢？
            self.lstm = nn.LSTM(self.e, self.u, bias=False).cuda(self.device)
            self.attn = nn.Linear(self.u, self.r).cuda(self.device)
            self.tagger = nn.Linear(self.r, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bias=False)
            self.attn = nn.Linear(self.u, self.r)
            self.tagger = nn.Linear(self.r, self.c)
        #self.hidden = self.init_hidden()
        #self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.Tensor(1, 1, self.lstm_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.lstm_dim)))

    def forward(self, words):
        #words.data.t_()
        if self.debug:
            print('words:', words.size())
        embeds = self.embed(words)
        if self.debug:
            print('embeds:', embeds.size())
        H, self.hidden = self.lstm(embeds)
        H = H.contiguous().view(-1, self.u)
        if self.debug:
            print('H:', H.size())
        A = self.attn(H)
        #A = H
        if self.debug:
            print('A:', A.size())
        #A = F.softmax(self.attn(H)) 严重影响收敛
        tag_space = self.tagger(A)
        #out = F.softmax(tag_space) #严重影响收敛
        out = tag_space
        if self.debug:
            print('out:', out.size())
        self.debug = False
        return out

    def get_tags(self, words):
        tag_scores = self.forward(words)
        _, tags = torch.max(tag_scores, dim=1)
        tags = tags.data.numpy().reshape((-1,))
        return tags

    def get_loss(self, tags, words):
        tag_scores = self.forward(words=words)
        loss = self.loss_function(tag_scores, tags)
        return loss


class SelfAttnTagger(Base):
    def __init__(self, param):
        super(SelfAttnTagger, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['tagset_size']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da, bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.da, self.r, bias=False).cuda(self.device)
            #self.mlp1 = nn.Linear(self.u, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            #self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, words):
        words = words.squeeze()
        words.data.t_()
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)
        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        Attn1 = self.tanh(self.attn1(H))
        if self.debug:
            print('Attn1', Attn1.size())

        Attn2 = self.attn2(Attn1)
        if self.debug:
            print('Attn2', Attn2.size())

        A = self.softmax(Attn2)
        A = A.view(batch, slen, self.r)
        if self.debug:
            print('A', A.size())
        A = A.view(batch * slen, self.r)
        mlp2 = self.mlp2(A)
        if self.debug:
            print('mlp2', mlp2.size())
        
        out = F.softmax(mlp2)
        self.debug = False
        return out

    def get_loss(self, tags, words):
        logit = self.forward(words)
        loss = F.cross_entropy(logit, tags)
        #loss = F.cross_entropy(self, tags)
        return loss


class NewSelfAttn(Base):
    # SelfAttn without batch_first
    def __init__(self, param):
        super(NewSelfAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['tagset_size']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.r)
            self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, words):
        embeds = self.embed(words)
        batch = embeds.size(1)
        slen = embeds.size(0)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)
        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        #A = self.tanh(self.attn1(H))
        A = self.attn1(H)
        if self.debug:
            print('A', A.size())

        #A = self.softmax(Attn2)
        A = A.view(slen, batch, self.r)
        H = H.view(slen, batch, self.u)
        H = torch.transpose(H, 0, 1)
        A = torch.transpose(A, 0, 1)
        A = torch.transpose(A, 1, 2)
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        if self.debug:
            print('M', M.size())

        M = M.view(batch * self.r, -1)
        mlp1 = F.relu(self.mlp1(M))
        mlp1 = mlp1.view(batch, -1)
        if self.debug:
            print('mlp1', mlp1.size())

        mlp2 = F.relu(self.mlp2(mlp1))
        if self.debug:
            print('mlp2', mlp2.size())

        out = F.softmax(mlp2)
        self.debug = False
        return out


class JKSelfAttn(Base):
    '''Self attention copy jkchen's tensorflow codes'''
    def __init__(self, param):
        super(JKSelfAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['tagset_size']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, dropout=param['dropout'], batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da, bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.da, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, dropout=param['dropout'], batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
        '''
        self.lstm.apply(self.weight_init)
        self.attn1.apply(self.weight_init)
        self.attn2.apply(self.weight_init)
        self.mlp1.apply(self.weight_init)
        self.mlp2.apply(self.weight_init)
        '''

    def weight_init(self, m):
        if isinstance(m, nn.LSTM):
            for _a in m.all_weights[0]:
                if len(_a.size()) == 1:
                    continue
                nn.init.xavier_uniform(_a)
        if isinstance(m, nn.Linear):
            m.weight.data = nn.init.xavier_uniform(m.weight.data)

    def dist(self, tensor):
        b = tensor.cpu().data.numpy()
        mean = b.sum(0) / b.shape[0]
        import numpy as np
        c = np.array([np.linalg.norm(b[i] - mean) for i in range(b.shape[0])])
        return np.std(c)

    def forward(self, words, log=False):
        words = words.squeeze()
        words.data.t_()
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)

        if log:
            print('H std:', self.dist(H))

        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        Attn1 = F.tanh(self.attn1(H))
        if self.debug:
            print('Attn1', Attn1.size())

        Attn2 = self.attn2(Attn1)
        if self.debug:
            print('Attn2', Attn2.size())

        A = F.softmax(Attn2)
        A = A.view(batch, slen, self.r)
        H = H.view(batch, slen, self.u)
        A = torch.transpose(A, 1, 2)
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        if self.debug:
            print('M', M.size())
        if log:
            print('M std:', self.dist(M))
            #print(M)

        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = mlp1
        self.debug = False
        return out


class SingleSelfAttn(Base):
    def __init__(self, param):
        super(SingleSelfAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['tagset_size']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, dropout=param['dropout'], batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da, bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.da, self.r, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, 1).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, dropout=param['dropout'], batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, 1)
            self.mlp2 = nn.Linear(self.r, self.c)
        '''
        self.lstm.apply(self.weight_init)
        self.attn1.apply(self.weight_init)
        self.attn2.apply(self.weight_init)
        self.mlp1.apply(self.weight_init)
        self.mlp2.apply(self.weight_init)
        '''

    def weight_init(self, m):
        if isinstance(m, nn.LSTM):
            for _a in m.all_weights[0]:
                if len(_a.size()) == 1:
                    continue
                nn.init.xavier_uniform(_a)
        if isinstance(m, nn.Linear):
            m.weight.data = nn.init.xavier_uniform(m.weight.data)

    def dist(self, tensor):
        b = tensor.cpu().data.numpy()
        mean = b.sum(0) / b.shape[0]
        import numpy as np
        c = np.array([np.linalg.norm(b[i] - mean) for i in range(b.shape[0])])
        return np.std(c)

    def forward(self, words, log=False):
        words = words.squeeze()
        words.data.t_()
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)
        H, self.hidden = self.lstm(embeds)

        if log:
            print('H std:', self.dist(H))

        if self.debug:
            print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        Attn1 = F.tanh(self.attn1(H))
        if self.debug:
            print('Attn1', Attn1.size())

        Attn2 = self.attn2(Attn1)
        if self.debug:
            print('Attn2', Attn2.size())

        A = F.softmax(Attn2)
        A = A.view(batch, slen, self.r)
        H = H.view(batch, slen, self.u)
        A = torch.transpose(A, 1, 2)
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        if self.debug:
            print('M', M.size())
        if log:
            print('M std:', self.dist(M))
            #print(M)

        M = M.view(batch * self.r, -1)
        mlp1 = F.relu(self.mlp1(M))
        #mlp1 = self.mlp1(M)
        mlp1 = mlp1.view(batch, -1)
        if self.debug:
            print('mlp1', mlp1.size())

        #mlp2 = F.relu(self.mlp2(mlp1))
        mlp2 = self.mlp2(mlp1)
        if self.debug:
            print('mlp2', mlp2.size())

        if log:
            print('mlp2 std:', self.dist(mlp2))
            print(mlp2)
        out = F.softmax(mlp2)
        self.debug = False
        return out


class SingleLSTM(Base):
    def __init__(self, param):
        super(SingleLSTM, self).__init__(param)
        V = param['vocab_size']
        D = param['embed_dim']
        C = param['tagset_size']
        self.hidden_dim = param['hidden_dim']

        self.embed = nn.Embedding(V, D)
        self.embed_dim = D
        self.use_cuda = param['use_cuda']
        self.device = param['device']

        '''
        if self.lower:
            self.embed_dim += CAP_DIM
        '''
        self.lstm = nn.LSTM(D, param['hidden_dim'])
        self.hidden2tag = nn.Linear(param['hidden_dim'], param['tagset_size'])
        self.hidden = self.init_hidden()
        self.loss_function = nn.NLLLoss()
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, words):
        words = words.squeeze()
        #print(type(words))
        #print(len(words))
        embeds = self.embed(words)
        #print(embeds.size())
        #print(embeds.view(len(words), 1, -1).size())
        
        '''
        if self.lower:
            caps = input['caps']
            input_caps = torch.FloatTensor(len(caps), CAP_DIM)
            input_caps.zero_()
            input_caps.scatter_(1, caps.view(-1,1) ,1)
            input_caps = autograd.Variable(input_caps)
            embeds = torch.cat((embeds, input_caps),1)
        '''

        lstm_out, self.hidden = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        #return tag_space[-1].view(1,-1)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores[-1].view(1, -1)
        #tag_scores = nn.LogSoftmax(tag_space)
        #return tag_scores[-1]


class SingleCNN(Base):
    def __init__(self, param):
        super(SingleCNN, self).__init__(param)
        V = param['vocab_size']
        Ks = [3, 4, 5]
        D = param['embed_dim']
        Co = 100
        Ci = 1
        self.use_cuda = param['use_cuda']
        C = param['tagset_size']
        print('V: %d, D: %d, C: %d, Co: %d, Ks, %s'%(V, D, C, Co, Ks))
        self.device = param['device']

        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(V, D).cuda(self.device)
            self.convs1 = [nn.Conv2d(Ci, Co, (K, D)).cuda(self.device) for K in Ks]
        else:
            self.embed = nn.Embedding(V, D)
            self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

        #self.loss_function = nn.NLLLoss()


    def init_hidden(self):
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, words):
        words.data.t_()
        if type(words) is torch.LongTensor:
            words = Variable(words)
        if self.debug:
            print('words.size: ', words.size())
        x = self.embed(words)
        if self.debug:
            print('embed.size: ', x.size())

        x = x.unsqueeze(1) # (N,Ci,W,D)
        if self.debug:
            print('embed.size: ', x.size())
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N,Co,W), ...]*len(Ks)
        if self.debug:
            for _x in x:
                print('relu.size: ', _x.size())
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N,Co), ...]*len(Ks)
        if self.debug:
            for _x in x:
                print('pool.size: ', _x.size())
        x = torch.cat(x, 1)
        if self.debug:
            print('cat.size: ', x.size())
        x = self.dropout(x) # (N,len(Ks)*Co)
        if self.debug:
            print('dropout.size: ', x.size())
        logit = self.fc1(x) # (N,C)
        if self.debug:
            print('fc1.size: ', logit.size())

        self.debug = False
        return logit

    '''
    def predict(self, **input):
        #TODO: caps 
        words = input['words']
        output = self.forward(words=words)
        _, tag_id = torch.max(output, dim=0)
        return tag_id

    def get_loss(self, tags, **input):
        words = input['words']
        logit = self.forward(words=words)
        loss = F.cross_entropy(logit, tags)
        #loss = F.cross_entropy(self, tags)
        return loss
    '''


"""
class ShareLSTM(Base):
    def __init__(self, param):
        super(ShareLSTM, self).__init__(param)
        V = param['vocab_size']
        D = param['embed_dim']
        C = param['tagset_size']
        self.hidden_dim = param['hidden_dim']
        self.embed = nn.Embedding(V, D)
        self.embed_dim = D

        '''
        if self.lower:
            self.embed_dim += CAP_DIM
        '''

        self.lstm = nn.LSTM(D, param['hidden_dim'])

        # The linear layer that maps from hidden state space to tag space
        self.w1 = nn.Linear(param['hidden_dim'], C)
        self.w2 = nn.Linear(param['hidden_dim'], 2)
        self.hidden = self.init_hidden()
        self.loss_function = nn.NLLLoss()


    def init_hidden(self):
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, words):
        embeds = self.embed(words)

        if self.lower:
            caps = input['input_caps']
            input_caps = torch.FloatTensor(len(caps), CAP_DIM)
            input_caps.zero_()
            input_caps.scatter_(1, caps.view(-1, 1), 1)
            input_caps = autograd.Variable(input_caps)
            embeds = torch.cat((embeds, input_caps), 1)

        lstm_out, self.hidden = self.lstm(embeds.view(len(words), 1, -1))
        if input['data_set'] == 1:
            tag_space = self.w1(lstm_out.view(len(words), -1))
        else:
            tag_space = self.w2(lstm_out.view(len(words), -1))

        #tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space)
        #tag_scores = nn.LogSoftmax(tag_space)
        return tag_scores[-1]

    def get_tags(self, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            output = self.forward(words,
                                  input_caps=input_caps,
                                  data_set=input['data_set'])
        else:
            output = self.forward(words=words)

        _, tag_id = torch.max(output, dim=0)

        return tag_id

    def get_loss(self, tags, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            tag_scores = self.forward(words=words, input_caps=input_caps, data_set=input['data_set'])
        else:
            tag_scores = self.forward(words=words)

        loss = self.loss_function(tag_scores, tags)
        return loss

    def get_loss_2(self, data_set, tags, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            tag_scores = self.forward(data_set, words=words, input_caps=input_caps)
        else:
            tag_scores = self.forward(data_set, words=words)

        loss = self.loss_function(tag_scores, tags)
        return loss

"""