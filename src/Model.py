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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import time
import random
np.set_printoptions(precision=3)
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

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Base(nn.Module):
    def __init__(self, param):
        super(Base, self).__init__()
        #self.device = 0
        self.state = None

    def init_embed(self, init_matrix):
        if self.use_cuda:
            self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix).cuda(self.device))
            self.embed_weight = self.embed.weight
        else:
            self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix))

    def train(self, mode=True):
        if mode is True:
            self.state = 'train'
            #print('train')
        super(Base, self).train(mode)

    def eval(self):
        self.state = 'eval'
        #print('eval')
        super(Base, self).eval()

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


def print_stat(v, s):
    return
    print('%s: mean: %.4f, max: %.4f, min: %.4f, std: %.4f' % (s, 
        v.mean().data[0], v.max().data[0], v.min().data[0], v.std().data[0]))


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

pos_enc = position_encoding_init(10000, 100)


class NewMemAttn(Base):
    def __init__(self, param):
        super(NewMemAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.nlayers = param['nlayers']
        self.epsilon = param['epsilon']
        self.p_lambda = param['p_lambda']
        self.p_lambda2 = param['p_lambda2']
        self.p_gamma = param['p_gamma']
        self.rl_batch = param['rl_batch']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.rl = 'rl' in param['params']
        self.pos = 'pos' in param['params']
        self.pause = 'pause' in param['params']
        self.fix_H = 'fix_H' in param['params']
        self.isolate_P = 'isolate_P' in param['params']
        self.last_h = 'last_h' in param['params']
        self.H_h = 'H_h' in param['params']
        self.test_rl = 'test_rl' in param['params']
        self.discount = 'discount' in param['params']
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            if self.pos:
                self.pos_len = 100
            else:
                self.pos_len = 0
            self.attn = nn.Linear(self.u + self.pos_len, 1, bias=False).cuda(self.device)
            self.gru = nn.GRUCell(self.u + self.pos_len, self.u + self.pos_len)
            self.mlp1 = nn.Linear(self.u + self.pos_len, self.c).cuda(self.device)
            if self.H_h:
                self.P = nn.Linear(self.u * 2 + self.pos_len, 2).cuda(self.device)
            else:
                self.P = nn.Linear(self.u + self.pos_len, 2).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.D = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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
        #H = embeds

        if self.debug:
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch, slen, self.u)
        if self.debug:
            print('H', H.size())

        if self.H_h:
            _a = torch.zeros(batch, slen)
            for _i, _l in enumerate(lens):
                for _j in range(_l):
                    _a[_i][_j] = 1
            _a = F.softmax(Variable(_a).cuda(self.device)).view(batch, 1, slen)
            H_sum = torch.bmm(_a, H).squeeze()


        """
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        '''
        if self.state is 'train':
            M = torch.bmm(A, H)
        elif self.state is 'eval':
            A = A.view(batch, slen)
            w, a = A.topk(1, dim=1)
            Ms = []
            a = a.view(-1)
            for i, _a in enumerate(a.data):
                Ms.append(H[i][_a].view(1, self.u))
            M = torch.cat(Ms)
        '''
        """

        q = torch.cat([self.attn.weight] * batch)
        if self.debug:
            print('q', q.size())

        Ds = []
        Ms = []
        Ps = []
        Ws = []
        DWs = []

        self.u += self.pos_len
        if self.pos:
            p = Variable(pos_enc[torch.LongTensor(range(slen) * batch)]).view(batch, slen, self.pos_len).cuda(self.device)
            H = torch.cat([H, p], 2)

        # TODO: fix H
        if self.fix_H:
            fix_H = Variable(H.data)
        for _i in range(self.nlayers):
            q = q.view(batch, self.u, 1)
            if self.fix_H:
                A = torch.bmm(fix_H, q).view(-1, slen)
            else:
                A = torch.bmm(H, q).view(-1, slen)
            A_double = F.softmax(A.double())
            A = F.softmax(A)


            if self.rl:
                for _j in range(self.rl_batch):
                    a, w = [], []
                    for i, l in enumerate(lens):
                        select = np.random.choice(["sample", "random"], p=[1.0 - self.epsilon, self.epsilon])
                        if select == 'sample':
                            #_w = A_double[i].data.squeeze().cpu()
                            #sample = list(torch.utils.data.sampler.WeightedRandomSampler(_w, 1))[0]
                            _w = A_double[i].data.squeeze().cpu().tolist()
                            #print('w_sum:', sum(_w))
                            sample = np.random.choice(range(slen), p=_w)
                            #print('i:', i, 'sample:', sample, 'slen:', slen, max(_w), min(_w))
                            if sample == 0 and ((_w[0] >= 0) is False):
                                print('A:', A)
                                print('A_double:', A_double)
                                print('H:', H)
                                print('q:', q)
                                exit()
                        elif select == 'random':
                            sample = random.randint(0, l-1)
                        a.append(sample)
                        w.append(A[i][sample])
                    w = torch.cat(w)
                    #a = Variable(torch.LongTensor(a))
                    #Ws.append(w)
                    Ws += [w]

                    _Ms = []
                    #a = a.view(-1)
                    #for i, _a in enumerate(a.data):
                    for i, _a in enumerate(a):
                        _Ms.append(H[i][_a].view(1, self.u))
                    M = torch.cat(_Ms)

                    q = q.squeeze()
                    q = self.gru(M, q)

                    _Ms = Ms + [M.view(batch, 1, self.u)]
                    _Ms = torch.sum(torch.cat(_Ms, 1), 1).squeeze()
                    if self.last_h:
                        _Ms = M
                    D = self.mlp1(_Ms)
                    #TODO: dropout or not?
                    DWs += [self.dropout(D)]
                    #DWs += [D]
  

            if self.rl:
                w, a = A.topk(1, dim=1)

                _Ms = []
                a = a.view(-1)
                for i, _a in enumerate(a.data):
                    _Ms.append(H[i][_a].view(1, self.u))
                M = torch.cat(_Ms)
            else:
                A = A.view(batch, 1, slen)
                M = torch.bmm(A, H).squeeze()

            Ms.append(M.view(batch, 1, self.u))

            q = q.squeeze()
            q = self.gru(M, q)

            _Ms = torch.sum(torch.cat(Ms, 1), 1).squeeze()
            if self.last_h:
                _Ms = M
            D = self.mlp1(_Ms)
            Ds += [self.dropout(D)]
            if self.debug:
                print('D', D.size())
                
            if self.pause:
                if self.isolate_P:
                    if self.H_h:
                        _Ms = Variable(_Ms.data)
                        _tmp = torch.cat([_Ms, Variable(H_sum.data) - _Ms], 1)
                        P = F.softmax(self.P(_tmp))
                    else:
                        P = F.softmax(self.P(Variable(_Ms.data)))
                else:
                    if self.H_h:
                        _tmp = torch.cat([_Ms, H_sum], 1)
                        #print('H_sum: ', H_sum)
                        #print('Ms: ', _Ms)
                        #print('tmp: ', _tmp)
                        P = F.softmax(self.P(_tmp))
                    else:
                        P = F.softmax(self.P(_Ms))
                Ps += [P]

            '''
            _Ms = torch.sum(torch.cat(Ms, 1), 1).squeeze()
            D = self.mlp1(_Ms)
            if self.debug:
                print('D', D.size())

            #out = F.softmax(D)
            #out1s.append(self.dropout(D))
            out1s += [self.dropout(out)]
            '''
        #out = F.softmax(D)
        #out1s.append(self.dropout(D))
  
        self.u -= self.pos_len
        self.debug = False

        if self.state is 'eval' and self.pause and self.rl:
            return Ds, Ps

        if self.pause and self.rl:
            return Ds, Ps, (Ws, DWs)
        if self.pause:
            return Ds, Ps
        elif self.rl:
            return Ds, (Ws, DWs)
        else:
            return Ds


class MemAttn(Base):
    def __init__(self, param):
        super(MemAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.nlayers = param['nlayers']
        self.epsilon = param['epsilon']
        self.p_lambda = param['p_lambda']
        self.p_lambda2 = param['p_lambda2']
        self.p_gamma = param['p_gamma']
        self.rl_batch = param['rl_batch']
        self.a_pen = param['a_pen']
        self.clip = param['clip']
        self.beta = param['beta']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.rl = 'rl' in param['params']
        self.pos = 'pos' in param['params']
        self.pause = 'pause' in param['params']
        self.fix_H = 'fix_H' in param['params']
        self.isolate_P = 'isolate_P' in param['params']
        self.last_h = 'last_h' in param['params']
        self.H_h = 'H_h' in param['params']
        self.test_rl = 'test_rl' in param['params']
        self.discount = 'discount' in param['params']
        self.bpause_even = 'bpause_even' in param['params']
        self.brl_even = 'brl_even' in param['params']
        self.rl_torch = 'rl_torch' in param['params']
        self.pause_torch = 'pause_torch' in param['params']
        self.q_D = 'q_D' in param['params']
        self.q_P = 'q_P' in param['params']
        self.hinge = 'hinge' in param['params']
        self.aoa = 'aoa' in param['params']
        self.all_ce = 'all_ce' in param['params']
        self.save_perf = 'save_perf' in param['params']
        self.cpx_Hh = 'cpx_Hh' in param['params']
        self.diff_Ds = 'diff_Ds' in param['params']
        self.max_ce = 'max_ce' in param['params']
        self.params = param['params']
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            if self.pos:
                self.pos_len = 100
            else:
                self.pos_len = 0
            self.attn = nn.Linear(self.u + self.pos_len, 1, bias=False).cuda(self.device)
            self.gru = nn.GRUCell(self.u + self.pos_len, self.u + self.pos_len).cuda(self.device)
            #self.gru = nn.Linear(self.u + self.pos_len, self.u + self.pos_len).cuda(self.device)
            if self.diff_Ds:
                self.mlp1 = ListModule(*[nn.Linear(self.u + self.pos_len, self.c).cuda(self.device) for _ in range(self.nlayers)])
            else:
                self.mlp1 = nn.Linear(self.u + self.pos_len, self.c).cuda(self.device)
            self.secattn = nn.Linear(self.u + self.pos_len, 1, bias=False).cuda(self.device)
            if self.cpx_Hh:
                self.P = nn.Linear(self.u * 4 + self.pos_len, 2).cuda(self.device)
            elif self.H_h:
                self.P = nn.Linear(self.u * 2 + self.pos_len, 2).cuda(self.device)
            else:
                self.P = nn.Linear(self.u + self.pos_len, 2).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.D = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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
        #H = embeds

        if self.debug:
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch, slen, self.u)
        if self.debug:
            print('H', H.size())

        if self.H_h or self.cpx_Hh:
            _a = torch.zeros(batch, slen)
            for _i, _l in enumerate(lens):
                for _j in range(_l):
                    _a[_i][_j] = 1
            _a = F.softmax(Variable(_a).cuda(self.device)).view(batch, 1, slen)
            H_sum = torch.bmm(_a, H).squeeze()


        """
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        '''
        if self.state is 'train':
            M = torch.bmm(A, H)
        elif self.state is 'eval':
            A = A.view(batch, slen)
            w, a = A.topk(1, dim=1)
            Ms = []
            a = a.view(-1)
            for i, _a in enumerate(a.data):
                Ms.append(H[i][_a].view(1, self.u))
            M = torch.cat(Ms)
        '''
        """

        Ds = []
        Ws = []
        Ps = []
        As = []
        qs = []
        MDs = []

        self.u += self.pos_len
        if self.pos:
            p = Variable(pos_enc[torch.LongTensor(range(slen) * batch)]).view(batch, slen, self.pos_len).cuda(self.device)
            H = torch.cat([H, p], 2)

        # TODO: fix H
        if self.fix_H:
            fix_H = Variable(H.data)

        episodes = []
        for _j in range(self.rl_batch):
            q = torch.cat([self.attn.weight] * batch)
            if self.debug:
                print('q', q.size())
            Ms = []
            actions = []
            for _i in range(self.nlayers):
                q = q.view(batch, self.u, 1)
                if self.fix_H:
                    A = torch.bmm(fix_H, q).view(-1, slen)
                else:
                    A = torch.bmm(H, q).view(-1, slen)
                A = F.softmax(A)
                A_double = F.softmax(A.double())
                if self.a_pen > 0:
                    As.append(A)

                A = A.view(batch, 1, slen)
                M = torch.bmm(A, H).squeeze()

                Ms.append(M.view(batch, 1, self.u))
                if self.last_h:
                    _Ms = M
                else:
                    _Ms = torch.sum(torch.cat(Ms, 1), 1).squeeze()

                q = q.squeeze()
                q = self.gru(_Ms, q)
                qs.append(q.view(batch, 1, self.u))

                if self.q_D:
                    if self.diff_Ds:
                        D = self.mlp1[_i](q)
                    else:
                        D = self.mlp1(q)
                else:
                    if self.diff_Ds:
                        D = self.mlp1[_i](_Ms)
                    else:
                        D = self.mlp1(_Ms)

                Ds += [D]
                if self.debug:
                    print('D', D.size())

                if self.pause:
                    if self.isolate_P:
                        if self.H_h:
                            if self.q_P:
                                _q = Variable(q.data)
                                _H = Variable(H_sum.data)
                                if self.cpx_Hh:
                                    _tmp = torch.cat([_q, _H, _q * _H, _H - _q], 1)
                                else:
                                    _tmp = torch.cat([_q, _H], 1)
                            else:
                                _Ms = Variable(_Ms.data)
                                _H = Variable(H_sum.data)
                                if self.cpx_Hh:
                                    _tmp = torch.cat([_Ms, _H, _Ms * _H, _H - _Ms], 1)
                                else:
                                    _tmp = torch.cat([_Ms, _H], 1)
                            P = F.softmax(self.P(_tmp))
                        elif self.cpx_Hh:
                            if self.q_P:
                                _q = Variable(q.data)
                                _H = Variable(H_sum.data)
                                _tmp = torch.cat([_q, _H, _q * _H, _H - _q], 1)
                            else:
                                _Ms = Variable(_Ms.data)
                                _H = Variable(H_sum.data)
                                _tmp = torch.cat([_Ms, _H, _Ms * _H, _H - _Ms], 1)
                            P = F.softmax(self.P(_tmp))
                        else:
                            if self.q_P:
                                P = F.softmax(self.P(Variable(q.data)))
                            else:
                                P = F.softmax(self.P(Variable(_Ms.data)))
                    else:
                        if self.H_h:
                            _H = H_sum
                            if self.q_P:
                                _q = q
                                if self.cpx_Hh:
                                    _tmp = torch.cat([_q, _H, _q * _H, _H - _q], 1)
                                else:
                                    _tmp = torch.cat([_q, _H], 1)
                            else:
                                if self.cpx_Hh:
                                    _tmp = torch.cat([_Ms, _H, _Ms * _H, _H - _Ms], 1)
                                else:
                                    _tmp = torch.cat([_Ms, _H], 1)
                            P = F.softmax(self.P(_tmp))
                        elif self.cpx_Hh:
                            _H = H_sum
                            if self.q_P:
                                _q = q
                                _tmp = torch.cat([_q, _H, _q * _H, _H - _q], 1)
                            else:
                                _tmp = torch.cat([_Ms, _H, _Ms * _H, _H - _Ms], 1)
                            P = F.softmax(self.P(_tmp))
                        else:
                            if self.q_P:
                                P = F.softmax(self.P(q))
                            else:
                                P = F.softmax(self.P(_Ms))
                        '''
                        if self.H_h:
                            _tmp = torch.cat([_Ms, H_sum], 1)
                            #print('H_sum: ', H_sum)
                            #print('Ms: ', _Ms)
                            #print('tmp: ', _tmp)
                            P = F.softmax(self.P(_tmp))
                        else:
                            if self.q_P:
                                P = F.softmax(self.P(q))
                            else:
                                P = F.softmax(self.P(_Ms))
                        '''
                    Ps += [P]

                '''
                _Ms = torch.sum(torch.cat(Ms, 1), 1).squeeze()
                D = self.mlp1(_Ms)
                if self.debug:
                    print('D', D.size())

                #out = F.softmax(D)
                #out1s.append(self.dropout(D))
                out1s += [self.dropout(out)]
                '''
            if self.rl_torch:
                episodes.append(actions)
            #out = F.softmax(D)
            #out1s.append(self.dropout(D))
            if self.aoa:
                Q = torch.cat(qs, 1).view(batch * self.nlayers, self.u) #batch * nlayers, u
                #print('Q:', Q.size())
                aQ = self.secattn(Q)
                aQ = F.softmax(aQ)
                aQ = aQ.view(batch, 1, self.nlayers)
                Q = Q.view(batch, self.nlayers, self.u)
                MQ = torch.bmm(aQ, Q).squeeze() # batch, self.u
                MDs.append(self.mlp1(MQ))
            if self.state is 'eval':
                break

        self.u -= self.pos_len
        self.debug = False

        if self.state is 'eval' and self.pause:
            return Ds, Ps

        if self.aoa:
            #print('MDs len:', len(MDs))
            return MDs
        if self.pause and self.rl:
            if self.a_pen > 0:
                return Ds, Ps, Ws, As
            else:
                if self.rl_torch:
                    return Ds, Ps, episodes
                else:
                    return Ds, Ps, Ws
        if self.pause:
            if self.a_pen > 0:
                return Ds, Ps, As
            else:
                return Ds, Ps
        elif self.rl:
            if self.rl_torch:
                return Ds, episodes
            else:
                return Ds, Ws
        else:
            if self.a_pen > 0:
                return Ds, As
            else:
                return Ds


class MemAttnOld(Base):
    def __init__(self, param):
        super(MemAttnOld, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.nlayers = param['nlayers']
        self.epsilon = param['epsilon']
        self.p_lambda = param['p_lambda']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.rl = 'rl' in param['params']
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.gru = nn.GRUCell(self.u, self.u, )
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch, slen, self.u)
        if self.debug:
            print('H', H.size())

        """
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        '''
        if self.state is 'train':
            M = torch.bmm(A, H)
        elif self.state is 'eval':
            A = A.view(batch, slen)
            w, a = A.topk(1, dim=1)
            Ms = []
            a = a.view(-1)
            for i, _a in enumerate(a.data):
                Ms.append(H[i][_a].view(1, self.u))
            M = torch.cat(Ms)
        '''
        """

        q = torch.cat([self.attn.weight] * batch)
        if self.debug:
            print('q', q.size())

        Ms = []
        # TODO: fix H
        H = Variable(H.data)
        for i in range(self.nlayers):
            q = q.view(batch, self.u, 1)
            A = torch.bmm(H, q).view(-1, slen)
            A = F.softmax(A)

            if self.rl:
                select = np.random.choice(["max", "random"], p=[1.0 - self.epsilon, self.epsilon])
                if select == 'max':
                    w, a = A.topk(1, dim=1)
                    ret2 = w
                elif select == 'random':
                    a, w = [], []
                    for i, l in enumerate(lens):
                        #sample = np.random.choice(np.arange(0, slen), p=A[i].data.cpu().numpy())
                        sample = random.randint(0, l-1)
                        #a.append(random.randint(0, l - 1))
                        """
                        if sample > l:
                            print(sample, l, A[i].data[l - 1], A[i].data[sample], A[i].data.max(), A[i].data.min())
                        """
                        a.append(sample)
                        w.append(A[i][a[-1]])
                    w = torch.cat(w)
                    a = Variable(torch.LongTensor(a).cuda(self.device))
                    ret2 = w
                _Ms = []
                a = a.view(-1)
                for i, _a in enumerate(a.data):
                    _Ms.append(H[i][_a].view(1, self.u))
                M = torch.cat(_Ms)
            else:
                A = A.view(batch, 1, slen)
                M = torch.bmm(A, H).squeeze()

            Ms.append(M.view(batch, 1, self.u))

            q = q.squeeze()
            q = self.gru(M, q)

        Ms = torch.sum(torch.cat(Ms, 1), 1).squeeze()
        mlp1 = self.mlp1(Ms)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out

class CNN(Base):
    def __init__(self, param):
        super(CNN, self).__init__(param)
        #self.args = param
        self.use_cuda = param['use_cuda']
        
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.c = param['c']
        Ci = 1
        Co = 50 # param['kernel_num']
        Ks = [1] # param['kernel_sizes']
        self.device = param['device']
        self.bilstm = param['bilstm']
        self.debug = True
        #print('V: %d, D: %d, C: %d, Co: %d, Ks, %s'%(V, D, C, Co, Ks))

        self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
        self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
        if self.bilstm:
            self.e = self.e * 2
        self.convs1 = [nn.Conv2d(Ci, Co, (K, self.e)).cuda(self.device) for K in Ks]
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(p=param['dropout'])
        self.fc1 = nn.Linear(len(Ks) * Co, self.c).cuda(self.device)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, log=False):
        embeds = self.embed(x) # (N,W,D)
        #x = embeds
        
        batch = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)

        lens = []
        for line in x.data:
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
        x, self.hidden = pad_packed_sequence(H, batch_first=True)

        #if self.args['static']:
        #    x = Variable(x)
        if self.debug:
            print('x0', x.size())

        x = x.unsqueeze(1) # (N,Ci,W,D)
        #print(x.is_cuda)
        if self.debug:
            print('x1', x.size())

        #print('convs1:', self.convs1[0].weight.sum().data[0])

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N,Co,W), ...]*len(Ks)
        if self.debug:
            print('x2', x[0].size())
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        #print('fc1:', self.fc1.weight.sum().data[0])
        self.debug = False
        return logit


class SingleAttnBoW(Base):
    def __init__(self, param):
        super(SingleAttnBoW, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
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
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.bow_fc = nn.Linear(self.v, self.u).cuda(self.device)
            self.mlp = nn.Linear(self.u * 2, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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

        Attn = self.attn(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        '''old
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
        #New start
        A = Attn.view(batch, slen, self.r)
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

        one_hot = Variable(torch.zeros(batch, self.v).scatter_(1, words.data.cpu(), 1).cuda(self.device))

        fc = self.bow_fc(one_hot)
        if self.debug:
            print('fc', fc.size())

        M = torch.cat([M, fc], 1)

        mlp1 = self.mlp(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class BoW(Base):
    def __init__(self, param):
        super(BoW, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        #self.da = param['da']
        #self.r = 1
        #self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.fc = nn.Linear(self.v, self.u).cuda(self.device)
            self.mlp = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            '''
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            '''
            self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, log=False):
        batch = words.size(0)
        one_hot = Variable(torch.zeros(batch, self.v).scatter_(1, words.data.cpu(), 1).cuda(self.device))

        fc = self.fc(one_hot)
        if self.debug:
            print('fc', fc.size())

        mlp = self.mlp(fc)
        if self.debug:
            print('mlp', mlp.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp)
        self.debug = False
        return out


class DocMultAttn(Base):
    def __init__(self, param):
        super(DocMultAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.need_stop = True
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
            self.w1 = nn.Linear(self.u, self.u, bias=False).cuda(self.device)
            self.w2 = nn.Linear(self.u, self.u, bias=True).cuda(self.device)
            self.s = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.a_sent = nn.Linear(self.u * 2, 1, bias=False).cuda(self.device)
            self.a_doc = nn.Linear(self.u * 2, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u * 2, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, stop, log=False):
        batch = words.size(0)
        lens = []
        sents = []
        u = self.u
        for line in words.data:
            length = 0
            sent = [0]
            for l in range(len(line) - 1, -1, -1):
                if line[l] != 1:
                    length = l + 1
                    break
            for l in range(length):
                if line[l] in stop:
                    sent.append(l + 1)
            sent.append(length)
            if length == 0:
                print('Error', line)
            lens.append(length)
            sents.append(sent)

        texts = []
        max_len = 0
        for line in range(len(sents)):
            l = len(sents[line])
            for i in range(0, l - 1):
                text = []
                crt_len = sents[line][i + 1] - sents[line][i]
                if crt_len > max_len:
                    max_len = crt_len
                for k in range(sents[line][i], sents[line][i + 1]):
                    w = words.data[line][k]
                    text.append(w)
                texts.append(text)

        lens = []
        for line in range(len(texts)):
            lens.append(len(texts[line]))
            texts[line] = texts[line] + [1] * (max_len - len(texts[line]))

        dec_idx = np.argsort(lens)[::-1]
        dec_lens = np.array(lens)[dec_idx]
        np_texts = np.array(texts)
        feature = Variable(torch.LongTensor(np_texts[dec_idx])).cuda(self.device)

        embeds = self.embed(feature)
        snum = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)

        packed = pack_padded_sequence(embeds, dec_lens, batch_first=True)
        H, _ = self.lstm(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', batch)
            print('snum', snum)
            print('slen', slen)
        H = H.contiguous().view(snum * slen, self.u)
        #H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        #----

        #W1 = F.tanh(self.w1(H).view(snum * slen, 1, self.u))
        W1 = self.w1(H).view(snum * slen, 1, self.u)
        W2 = self.w2(H).view(snum, slen, self.u)

        mask1 = torch.ones(snum, slen)
        for i in range(mask1.size(0)):
            for j in range(dec_lens[i], mask1.size(1)):
                mask1[i][j] = 0
        mask1 = Variable(mask1).cuda(self.device)

        WR = []
        for i in range(slen):
            mask2 = [mask1[:, i].contiguous().view(-1, 1)] * slen
            mask2 = torch.cat(mask2, 1)
            _w2i = [W2[:, i, :].contiguous().view(snum, 1, u)] * slen
            #W2i = F.tanh(torch.cat(_w2i, 1).view(snum * slen, self.u, 1))
            W2i = torch.cat(_w2i, 1).view(snum * slen, self.u, 1)
            theta = torch.bmm(W1, W2i).view(snum, slen)
            #theta = F.tanh(theta.view(snum * slen, u))
            #theta = F.tanh(self.s(theta).view(snum, slen))
            theta = theta * mask1 * mask2
            theta = F.softmax(theta)
            theta = theta * mask2
            theta = theta.view(snum, 1, slen)
            H = H.view(snum, slen, u)
            M = torch.bmm(theta, H)
            M = M.squeeze()
            tempt = torch.cat([H[:, i, :], M], 1)
            WR.append(tempt.view(snum, 1, u * 2))

        WR = torch.cat(WR, 1).view(snum * slen, u * 2)
        del(H)

        #-----

        A_sent = self.a_sent(WR)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())
        if log:
            print('Attn max: %f, min: %f' % (A_sent.max().data[0], A_sent.min().data[0]))

        '''old
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
        #New start
        #A = A_sent.view(batch, slen, 1)
        #A = torch.transpose(A, 1, 2)
        #A = A.contiguous().view(-1, slen)
        A = A_sent.view(snum, slen)
        A = F.softmax(A)
        A = A.view(snum, 1, slen)
        #New end

        WR = WR.view(snum, slen, self.u * 2)

        if self.debug:
            print('A', A.size())
            print('WR', WR.size())

        M = torch.bmm(A, WR)
        del(WR)
        if self.debug:
            print('M', M.size())
            #print(M)

        dic_idx = {}
        for i in range(len(dec_idx)):
            dic_idx[dec_idx[i]] = i

        max_sent_len = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            if n > max_sent_len:
                max_sent_len = n

        H_docs = []
        idx = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            tempt = []
            for i in range(n):
                tempt.append(M[dic_idx[idx]].view(1, -1))
                idx += 1
            for i in range(max_sent_len - n):
                tempt.append(Variable(torch.zeros(1, self.u * 2).float().cuda(self.device)))
            #print(tempt)
            H_docs.append(torch.cat(tempt).view(1, max_sent_len, self.u * 2))
        assert(idx == len(dec_lens))
        H_docs = torch.cat(H_docs).view(batch * max_sent_len, self.u * 2)

        A_doc = self.a_doc(H_docs)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())

        A = A_doc.view(batch, max_sent_len)
        A = F.softmax(A)
        A = A.view(batch, 1, max_sent_len)
        #New end

        H_docs = H_docs.view(batch, max_sent_len, self.u * 2)

        if self.debug:
            print('A', A.size())
            print('H_docs', H_docs.size())

        M = torch.bmm(A, H_docs)
 
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class DocThetaAttn(Base):
    def __init__(self, param):
        super(DocThetaAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.need_stop = True
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
            self.w1 = nn.Linear(self.u, self.u, bias=False).cuda(self.device)
            self.w2 = nn.Linear(self.u, self.u, bias=True).cuda(self.device)
            self.s = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.a_sent = nn.Linear(self.u * 2, 1, bias=False).cuda(self.device)
            self.a_doc = nn.Linear(self.u * 2, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u * 2, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, stop, log=False):
        batch = words.size(0)
        lens = []
        sents = []
        u = self.u
        for line in words.data:
            length = 0
            sent = [0]
            for l in range(len(line) - 1, -1, -1):
                if line[l] != 1:
                    length = l + 1
                    break
            for l in range(length):
                if line[l] in stop:
                    sent.append(l + 1)
            sent.append(length)
            if length == 0:
                print('Error', line)
            lens.append(length)
            sents.append(sent)

        texts = []
        max_len = 0
        for line in range(len(sents)):
            l = len(sents[line])
            for i in range(0, l - 1):
                text = []
                crt_len = sents[line][i + 1] - sents[line][i]
                if crt_len > max_len:
                    max_len = crt_len
                for k in range(sents[line][i], sents[line][i + 1]):
                    w = words.data[line][k]
                    text.append(w)
                texts.append(text)

        lens = []
        for line in range(len(texts)):
            lens.append(len(texts[line]))
            texts[line] = texts[line] + [1] * (max_len - len(texts[line]))

        dec_idx = np.argsort(lens)[::-1]
        dec_lens = np.array(lens)[dec_idx]
        np_texts = np.array(texts)
        feature = Variable(torch.LongTensor(np_texts[dec_idx])).cuda(self.device)

        embeds = self.embed(feature)
        snum = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)

        packed = pack_padded_sequence(embeds, dec_lens, batch_first=True)
        H, _ = self.lstm(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', batch)
            print('snum', snum)
            print('slen', slen)
        H = H.contiguous().view(snum * slen, self.u)
        #H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        #----

        W1 = self.w1(H).view(snum, slen, self.u)
        W2 = self.w2(H).view(snum, slen, self.u)

        mask1 = torch.ones(snum, slen)
        for i in range(mask1.size(0)):
            for j in range(dec_lens[i], mask1.size(1)):
                mask1[i][j] = 0
        mask1 = Variable(mask1).cuda(self.device)

        WR = []
        for i in range(slen):
            mask2 = [mask1[:, i].contiguous().view(-1, 1)] * slen
            mask2 = torch.cat(mask2, 1)
            _w2i = [W2[:, i, :].contiguous().view(snum, 1, u)] * slen
            W2i = torch.cat(_w2i, 1)
            theta = W1 + W2i
            theta = F.tanh(theta.view(snum * slen, u))
            theta = F.tanh(self.s(theta).view(snum, slen))
            theta = theta * mask1 * mask2
            theta = F.softmax(theta)
            theta = theta * mask2
            theta = theta.view(snum, 1, slen)
            H = H.view(snum, slen, u)
            M = torch.bmm(theta, H)
            M = M.squeeze()
            tempt = torch.cat([H[:, i, :], M], 1)
            WR.append(tempt.view(snum, 1, u * 2))

        WR = torch.cat(WR, 1).view(snum * slen, u * 2)
        del(H)

        #-----

        A_sent = self.a_sent(WR)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())
        if log:
            print('Attn max: %f, min: %f' % (A_sent.max().data[0], A_sent.min().data[0]))

        '''old
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
        #New start
        #A = A_sent.view(batch, slen, 1)
        #A = torch.transpose(A, 1, 2)
        #A = A.contiguous().view(-1, slen)
        A = A_sent.view(snum, slen)
        A = F.softmax(A)
        A = A.view(snum, 1, slen)
        #New end

        WR = WR.view(snum, slen, self.u * 2)

        if self.debug:
            print('A', A.size())
            print('WR', WR.size())

        M = torch.bmm(A, WR)
        del(WR)
        if self.debug:
            print('M', M.size())
            #print(M)

        dic_idx = {}
        for i in range(len(dec_idx)):
            dic_idx[dec_idx[i]] = i

        max_sent_len = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            if n > max_sent_len:
                max_sent_len = n

        H_docs = []
        idx = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            tempt = []
            for i in range(n):
                tempt.append(M[dic_idx[idx]].view(1, -1))
                idx += 1
            for i in range(max_sent_len - n):
                tempt.append(Variable(torch.zeros(1, self.u * 2).float().cuda(self.device)))
            #print(tempt)
            H_docs.append(torch.cat(tempt).view(1, max_sent_len, self.u * 2))
        assert(idx == len(dec_lens))
        H_docs = torch.cat(H_docs).view(batch * max_sent_len, self.u * 2)

        A_doc = self.a_doc(H_docs)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())

        A = A_doc.view(batch, max_sent_len)
        A = F.softmax(A)
        A = A.view(batch, 1, max_sent_len)
        #New end

        H_docs = H_docs.view(batch, max_sent_len, self.u * 2)

        if self.debug:
            print('A', A.size())
            print('H_docs', H_docs.size())

        M = torch.bmm(A, H_docs)
 
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class SentThetaAttn(Base):
    def __init__(self, param):
        super(SentThetaAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
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
            self.w1 = nn.Linear(self.u, self.u, bias=False).cuda(self.device)
            self.w2 = nn.Linear(self.u, self.u, bias=True).cuda(self.device)
            self.s = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn = nn.Linear(self.u * 2, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u * 2, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, log=False):
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        u = self.u
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

        W1 = self.w1(H).view(batch, slen, self.u)
        W2 = self.w2(H).view(batch, slen, self.u)

        mask1 = torch.ones(batch, slen)
        for i in range(mask1.size(0)):
            for j in range(lens[i], mask1.size(1)):
                mask1[i][j] = 0
        mask1 = Variable(mask1).cuda(self.device)

        WR = []
        for i in range(slen):
            mask2 = [mask1[:, i].contiguous().view(-1, 1)] * slen
            mask2 = torch.cat(mask2, 1)
            _w2i = [W2[:, i, :].contiguous().view(batch, 1, u)] * slen
            W2i = torch.cat(_w2i, 1)
            theta = W1 + W2i
            theta = theta.view(batch * slen, u)
            theta = self.s(theta).view(batch, slen)
            theta = theta * mask1 * mask2
            theta = F.softmax(theta)
            theta = theta * mask2
            theta = theta.view(batch, 1, slen)
            H = H.view(batch, slen, u)
            M = torch.bmm(theta, H)
            M = M.squeeze()
            tempt = torch.cat([H[:,i,:], M], 1)
            WR.append(tempt.view(batch, 1, u * 2))

        WR = torch.cat(WR, 1).view(batch * slen, u * 2)

        Attn = self.attn(WR)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        A = Attn.view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
        #New end

        WR = WR.view(batch, slen, self.u * 2)

        if self.debug:
            print('A', A.size())
            print('H', WR.size())

        M = torch.bmm(A, WR)
        if self.debug:
            print('M', M.size())
            #print(M)

        #M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class DocAttn(Base):
    def __init__(self, param):
        super(DocAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.need_stop = True
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
            self.a_sent = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.a_doc = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, stop, log=False):
        batch = words.size(0)
        lens = []
        sents = []
        for line in words.data:
            length = 0
            sent = [0]
            for l in range(len(line) - 1, -1, -1):
                if line[l] != 1:
                    length = l + 1
                    break
            for l in range(length):
                if line[l] in stop:
                    sent.append(l + 1)
            sent.append(length)
            if length == 0:
                print('Error', line)
            lens.append(length)
            sents.append(sent)

        texts = []
        max_len = 0
        for line in range(len(sents)):
            l = len(sents[line])
            for i in range(0, l - 1):
                text = []
                crt_len = sents[line][i + 1] - sents[line][i]
                if crt_len > max_len:
                    max_len = crt_len
                for k in range(sents[line][i], sents[line][i + 1]):
                    w = words.data[line][k]
                    text.append(w)
                texts.append(text)

        lens = []
        for line in range(len(texts)):
            lens.append(len(texts[line]))
            texts[line] = texts[line] + [1] * (max_len - len(texts[line]))

        dec_idx = np.argsort(lens)[::-1]
        dec_lens = np.array(lens)[dec_idx]
        np_texts = np.array(texts)
        feature = Variable(torch.LongTensor(np_texts[dec_idx])).cuda(self.device)

        embeds = self.embed(feature)
        snum = embeds.size(0)
        slen = embeds.size(1)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)

        packed = pack_padded_sequence(embeds, dec_lens, batch_first=True)
        H, _ = self.lstm(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', batch)
            print('snum', snum)
            print('slen', slen)
        H = H.contiguous().view(snum * slen, self.u)
        #H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        A_sent = self.a_sent(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())
        if log:
            print('Attn max: %f, min: %f' % (A_sent.max().data[0], A_sent.min().data[0]))

        '''old
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
        #New start
        #A = A_sent.view(batch, slen, 1)
        #A = torch.transpose(A, 1, 2)
        #A = A.contiguous().view(-1, slen)
        A = A_sent.view(snum, slen)
        A = F.softmax(A)
        A = A.view(snum, 1, slen)
        #New end

        H = H.view(snum, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        if self.debug:
            print('M', M.size())
            #print(M)

        dic_idx = {}
        for i in range(len(dec_idx)):
            dic_idx[dec_idx[i]] = i

        max_sent_len = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            if n > max_sent_len:
                max_sent_len = n

        H_docs = []
        idx = 0
        for i in range(len(sents)):
            n = len(sents[i]) - 1
            tempt = []
            for i in range(n):
                tempt.append(M[dic_idx[idx]].view(1, -1))
                idx += 1
            for i in range(max_sent_len - n):
                tempt.append(Variable(torch.zeros(1, self.u).float().cuda(self.device)))
            #print(tempt)
            H_docs.append(torch.cat(tempt).view(1, max_sent_len, self.u))
        assert(idx == len(dec_lens))
        H_docs = torch.cat(H_docs).view(batch * max_sent_len, self.u)

        A_doc = self.a_doc(H_docs)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', A_sent.size())

        A = A_doc.view(batch, max_sent_len)
        A = F.softmax(A)
        A = A.view(batch, 1, max_sent_len)
        #New end

        H_docs = H_docs.view(batch, max_sent_len, self.u)

        if self.debug:
            print('A', A.size())
            print('H_docs', H_docs.size())

        M = torch.bmm(A, H_docs)
 
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class DecompAttn(Base):
    def __init__(self, param):
        super(DecompAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c, bias=False).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn = ListModule(*attn)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        #print('batch', self.batch)
        #print('slen', slen)
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

        tA = self.attn[1](H).view(batch, slen, 1)
        tA = torch.transpose(tA, 1, 2)
        tA = tA.contiguous().view(-1, slen)
        tA = F.softmax(tA)
        tA = tA.view(batch, slen)

        sA = self.attn[0](H).view(batch, slen, 1)
        sA = torch.transpose(sA, 1, 2)
        sA = sA.contiguous().view(-1, slen)
        sA = F.softmax(sA)
        sA = sA.view(batch, slen)

        '''
        A = Variable(torch.zeros(batch, slen).float()).cuda(self.device)
        for s in range(slen):
            for t in range(slen):
                A[:, s] = A[:, s] + (sA[:, s] * tA[:, t])
        '''

        width = 20

        _sA = []
        for i in range(batch):
            _sA.append(torch.cat([sA[i].view(1, -1)] * width * 2).view(1, width * 2, slen))
        sA = torch.cat(_sA)

        rshift = torch.diag(torch.ones(slen))
        rshift[0] = 0
        rshift = Variable(torch.cat([rshift[1:], rshift[0].view(1, -1)]).float()).cuda(self.device)
        lshift = torch.diag(torch.ones(slen))
        lshift[-1] = 0
        lshift = Variable(torch.cat([lshift[-1].view(1, -1), lshift[:-1]]).float()).cuda(self.device)

        if self.debug:
            print('sA', sA.size())
            print('rshift', rshift.size())
            print('lshift', lshift.size())
            #print('tA', tA)
            #print('shift', shift)

        _tA = []
        r_tA = tA
        l_tA = tA
        for i in range(width):
            r_tA = torch.mm(r_tA, rshift)
            l_tA = torch.mm(l_tA, lshift)
            _tA.append(r_tA.view(batch, 1, slen))
            _tA.append(l_tA.view(batch, 1, slen))
        tA = torch.cat(_tA, 1)

        if self.debug:
            print('sA', sA.size())
            print('tA', tA.size())

        A = sA * tA
        A = torch.sum(A, 1).squeeze()

        #A = Variable(A)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
        
        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class SecOrdAttn(Base):
    def __init__(self, param):
        super(SecOrdAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c, bias=False).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn = ListModule(*attn)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        #print('batch', self.batch)
        #print('slen', slen)
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

        tA = self.attn[1](H).view(batch, slen, 1)
        tA = torch.transpose(tA, 1, 2)
        tA = tA.contiguous().view(-1, slen)
        tA = F.softmax(tA)
        tA = tA.view(batch, slen)

        sA = self.attn[0](H).view(batch, slen, 1)
        sA = torch.transpose(sA, 1, 2)
        sA = sA.contiguous().view(-1, slen)
        sA = F.softmax(sA)
        sA = sA.view(batch, slen)

        '''
        A = Variable(torch.zeros(batch, slen).float()).cuda(self.device)
        for s in range(slen):
            for t in range(slen):
                A[:, s] = A[:, s] + (sA[:, s] * tA[:, t])
        '''

        width = 20

        _sA = []
        for i in range(batch):
            _sA.append(torch.cat([sA[i].view(1, -1)] * width * 2).view(1, width * 2, slen))
        sA = torch.cat(_sA)

        rshift = torch.diag(torch.ones(slen))
        rshift[0] = 0
        rshift = Variable(torch.cat([rshift[1:], rshift[0].view(1, -1)]).float()).cuda(self.device)
        lshift = torch.diag(torch.ones(slen))
        lshift[-1] = 0
        lshift = Variable(torch.cat([lshift[-1].view(1, -1), lshift[:-1]]).float()).cuda(self.device)

        if self.debug:
            print('sA', sA.size())
            print('rshift', rshift.size())
            print('lshift', lshift.size())
            #print('tA', tA)
            #print('shift', shift)

        _tA = []
        r_tA = tA
        l_tA = tA
        for i in range(width):
            r_tA = torch.mm(r_tA, rshift)
            l_tA = torch.mm(l_tA, lshift)
            _tA.append(r_tA.view(batch, 1, slen))
            _tA.append(l_tA.view(batch, 1, slen))
        tA = torch.cat(_tA, 1)

        if self.debug:
            print('sA', sA.size())
            print('tA', tA.size())

        A = sA * tA
        A = torch.sum(A, 1).squeeze()

        #A = Variable(A)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
        
        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class SPLSTM(Base):
    def __init__(self, param):
        super(SPLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
        self.batch = param['batch_size']
        self.device = param['device']
        self.data_set = param['data_set']

        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.slstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            lstm, mlp = [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                lstm.append(nn.LSTM(self.e, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device))
                mlp.append(nn.Linear(self.u * 2, c).cuda(self.device))
            self.plstm = ListModule(*lstm)
            self.mlp = ListModule(*mlp)
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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
        sH, _ = self.slstm(packed)
        sH, _ = pad_packed_sequence(sH, batch_first=True)

        if is_tagger:
            sH = sH.contiguous().view(batch * slen, self.u)
            tag_space = self.tagger(sH)
            return tag_space

        packed = pack_padded_sequence(embeds, lens, batch_first=True)
        pH, _ = self.plstm[idx](packed)
        pH, _ = pad_packed_sequence(pH, batch_first=True)

        if self.debug:
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
            print('sH', sH.size())
            print('pH', pH.size())
        #sH = sH.contiguous().view(batch, slen, self.u)
        #pH = pH.contiguous().view(batch, slen, self.u)

        #sH = torch.mean(sH, 1).squeeze()
        #pH = torch.mean(pH, 1).squeeze()
        #sH = sH[:, 0, :]
        #pH = pH[:, 0, :]
        '''
        _sH = []
        for i in range(batch):
            _sH.append(sH[i, lens[i] - 1])
        sH = torch.cat(_sH).view(batch, self.u)

        _pH = []
        for i in range(batch):
            _pH.append(pH[i, lens[i] - 1])
        pH = torch.cat(_pH).view(batch, self.u)
        '''
        pH = torch.mean(pH.contiguous().view(batch, slen, self.u), 1).squeeze()
        sH = torch.mean(sH.contiguous().view(batch, slen, self.u), 1).squeeze()

        if self.debug:
            print('sH', sH.size())
            print('pH', pH.size())
        M = torch.cat([sH, pH], 1)
        if self.debug:
            print('M', M.size())
        mlp = self.mlp[idx](M)
        if self.debug:
            print('mlp', mlp.size())

        out = self.dropout(mlp)
        self.debug = False

        return out



class StackLSTM(Base):
    def __init__(self, param):
        super(StackLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
        self.batch = param['batch_size']
        self.device = param['device']
        self.data_set = param['data_set']

        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.slstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            lstm, mlp = [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                lstm.append(nn.LSTM(self.u, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device))
                mlp.append(nn.Linear(self.u, c).cuda(self.device))
            self.plstm = ListModule(*lstm)
            self.mlp = ListModule(*mlp)
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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
        sH, _ = self.slstm(packed)
        sH, _ = pad_packed_sequence(sH, batch_first=True)

        if is_tagger:
            sH = sH.contiguous().view(batch * slen, self.u)
            tag_space = self.tagger(sH)
            return tag_space

        packed = pack_padded_sequence(sH, lens, batch_first=True)
        pH, _ = self.plstm[idx](packed)
        pH, _ = pad_packed_sequence(pH, batch_first=True)

        if self.debug:
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
            print('sH', sH.size())
            print('pH', pH.size())
        #sH = sH.contiguous().view(batch, slen, self.u)
        #pH = pH.contiguous().view(batch, slen, self.u)

        #sH = torch.mean(sH, 1).squeeze()
        #pH = torch.mean(pH, 1).squeeze()
        #sH = sH[:, 0, :]
        #pH = pH[:, 0, :]
        '''
        _sH = []
        for i in range(batch):
            _sH.append(sH[i, lens[i] - 1])
        sH = torch.cat(_sH).view(batch, self.u)
        _pH = []
        for i in range(batch):
            _pH.append(pH[i, lens[i] - 1])
        M = torch.cat(_pH).view(batch, self.u)
        '''

        M = torch.mean(pH.contiguous().view(batch, slen, self.u), 1).squeeze()

        if self.debug:
            print('sH', sH.size())
            print('pH', pH.size())
            print('M', M.size())
        #M = torch.cat([sH, pH], 1)
        #if self.debug:
        #    print('M', M.size())
        #mlp = self.mlp[idx](pH)
        mlp = self.mlp[idx](M)
        if self.debug:
            print('mlp', mlp.size())

        out = self.dropout(mlp)
        self.debug = False

        return out


class MultiDynAttn(Base):
    def __init__(self, param):
        super(MultiDynAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.c = param['cs']
        #self.r = len(self.c)
        self.n = len(self.c)
        #TODO, plus tagger in single attn
        have_tagger = False
        '''
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        '''
        self.bilstm = param['bilstm']
        self.batch = param['batch_size']
        self.device = param['device']
        self.data_set = param['data_set']

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
            attn, mlp1, mlp2 = [], [], []
            for i, d in enumerate(self.data_set):
                if self.c[i] > 20:
                    self.tagger = nn.Linear(self.u, self.c[i], bias=False).cuda(self.device)
                    continue
                elif 'topic' in d:
                    self.mlp2 = nn.Linear(self.u, self.c[i]).cuda(self.device)
            self.attn2 = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn2q = nn.Linear(self.u, self.u, bias=False).cuda(self.device)
            self.b = ListModule(*[nn.Linear(self.u, 1, bias=False).cuda(self.device) for _ in range(self.n)])
            self.mlp1 = nn.Linear(self.u, self.c[0]).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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

        M = torch.bmm(A, H).squeeze()

        if self.debug:
            print('A', A.size())
            print('H', H.size())
            print('M', M.size())

        #if idx == len(self.n) - 1:
        if 'topic' in self.data_set[idx]:
            mlp2 = self.mlp2(M)
            if self.debug:
                print('mlp2', mlp2.size())
            out = self.dropout(mlp2)
            self.debug = False
            return out

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
        q = self.attn2q(M).view(batch, self.u, 1) + torch.cat([self.b[idx].weight] * batch).view(batch, self.u, 1)
        A = torch.bmm(H, q).view(batch, slen, 1)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
        H = H.view(batch, slen, self.u)
        M = torch.bmm(A, H).squeeze()
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

        return out



class AttnOverAttn(Base):
    def __init__(self, param):
        super(AttnOverAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.c = param['cs']
        #self.r = len(self.c)
        self.n = len(self.c)
        #TODO, plus tagger in single attn
        have_tagger = False
        '''
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        '''
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            if len(self.c) > 2:
                self.tagger = nn.Linear(self.u, self.c[2], bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn2q = nn.Linear(self.u, self.u, bias=False).cuda(self.device)
            self.b = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c[0]).cuda(self.device)
            self.mlp2 = nn.Linear(self.u, self.c[1]).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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

        M = torch.bmm(A, H).squeeze()

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
            return out

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
        M = torch.bmm(A, H).squeeze()
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

        return out


class AttnOverAttnSingle(Base):
    def __init__(self, param):
        super(AttnOverAttnSingle, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.c = param['c']
        #self.r = len(self.c)
        #self.n = len(self.c)
        #TODO, plus tagger in single attn
        have_tagger = False
        '''
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        '''
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            self.attn2 = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn2q = nn.Linear(self.u, self.u).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            #self.mlp2 = nn.Linear(self.u, self.cs[1]).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
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
        M = torch.bmm(A, H).squeeze()
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

        return out


class DualAttn(Base):
    def __init__(self, param):
        super(DualAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.c = param['c']
        #self.r = len(self.cs)
        self.n = 2
        #TODO, plus tagger in single attn
        have_tagger = False
        '''
        for c in self.cs:
            if c > 20:
                have_tagger = True
        '''
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            #self.lstm2 = nn.LSTM(self.u, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            self.top_n = 10
            self.attn1 = []
            self.attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
            self.attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
            self.g_fc1 = nn.Linear(self.u * 2, 256).cuda(self.device)
            #g_fc2s.append(nn.linear(256, 256).cuda(self.device))
            #g_fc3s.append(nn.linear(256, 256).cuda(self.device))
            self.attn2 = nn.Linear(256, 1, bias=False).cuda(self.device)
            #f_fc1s.append(nn.linear(256, 256).cuda(self.device))
            #f_fc2s.append(nn.linear(256, 256).cuda(self.device))
            self.f_fc3 = nn.Linear(256, self.c).cuda(self.device)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, is_tagger=False, log=False, penalty=False):
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        #print(slen)
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

        print_stat(H, 'H')
        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        if is_tagger:
            tag_space = self.tagger(H)
            return tag_space

        a = []
        for i in range(self.n):
            _a = self.attn1[i](H)
            #print('_a:', _a.size())
            _a = _a.view(batch, slen)
            _a = F.softmax(_a)
            # _a: batch, 1, slen
            a.append(_a)
        H = H.view(batch, slen, self.u)

        '''g'''

        if log:
            t1 = time.time()

        '''
        h_input = torch.Tensor(batch, top_n**2, self.u * 2)
        ws0, as0 = a[0].topk(top_n, dim=1)
        ws1, as1 = a[1].topk(top_n, dim=1)
        for n in range(batch):
            #print(a[0].size(), top_n)
            line = 0
            for wi, i in zip(ws0[n].data, as0[n].data):
                hi = H[n, i] * wi
                for wj, j in zip(ws1[n].data, as1[n].data):
                    hj = H[n, j] * wj
                    #print(torch.cat([hi, hj]).data.size())
                    h_input[n][line] = torch.cat([hi, hj]).data
                    line += 1
        h_input = Variable(h_input.view(batch * (top_n**2), self.u * 2)).cuda(self.device)
        '''

        hs = []
        top_n = self.top_n
        if slen < top_n:
            print(slen)
            top_n = slen
        ws0, as0 = a[0].topk(top_n, dim=1)
        ws1, as1 = a[1].topk(top_n, dim=1)
        '''
        pos_enc = Variable(position_encoding_init(10000, 10))
        H_sum = Variable(torch.zeros(batch, slen, self.u).float()).cuda(self.device)
        H_sum[:, 0, :] = H[:, 0, :]
        for i in range(1, slen):
            H_sum[:, i, :] = H[:, i, :] + H_sum[:, i - 1, :]
        '''

        for n in range(batch):
            h0 = H[n][as0[n].data]
            #h0 = torch.cat([ws0[n].view(top_n, 1)] * self.u, 1) * h0
            # h0: topn * u
            h0 = torch.cat([h0] * top_n, 1)
            # h0: topn * (u * topn)
            h0 = h0.view(top_n ** 2, self.u)
            #seq0 = torch.cat([as0[n].view(top_n, 1)] * top_n, 1).view(top_n**2)
            #print(type(seq0.data))
            #print('slen:', slen)
            #p0 = pos_enc[seq0.data.cpu()].cuda(self.device)
            #H_sum0 = H_sum[n][seq0.data].cuda(self.device)
            h1 = H[n][as1[n].data]
            #h1 = torch.cat([ws1[n].view(top_n, 1)] * self.u, 1) * h1
            # h1: topn * u
            h1 = torch.cat([h1] * top_n)
            # h1: (topn * topn) * u
            #seq1 = torch.cat([as1[n].view(top_n, 1)] * top_n).view(top_n**2)
            #p1 = pos_enc[seq1.data.cpu()].cuda(self.device)
            #H_sum1 = H_sum[n][seq1.data].cuda(self.device)
            #H_diff = torch.abs(H_sum1 - H_sum0)
            #diff = p0 - p1
            #dist = torch.abs(seq0 - seq1).float().cuda(self.device)
            #print(type(h0), type(h1), type(dist))
            #h1 = torch.cat([h1, dist.view(-1, 1)], 1)
            hs.append(torch.cat([h0, h1], 1))
            #hs.append(torch.cat([h0, h1, H_diff, p0, p1], 1))
        h_input = torch.cat(hs).view(batch * (top_n**2), self.u * 2)

        if log:
            t2 = time.time()
            print('loop:', t2 - t1)

        x_ = self.g_fc1(h_input)
        x_ = F.relu(x_)
        #x_ = self.g_fc2(x_)
        #x_ = F.relu(x_)
        #x_ = self.g_fc3(x_)
        #x_ = F.relu(x_)
        x_ = x_.view(batch * top_n ** 2, 256)
        #x_ = torch.transpose(x_, 1, 2)
        a_ = self.attn2(x_)
        a_ = a_.view(batch, top_n ** 2)
        a_ = F.softmax(a_)
        a_ = a_.view(batch, 1, top_n ** 2)
        x_ = x_.view(batch, top_n ** 2, 256)
        x_ = torch.bmm(a_, x_)
        #x_ = torch.sum(x_, 1)
        x_ = x_.squeeze()

        if log:
            t3 = time.time()
            print('g:', t3 - t2)

        '''f'''

        #x_ = self.f_fc1(x_)
        #x_ = F.relu(x_)
        #x_ = self.f_fc2(x_)
        #x_ = F.relu(x_)
        x_ = F.dropout(x_)
        x_ = self.f_fc3(x_)

        if log:
            t4 = time.time()
            print('f:', t4 - t3)

        if self.debug:
            print('x', x_.size())

        #out = F.softmax(mlp1)
        self.debug = False

        if penalty > 0:
            a1 = a[0].view(batch, 1, -1)
            a2 = a[1].view(batch, -1, 1)
            #a1 = a1.transpose(1, 2)
            diff = torch.norm(torch.bmm(a1, a2))

            return x_, diff

        return x_


class RelateAttn(Base):
    def __init__(self, param):
        super(RelateAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
        self.batch = param['batch_size']
        self.device = param['device']
        self.window = ('window' in param['params'])
        if self.window:
            self.window_size = param['window_size']
        self.pos = ('pos' in param['params'])
        self.tgf = ('tgf' in param['params'])
        self.sent = ('sent' in param['params'])
        self.rl = ('rl' in param['params'])
        if self.sent:
            self.need_stop = True

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
            #self.lstm2 = nn.LSTM(self.u, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            attn1, attn2, mlp2 = [], [], []
            g_fc1s, g_fc2s, g_fc3s = [], [], []
            f_fc1s, f_fc2s, f_fc3s = [], [], []
            #TODO: self.top_n = 10
            self.top_n = 10
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #attn2.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                if self.pos:
                    g_fc1s.append(nn.Linear(self.u * 2 + 20, 256).cuda(self.device))
                else:
                    g_fc1s.append(nn.Linear(self.u * 2, 256).cuda(self.device))
                if self.tgf:
                    g_fc2s.append(nn.Linear(256, 256).cuda(self.device))
                    g_fc3s.append(nn.Linear(256, 256).cuda(self.device))
                attn2.append(nn.Linear(256, 1, bias=False).cuda(self.device))
                if self.tgf:
                    f_fc1s.append(nn.Linear(256, 256).cuda(self.device))
                    f_fc2s.append(nn.Linear(256, 256).cuda(self.device))
                f_fc3s.append(nn.Linear(256, c).cuda(self.device))
                if c > 10:
                    self.mlp = nn.Linear(self.u, c).cuda(self.device)
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                #mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn1 = ListModule(*attn1)
            self.attn2 = ListModule(*attn2)
            #self.attn2 = ListModule(*attn2)
            #self.mlp1 = ListModule(*mlp1)
            self.g_fc1 = ListModule(*g_fc1s)
            if self.tgf:
                self.g_fc2 = ListModule(*g_fc2s)
                self.g_fc3 = ListModule(*g_fc3s)
                self.f_fc1 = ListModule(*f_fc1s)
                self.f_fc2 = ListModule(*f_fc2s)
            self.f_fc3 = ListModule(*f_fc3s)
            #self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, stop=[], log=False, penalty=False):
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        #print(slen)
        if self.debug:
            print('embeds', embeds.size())
        #embeds = torch.transpose(embeds, 0, 1)

        lens = []
        sents = []
        max_sents = 0
        for line in words.data:
            length = 0
            sent = [0]
            for l in range(len(line) - 1, -1, -1):
                if line[l] != 1:
                    length = l + 1
                    break
            for l in range(length):
                if line[l] in stop:
                    sent.append(l + 1)
            sent.append(length)
            if len(sent) - 1 > max_sents:
                max_sents = len(sent) - 1
            if length == 0:
                print('Error', line)
            lens.append(length)
            sents.append(sent)

        packed = pack_padded_sequence(embeds, lens, batch_first=True)
        H, _ = self.lstm(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)

        print_stat(H, 'H')
        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        if is_tagger:
            tag_space = self.tagger(H)
            return tag_space

        if idx == 1:
            A = self.attn1[idx](H).view(batch, slen)
            A = F.softmax(A)
            A = A.view(batch, 1, slen)
            H = H.view(batch, slen, self.u)
            M = torch.bmm(A, H).squeeze()
            out = self.mlp(M)
            out = self.dropout(out)
            return out

        a = []
        for i in range(self.n):
            _a = self.attn1[i](H)
            #print('_a:', _a.size())
            _a = _a.view(batch, slen)
            _a = F.softmax(_a)
            # _a: batch, 1, slen
            a.append(_a)
        H = H.view(batch, slen, self.u)

        '''g'''

        if log:
            t1 = time.time()

        '''
        h_input = torch.Tensor(batch, top_n**2, self.u * 2)
        ws0, as0 = a[0].topk(top_n, dim=1)
        ws1, as1 = a[1].topk(top_n, dim=1)
        for n in range(batch):
            #print(a[0].size(), top_n)
            line = 0
            for wi, i in zip(ws0[n].data, as0[n].data):
                hi = H[n, i] * wi
                for wj, j in zip(ws1[n].data, as1[n].data):
                    hj = H[n, j] * wj
                    #print(torch.cat([hi, hj]).data.size())
                    h_input[n][line] = torch.cat([hi, hj]).data
                    line += 1
        h_input = Variable(h_input.view(batch * (top_n**2), self.u * 2)).cuda(self.device)
        '''

        hs = []
        pos_enc = Variable(position_encoding_init(10000, 10))
        '''
        H_sum = Variable(torch.zeros(batch, slen, self.u).float()).cuda(self.device)
        H_sum[:, 0, :] = H[:, 0, :]
        for i in range(1, slen):
            H_sum[:, i, :] = H[:, i, :] + H_sum[:, i - 1, :]
        '''

        if self.window:
            window = self.window_size
            if slen < self.window_size:
                window = slen
            for n in range(batch):
                a0 = a[0][n].data
                a1 = a[1][n].data
                tops, top0s, top1s = [], [], []
                for start in range(slen):
                    if start + window > slen:
                        break
                    _, top0 = a0[start: start + window].topk(1)
                    _, top1 = a1[start: start + window].topk(1)
                    top0 = top0[0]
                    top1 = top1[0]
                    top0 += start
                    top1 += start
                    tops.append((top0, top1))
                    top0s.append(top0)
                    top1s.append(top1)
                tlen = len(top0s)
                #print(top0s)
                #print(top1s)
                top0s = torch.LongTensor(top0s).cuda(self.device)
                top1s = torch.LongTensor(top1s).cuda(self.device)
                h0 = H[n][top0s].view(tlen, self.u)
                h1 = H[n][top1s].view(tlen, self.u)
                hs.append(torch.cat([h0, h1], 1))
        elif self.sent:
            for n in range(batch):
                a0 = a[0][n].data
                a1 = a[1][n].data
                tops, top0s, top1s = [], [], []
                for start, end in zip(sents[n][:-1], sents[n][1:]):
                    _, top0 = a0[start: end].topk(1)
                    _, top1 = a1[start: end].topk(1)
                    top0 = top0[0] + start
                    top1 = top1[0] + start
                    tops.append((top0, top1))
                    top0s.append(top0)
                    top1s.append(top1)
                tlen = len(top0s)
                #print tlen,
                #print(top0s)
                #print(top1s)
                top0s = torch.LongTensor(top0s).cuda(self.device)
                top1s = torch.LongTensor(top1s).cuda(self.device)
                h0 = H[n][top0s].view(tlen, self.u)
                h1 = H[n][top1s].view(tlen, self.u)
                if tlen < max_sents:
                    h0 = torch.cat([h0, Variable(torch.zeros((max_sents - tlen), self.u).cuda(self.device))]).view(max_sents, self.u)
                    h1 = torch.cat([h1, Variable(torch.zeros((max_sents - tlen), self.u).cuda(self.device))]).view(max_sents, self.u)
                hs.append(torch.cat([h0, h1], 1))
                tlen = max_sents
        else:
            top_n = self.top_n
            if slen < top_n:
                print(slen)
                top_n = slen
            ws0, as0 = a[0].topk(top_n, dim=1)
            ws1, as1 = a[1].topk(top_n, dim=1)
            #ret2 = ws0 / a[0].sum(1) + ws1 / a[1].sum(1)
            ret2 = torch.log(ws0.sum(1)) + torch.log(ws1.sum(1))
            #print a[0].sum(1)
            #print 'ret2', ret2
            for n in range(batch):
                h0 = H[n][as0[n].data]
                #h0 = torch.cat([ws0[n].view(top_n, 1)] * self.u, 1) * h0
                # h0: topn * u
                h0 = torch.cat([h0] * top_n, 1)
                # h0: topn * (u * topn)
                h0 = h0.view(top_n ** 2, self.u)
                seq0 = torch.cat([as0[n].view(top_n, 1)] * top_n, 1).view(top_n**2)
                #print(type(seq0.data))
                #print('slen:', slen)
                p0 = pos_enc[seq0.data.cpu()].cuda(self.device)
                #H_sum0 = H_sum[n][seq0.data].cuda(self.device)
                h1 = H[n][as1[n].data]
                #h1 = torch.cat([ws1[n].view(top_n, 1)] * self.u, 1) * h1
                # h1: topn * u
                h1 = torch.cat([h1] * top_n)
                # h1: (topn * topn) * u
                seq1 = torch.cat([as1[n].view(top_n, 1)] * top_n).view(top_n**2)
                p1 = pos_enc[seq1.data.cpu()].cuda(self.device)
                #H_sum1 = H_sum[n][seq1.data].cuda(self.device)
                #H_diff = torch.abs(H_sum1 - H_sum0)
                #diff = p0 - p1
                #dist = torch.abs(seq0 - seq1).float().cuda(self.device)
                #print(type(h0), type(h1), type(dist))
                #h1 = torch.cat([h1, dist.view(-1, 1)], 1)
                #hs.append(torch.cat([h0, h1, p0, p1], 1))
                #hs.append(torch.cat([h0, h1], 1))
                if self.pos:
                    hs.append(torch.cat([h0, h1, p0, p1], 1))
                else:
                    hs.append(torch.cat([h0, h1], 1))
                tlen = top_n ** 2
        #print tlen,
        h_input = torch.cat(hs).view(batch * tlen, -1)

        if log:
            t2 = time.time()
            print('loop:', t2 - t1)

        x_ = self.g_fc1[idx](h_input)
        x_ = F.relu(x_)
        if self.tgf:
            x_ = self.g_fc2[idx](x_)
            x_ = F.relu(x_)
            x_ = self.g_fc3[idx](x_)
            x_ = F.relu(x_)
        x_ = x_.view(batch * tlen, 256)
        #x_ = x_.view(batch * top_n ** 2, -1)
        #x_ = torch.transpose(x_, 1, 2)
        a_ = self.attn2[idx](x_)
        a_ = a_.view(batch, tlen)
        a_ = F.softmax(a_)
        a_ = a_.view(batch, 1, tlen)
        x_ = x_.view(batch, tlen, -1)
        x_ = torch.bmm(a_, x_)
        #x_ = torch.sum(x_, 1)
        x_ = x_.squeeze()

        if log:
            t3 = time.time()
            print('g:', t3 - t2)

        '''f'''

        if self.tgf:
            x_ = self.f_fc1[idx](x_)
            x_ = F.relu(x_)
            x_ = self.f_fc2[idx](x_)
            x_ = F.relu(x_)
        x_ = self.f_fc3[idx](x_)
        #print('Before Dropout', x_.sum().data[0])
        x_ = self.dropout(x_)
        #print('After Dropout', x_.sum().data[0])

        if log:
            t4 = time.time()
            print('f:', t4 - t3)

        if self.debug:
            print('x', x_.size())

        #out = F.softmax(mlp1)
        self.debug = False

        if penalty:
            a1 = a[0].view(batch, 1, -1)
            a2 = a[1].view(batch, -1, 1)
            #a1 = a1.transpose(1, 2)
            diff = torch.norm(torch.bmm(a1, a2))

            return x_, diff

        if self.rl:
            return x_, ret2

        return x_


class RelateNet(Base):
    def __init__(self, param):
        super(RelateNet, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            #self.lstm2 = nn.LSTM(self.u, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            attn1, attn2, mlp2 = [], [], []
            g_fc1s, g_fc2s, g_fc3s = [], [], []
            f_fc1s, f_fc2s, f_fc3s = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #attn2.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                g_fc1s.append(nn.Linear(self.u * 2, 256).cuda(self.device))
                g_fc2s.append(nn.Linear(256, 256).cuda(self.device))
                g_fc3s.append(nn.Linear(256, 256).cuda(self.device))
                f_fc1s.append(nn.Linear(256, 256).cuda(self.device))
                f_fc2s.append(nn.Linear(256, 256).cuda(self.device))
                f_fc3s.append(nn.Linear(256, c).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                #mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn1 = ListModule(*attn1)
            #self.attn2 = ListModule(*attn2)
            #self.mlp1 = ListModule(*mlp1)
            self.g_fc1 = ListModule(*g_fc1s)
            self.g_fc2 = ListModule(*g_fc2s)
            self.g_fc3 = ListModule(*g_fc3s)
            self.f_fc1 = ListModule(*f_fc1s)
            self.f_fc2 = ListModule(*f_fc2s)
            self.f_fc3 = ListModule(*f_fc3s)
            #self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
        embeds = self.embed(words)
        batch = embeds.size(0)
        slen = embeds.size(1)
        #print(slen)
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

        print_stat(H, 'H')
        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        if self.debug:
            #print('H', H.size())
            print('u', self.u)
            print('batch', self.batch)
            print('slen', slen)
        H = H.contiguous().view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        if is_tagger:
            tag_space = self.tagger(H)
            return tag_space

        a = []
        for i in range(self.n):
            _a = self.attn1[i](H)
            _a = _a.view(batch, slen)
            _a = F.softmax(_a)
            # _a: batch, 1, slen
            a.append(_a)
        H = H.view(batch, slen, self.u)

        '''g'''

        if log:
            t1 = time.time()

        '''
        h_input = torch.Tensor(batch, top_n**2, self.u * 2)
        ws0, as0 = a[0].topk(top_n, dim=1)
        ws1, as1 = a[1].topk(top_n, dim=1)
        for n in range(batch):
            #print(a[0].size(), top_n)
            line = 0
            for wi, i in zip(ws0[n].data, as0[n].data):
                hi = H[n, i] * wi
                for wj, j in zip(ws1[n].data, as1[n].data):
                    hj = H[n, j] * wj
                    #print(torch.cat([hi, hj]).data.size())
                    h_input[n][line] = torch.cat([hi, hj]).data
                    line += 1
        h_input = Variable(h_input.view(batch * (top_n**2), self.u * 2)).cuda(self.device)
        '''

        hs = []
        top_n = 20
        if slen < top_n:
            print(slen)
            top_n = slen
        ws0, as0 = a[0].topk(top_n, dim=1)
        ws1, as1 = a[1].topk(top_n, dim=1)
        for n in range(batch):
            h0 = H[n][as0[n].data]
            h0 = torch.cat([ws0[0].view(top_n, 1)] * self.u, 1) * h0
            h0 = torch.cat([h0] * top_n, 1)
            h0 = h0.view(top_n ** 2, self.u)
            seq0 = torch.cat([as0[n].view(top_n, 1)] * top_n, 1).view(top_n**2)
            h1 = H[n][as1[n].data]
            h1 = torch.cat([ws1[0].view(top_n, 1)] * self.u, 1) * h1
            h1 = torch.cat([h1] * top_n)
            seq1 = torch.cat([as0[n].view(top_n, 1)] * top_n).view(top_n**2)
            dist = torch.abs(seq0 - seq1).float().cuda(self.device)
            #print(h0.size(), h1.size(), dist.size())
            #print(type(h0), type(h1), type(dist))
            #h1 = torch.cat([h1, dist.view(-1, 1)], 1)
            hs.append(torch.cat([h0, h1], 1))
        h_input = torch.cat(hs).view(batch * (top_n**2), self.u * 2)

        if log:
            t2 = time.time()
            print('loop:', t2 - t1)

        x_ = self.g_fc1[idx](h_input)
        x_ = F.relu(x_)
        x_ = self.g_fc2[idx](x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3[idx](x_)
        x_ = F.relu(x_)
        x_ = x_.view(batch, top_n ** 2, 256)
        x_ = torch.sum(x_, 1)
        x_ = x_.squeeze()

        if log:
            t3 = time.time()
            print('g:', t3 - t2)

        '''f'''

        x_ = self.f_fc1[idx](x_)
        x_ = F.relu(x_)
        x_ = self.f_fc2[idx](x_)
        x_ = F.relu(x_)
        x_ = F.dropout(x_)
        x_ = self.f_fc3[idx](x_)

        if log:
            t4 = time.time()
            print('f:', t4 - t3)

        if self.debug:
            print('x', x_.size())

        #out = F.softmax(mlp1)
        self.debug = False

        return x_


class AddTwoLSTM(Base):
    def __init__(self, param):
        super(AddTwoLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            self.lstm2 = nn.LSTM(self.u, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            attn1, attn2, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                attn2.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn1 = ListModule(*attn1)
            self.attn2 = ListModule(*attn2)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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
        #Attn = Variable(torch.zeros(batch * slen, self.n))
        #if self.use_cuda:
        #    Attn = Attn.cuda(self.device)
        Ms = []
        for i in range(self.n):
            #if i == idx:
            #    self.attn1[i].weight.requires_grad = True
            #else:
            #    self.attn1[i].weight.requires_grad = False
            #Attn[:, i] = self.attn[i](H)
            a = self.attn1[i](H)
            a = a.view(batch, slen)
            a = F.softmax(a)
            a = a.view(batch * slen, 1)
            A = []
            for j in range(self.u):
                A.append(a)
            A = torch.cat(A, 1)
            Ms.append(A * H)
        M = Ms[0] + Ms[1]

        if self.debug:
            print('Ms', len(Ms))
            print('A', A.size())
            print('M', M.size())
            print('H', H.size())
        M = M.view(batch, slen, -1)
        if self.debug:
            print('M', M.size())

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''

        packed = pack_padded_sequence(M, lens, batch_first=True)
        H, _ = self.lstm2(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)
        if self.debug:
            print('H', H.size())

        H = H.contiguous().view(batch * slen, self.u)
        Attn = self.attn2[idx](H)

        A = Attn.view(batch, slen, 1)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(batch, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class PosTwoLSTM(Base):
    def __init__(self, param):
        super(PosTwoLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            self.lstm2 = nn.LSTM(self.u * self.n, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            attn1, attn2, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                attn2.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            #```self.mlp3 = nn.Linear(self.u * 2, self.u).cuda(self.device)
            self.attn1 = ListModule(*attn1)
            self.attn2 = ListModule(*attn2)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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
        #Attn = Variable(torch.zeros(batch * slen, self.n))
        #if self.use_cuda:
        #    Attn = Attn.cuda(self.device)
        Ms = []
        for i in range(self.n):
            #if i == idx:
            #    self.attn1[i].weight.requires_grad = True
            #else:
            #    self.attn1[i].weight.requires_grad = False
            #Attn[:, i] = self.attn[i](H)
            a = self.attn1[i](H).view(batch, slen)
            a = F.softmax(a).view(batch, slen, 1)
            #a = a.view(batch, slen, 1)
            a = a.expand(batch, slen, self.u)
            a = a.contiguous().view(batch * slen, self.u)
            #c = Variable(torch.ones(batch, 1, self.u).float().cuda(self.device))
            Ms.append(torch.mul(a, H))
            '''
            A = []
            for j in range(self.u):
                A.append(a)
            A = torch.cat(A, 1)
            Ms.append(A * H)
            '''
            #break
        M = torch.cat(Ms, 1)

        if self.debug:
            print('Ms', len(Ms))
            #print('A', A.size())
            print('M', M.size())
            print('H', H.size())
        M = M.view(batch, slen, -1)
        if self.debug:
            print('M', M.size())

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''

        packed = pack_padded_sequence(M, lens, batch_first=True)
        H, _ = self.lstm2(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)
        if self.debug:
            print('H', H.size())
        #H = self.mlp3(M)

        H = H.contiguous().view(batch * slen, self.u)
        Attn = self.attn2[idx](H)

        A = Attn.view(batch, slen, 1)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class CatTwoLSTM(Base):
    def __init__(self, param):
        super(CatTwoLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            self.lstm2 = nn.LSTM(self.u * self.n, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            attn1, attn2, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                attn2.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            #```self.mlp3 = nn.Linear(self.u * 2, self.u).cuda(self.device)
            self.attn1 = ListModule(*attn1)
            self.attn2 = ListModule(*attn2)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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
        #Attn = Variable(torch.zeros(batch * slen, self.n))
        #if self.use_cuda:
        #    Attn = Attn.cuda(self.device)
        Ms = []
        for i in range(self.n):
            #if i == idx:
            #    self.attn1[i].weight.requires_grad = True
            #else:
            #    self.attn1[i].weight.requires_grad = False
            #Attn[:, i] = self.attn[i](H)
            a = self.attn1[i](H).view(batch, slen)
            a = F.softmax(a).view(batch, slen, 1)
            #a = a.view(batch, slen, 1)
            a = a.expand(batch, slen, self.u)
            a = a.contiguous().view(batch * slen, self.u)
            #c = Variable(torch.ones(batch, 1, self.u).float().cuda(self.device))
            Ms.append(torch.mul(a, H))
            '''
            A = []
            for j in range(self.u):
                A.append(a)
            A = torch.cat(A, 1)
            Ms.append(A * H)
            '''
            #break
        M = torch.cat(Ms, 1)

        if self.debug:
            print('Ms', len(Ms))
            #print('A', A.size())
            print('M', M.size())
            print('H', H.size())
        M = M.view(batch, slen, -1)
        if self.debug:
            print('M', M.size())

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''

        packed = pack_padded_sequence(M, lens, batch_first=True)
        H, _ = self.lstm2(packed)
        H, self.hidden = pad_packed_sequence(H, batch_first=True)
        if self.debug:
            print('H', H.size())
        #H = self.mlp3(M)

        H = H.contiguous().view(batch * slen, self.u)
        Attn = self.attn2[idx](H)

        A = Attn.view(batch, slen, 1)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class ASPCatAttn(Base):
    def __init__(self, param):
        super(ASPCatAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            plstm = []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(2, 1, bias=False).cuda(self.device))
                if self.bilstm:
                    plstm.append(nn.LSTM(self.e, self.u / 2, bidirectional=self.bilstm, batch_first=True).cuda(self.device))
                else:
                    plstm.append(nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device))
                mlp2.append(nn.Linear(self.u * 2, c).cuda(self.device))
            self.plstm = ListModule(*plstm)
            self.share_attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.D = nn.Linear(self.u, self.n + 1).cuda(self.device)
            #self.attn = ListModule(*attn)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False, train_disc=False):
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

        print_stat(H, 'H')
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
        a1 = self.attn[idx](H)
        a2 = self.share_attn(H)
        #Attn = torch.cat([a1, a2])

        '''
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))
        '''

        #A = Attn.view(batch, slen, 1)
        #A = torch.transpose(A, 1, 2)
        #A = A.contiguous().view(-1, slen)
        #A = F.softmax(A)
        #A = A.view(batch, 1, slen)
        H = H.view(batch, slen, self.u)
        a1 = a1.view(batch, slen, 1)
        a1 = torch.transpose(a1, 1, 2)
        a1 = a1.contiguous().view(-1, slen)
        a1 = F.softmax(a1)
        a1 = a1.view(batch, 1, slen)

        a2 = a2.view(batch, slen, 1)
        a2 = torch.transpose(a2, 1, 2)
        a2 = a2.contiguous().view(-1, slen)
        a2 = F.softmax(a2)
        a2 = a2.view(batch, 1, slen)

        if self.debug:
            #print('Attn', Attn.size())
            print('H', H.size())
            print('a1', a1.size())
            print('a2', a2.size())

        h1 = torch.bmm(a1, H)
        h2 = torch.bmm(a2, H)

        if self.debug:
            #print('A', A.size())
            print('h1', h1.size())
            print('h2', h2.size())

        if train_disc:
            disc = self.D(h2.squeeze())
            #print('dis', disc.size())
            return disc
        #M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        #M = torch.transpose(M, 1, 2).contiguous().view(-1, 2)
        #if self.debug:
        #    print('M', M.size())
        #mlp1 = self.mlp1[idx](M)
        #if self.debug:
        #    print('mlp1', mlp1.size())
        #if log and self.use_cuda:
        #    print('mlp1', self.mlp1[idx].weight.data.cpu().numpy())
        #mlp1 = mlp1.view(batch, self.u)
        M = torch.cat([h1, h2], 2)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp2', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        if penalty is True:
            #pen = 0
            #a = M.view(batch, self.u, self.n)[:, :, idx]
            #b = mlp1
            #target = Variable(torch.zeros(batch, self.u))
            #if self.use_cuda:
            #    target = target.cuda(self.device)
            #l1_crit = nn.L1Loss()
            #pen += l1_crit(a - b, target)
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
            a1 = a1.view(batch, -1, 1)
            a2 = a2.view(batch, -1, 1)
            a1 = a1.transpose(1, 2)
            diff = torch.norm(torch.bmm(a1, a2))
            #print('pen', pen)
            adv = self.D(h2.squeeze())
            #print('dis', disc.size())
            return out, adv, diff

        return out


class FSSingleAttn(Base):
    def __init__(self, param):
        super(FSSingleAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.mlp2 = nn.Linear(self.u, c).cuda(self.device)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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

        A = self.attn(H).view(batch, slen, 1)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 1, slen)
        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
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
        mlp2 = self.mlp2(M)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        return out


class MultiSingleAttn(Base):
    def __init__(self, param):
        super(MultiSingleAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c, bias=False).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                #mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn = ListModule(*attn)
            #self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            '''
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
            '''
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(batch, self.u)
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
        mlp2 = self.mlp2[idx](M)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
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

        return out


class ASPOneAttn(Base):
    def __init__(self, param):
        super(ASPOneAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                mlp1.append(nn.Linear(2, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.share_attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn = ListModule(*attn)
            self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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
        a1 = self.attn[idx](H)
        a2 = self.share_attn(H)
        Attn = torch.cat([a1, a2], 1)

        if self.debug:
            print('Attn', Attn.size())
            print('a1', a2.size())
            print('a2', a1.size())
        '''
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))
        '''

        A = Attn.view(batch, slen, 2)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, 2, slen)
        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(-1, 2)
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1[idx](M)
        if self.debug:
            print('mlp1', mlp1.size())
        if log and self.use_cuda:
            print('mlp1', self.mlp1[idx].weight.data.cpu().numpy())
        mlp1 = mlp1.view(batch, self.u)
        mlp2 = self.mlp2[idx](mlp1)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
        self.debug = False

        if penalty is True:
            #pen = 0
            #a = M.view(batch, self.u, self.n)[:, :, idx]
            #b = mlp1
            #target = Variable(torch.zeros(batch, self.u))
            #if self.use_cuda:
            #    target = target.cuda(self.device)
            #l1_crit = nn.L1Loss()
            #pen += l1_crit(a - b, target)
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
            a1 = a1.view(batch, -1, 1)
            a2 = a2.view(batch, -1, 1)
            a1 = a1.transpose(1, 2)
            pen = torch.norm(torch.bmm(a1, a2))
            #print('pen', pen)
            return out, pen

        return out



class ShareSingleAttn(Base):
    def __init__(self, param):
        super(ShareSingleAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.cs = param['cs']
        self.r = len(self.cs)
        self.n = len(self.cs)
        #TODO, plus tagger in single attn
        have_tagger = False
        for c in self.cs:
            if c > 20:
                have_tagger = True
        if have_tagger:
            self.n -= 1
            self.r -= 1
        self.bilstm = param['bilstm']
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
            attn, mlp1, mlp2 = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn.append(nn.Linear(self.u, 1, bias=False).cuda(self.device))
                mlp1.append(nn.Linear(self.n, 1, bias=False).cuda(self.device))
                mlp2.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn = ListModule(*attn)
            self.mlp1 = ListModule(*mlp1)
            self.mlp2 = ListModule(*mlp2)
            for i in range(len(self.mlp1)):
                weight = torch.zeros(self.mlp1[i].weight.size())
                weight[0][i] = 1.0
                self.mlp1[i].weight = nn.Parameter(torch.FloatTensor(weight))
                #self.mlp1[i].weight.requires_grad = False
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False, penalty=False):
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

        print_stat(H, 'H')
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
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))
        '''

        A = Attn.view(batch, slen, self.n)
        A = torch.transpose(A, 1, 2)
        A = A.contiguous().view(-1, slen)
        A = F.softmax(A)
        A = A.view(batch, self.n, slen)
        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)

        '''
        M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        '''
        M = torch.transpose(M, 1, 2).contiguous().view(-1, self.n)
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1[idx](M)
        if self.debug:
            print('mlp1', mlp1.size())
        if log and self.use_cuda:
            print('mlp1', self.mlp1[idx].weight.data.cpu().numpy())
        mlp1 = mlp1.view(batch, self.u)
        mlp2 = self.mlp2[idx](mlp1)
        if self.debug:
            print('mlp1', mlp2.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)
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

        return out


class TreeAttn(Base):
    def __init__(self, param):
        super(TreeAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.epsilon = param['epsilon']
        self.p_lambda = param['p_lambda']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.rl = 'rl' in param['params']
        self.pos_len = 10
        self.debug = True
        self.incre = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.attn2 = nn.Linear((self.u + self.pos_len) * 2, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.mlp2 = nn.Linear((self.u + self.pos_len) * 2, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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
        self.train_type = 'MID' # TWO, MID, MIX
        self.select = 'epsilon' # random, max
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
        #print(H[-1][-1])
        H = H.contiguous().view(batch * slen, self.u)
        #H = H.view(batch * slen, self.u)

        H = Variable(H.data) # No grad

        if self.debug:
            print('H', H.size())

        if self.train_type is 'MID':
            Attn = self.attn(Variable(H.data))
        else:
            Attn = self.attn(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        self.n_sample = 1
        #New start
        logit = []
        logit2 = []
        A = Attn.view(batch, slen)
        A = F.softmax(A)
        w, a = A.topk(1, dim=1)
        H = H.view(batch, slen, self.u)

        '''start'''
        if 'incre' in dir(self) and self.incre:
            Ms = []
            a = a.view(-1)
            for i, _a in enumerate(a.data):
                Ms.append(H[i][_a].view(1, self.u))
            M = torch.cat(Ms)

            if self.debug:
                print('A', A.size())
                print('H', H.size())
                print('M', M.size())

            M = M.squeeze()
            if self.debug:
                print('M', M.size())
            mlp1 = self.mlp1(M)
            if self.debug:
                print('mlp1', mlp1.size())

            #out = F.softmax(mlp1)
            out0 = self.dropout(mlp1)
        '''end'''

        pos_enc = Variable(position_encoding_init(10000, self.pos_len))
        p = pos_enc[torch.LongTensor(range(slen) * batch)].view(batch, slen, self.pos_len).cuda(self.device)
        H = torch.cat([H, p], 2)
        if self.debug:
            print('H', H.size())

        Ms = []
        a = a.view(-1)
        for i, _a in enumerate(a.data):
            _H = torch.cat([H[i][_a].view(1, 1, self.u + self.pos_len)] * slen, 1)
            Ms.append(_H)
        M = torch.cat(Ms, 0)
        if self.debug:
            print('M', M.size())

        H = torch.cat([H, M], 2).view(batch * slen, (self.u + self.pos_len) * 2)
        if self.debug:
            print('H', H.size())

        A = self.attn2(Variable(H.data)).view(batch, slen)
        A = F.softmax(A)

        del a

        if self.state is 'train':
            if self.select == 'epsilon':
                select = np.random.choice(["max", "random"], p=[1.0 - self.epsilon, self.epsilon])
            else:
                select = self.select
            if select == 'max':
                w, a = A.topk(1, dim=1)
                ret2 = w
            elif select == 'random':
                a, w = [], []
                for i, l in enumerate(lens):
                    #print(A[i].data.cpu().numpy())
                    #sample = np.random.choice(np.arange(0, slen), p=A[i].data.cpu().numpy())
                    sample = random.randint(0, l-1)
                    #a.append(random.randint(0, l - 1))
                    """
                    if sample > l:
                        print(sample, l, A[i].data[l - 1], A[i].data[sample], A[i].data.max(), A[i].data.min())
                    """
                    a.append(sample)
                    w.append(A[i][a[-1]])
                #w = Variable(torch.FloatTensor(w).cuda(self.device))
                w = torch.cat(w)
                #print(w.size())
                a = Variable(torch.LongTensor(a).cuda(self.device))
                ret2 = w
        #else:
        elif self.state == 'eval':
            w, a = A.topk(1, dim=1)
            #print('w', type(w), 'a', type(a))
            #print(w.size())
            ret2 = w

        H = H.view(batch, slen, (self.u + self.pos_len) * 2)
        if self.train_type is 'TWO':
            H = Variable(H.data) # No Grad for classification

        Ms = []
        a = a.view(-1)
        for i, _a in enumerate(a.data):
            Ms.append(H[i][_a].view(1, (self.u + self.pos_len) * 2))
        M = torch.cat(Ms)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        if self.debug:
            print('M', M.size())
            #print(M)

        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp2 = self.mlp2(M)
        if self.debug:
            print('mlp2', mlp2.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp2)

        self.debug = False
        if self.rl:
            if self.incre and self.state == 'train':
                return out0, out, ret2
            else:
                return out, ret2
        return out


class MaxAttn(Base):
    def __init__(self, param):
        super(MaxAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.device = param['device']
        self.epsilon = param['epsilon']
        self.p_lambda = param['p_lambda']
        self.search_n = param['search_n']
        if 'debug' in param:
            self.debug = param['debug']
        else:
            self.debug = False
        self.rl = 'rl' in param['params']
        self.debug = True
        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True).cuda(self.device)
            if self.bilstm:
                self.u = self.u * 2
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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
        self.train_type = 'MID' # TWO, MID, MIX
        self.select = 'epsilon' # random, max, epsilon
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
        #print(H[-1][-1])
        H = H.contiguous().view(batch * slen, self.u)
        #H = H.view(batch * slen, self.u)
        if self.debug:
            print('H', H.size())

        if self.train_type is 'MID':
            Attn = self.attn(Variable(H.data))
        else:
            Attn = self.attn(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        self.n_sample = 1
        #New start
        logit = []
        logit2 = []
        A = Attn.view(batch, slen)
        A = F.softmax(A)

        #New end
        if self.state == 'train':
            if self.select is 'epsilon':
                select = np.random.choice(["max", "random"], p=[1.0 - self.epsilon, self.epsilon])
            else:
                select = self.select
            if select == 'max':
                if self.search_n <= 1:
                    w, a = A.topk(1, dim=1)
                    ret2 = w
                else:
                    w, a = A.topk(self.search_n, dim=1)

                    H = H.view(batch, slen, self.u)
                    if self.train_type is 'TWO':
                        H = Variable(H.data) # No Grad for classification

                    Ms = []
                    for i, _batch in enumerate(a.data):
                        _Ms = []
                        for _a in _batch:
                            _Ms.append(H[i][_a].view(1, 1, self.u))
                        _Ms = torch.cat(_Ms, 1)
                        Ms.append(_Ms)
                    M = torch.cat(Ms)

                    M = M.view(batch * self.search_n, self.u)
                    if self.debug:
                        print('M', M.size())
                    mlp1 = self.mlp1(M)
                        
                    if self.debug:
                        print('mlp1', mlp1.size())

                    soft = F.softmax(mlp1)
                    soft = torch.max(soft, 1)[0]
                    _, max_idx = torch.topk(soft.view(batch, self.search_n), 1)
                    ret2 = torch.cat([w[i][idx].view(1, 1) for i, idx in enumerate(max_idx.data)])
                    mlp1 = mlp1.view(batch, self.search_n, self.c)
                    ret = torch.cat([mlp1[i][idx].view(1, self.c) for i, idx in enumerate(max_idx.data)])
                    
                    self.debug = False
                    if self.rl:
                        return ret, ret2
                    return out
            elif select == 'random':
                a, w = [], []
                for i, l in enumerate(lens):
                    #print(A[i].data.cpu().numpy())
                    #sample = np.random.choice(np.arange(0, slen), p=A[i].data.cpu().numpy())
                    sample = random.randint(0, l-1)
                    #a.append(random.randint(0, l - 1))
                    """
                    if sample > l:
                        print(sample, l, A[i].data[l - 1], A[i].data[sample], A[i].data.max(), A[i].data.min())
                    """
                    a.append(sample)
                    w.append(A[i][a[-1]])
                #w = Variable(torch.FloatTensor(w).cuda(self.device))
                w = torch.cat(w)
                #print(w.size())
                a = Variable(torch.LongTensor(a).cuda(self.device))
                ret2 = w
        elif self.state == 'eval' and self.search_n <= 1:
            w, a = A.topk(1, dim=1)
            #print('w', type(w), 'a', type(a))
            #print(w.size())
            ret2 = w
        elif self.state == 'eval' and self.search_n > 1:
            w, a = A.topk(self.search_n, dim=1)
            ret2 = w

            H = H.view(batch, slen, self.u)
            if self.train_type is 'TWO':
                H = Variable(H.data) # No Grad for classification

            Ms = []
            for i, _batch in enumerate(a.data):
                _Ms = []
                for _a in _batch:
                    _Ms.append(H[i][_a].view(1, 1, self.u))
                _Ms = torch.cat(_Ms, 1)
                Ms.append(_Ms)
            M = torch.cat(Ms)

            M = M.view(batch * self.search_n, self.u)
            if self.debug:
                print('M', M.size())
            mlp1 = self.mlp1(M)
                
            if self.debug:
                print('mlp1', mlp1.size())

            soft = F.softmax(mlp1)
            soft = torch.max(soft, 1)[0]
            _, max_idx = torch.topk(soft.view(batch, self.search_n), 1)
            mlp1 = mlp1.view(batch, self.search_n, self.c)
            ret = torch.cat([mlp1[i][idx].view(1, self.c) for i, idx in enumerate(max_idx.data)])
            
            self.debug = False
            if self.rl:
                return ret, ret2
            return out

        H = H.view(batch, slen, self.u)
        if self.train_type is 'TWO':
            H = Variable(H.data) # No Grad for classification

        Ms = []
        a = a.view(-1)
        for i, _a in enumerate(a.data):
            Ms.append(H[i][_a].view(1, self.u))
        M = torch.cat(Ms)

        if self.debug:
            print('A', A.size())
            print('H', H.size())
            print('M', M.size())

        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)

        self.debug = False
        if self.rl:
            return out, ret2
        return out


class SingleAttn(Base):
    def __init__(self, param):
        super(SingleAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
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
            self.attn = nn.Linear(self.u, 1, bias=False).cuda(self.device)
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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

        Attn = self.attn(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        '''old
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))
        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
        #New start
        A = Attn.view(batch, slen)
        A = F.softmax(A)
        A = A.view(batch, self.r, slen)
        #New end

        H = H.view(batch, slen, self.u)

        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        '''
        if self.state is 'train':
            M = torch.bmm(A, H)
        elif self.state is 'eval':
            A = A.view(batch, slen)
            w, a = A.topk(1, dim=1)
            Ms = []
            a = a.view(-1)
            for i, _a in enumerate(a.data):
                Ms.append(H[i][_a].view(1, self.u))
            M = torch.cat(Ms)
        '''

        if self.debug:
            print('M', M.size())
            #print(M)

        #M = torch.mean(M, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class SingleLSTM(Base):
    def __init__(self, param):
        super(SingleLSTM, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = 1
        self.bilstm = param['bilstm']
        self.c = param['c']
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
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])

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

        '''
        Attn = self.attn(H)
        #Attn1 = self.attn1(H)
        if self.debug:
            print('Attn', Attn.size())
        if log:
            print('Attn max: %f, min: %f' % (Attn.max().data[0], Attn.min().data[0]))

        #New start
        A = Attn.view(batch, slen, self.r)
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
        '''
        H = H.view(batch, slen, self.u)

        M = torch.mean(H, 1)
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        mlp1 = self.mlp1(M)
        if self.debug:
            print('mlp1', mlp1.size())

        #out = F.softmax(mlp1)
        out = self.dropout(mlp1)
        self.debug = False
        return out


class SelfAttn(Base):
    '''Self attention copy jkchen's tensorflow codes'''
    def __init__(self, param):
        super(SelfAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['c']
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
            self.mlp1 = nn.Linear(self.u, self.c).cuda(self.device)
            self.mlp2 = nn.Linear(self.r, self.c).cuda(self.device)
            self.dropout = nn.Dropout(p=param['dropout'])
        else:
            self.embed = nn.Embedding(self.v, self.e, padding_idx=1)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1 = nn.Linear(self.u, self.da)
            self.attn2 = nn.Linear(self.da, self.r)
            self.mlp1 = nn.Linear(self.u, self.c)
            self.mlp2 = nn.Linear(self.r, self.c)
            self.dropout = nn.Dropout(p=param['dropout'])
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

        Attn1 = F.tanh(self.attn1(H))
        #Attn1 = self.attn1(H)
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
        return out


class AttnTagger(Base):
    def __init__(self, param):
        super(AttnTagger, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.c = param['c']
        self.batch = param['batch_size']
        self.debug = True
        self.device = param['device']

        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            #self.lstm = nn.LSTM(self.e, self.u, bias=False).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, batch_first=True, bias=False).cuda(self.device)
            self.attn1 = nn.Linear(self.u * 2, self.da, bias=False).cuda(self.device)
            self.attn2 = nn.Linear(self.da, self.r, bias=False).cuda(self.device)
            self.mlp = nn.Linear(self.u * 2, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bias=False)
            self.attn1 = nn.Linear(self.u * 2, self.da, bias=False)
            self.attn2 = nn.Linear(self.da, self.r, bias=False)
            self.mlp = nn.Linear(self.u * 2, self.c)
        #self.hidden = self.init_hidden()
        #self.loss_function = nn.NLLLoss()

    def forward(self, words, log=None):
        # TODO: Padding
        if self.debug:
            print('words:', words.size())
        embeds = self.embed(words)
        if self.debug:
            print('embeds:', embeds.size())

        H, self.hidden = self.lstm(embeds)
        batch, slen, u = H.size()
        out = []
        for i in range(slen):
            h = H[:, i, :]
            h = h.repeat(slen, 1).view(slen, batch, u)
            h = h.transpose(0, 1)
            Hi = torch.cat([h, h], 2)

            Hi = Hi.contiguous().view(-1, u * 2)
            if self.debug:
                print('H:', Hi.size())

            Attn1 = F.tanh(self.attn1(Hi))
            if self.debug:
                print('Attn1', Attn1.size())

            #print_stat(self.attn1.weight, 'W1')

            Attn2 = self.attn2(Attn1)
            if self.debug:
                print('Attn2', Attn2.size())

            #print_stat(self.attn2[0].weight, 'W2')

            A = F.softmax(Attn2)
            A = A.view(batch, slen, self.r)
            Hi = Hi.view(batch, slen, u * 2)
            A = torch.transpose(A, 1, 2)
            if self.debug:
                print('A', A.size())
                print('H', H.size())

            M = torch.bmm(A, Hi)
            if self.debug:
                print('M', M.size())

            M = torch.mean(M, 1)
            M = M.squeeze()
            if self.debug:
                print('M', M.size())
            mlp = self.mlp(M)
            if self.debug:
                print('mlp1', mlp.size())
            self.debug = False

            #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
            #out = F.dropout(mlp)
            out.append(mlp)

        ans = []
        for j in range(out[0].size(0)):
            lines = []
            for k in range(len(out)):
                lines.append(out[k][j].unsqueeze(0))
            ans.append(torch.cat(lines))
        ans = torch.cat(ans)
        return ans


class MultiAllAttn(Base):
    '''Self attention copy jkchen's tensorflow codes'''
    def __init__(self, param):
        super(MultiAllAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.cs = param['cs']
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
            mlp = []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                mlp.append(nn.Linear(self.u, c).cuda(self.device))
            self.mlp = ListModule(*mlp)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False):
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

        print_stat(H, 'H')
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

        H = H.view(batch, slen, self.u)
        A = torch.transpose(H, 1, 2)

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
        return out


class MultiSelfAttn(Base):
    '''Self attention copy jkchen's tensorflow codes'''
    def __init__(self, param):
        super(MultiSelfAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.cs = param['cs']
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
            attn1, attn2, mlp = [], [], []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                attn1.append(nn.Linear(self.u, self.da, bias=False).cuda(self.device))
                attn2.append(nn.Linear(self.da, self.r, bias=False).cuda(self.device))
                mlp.append(nn.Linear(self.u, c).cuda(self.device))
            self.attn1 = ListModule(*attn1)
            self.attn2 = ListModule(*attn2)
            self.mlp = ListModule(*mlp)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False):
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

        print_stat(H, 'H')
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

        Attn1 = F.tanh(self.attn1[idx](H))
        if self.debug:
            print('Attn1', Attn1.size())
        if log:
            print('Attn1 max: %f, min: %f' % (Attn1.max().data[0], Attn1.min().data[0]))

        print_stat(self.attn1[0].weight, 'W1')

        Attn2 = self.attn2[idx](Attn1)
        if self.debug:
            print('Attn2', Attn2.size())
        if log:
            print('Attn2 max: %f, min: %f' % (Attn2.max().data[0], Attn2.min().data[0]))

        '''old wrong
        A = F.softmax(Attn2)
        if log:
            print('A max: %f, min: %f' % (A.max().data[0], A.min().data[0]))

        A = A.view(batch, slen, self.r)
        A = torch.transpose(A, 1, 2)
        '''
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
        return out


class ShareLastAttn(Base):
    '''Self attention copy jkchen's tensorflow codes'''
    def __init__(self, param):
        super(ShareLastAttn, self).__init__(param)
        self.use_cuda = param['use_cuda']
        self.v = param['vocab_size']
        self.e = param['embed_dim']
        self.u = param['lstm_dim']
        self.da = param['da']
        self.r = param['r']
        self.bilstm = param['bilstm']
        self.cs = param['cs']
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
            mlp = []
            for c in self.cs:
                if c > 20:
                    self.tagger = nn.Linear(self.u, c).cuda(self.device)
                    continue
                #mlp.append(nn.Linear(self.u, c).cuda(self.device))
            self.mlp = nn.Linear(self.u, c).cuda(self.device)
            #self.mlp = ListModule(*mlp)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, bidirectional=self.bilstm, batch_first=True)
            if self.bilstm:
                self.u = self.u * 2
            self.attn1, self.attn2, self.mlp = [], [], []
            for c in self.cs:
                self.attn1.append(nn.Linear(self.u, self.da, bias=False))
                self.attn2.append(nn.Linear(self.da, self.r, bias=False))
                self.mlp.append(nn.Linear(self.u, c))
        self.dropout = nn.Dropout(p=param['dropout'])

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

    def forward(self, words, idx, is_tagger=False, log=False):
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
        if self.debug:
            print('A', A.size())
            print('H', H.size())

        M = torch.bmm(A, H)
        if self.debug:
            print('M', M.size())

        M = torch.mean(M, 1)
        '''
        H = H.view(batch, slen, self.u)
        M = H[:, 0, :]
        M = M.squeeze()
        if self.debug:
            print('M', M.size())
        #mlp = self.mlp[idx](M)
        mlp = self.mlp(M)
        if self.debug:
            print('mlp1', mlp.size())

        #print('a1:', self.mlp[0].weight.sum().data[0], 'a2:', self.mlp[1].weight.sum().data[0])
        #out = F.softmax(mlp1)
        out = self.dropout(mlp)
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
        self.c = param['c']
        self.batch = param['batch_size']
        self.debug = True
        self.device = param['device']

        if self.use_cuda:
            self.embed = nn.Embedding(self.v, self.e).cuda(self.device)
            #self.lstm = nn.LSTM(self.e, self.u, bias=False).cuda(self.device)
            self.lstm = nn.LSTM(self.e, self.u, batch_first=True, bias=False).cuda(self.device)
            #self.attn = nn.Linear(self.u, self.r).cuda(self.device)
            self.tagger = nn.Linear(self.u, self.c).cuda(self.device)
        else:
            self.embed = nn.Embedding(self.v, self.e)
            self.lstm = nn.LSTM(self.e, self.u, batch_first=True, bias=False)
            #self.attn = nn.Linear(self.u, self.r)
            self.tagger = nn.Linear(self.u, self.c)
        #self.hidden = self.init_hidden()
        #self.loss_function = nn.NLLLoss()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.Tensor(1, 1, self.lstm_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.lstm_dim)))

    def forward(self, words, log=None):
        #words.data.t_()
        if self.debug:
            print('words:', words.size())
        embeds = self.embed(words)
        if self.debug:
            print('embeds:', embeds.size())

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

        #embeds = pack_padded_sequence(embeds, lens, batch_first=True)
        #packed = pack_padded_sequence(embeds, lens)
        H, self.hidden = self.lstm(embeds)
        #H, self.hidden = pad_packed_sequence(H, batch_first=True)
        #########H, self.hidden = pad_packed_sequence(H)

        #print('lstm:', self.lstm.weight_ih_l0.sum().data[0])

        H = H.contiguous().view(-1, self.u)
        if self.debug:
            print('H:', H.size())
        #A = self.attn(H)
        A = H
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





class OldSelfAttn(Base):
    def __init__(self, param):
        super(OldSelfAttn, self).__init__(param)
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


'''
class SingleLSTM(Base):
    def __init__(self, param):
        super(SingleLSTM, self).__init__(param)
        V = param['vocab_size']
        D = param['embed_dim']
        C = param['c']
        self.hidden_dim = param['lstm_dim']

        self.embed = nn.Embedding(V, D)
        self.embed_dim = D
        self.use_cuda = param['use_cuda']
        self.device = param['device']

        self.lstm = nn.LSTM(D, param['lstm_dim'])
        self.hidden2tag = nn.Linear(param['lstm_dim'], param['c'])
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
        

        lstm_out, self.hidden = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        #return tag_space[-1].view(1,-1)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores[-1].view(1, -1)
        #tag_scores = nn.LogSoftmax(tag_space)
        #return tag_scores[-1]
'''


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