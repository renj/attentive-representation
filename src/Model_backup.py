import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
from loader import CAP_DIM


class Base(nn.Module):
    def __init__(self, param):
        super(Base, self).__init__()

    def init_embed(self, init_matrix):
        self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix))

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


class SingleLSTM(Base):
    def __init__(self, param):
        super(SingleLSTM, self).__init__(param)
        V = param['vocab_size']
        D = param['embed_dim']
        C = param['tagset_size']
        self.lower = param['lower']
        self.hidden_dim = param['hidden_dim']
        self.embed = nn.Embedding(V, D)
        self.embed_dim = D

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

    def forward(self, **input):
        words = input['words'].squeeze()
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

    '''
    def get_tags(self, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            output = self.forward(words = words,
                                  input_caps = input_caps)
        else:
            output = self.forward(words = words)

        #top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
        #tag_id = top_i[0][0]
        #tag = dictionaries['id_to_tag'][tag_id]

        _, tag_id = torch.max(output, dim=0)
        #tags = tags.data.numpy().reshape((-1,))
        return tag_id

    def get_loss(self, tags, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            tag_scores = self.forward(words = words,
                                  input_caps = input_caps)
        else:
            tag_scores = self.forward(words = words)

        loss = self.loss_function(tag_scores, tags)
        return loss
    '''


class SingleCNN(Base):
    def __init__(self, param):
        super(SingleCNN, self).__init__(param)
        V = param['vocab_size']
        Ks = [3, 4, 5]
        D = param['embed_dim']
        Co = 100
        Ci = 1
        C = param['tagset_size']
        self.lower = param['lower']
        self.embed = nn.Embedding(V, D)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]

        # The linear layer that maps from hidden state space to tag space
        #self.w1 = nn.Linear(param['hidden_dim'], param['tagset_size'])
        #self.w2 = nn.Linear(param['hidden_dim'], 2)
        #self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

        self.loss_function = nn.NLLLoss()


    def init_hidden(self):
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, **input):
        words = input['words']
        x = self.embed(words)

        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)

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


class ShareLSTM(Base):
    def __init__(self, param):
        super(ShareLSTM, self).__init__(param)
        V = param['vocab_size']
        D = param['embed_dim']
        C = param['tagset_size']
        self.lower = param['lower']
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

    def forward(self, **input):
        words = input['words']
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
            output = self.forward(words=words,
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


class LSTMTagger(nn.Module):
    def __init__(self, param):
        super(LSTMTagger, self).__init__()
        self.lower = param['lower']
        self.hidden_dim = param['hidden_dim']

        self.embed = nn.Embedding(param['vocab_size'],
                                            param['embed_dim'])

        self.embed_dim = param['embed_dim']
        if self.lower:
            self.embed_dim += CAP_DIM

        # The LSTM takes word embeddings and captical embedding as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embed_dim, param['hidden_dim'])

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(param['hidden_dim'], param['tagset_size'])
        self.hidden = self.init_hidden()
        self.loss_function = nn.NLLLoss()

    def init_embed(self, init_matrix):
        self.embed.weight = nn.Parameter(torch.FloatTensor(init_matrix))

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)),
                autograd.Variable(torch.Tensor(1, 1, self.hidden_dim)))

    def forward(self, **input):
        words = input['words']
        embeds = self.embed(words)
        
        if self.lower:
            caps = input['input_caps']
            input_caps = torch.FloatTensor(len(caps), CAP_DIM)
            input_caps.zero_()
            input_caps.scatter_(1, caps.view(-1,1) ,1)
            input_caps = autograd.Variable(input_caps)
            embeds = torch.cat((embeds, input_caps),1)

        lstm_out, self.hidden = self.lstm(embeds.view(len(words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(words), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

    def get_tags(self, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            tag_scores = self.forward(words = words,
                                  input_caps = input_caps)
        else:
            tag_scores = self.forward(words = words)
        
        _, tags = torch.max(tag_scores, dim=1)
        tags = tags.data.numpy().reshape((-1,))
        return tags

    def get_loss(self, tags, **input):
        words = input['words']

        if self.lower:
            input_caps = input['input_caps']
            tag_scores = self.forward(words = words,
                                  input_caps = input_caps)
        else:
            tag_scores = self.forward(words=words)

        loss = self.loss_function(tag_scores, tags)
        return loss
