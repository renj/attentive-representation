import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNN_Text(nn.Module):
    
    def __init__(self, param):
        super(CNN_Text,self).__init__()
        V = param['vocab_size']
        Ks = [3, 4, 5]
        D = param['embed_dim']
        Co = 100
        Ci = 1
        self.use_cuda = param['use_cuda']
        C = param['tagset_size']

        print('V: %d, D: %d, C: %d, Co: %d, Ks, %s'%(V, D, C, Co, Ks))

        self.embed = nn.Embedding(V, D).cuda()
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)).cuda() for K in Ks]
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, **input):
        x = input['words']
        x = self.embed(x) # (N,W,D)
        
        x = x.unsqueeze(1) # (N,Ci,W,D)
        #print(x.is_cuda)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)
        return logit
