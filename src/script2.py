import os
from pylab import frange
import pp


def cmdline(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, n_epochs, batch_size, lstm_dim, da, r, optim, dropout, bilstm=True, use_cuda=True, device=0):
    cmd = '/home/rjzhen/anaconda2/bin/python main.py with ex_name=%s data_set=%s pre_emb=%r vocab_size=%d embed_dim=%d lr=%f use_model=%s use_cuda=%r n_epochs=%d batch_size=%d lstm_dim=%d da=%d r=%d dropout=%f bilstm=%r optim=%s device=%d' % (ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, use_cuda, n_epochs, batch_size, lstm_dim, da, r, dropout, bilstm, optim, device)
    return cmd


def execute(cmd):
    print 'execute', cmd
    ret = os.popen(cmd).read()
    #ret = os.system(cmd)
    '''
    if ret == 256:
        print 'KeyboardInterrupt'
        exit()
    return ret
    '''
    print ret
    return 1


ppservers = ()
job_server = pp.Server(ppservers=ppservers)
job_server.set_ncpus(1)
print "Starting pp with", job_server.get_ncpus(), "workers"


#ex_name = 'SelfAttnTagger'
ex_name = 'JKSelfAttn'
pre_emb = True
vocab_size = 19540
use_model = 'JKSelfAttn'
n_epochs = 50
embed_dim = 100

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

lr = 0.001
idx = 0
jobs = []
device = 0
optim = 'adam'

data_sets = ['product/kitchen', 'product/dvd']
#['product/books', 'product/dvd']

for data_set in data_sets:
    cmd = cmdline(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, n_epochs, batch_size, lstm_dim, da, r, optim, dropout, bilstm=bilstm, device=device)
    idx += 1
    jobs.append([idx, cmd, job_server.submit(execute, (cmd,), )])

'''
for bilstm in [True, False]:
    for lr in [0.001, 0.01, 0.1]:
        for r in [20, 30, 50]:
            for lstm_dim in [100, 200]:
                for dropout in [0.7, 0.5, 0.3]:
                    for batch_size in [16, 32, 64]:
                        cmd = cmdline(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, n_epochs, batch_size, lstm_dim, da, r, optim, dropout, bilstm=bilstm, device=device)
                        if idx == 0:
                            print(cmd)
                        idx += 1
                        jobs.append([idx, cmd, job_server.submit(execute, (cmd,), )])
'''

'''

for optim in ['adam']:
    for bilstm in [True]:
        for r in [10]:
            for da in [50]:
                for lstm_dim in [50]:
                    for lr in [0.0005]:
                        for batch_size in [4]:
                            cmd = cmdline(ex_name, data_set, pre_emb, vocab_size, embed_dim, lr, use_model, n_epochs, batch_size, lstm_dim, da, r, optim, bilstm)
                            idx += 1
                            jobs.append([idx, cmd, job_server.submit(execute, (cmd,), )])
'''

for idx, cmd, job in jobs:
    print '%d/%d: %s Return: %d' % (idx, len(jobs), cmd, job())

job_server.print_stats()
