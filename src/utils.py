import os
import re
import numpy as np
import torch
from torch import autograd
from torch.autograd import Variable
import codecs
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import torch.nn.functional as F
import math
import torch.nn as nn
import sys
sys.path.append('/home/rjzhen/anaconda2/lib/python2.7/site-packages/torchtext-0.1.1-py2.7.egg')
from torchtext import data
from torchtext import datasets
import numpy as np
import math


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico, vocabulary_size=2000):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), 
            key=lambda x: (-x[1], x[0]))[:vocabulary_size]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def read_pre_training(emb_path):
    """
    Read pre-train word embeding
    The detail of this dataset can be found in the following link
    https://nlp.stanford.edu/projects/glove/
    """
    print('Preparing pre-train dictionary')
    emb_dictionary = {}
    for line in codecs.open(emb_path, 'r', 'utf-8'):
        temp = line.split()
        emb_dictionary[temp[0]] = np.asarray(temp[1:], dtype=np.float16)
    return emb_dictionary


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def make_batch(_data, key, size, vs):
    seq = list(range(len(_data)))
    #random.shuffle(seq)
    sents = []
    for j in range(int(math.ceil(1.0 * len(_data) / size))):
        tempt = []
        max_len = 0
        for idx in range(j * size, (j + 1) * size):
            if idx >= len(_data):
                tempt.append([])
                continue
            s = _data[seq[idx]][key]
            if len(s) > max_len:
                max_len = len(s)
            tempt.append(s)
        new_tempt = []
        for idx in range(len(tempt)):
            new_tempt.append(tempt[idx] + ([0] * (max_len - len(tempt[idx]))))
        sents.append(torch.LongTensor(new_tempt))
    return sents





def model_penalty(model):
    '''
    ps = []
    for a in model.attn1:
        ps.append(a._parameters['weight'])
    ret = torch.dot(ps[0], ps[1])
    if ret.data[0] < 0:
        ret.data[0] = abs(ret.data[0])
    '''
    w1 = model.attn1[0].weight
    w2 = model.attn1[1].weight
    '''
    ret = torch.norm(torch.mm(w1, w2.t()))
    '''
    ret = torch.norm(w1 - w2)
    return ret


def print_stat(v, s):
    return
    print('%s: mean: %.4f, max: %.4f, min: %.4f, std: %.4f' % (s, v.mean().data[0], v.max().data[0], v.min().data[0], v.std().data[0]))


def set_grad(layer, b):
    for k in layer._parameters.keys():
        if layer._parameters[k] is not None:
            #print(layer, k, b)
            layer._parameters[k].requires_grad = b


def fine_tune(n_epochs, division, train_iters, dev_iters, test_iters, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=False, adv_train_iters=None, adv_label_field=None, penalty=0.0, params=[], stop=[]):

    #is_da = (str(model.__class__)[8:-2] == 'Model.AttnOverAttn')
    is_da = (str(model.__class__)[8:-2] == 'Model.MultiDynAttn')
    if is_da:
        print('Finetune for dynamic attention')
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optim == 'adam':
        if adversarial:
            ignored_params = list(map(id, model.D.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
            optimizer_D = torch.optim.Adam(model.D.parameters(), lr=lr, weight_decay=l2_reg)
        elif lock_params:
            print('lock params')
            '''RelateNet
            ignored_params = list(map(id, model.embed.parameters()))
            ignored_params += list(map(id, model.lstm.parameters()))
            ignored_params += list(map(id, model.attn1[0].parameters()))
            ignored_params += list(map(id, model.attn1[1].parameters()))
            '''
            ignored_params = list(map(id, model.embed.parameters()))
            if 'lstm' in dir(model):
                ignored_params += list(map(id, model.lstm.parameters()))
            else:
                ignored_params += list(map(id, model.slstm.parameters()))
            #ignored_params += list(map(id, model.attn1.parameters()))
            #ignored_params += list(map(id, model.attn.parameters()))
            #ignored_params += list(map(id, model.mlp2.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    else:
        print('wrong optimizer')
        exit()

    _run.info['test_loss'] = []
    _run.info['test_accuracy'] = []
    _run.info['dev_loss'] = []
    _run.info['dev_accuracy'] = []
    _run.info['division'] = division
    _run.info['train_loss'] = []
    _run.info['train_accuracy'] = []
    _run.info['train_pen'] = []

    seq = []
    for i in range(len(train_iters)):
        _run.info[('%s_test_loss' % data_set[i])] = []
        _run.info[('%s_test_accuracy' % data_set[i])] = []
        _run.info[('%s_dev_loss' % data_set[i])] = []
        _run.info[('%s_dev_accuracy' % data_set[i])] = []
        if 'top_six' in params:
            if str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                if i >= 6 and 'topic' not in model.data_set[i]:
                    continue
            else:
                if i >= 6:
                    continue 
        elif 'end_ten' in params:
            if str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                if i < 6 and 'topic' not in model.data_set[i]:
                    continue
            else:
                if i < 6:
                    continue 
        if is_da and ('topic' in model.data_set[i] or 'product' not in model.data_set[i]):
            continue
        seq += ([i] * int(len(train_iters[i])))

    #random.shuffle(seq)
    seq = []
    for i in range(len(train_iters[0])):
        for j in range(len(train_iters)):
            seq += [j]

    def nextone(a):
        for _a in a:
            return _a

    steps = 0
    max_accs = [0] * len(train_iters)
    for epoch in range(n_epochs):
        for idx in seq:
            model.train()
            batch = nextone(train_iters[idx])
            optimizer.zero_grad()
            feature, target = batch.text, batch.label
            if feature.size()[0] <= 4:
                continue
            feature.data.t_()
            is_tagger = False
            if len(target.size()) > 1:
                target.data.t_()
                target = target.contiguous().view(-1)
                is_tagger = True

            if use_cuda:
                feature = feature.cuda(model.device)
                target = target.cuda(model.device)

            logit = model.forward(feature, idx, is_tagger)
            loss = F.cross_entropy(logit, target)

            '''
            old_q0 = model.attn1[0].weight.data[0].cpu()
            old_q1 = model.attn1[1].weight.data[0].cpu()
            old_q3 = model.attn2[0].weight.data[0].cpu()
            old_embed = model.embed.weight.data[0].cpu()
            old_lstm = model.lstm.weight_ih_l0.data[0].cpu()
            '''

            loss.backward()
            optimizer.step()

            if steps % division == 0:
                '''
                print('attn0:', sum(old_q0 - model.attn1[0].weight.data[0].cpu()))
                print('attn1:', sum(old_q1 - model.attn1[1].weight.data[0].cpu()))
                print('attn2:', sum(old_q3 - model.attn2[0].weight.data[0].cpu()))
                print('lstm:', sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()))
                print('embed:', sum(old_embed - model.embed.weight.data[0].cpu()))
                '''
                #print('lstm:', sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()))

                print('*************New Evaluation**********')
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 1.0 * corrects / target.size(0) * 100.0
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.data[0], accuracy, corrects, target.size(0)))
                print(' ')
                _run.info['train_loss'].append(loss.data[0])
                _run.info['train_accuracy'].append(accuracy)

                if str(model.__class__)[8:-2] == 'Model.RelateAttn' or str(model.__class__)[8:-2] == 'Model.CatTwoLSTM'  or str(model.__class__)[8:-2] == 'Model.AttnOverAttn':
                    mean_accs = max_accs[0]
                elif str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                    if 'top_six' in params:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i < 6])
                    elif 'end_ten' in params:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i >= 6])
                    else:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i]])
                else:
                    if 'top_six' in params:
                        mean_accs = sum(max_accs[:6]) * 1.0 / len(max_accs[:6])
                    elif 'end_ten' in params:
                        mean_accs = sum(max_accs[6:]) * 1.0 / len(max_accs[6:])
                    else:
                        mean_accs = sum(max_accs) * 1.0 / len(max_accs)

                curr_accs = []
                new_max = False
                for i in range(len(data_set)):
                    test_acc = evaluate(model, test_iters, use_cuda, _run, max_accs[i], stop=stop, idx=i, data_set=data_set, dtype='test')
                    curr_accs.append(test_acc)
                    if test_acc > max_accs[i]:
                        new_max = True
                        max_accs[i] = test_acc
                    #evaluate(model, dev_iters, use_cuda, _run, max_accs[i], idx=i, data_set=data_set, dtype='dev')

                if str(model.__class__)[8:-2] == 'Model.RelateAttn' or str(model.__class__)[8:-2] == 'Model.CatTwoLSTM'  or str(model.__class__)[8:-2] == 'Model.AttnOverAttn':
                    curr_mean_accs = curr_accs[0]
                elif str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                    if 'top_six' in params:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i < 6])
                    elif 'end_ten' in params:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i >= 6])
                    else:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i]])
                else:
                    if 'top_six' in params:
                        curr_mean_accs = sum(curr_accs[:6]) * 1.0 / len(curr_accs[:6])
                    elif 'end_ten' in params:
                        curr_mean_accs = sum(curr_accs[6:]) * 1.0 / len(curr_accs[6:])
                    else:
                        curr_mean_accs = sum(curr_accs) * 1.0 / len(curr_accs)

                if curr_mean_accs > mean_accs and curr_mean_accs > 30 and save_model:
                    #f_name = '%.4f_%.4f_%.2f.torch' % (p_lambda, p_gamma, curr_mean_accs)
                    f_name = '%.2f.torch' % (curr_mean_accs)
                    if penalty != 0:
                        f_name = '%.3f' % (penalty) + '_' + f_name
                    for d in data_set:
                        d = d.split('/')
                        if len(d) > 1:
                            d = d[1]
                        else:
                            d = d[0]
                        f_name = d + '_' + f_name
                    for p in params:
                        f_name = p + '_' + f_name
                    if lock_params:
                        f_name = 'lock' + '_' + f_name
                    else:
                        f_name = 'unlock' + '_' + f_name
                    f_name = 'fine_tune' + '_' + f_name
                    f_name = str(model.__class__)[8:-2] + '_' + f_name

                    dont_save = False
                    files = os.listdir('../model/')
                    for f in files:
                        if f.startswith(f_name[:-12]):
                            if float(f[-11:-6]) < curr_mean_accs:
                                os.remove('../model/' + f)
                            else:
                                dont_save = True
                    if dont_save is False:
                        torch.save(model.cpu(), '../model/' + f_name)
                        model.cuda(model.device)
                    print('********New best accs*********')
                    for i in range(len(data_set)):
                        sys.stdout.write('%s: %.2f\t' % (data_set[i], curr_accs[i]))
                    print(' ')
                    print('save model', f_name)
                else:
                    print('avg acc: %.2f, max acc: %.2f, p_lambda: %.4f, p_gamma: %.4f' % (curr_mean_accs, mean_accs, p_lambda, p_gamma))
                    print('Finetune, lock:'+str(lock_params))
                #if curr_mean_accs > mean_accs:
                #    max_accs = curr_accs

            steps += 1


def mem_evaluate(model, dev_iters, use_cuda=True, _run=None, max_acc=0, max_perfect_acc=0, max_last_acc=0, idx=-1, data_set=None, stop=None, dtype='test'):
    #TODO: Multi-task
    model.eval()
    avg_loss = 0
    corrects = 0
    perfect_corrects = 0
    size = 0
    if idx != -1:
        dev_iter = dev_iters[idx]
    else:
        dev_iter = dev_iters
    #dev_iter = dev_iters
    cnt = 0

    from collections import defaultdict
    position_dict = defaultdict(int)
    layer_corrects = [0] * model.nlayers
    layer_vector = [0] * model.nlayers
    layer_first = [0] * model.nlayers
    all_corrects = 0

    for batch in dev_iter:
        cnt += 1
        feature, target = batch.text, batch.label
        feature.data.t_()
        #if cnt % 100 == 0:
        #    print('cnt:', cnt, 'feature.size:', feature.size())
        is_tagger = False
        if len(target.size()) > 1:
            target.data.t_()  # batch first, index align
            target = target.contiguous().view(-1)
            is_tagger = True
        if use_cuda:
            feature, target = feature.cuda(model.device), target.cuda(model.device)

        if 'pause' in dir(model) and model.pause:
            Ds, Ps = model.forward(feature)

            #print('P', Ps[0])
            _sum = 0
            stop_at = []
            _D = None
            _stop = []
            for i in range(Ds[0].size(0)):
                found = False
                for j, P in enumerate(Ps):
                    #print('PP', P[i][1].data[0])
                    if P[i][1].data[0] > 0.5:
                        _stop = j
                        found = True
                        break
                if found is False:
                    _stop = len(Ps) - 1
                position_dict[_stop] += 1
                if _D is None:
                    _D = Ds[_stop][i].view(1, -1)
                else:
                    _D = torch.cat([_D, Ds[_stop][i].view(1, -1)], 0)
                #print('_D', _D.size())
            #print('_D', _D)
            logit = _D
        else:
            Ds = model.forward(feature)
            logit = Ds[-1]
        for _idx, D in enumerate(Ds):
            layer_corrects[_idx] += (torch.max(D, 1)[1].view(target.size()).data == target.data).sum()
            layer_vector[_idx] = (torch.max(D, 1)[1].view(target.size()).data == target.data)
        all_corrects = layer_vector[0]
        for _idx, lv in enumerate(layer_vector):
            if _idx == 0:
                layer_first[_idx] += all_corrects.sum()
            else:
                layer_first[_idx] += (all_corrects | lv).sum() - all_corrects.sum()
            all_corrects = all_corrects | lv
        perfect_corrects += all_corrects.sum()

        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

        size += target.size(0)

    #if ('pause' in dir(model) and model.pause) or ('hinge' in dir(model) and model.hinge):
    if dtype is 'test' and model.pause:
        print('position_dict: %s' % dict(position_dict))
    layer_corrects = [1.0 * c/size * 100.0 for c in layer_corrects]
    perfect_corrects = 1.0 * perfect_corrects / size * 100.0
    avg_loss = 1.0 * avg_loss / size
    accuracy = 1.0 * corrects / size * 100.0

    if idx == -1:
        d = data_set
    else:
        d = data_set[idx]
    if dtype is 'test':
        #print('%s:  acc: %.2f, perf: %.2f, last: %.2f loss: %.2f  acc: %.2f(%d/%d)' % (dtype, accuracy, corrects, size, max_acc, max_perfect_acc, max_last_acc, avg_loss))
        print('%s:  acc: %.2f, perf: %.2f, last: %.2f loss: %.2f' % (dtype, accuracy, perfect_corrects, layer_corrects[-1], avg_loss))
        print('%s:  max: %.2f, perf: %.2f, last: %.2f' % (dtype, max_acc, max_perfect_acc, max_last_acc))
    model.train()

    if dtype is 'test':
        #print('layer corrects %s, perfect corrects: %.2f' % (layer_corrects,perfect_corrects))
        print('%s\tlayer acc %s' % (dtype, layer_corrects))
    print('%s\tlayer corr: %s' % (dtype, layer_first))

    '''
    if _run is None:
        return accuracy
    if idx == -1:
        _run.info[dtype + '_loss'].append(avg_loss)
        _run.info[dtype + '_accuracy'].append(accuracy)
    else:
        _run.info['%s_%s_loss' % (d, dtype)].append(avg_loss)
        _run.info['%s_%s_accuracy' % (d, dtype)].append(accuracy)
    '''
    return accuracy, perfect_corrects, layer_corrects[-1]


def mem_train(n_epochs, division, train_iter, dev_iter, test_iter, model, lr, optim, use_cuda, l2_reg, data_set, mr_test_data, imdb_test_data, save_model, stop, p_gamma, _run, lock_params=False, params=[]):
    #TODO: support multi-task
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optim == 'adam':
        if lock_params:
            ignored_params = list(map(id, model.lstm.parameters()))
            ignored_params += list(map(id, model.attn.parameters()))
            ignored_params += list(map(id, model.mlp1.parameters()))
            ignored_params += list(map(id, model.embed.parameters()))
            ignored_params += list(map(id, model.gru.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
            #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5, weight_decay=l2_reg)
    else:
        print('wrong optimizer')
        exit()

    _run.info['test_loss'] = []
    _run.info['test_accuracy'] = []
    _run.info['dev_loss'] = []
    _run.info['dev_accuracy'] = []
    _run.info['division'] = division
    _run.info['train_loss'] = []
    _run.info['train_accuracy'] = []

    model.train()
    steps = 0
    max_acc = 0
    max_perfect_acc = 0
    max_last_acc = 0
    max_grad = -999.99
    for epoch in range(n_epochs):
        for batch in train_iter:
            steps += 1
            #print('steps:', steps)
            optimizer.zero_grad()
            #model.zero_grad()
            feature, target = batch.text, batch.label
            feature.data.t_()
            if feature.size()[0] <= 4:
                continue
            #feature.data.t_(), target.data.sub_(1)
            if len(target.size()) > 1:
                target.data.t_()  # batch first, index align
                target = target.contiguous().view(-1)

            if use_cuda:
                feature = feature.cuda(model.device)
                target = target.cuda(model.device)

            loss = 0

            if model.pause:
                if model.a_pen > 0:
                    Ds, Ps, As = model.forward(feature)
                else:
                    Ds, Ps = model.forward(feature)
            else:
                if model.a_pen > 0:
                    Ds, As = model.forward(feature)
                else:
                    Ds = model.forward(feature)

            logit = model.dropout(Ds[-1])
            batch_size = Ds[0].size(0)
            reg = (1.0 / (batch_size * model.nlayers))

            for _idx, D in enumerate(Ds):
                Ds[_idx] = model.dropout(D)

            if model.max_ce:
                Rs = []
                for _idx, D in enumerate(Ds):
                    target_mat = torch.zeros(Ds[-1].size()).float()
                    for i in range(target_mat.size(0)):
                        target_mat[i][target[i % target.size(0)].data[0]] = 1
                    target_mat = Variable(target_mat).cuda(model.device)
                    #print((D > 0.5).float())
                    reward = F.softmax(D) * target_mat
                    reward = reward.sum(1).data
                    if model.discount:
                        reward = (p_gamma **_idx) * reward
                    Rs.append(reward)
                Rs = torch.cat(Rs, 1)
                #print('Rs', Rs.size())
                if steps % division == 0:
                    print('Rs', Rs)
                _w, top_col = Rs.topk(1, dim=1)
                #print('_w', _w, 'top_col', top_col)

                _D = None
                from collections import defaultdict
                position_dict = defaultdict(int)
                #print(top_col.view(-1).tolist())
                for i, _stop in enumerate(top_col.view(-1).tolist()):
                    #print(i, _stop)
                    #_stop = model.nlayers - 1
                    #_stop = 0
                    position_dict[_stop] += 1
                    if _D is None:
                        _D = Ds[_stop][i].view(1, -1)
                    else:
                        _D = torch.cat([_D, Ds[_stop][i].view(1, -1)], 0)
                if steps % division == 0:
                    print('position_dict: %s'%position_dict)
                if steps % division == 0:
                    print('_D', _D)
                loss += reg * batch_size * model.nlayers * F.cross_entropy(_D, target)
                logit = _D
                #loss += reg * batch_size * model.nlayers * F.cross_entropy(model.dropout(Ds[-1]), target)

            elif model.all_ce or model.pause:
                for D in Ds:
                    continue 
                    loss += reg * batch_size * F.cross_entropy(model.dropout(D), target)
            else:
                loss += reg * batch_size * model.nlayers * F.cross_entropy(model.dropout(Ds[-1]), target)

            if 'hinge' in dir(model) and model.hinge:
                ls = [F.cross_entropy(l, target) for l in Ds]
                for _idx in range(1, len(ls)):
                    loss += model.beta * max(ls[_idx] - ls[_idx-1], 0)


            if 'pause' in dir(model) and model.pause:
                Rs = []
                for _idx, D in enumerate(Ds):
                    target_mat = torch.zeros(Ds[-1].size()).float()
                    for i in range(target_mat.size(0)):
                        target_mat[i][target[i % target.size(0)].data[0]] = 1
                    target_mat = Variable(target_mat).cuda(model.device)
                    #print((D > 0.5).float())
                    reward = (D > 0.5).float() * target_mat
                    #reward = F.softmax(D) * target_mat
                    reward = reward.sum(1).data
                    if model.discount:
                        reward = (p_gamma **_idx) * reward
                    Rs.append(reward)

                if model.bpause_even:
                    Bs = [[] for _ in range(model.rl_batch)]
                    for _idx, R in enumerate(Rs):
                        Bs[int(_idx / model.nlayers)].append(R)
                    for _idx in range(model.rl_batch):
                        #print(_idx, sum(Bs[_idx]), len(Bs[_idx]))
                        #print('Bs:', sum(Bs[_idx]) / len(Bs[_idx]), sum(Rs)/len(Rs))
                        #Bs[_idx] = sum(Bs[_idx]) / len(Bs[_idx])
                        Bs[_idx] = sum(Rs)/len(Rs)

                if steps % division == 0:
                    print('Rs:', [r.mean() for r in Rs], 'Bs:', [b.mean() for b in Bs])

                _sum = 0
                #for _idx, D, P in zip(range(len(Ds)), Ds, Ps):
                for _idx, R, P in zip(range(len(Ps)), Rs, Ps):
                    #loss += reg * F.cross_entropy(D, target)
                    '''
                    target_mat = torch.zeros(Ds[-1].size()).float()
                    for i in range(target_mat.size(0)):
                        target_mat[i][target[i % target.size(0)].data[0]] = 1
                    target_mat = Variable(target_mat).cuda(model.device)
                    R = F.softmax(D) * target_mat
                    R = Variable(R.data).sum(1)
                    '''
                    if model.bpause_even:
                        B = Bs[int(_idx / model.nlayers)]
                    else:
                        B = 0.5
                    if _idx % model.nlayers == 0:
                        #loss2 = slen_reg * (torch.neg(torch.log(P[:,1])) * (R - 0.5)).sum()
                        loss2 = reg * (torch.neg(torch.log(P[:,1])) * Variable(R - B)).sum()
                        #loss2 = reg * (torch.log(P[:,1]) * (R - B)).sum()
                        #loss2 = slen_reg * (torch.neg(torch.log(P[:,1])) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                        _sum = 0
                        #if steps % (division) == 0:
                        #    print(R-B)
                    elif (_idx % model.nlayers) == model.nlayers - 1:
                        loss2 = reg * (torch.neg(1 - _sum) * Variable(R - B)).sum()
                        #loss2 = reg * ((1 - _sum) * (R - B)).sum()
                        #loss2 = slen_reg * (torch.neg(_sum) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                    else:
                        #loss2 = slen_reg * (torch.neg(_sum + torch.log(P[:,1])) * (R - 0.5)).sum()
                        loss2 = reg * (torch.neg(_sum + torch.log(P[:,1])) * Variable(R - B)).sum()
                        #loss2 = reg * ((_sum + torch.log(P[:,1])) * (R - B)).sum()
                        #loss2 = slen_reg * (torch.neg(_sum + torch.log(P[:,1])) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                    _sum += torch.log(P[:,0])
                    loss += model.p_lambda2 * loss2
                    #diff2 = torch.norm(model.attn1[0].weight.data[0] - model.attn1[1].weight.data[0], 2)

            if model.a_pen > 0:
                I = Variable(torch.diag(torch.ones(batch_size))).cuda(model.device)
                for a1, a2 in zip(As[:-1], As[1:]):
                    #loss += a_pen * torch.norm(a1 - a2)
                    _pen = torch.norm(torch.mm(a1, a2.t()) - I, 2)
                    loss += model.a_pen * _pen

            if 'gru' in dir(model):
                old_q = model.attn.weight.data[0].cpu()
                old_lstm = model.lstm.weight_ih_l0.data[0].cpu()
                old_embed = model.embed.weight.data[0].cpu()
                #old_mlp1 = model.mlp1.weight.data[0].cpu()
                old_gru = model.gru.weight_ih.data[0].cpu()
                old_P = model.P.weight.data[0].cpu()
                #old_gru = model.gru.weight.data[0].cpu()

            loss.backward()

            if steps % division == 0:
                _max = max([p.grad.data.max() for p in model.parameters() if p.grad is not None])

            if 'clip' in dir(model) and model.clip > 0.0:
                torch.nn.utils.clip_grad_norm(model.parameters(), model.clip)

            if steps % division == 0:
                __max = max([p.grad.data.max() for p in model.parameters() if p.grad is not None])
                print('_max:%.3f\t__max:%.3f'%(_max, __max))

            max_grad = max(max_grad, max([p.grad.data.max() for p in model.parameters() if p.grad is not None]))
            if steps % division == 0:
                print('********** max gradient: %.4f ************' % max_grad)

            def weight_gradient_error(w):
                _max = w.grad.data.max()
                _min = w.grad.data.min()
                if math.isnan(_max) or math.isnan(_min) or (_max < _min) or (abs(_max) > 1000):
                    return True
                else:
                    return False

            if 'gru' in dir(model):
                exit_flag = False
                if model.gru.weight_ih.grad is not None:
                    if weight_gradient_error(model.gru.weight_ih):
                        _max = model.gru.weight_ih.grad.data.max()
                        _min = model.gru.weight_ih.grad.data.min()
                        print('******** Weight Gradient Error *********')
                        print('GRU gradient', _max, _min, _max < _min, abs(_max) > 1000)
                        exit_flag = True
                if weight_gradient_error(model.lstm.weight_ih_l0):
                    print('******** Weight Gradient Error *********')
                    _max = model.lstm.weight_ih_l0.grad.data.max()
                    _min = model.lstm.weight_ih_l0.grad.data.min()
                    print('LSTM gradient', _max, _min, _max < _min, abs(_max) > 1000)
                    exit_flag = True
                if exit_flag:
                    exit()

            optimizer.step()

            if 'gru' in dir(model):
                q_max = model.attn.weight.data.max()
                lstm_max = model.lstm.weight_ih_l0.data.max()
                embed_max = model.embed.weight.data.max()
                #mlp1_max = model.mlp1.weight.data.max()
                P_max = model.P.weight.data.max()
                gru_max = model.gru.weight_ih.data.max()
                #gru_max = model.gru.weight.data.max()
            '''
            print('--------- max_grad', max_grad, ' ----------')
            print('GRU max', gru_max, math.isnan(gru_max))
            print('q max', q_max, math.isnan(q_max))
            print('lstm max', lstm_max, math.isnan(lstm_max))
            print('embed max', embed_max, math.isnan(embed_max))
            print('mlp1 max', mlp1_max, math.isnan(mlp1_max))
            print('P max', P_max, math.isnan(P_max))
            '''


            '''
            for p in model.parameters():
                if p.grad is None:
                    continue
                p.data.add_(lr, p.grad.data)
            '''
            '''
            if steps % (division/2) == 0 and 'epsilon' in dir(model) and model.epsilon:
                print('loss: %.3f\t attn: %.3f\t lstm: %.3f\t embed: %.3f\t mlp1: %.3f' % (loss2.data.cpu().sum(), sum(old_q - model.attn.weight.data[0].cpu()), sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()), sum(old_embed - model.embed.weight.data[0].cpu()), sum(old_mlp1 - model.mlp1.weight.data[0].cpu())))
            '''

            if steps % division == 0:
                if 'gru' in dir(model):
                    if 'nlayers' in dir(model):
                        print('nlayers: %d\t rl_batch: %d' % (model.nlayers, model.rl_batch))
                    if 'epsilon' in dir(model):
                        print('epsilon: %.2f\t p_lambda: %.4f\t p_lambda2: %.4f' % (model.epsilon, model.p_lambda, model.p_lambda2))
                    print('params: %s'%model.params)
                    print('attn: %.3f\t lstm: %.3f\t embed: %.3f' % (sum(old_q - model.attn.weight.data[0].cpu()), sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()), sum(old_embed - model.embed.weight.data[0].cpu())))
                    print('gru: %.3f\t P: %.3f\t' % (sum(old_gru - model.gru.weight_ih.data[0].cpu()), sum(old_P - model.P.weight.data[0].cpu())))
                    '''
                    print('gru: %.2f \t attn: %.2f\t lstm: %.2f\t embed: %.2f' % (sum(model.gru.weight_ih.data[0].cpu()), sum(model.attn.weight.data[0].cpu()), sum(model.lstm.weight_ih_l0.data[0].cpu()), sum(model.embed.weight.data[0].cpu())))
                    '''

                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                ans = torch.max(logit, 1)[1].view(target.size()).data
                ans_dict = list2dict(ans.cpu())
                print(ans_dict)

                accuracy = 1.0 * corrects / target.size(0) * 100.0
                #sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.data[0], accuracy, corrects, target.size(0)))
                print('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.data[0], accuracy, corrects, target.size(0)))
                _run.info['train_loss'].append(loss.data[0])
                _run.info['train_accuracy'].append(accuracy)

                acc, perfect_acc, last_acc = mem_evaluate(model, test_iter, use_cuda, _run, max_acc, max_perfect_acc, max_last_acc, stop=stop, dtype='test')
                if mr_test_data is not None:
                    mr_acc, mr_perfect_acc, mr_last_acc = mem_evaluate(model, mr_test_data, use_cuda, _run, max_acc, max_perfect_acc, max_last_acc, stop=stop, dtype='mr')
                    imdb_acc, imdb_perfect_acc, imdb_last_acc = mem_evaluate(model, imdb_test_data, use_cuda, _run, max_acc, max_perfect_acc, max_last_acc, stop=stop, dtype='imdb')
                    print('mr acc: %.2f\t imdb acc: %.2f' % (mr_acc, imdb_acc))
                if model.save_perf:
                    acc = perfect_acc

                if save_model and (max_acc < acc) and acc > 40.0:
                    f_name = '%.2f.torch' % (acc)
                    if 'search_n' in dir(model) and model.search_n > 1:
                        f_name = '%.d_' % (model.search_n) + f_name
                    if 'epsilon' in dir(model) and model.epsilon:
                        f_name = '%.2f_%.2f_' % (model.epsilon, model.p_lambda) + f_name
                    if 'rl_batch' in dir(model):
                        f_name = '%d_' % (model.rl_batch) + f_name
                    if 'nlayers' in dir(model):
                        f_name = '%d_' % (model.nlayers) + f_name
                    for p in sorted(model.params):
                        f_name = p + '_' + f_name
                    if lock_params:
                        f_name = 'transfer' + '_' + f_name
                    for d in data_set:
                        d = d.split('/')
                        if len(d) > 1:
                            d = d[1]
                        else:
                            d = d[0]
                        f_name = d + '_' + f_name
                    f_name = str(model.__class__)[8:-2] + '_' + f_name

                    dont_save = False
                    files = os.listdir('../model/')
                    for f in files:
                        if f.startswith(f_name[:-12]):
                            if float(f[-11:-6]) < acc:
                                os.remove('../model/' + f)
                            else:
                                dont_save = True
                    if dont_save is False:
                        torch.save(model.cpu(), '../model/' + f_name)
                        model.cuda(model.device)
                    print('save model', f_name)
                if max_acc < acc:
                    max_acc = acc
                if max_perfect_acc < perfect_acc:
                    max_perfect_acc = perfect_acc 
                if max_last_acc < last_acc:
                    max_last_acc = last_acc



def multi_train(n_epochs, division, train_iters, dev_iters, test_iters, model, lr, optim, use_cuda, l2_reg, data_set, p_lambda, p_gamma, save_model, adversarial, _run, lock_params=False, adv_train_iters=None, adv_label_field=None, penalty=0.0, params=[], stop=[], rec_acc=False):
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optim == 'adam':
        if adversarial:
            ignored_params = list(map(id, model.D.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
            optimizer_D = torch.optim.Adam(model.D.parameters(), lr=lr, weight_decay=l2_reg)
        elif lock_params:
            print('lock params')
            '''RelateNet
            ignored_params = list(map(id, model.embed.parameters()))
            ignored_params += list(map(id, model.lstm.parameters()))
            ignored_params += list(map(id, model.attn1[0].parameters()))
            ignored_params += list(map(id, model.attn1[1].parameters()))
            '''
            ignored_params = list(map(id, model.embed.parameters()))
            ignored_params += list(map(id, model.lstm.parameters()))
            #ignored_params += list(map(id, model.attn1.parameters()))
            #ignored_params += list(map(id, model.attn.parameters()))
            #ignored_params += list(map(id, model.mlp2.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    else:
        print('wrong optimizer')
        exit()

    _run.info['test_loss'] = []
    _run.info['test_accuracy'] = []
    _run.info['dev_loss'] = []
    _run.info['dev_accuracy'] = []
    _run.info['division'] = division
    _run.info['train_loss'] = []
    _run.info['train_accuracy'] = []
    _run.info['train_pen'] = []

    seq = []
    for i in range(len(train_iters)):
        _run.info[('%s_test_loss' % data_set[i])] = []
        _run.info[('%s_test_accuracy' % data_set[i])] = []
        _run.info[('%s_dev_loss' % data_set[i])] = []
        _run.info[('%s_dev_accuracy' % data_set[i])] = []
        if 'top_six' in params:
            if str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                if i >= 6 and 'topic' not in model.data_set[i]:
                    continue
            else:
                if i >= 6:
                    continue 
        elif 'end_ten' in params:
            if str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                if i < 6 and 'topic' not in model.data_set[i]:
                    continue
            else:
                if i < 6:
                    continue 
        if str(model.__class__)[8:-2] == 'Model.AttnOverAttn' and i == 0:
            seq += ([i] * (int(len(train_iters[i])) * 16))
        elif str(model.__class__)[8:-2] == 'Model.MultiDynAttn' and 'topic' not in model.data_set[i] and 'product' in model.data_set[i]:
            seq += ([i] * (int(len(train_iters[i])) * (len(model.data_set)+1)))
        else:
            if model.data_set[0] == model.data_set[1] and i == 1:
                print('the same')
                break
            seq += ([i] * int(len(train_iters[i])))

    random.shuffle(seq)

    def nextone(a):
        for _a in a:
            return _a

    if rec_acc:
        rec_file_name = '.txt'
        for d in data_set:
            d = d.split('/')
            if len(d) > 1:
                d = d[1]
            else:
                d = d[0]
            rec_file_name = '_' + d + rec_file_name
        rec_file_name = '../model/Rec_' + str(model.__class__)[8:-2] + '_'+str(model.u)+'_'+rec_file_name
        rec_file = open(rec_file_name, 'a')
        rec_file.write('new\n')
        rec_file.close()

    steps = 0
    max_accs = [0] * len(train_iters)
    for epoch in range(n_epochs):
        for idx in seq:
            #if idx == 1:
            #    continue
            model.train()
            batch = nextone(train_iters[idx])
            optimizer.zero_grad()
            feature, target = batch.text, batch.label
            if feature.size()[0] <= 4:
                continue
            feature.data.t_()
            is_tagger = False
            if len(target.size()) > 1:
                target.data.t_()
                target = target.contiguous().view(-1)
                is_tagger = True

            if use_cuda:
                feature = feature.cuda(model.device)
                target = target.cuda(model.device)

            if adversarial and (p_lambda != 0 or p_gamma != 0):
                logit, adv, diff = model.forward(feature, idx, is_tagger, penalty=True)
                #task_target
                adv_label = adv_label_field.vocab.stoi[data_set[idx][8:]]
                adv_target = [adv_label] * target.size(0)
                adv_target = Variable(torch.LongTensor(adv_target)).cuda(model.device)
                adv_loss = torch.neg(F.cross_entropy(adv, adv_target))
                #p = F.softmax(adv)
                #adv = torch.sum(p * torch.log(p))
                #print('cross_entropy', F.cross_entropy(logit, target).size())
                #print('pen', pen.size())
                loss = F.cross_entropy(logit, target) + p_lambda * adv_loss + p_gamma * diff
            elif penalty != 0:
                if 'sent' in dir(model) and model.sent:
                    logit, diff = model.forward(feature, idx, is_tagger, stop=stop, penalty=True)
                else:
                    logit, diff = model.forward(feature, idx, is_tagger, penalty=True)
                #print 'logit:', logit
                print('diff:', diff.data[0])
                loss = F.cross_entropy(logit, target) + penalty * diff
            else:
                if 'sent' in dir(model) and model.sent:
                    logit = model.forward(feature, idx, is_tagger, stop=stop)
                    loss = F.cross_entropy(logit, target)
                elif 'rl' in dir(model) and model.rl:
                    logit, logit2 = model.forward(feature, idx, is_tagger)
                    loss = F.cross_entropy(logit, target)
                    target_2d = torch.zeros(logit.size()).float()
                    for i in range(target.size(0)):
                        target_2d[i][target[i].data[0]] = 1
                    target_2d = Variable(target_2d).cuda(model.device)
                    mul = F.softmax(logit) * target_2d
                    mul = Variable(mul.data)
                    #print 'logit', F.softmax(logit)
                    #print 'target_2d', target_2d
                    #print 'mul1', mul
                    mul = torch.log(mul.sum(1))
                    #print 'mul2', mul
                    loss2 = 0.01 * (logit2 * mul).sum()
                    #print model.attn1[0].weight.data[0]
                    diff2 = torch.norm(model.attn1[0].weight.data[0] - model.attn1[1].weight.data[0], 2)
                    print('loss2', loss2.data[0], 'diff', diff2)
                    loss += loss2

                else:
                    logit = model.forward(feature, idx, is_tagger)
                    loss = F.cross_entropy(logit, target)

            if adversarial:
                old_D = model.D.weight

            '''
            old_q0 = model.attn1[0].weight.data[0].cpu()
            old_q1 = model.attn1[1].weight.data[0].cpu()
            old_q3 = model.attn2[0].weight.data[0].cpu()
            old_embed = model.embed.weight.data[0].cpu()
            old_lstm = model.lstm.weight_ih_l0.data[0].cpu()
            '''

            loss.backward()
            optimizer.step()

            #print 'loss', loss.data[0]

            #print(type(target))
            if adversarial:
                new_D = model.D.weight
                assert(old_D.max().data[0] == new_D.max().data[0])
                optimizer_D.zero_grad()
                #print('target', target.size())
                #print(target)
                #target_list = []
                #features = []
                #max_len = 0
                batch = nextone(adv_train_iters)
                feature, target = batch.text, batch.label
                feature.data.t_()
                if use_cuda:
                    feature = feature.cuda(model.device)
                    target = target.cuda(model.device)

                #old_lstm = model.lstm.weight_ih_l0

                logit = model.forward(feature, idx, is_tagger, train_disc=True)
                _loss = F.cross_entropy(logit, target)
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 1.0 * corrects / target.size(0) * 100.0
                print('loss: %.4f\tadv_loss: %.4f\taccuracy: %.4f' % (loss.data[0], _loss.data[0], accuracy))
                _loss.backward()
                optimizer_D.step()

                #new_lstm = model.lstm.weight_ih_l0
                #assert(old_lstm.max().data[0] == new_lstm.max().data[0])

            if steps % division == 0:
                '''
                print('attn0:', sum(old_q0 - model.attn1[0].weight.data[0].cpu()))
                print('attn1:', sum(old_q1 - model.attn1[1].weight.data[0].cpu()))
                print('attn2:', sum(old_q3 - model.attn2[0].weight.data[0].cpu()))
                print('lstm:', sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()))
                print('embed:', sum(old_embed - model.embed.weight.data[0].cpu()))
                '''
                #print('lstm:', sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()))

                print('*************New Evaluation**********')
                if adversarial and (p_lambda != 0 or p_gamma != 0):
                    print('loss:', loss.data[0], 'adv:', adv.data[0], 'diff:', diff.data[0])
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 1.0 * corrects / target.size(0) * 100.0
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.data[0], accuracy, corrects, target.size(0)))
                print(' ')
                _run.info['train_loss'].append(loss.data[0])
                _run.info['train_accuracy'].append(accuracy)
                #_run.info['train_pen'].append(pen.data[0])
                if len(params) > 0:
                    print(params)

                #if str(model.__class__)[8:-2] == 'Model.RelateAttn' or str(model.__class__)[8:-2] == 'Model.CatTwoLSTM'  or str(model.__class__)[8:-2] == 'Model.AttnOverAttn':
                if str(model.__class__)[8:-2] == 'Model.RelateAttn' or str(model.__class__)[8:-2] == 'Model.CatTwoLSTM':
                    mean_accs = max_accs[0]
                elif str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                    if 'top_six' in params:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i < 6])
                    elif 'end_ten' in params:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i >= 6])
                    else:
                        mean_accs = np.mean([_acc for i, _acc in enumerate(max_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i]])
                else:
                    if 'top_six' in params:
                        mean_accs = sum(max_accs[:6]) * 1.0 / len(max_accs[:6])
                    elif 'end_ten' in params:
                        mean_accs = sum(max_accs[6:]) * 1.0 / len(max_accs[6:])
                    else:
                        mean_accs = sum(max_accs) * 1.0 / len(max_accs)

                curr_accs = []
                dev_accs = []
                if model.data_set[0] == model.data_set[1]:
                    mean_accs = max_accs[0]
                for i in range(len(data_set)):
                    test_acc = evaluate(model, test_iters, use_cuda, _run, max_accs[i], stop=stop, idx=i, data_set=data_set, dtype='test')
                    curr_accs.append(test_acc)
                    #dev_acc = evaluate(model, dev_iters, use_cuda, _run, max_accs[i], idx=i, data_set=data_set, dtype='dev')
                    dev_acc = 0
                    dev_accs.append(dev_acc)

                if str(model.__class__)[8:-2] == 'Model.RelateAttn' or str(model.__class__)[8:-2] == 'Model.CatTwoLSTM' or str(model.__class__)[8:-2] == 'Model.AttnOverAttn':
                    curr_mean_accs = curr_accs[0]
                elif str(model.__class__)[8:-2] == 'Model.MultiDynAttn':
                    if 'top_six' in params:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i < 6])
                    elif 'end_ten' in params:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i] and i >= 6])
                    else:
                        curr_mean_accs = np.mean([_acc for i, _acc in enumerate(curr_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i]])
                    dev_mean_accs = np.mean([_acc for i, _acc in enumerate(dev_accs) if 'product' in model.data_set[i] and 'topic' not in model.data_set[i]])
                else:
                    if 'top_six' in params:
                        curr_mean_accs = sum(curr_accs[:6]) * 1.0 / len(curr_accs[:6])
                    elif 'end_ten' in params:
                        curr_mean_accs = sum(curr_accs[6:]) * 1.0 / len(curr_accs[6:])
                    else:
                        curr_mean_accs = sum(curr_accs) * 1.0 / len(curr_accs)
                    dev_mean_accs = sum(dev_accs) * 1.0 / len(dev_accs)

                if model.data_set[0] == model.data_set[1]:
                    curr_mean_accs = curr_accs[0]

                if rec_acc:
                    rec_file = open(rec_file_name, 'a')
                    rec_file.write('test %.4f\n'%curr_mean_accs)
                    rec_file.write('dev %.4f\n'%dev_mean_accs)
                    rec_file.write('loss %.4f\n'%loss.data[0])
                    rec_file.close()

                if curr_mean_accs > mean_accs and curr_mean_accs > 30 and save_model:
                    #f_name = '%.4f_%.4f_%.2f.torch' % (p_lambda, p_gamma, curr_mean_accs)
                    f_name = '%.2f.torch' % (curr_mean_accs)
                    if penalty != 0:
                        f_name = '%.3f' % (penalty) + '_' + f_name
                    for d in data_set:
                        d = d.split('/')
                        if len(d) > 1:
                            d = d[1]
                        else:
                            d = d[0]
                        f_name = d + '_' + f_name
                    for p in params:
                        f_name = p + '_' + f_name
                    if lock_params:
                        f_name = 'lock' + '_' + f_name
                    if model.u != 400:
                        f_name = str(model.u) + '_' + f_name
                    f_name = str(model.__class__)[8:-2] + '_' + f_name

                    dont_save = False
                    files = os.listdir('../model/')
                    for f in files:
                        if f.startswith(f_name[:-12]):
                            if float(f[-11:-6]) < curr_mean_accs:
                                os.remove('../model/' + f)
                            else:
                                dont_save = True
                    if dont_save is False:
                        torch.save(model.cpu(), '../model/' + f_name)
                        model.cuda(model.device)
                    print('********New best accs*********')
                    for i in range(len(data_set)):
                        sys.stdout.write('%s: %.2f\t' % (data_set[i], curr_accs[i]))
                    print(' ')
                    print('save model', f_name)
                else:
                    print('avg acc: %.2f, max acc: %.2f, p_lambda: %.4f, p_gamma: %.4f' % (curr_mean_accs, mean_accs, p_lambda, p_gamma))
                if curr_mean_accs > mean_accs:
                    max_accs = curr_accs

            steps += 1


def list2dict(ans):
    ans_dict = {}
    for _a in ans:
        if _a not in ans_dict:
            ans_dict[_a] = 0
        ans_dict[_a] += 1
    return ans_dict


def evaluate(model, dev_iters, use_cuda=True, _run=None, max_acc=0, idx=-1, data_set=None, stop=None, dtype='test'):
    #TODO: Multi-task
    model.eval()
    avg_loss = 0
    corrects = 0
    perfect_corrects = 0
    size = 0
    if idx != -1:
        dev_iter = dev_iters[idx]
    else:
        dev_iter = dev_iters
    #dev_iter = dev_iters
    cnt = 0
    if ('pause' in dir(model) and model.pause) or ('hinge' in dir(model) and model.hinge):
        from collections import defaultdict
        position_dict = defaultdict(int)
        layer_corrects = [0] * model.nlayers
        layer_vector = [0] * model.nlayers
        all_corrects = 0
    for batch in dev_iter:
        cnt += 1
        feature, target = batch.text, batch.label
        feature.data.t_()
        #if cnt % 100 == 0:
        #    print('cnt:', cnt, 'feature.size:', feature.size())
        is_tagger = False
        if len(target.size()) > 1:
            target.data.t_()  # batch first, index align
            target = target.contiguous().view(-1)
            is_tagger = True
        if use_cuda:
            feature, target = feature.cuda(model.device), target.cuda(model.device)

        if idx != -1:
            if 'need_stop' in dir(model) and model.need_stop is True:
                logit = model.forward(feature, idx, stop=stop, is_tagger=is_tagger, log=False)
            else:
                if idx == 0:
                    if 'rl' in dir(model) and model.rl:
                        logit, logit2 = model.forward(feature, idx, is_tagger, log=False)
                    else:
                        logit = model.forward(feature, idx, is_tagger, log=False)
                else:
                    logit = model.forward(feature, idx, is_tagger, log=False)
        else:
            #if model.need_stop is True:
            if 'need_stop' in dir(model) and model.need_stop is True:
                logit = model.forward(feature, stop)
            elif 'pause' in dir(model) and model.pause:
                Ds, Ps = model.forward(feature)
                for _idx, D in enumerate(Ds):
                    layer_corrects[_idx] += (torch.max(D, 1)
                         [1].view(target.size()).data == target.data).sum()

                #print('P', Ps[0])
                _sum = 0
                stop_at = []
                _D = None
                _stop = []
                for i in range(Ds[0].size(0)):
                    found = False
                    for j, P in enumerate(Ps):
                        #print('PP', P[i][1].data[0])
                        if P[i][1].data[0] > 0.5:
                            _stop = j
                            found = True
                            break
                    if found is False:
                        _stop = len(Ps) - 1
                    position_dict[_stop] += 1
                    if _D is None:
                        _D = Ds[_stop][i].view(1, -1)
                    else:
                        _D = torch.cat([_D, Ds[_stop][i].view(1, -1)], 0)
                    #print('_D', _D.size())
                #print('_D', _D)
                logit = _D
            elif 'rl' in dir(model) and model.rl:
                logit, logit2 = model.forward(feature)
                #print('logit:', logit)
                #print('logit2:', logit2)
                logit = logit[-1]
                #logit2 = logit2[-1]
            else:
                logit = model.forward(feature)
                #if 'hinge' in dir(model) and model.hinge:
                #    if model.pause is False:
        if 'pause' in dir(model):
            for _idx, D in enumerate(logit):
                layer_corrects[_idx] += (torch.max(D, 1)
                     [1].view(target.size()).data == target.data).sum()
                layer_vector[_idx] = (torch.max(D, 1)[1].view(target.size()).data == target.data)
            all_corrects = layer_vector[0]
            for lv in layer_vector:
                all_corrects = all_corrects | lv
            perfect_corrects += all_corrects.sum()

        if type(logit) is tuple:
            logit, logit2 = logit
        if type(logit) is list:
            logit = logit[-1]
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

        size += target.size(0)

    #if ('pause' in dir(model) and model.pause) or ('hinge' in dir(model) and model.hinge):
    if 'pause' in dir(model):
        print('position_dict', dict(position_dict))
        layer_corrects = [1.0 * c/size * 100.0 for c in layer_corrects]
        last_layer_corrects = [-1]
        perfect_corrects = 1.0 * perfect_corrects / size * 100.0
        print('layer corrects', layer_corrects, 'all corrects: %.2f' % perfect_corrects)

    avg_loss = 1.0 * avg_loss / size
    accuracy = 1.0 * corrects / size * 100.0
    #if accuracy > max_acc:
    #    max_acc = accuracy
    #sys.stdout.write('\r%s - loss: {:.3f}  acc: {:.4f}%({}/{}), max: {:.2f}'.format(dtype, avg_loss, accuracy, corrects, size, max_acc))
    if idx == -1:
        d = data_set
    else:
        d = data_set[idx]
    #print(data_set[idx], corrects, size, accuracy)
    print('\r%s:%s - loss: %.2f  acc: %.2f(%d/%d), max: %.2f' % (d, dtype, avg_loss, accuracy, corrects, size, max_acc))
    model.train()

    if _run is None:
        return accuracy
    if idx == -1:
        _run.info[dtype + '_loss'].append(avg_loss)
        _run.info[dtype + '_accuracy'].append(accuracy)
    else:
        _run.info['%s_%s_loss' % (d, dtype)].append(avg_loss)
        _run.info['%s_%s_accuracy' % (d, dtype)].append(accuracy)
    return accuracy




def train(n_epochs, division, train_iter, dev_iter, test_iter, model, lr, optim, use_cuda, l2_reg, data_set, save_model, stop, _run, lock_params=False, params=[]):
    #TODO: support multi-task
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg)
    elif optim == 'adam':
        if lock_params:
            ignored_params = list(map(id, model.lstm.parameters()))
            ignored_params += list(map(id, model.attn.parameters()))
            ignored_params += list(map(id, model.mlp1.parameters()))
            ignored_params += list(map(id, model.embed.parameters()))
            ignored_params += list(map(id, model.gru.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([{'params': base_params}], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
            #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.5, weight_decay=l2_reg)
    else:
        print('wrong optimizer')
        exit()

    _run.info['test_loss'] = []
    _run.info['test_accuracy'] = []
    _run.info['dev_loss'] = []
    _run.info['dev_accuracy'] = []
    _run.info['division'] = division
    _run.info['train_loss'] = []
    _run.info['train_accuracy'] = []

    model.train()
    steps = 0
    max_acc = 0
    max_grad = -999.99
    for epoch in range(n_epochs):
        for batch in train_iter:
            steps += 1
            #print('steps:', steps)
            optimizer.zero_grad()
            #model.zero_grad()
            feature, target = batch.text, batch.label
            feature.data.t_()
            if feature.size()[0] <= 4:
                continue
            #feature.data.t_(), target.data.sub_(1)
            if len(target.size()) > 1:
                target.data.t_()  # batch first, index align
                target = target.contiguous().view(-1)

            if use_cuda:
                feature = feature.cuda(model.device)
                target = target.cuda(model.device)

            if 'need_stop' in dir(model) and model.need_stop is True:
                logit = model.forward(feature, stop)
                loss = F.cross_entropy(logit, target)
            elif ('rl' in dir(model) and model.rl) or ('pause' in dir(model) and model.pause):
                if model.rl and model.pause:
                    if model.rl_torch:
                        Ds, Ps, episodes = model.forward(feature)
                    else:
                        Ds, Ps, Ws = model.forward(feature)
                elif model.rl:
                    if model.rl_torch:
                        Ds, episodes = model.forward(feature)
                    else:
                        Ds, Ws = model.forward(feature)
                elif model.pause:
                    Ds, Ps = model.forward(feature)
                logit = model.dropout(Ds[-1])
                batch_size = Ds[0].size(0)
                reg = (1.0 / (batch_size * model.rl_batch * model.nlayers))

                loss = 0
                Rs = []
                for _idx, D in enumerate(Ds):
                    if model.pause:
                        loss += reg * batch_size * F.cross_entropy(model.dropout(D), target)
                    else:
                        loss += reg * batch_size * model.nlayers * F.cross_entropy(model.dropout(D), target)
                    target_mat = torch.zeros(Ds[-1].size()).float()
                    for i in range(target_mat.size(0)):
                        target_mat[i][target[i % target.size(0)].data[0]] = 1
                    target_mat = Variable(target_mat).cuda(model.device)
                    reward = F.softmax(D) * target_mat
                    reward = reward.sum(1).data
                    Rs.append(reward)

                if model.pause:
                    #print('new pause process')
                    if model.bpause_even:
                        Bs = [[] for _ in range(model.rl_batch)]
                        for _idx, R in enumerate(Rs):
                            Bs[int(_idx / model.nlayers)].append(R)
                        for _idx in range(model.rl_batch):
                            #print(_idx, sum(Bs[_idx]), len(Bs[_idx]))
                            #print('Bs:', sum(Bs[_idx]) / len(Bs[_idx]), sum(Rs)/len(Rs))
                            #Bs[_idx] = sum(Bs[_idx]) / len(Bs[_idx])
                            Bs[_idx] = sum(Rs)/len(Rs)

                    _sum = 0
                    #for _idx, D, P in zip(range(len(Ds)), Ds, Ps):
                    for _idx, R, P in zip(range(len(Ps)), Rs, Ps):
                        #loss += reg * F.cross_entropy(D, target)
                        '''
                        target_mat = torch.zeros(Ds[-1].size()).float()
                        for i in range(target_mat.size(0)):
                            target_mat[i][target[i % target.size(0)].data[0]] = 1
                        target_mat = Variable(target_mat).cuda(model.device)
                        R = F.softmax(D) * target_mat
                        R = Variable(R.data).sum(1)
                        '''
                        if model.bpause_even:
                            B = Bs[int(_idx / model.nlayers)]
                        else:
                            B = 0.5
                        # TODO
                        if model.pause_torch:
                            p_action = P.multinomial()
                            
                        else:
                            if _idx % model.nlayers == 0:
                                #loss2 = slen_reg * (torch.neg(torch.log(P[:,1])) * (R - 0.5)).sum()
                                loss2 = reg * (torch.neg(torch.log(P[:,1])) * Variable(R - B)).sum()
                                #loss2 = reg * (torch.log(P[:,1]) * (R - B)).sum()
                                #loss2 = slen_reg * (torch.neg(torch.log(P[:,1])) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                                _sum = 0
                                #if steps % (division) == 0:
                                #    print(R-B)
                            elif (_idx % model.nlayers) == model.nlayers - 1:
                                loss2 = reg * (torch.neg(1 - _sum) * Variable(R - B)).sum()
                                #loss2 = reg * ((1 - _sum) * (R - B)).sum()
                                #loss2 = slen_reg * (torch.neg(_sum) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                            else:
                                #loss2 = slen_reg * (torch.neg(_sum + torch.log(P[:,1])) * (R - 0.5)).sum()
                                loss2 = reg * (torch.neg(_sum + torch.log(P[:,1])) * Variable(R - B)).sum()
                                #loss2 = reg * ((_sum + torch.log(P[:,1])) * (R - B)).sum()
                                #loss2 = slen_reg * (torch.neg(_sum + torch.log(P[:,1])) * (R/Bs[int(_idx / model.nlayers)] - 1)).sum()
                            _sum += torch.log(P[:,0])
                            loss += model.p_lambda2 * loss2
                            #diff2 = torch.norm(model.attn1[0].weight.data[0] - model.attn1[1].weight.data[0], 2)

                if model.rl:
                    #print('new rl process')
                    _sum = 0
                    if model.brl_even:
                        Bs = [[] for _ in range(model.nlayers)]
                        for _idx, R in enumerate(Rs):
                            Bs[_idx % model.nlayers].append(R)
                        for _idx in range(model.nlayers):
                            Bs[_idx] = sum(Bs[_idx]) / len(Bs[_idx])

                    #for _idx, D, W in zip(range(len(Ds)), Ds, Ws):
                    if model.rl_torch:
                        for _idx, R, actions in zip(range(len(Rs)), Rs, episodes):
                            if model.brl_even:
                                if model.pause:
                                    B = Bs[_idx % model.nlayers]
                                else:
                                    B = sum(Rs) / len(Rs)
                            else:
                                B = 0.5
                            for action in actions:
                                action.reinforce(model.p_lambda / model.rl_batch * (R - B))
                    else:
                        for _idx, R, W in zip(range(len(Rs)), Rs, Ws):
                            '''
                            loss += F.cross_entropy(D, target)
                            target_mat = torch.zeros(D.size()).float()
                            for i in range(target_mat.size(0)):
                                target_mat[i][target[i % target.size(0)].data[0]] = 1
                            target_mat = Variable(target_mat).cuda(model.device)
                            R = F.softmax(D) * target_mat
                            R = Variable(R.data).sum(1)
                            '''
                            if _idx % model.nlayers == 0:
                                _sum = 0
                            #reg = (1.0 / D.size(0))
                            #_sum = torch.log(W)
                            _sum += torch.log(W)
                            if model.brl_even:
                                if model.pause:
                                    B = Bs[_idx % model.nlayers]
                                else:
                                    B = sum(Rs)/len(Rs)
                                _loss = reg * (torch.neg(_sum) * Variable(R - B)).sum()
                                #_loss = reg * (_sum.clone() * (R - Bs[_idx % model.nlayers])).sum()
                                '''
                                R_list = [ '%.2f' % elem for elem in R.data.tolist()[0] ]
                                B_list = [ '%.2f' % elem for elem in B.data.tolist()[0] ]
                                print('R:', R.data.tolist())
                                print(_idx, _idx%model.nlayers, 'R:', R_list)
                                print(_idx, _idx%model.nlayers, 'B:', B_list)
                                '''
                                #print('R-B:', R - Bs[_idx % model.nlayers])
                            else:
                                _loss = reg * (torch.neg(_sum) * (R - 0.5)).sum()
                                #_loss = reg * (_sum.clone() * (R - 0.5)).sum()

                            if model.pause:
                                loss += model.p_lambda * _loss
                            else:
                                if (_idx % model.nlayers) == model.nlayers - 1:
                                    loss += model.p_lambda * _loss
            else:
                logit = model.forward(feature)
                if type(logit) is tuple:
                    logit, logit2 = logit
                if type(logit) is list:
                    if 'hinge' in dir(model) and model.hinge:
                        ls = [F.cross_entropy(l, target) for l in logit]
                    logit = logit[-1]
                if str(model.__class__)[8:-2] == 'Model.MemAttn':
                    logit = model.dropout(logit)
                loss = F.cross_entropy(logit, target)
                if 'hinge' in dir(model) and model.hinge:
                    for _idx in range(1, len(ls)):
                        loss += model.beta * max(ls[_idx] - ls[_idx-1], 0)

            if 'gru' in dir(model):
                old_q = model.attn.weight.data[0].cpu()
                old_lstm = model.lstm.weight_ih_l0.data[0].cpu()
                old_embed = model.embed.weight.data[0].cpu()
                old_mlp1 = model.mlp1.weight.data[0].cpu()
                old_gru = model.gru.weight_ih.data[0].cpu()
                old_P = model.P.weight.data[0].cpu()
                #old_gru = model.gru.weight.data[0].cpu()

            if'rl_torch' in dir(model) and model.rl_torch:
                #final_nodes = [loss.cpu()] + [actions[0].cpu()]
                final_nodes = [loss] + actions
                #final_nodes = [loss]
                gradients = [torch.ones(1).cuda(model.device)] + [None] * len(actions)
                #gradients = [torch.ones(1).cuda(model.device)]

                autograd.backward(final_nodes, gradients)
                '''
                try:
                    autograd.backward(final_nodes, gradients)
                except Exception, _data:
                    print _data
                    from IPython import embed
                    embed()
                '''
            else:
                loss.backward()
            if steps % division == 0:
                _max = max([p.grad.data.max() for p in model.parameters() if p.grad is not None])

            if 'clip' in dir(model) and model.clip > 0.0:
                torch.nn.utils.clip_grad_norm(model.parameters(), model.clip)

            if steps % division == 0:
                __max = max([p.grad.data.max() for p in model.parameters() if p.grad is not None])
                print('_max:%.3f\t__max:%.3f'%(_max, __max))

            max_grad = max(max_grad, max([p.grad.data.max() for p in model.parameters() if p.grad is not None]))
            if steps % division == 0:
                print('******** max gradient:', max_grad, ' **********')

            def weight_gradient_error(w):
                _max = w.grad.data.max()
                _min = w.grad.data.min()
                if math.isnan(_max) or math.isnan(_min) or (_max < _min) or (abs(_max) > 1000):
                    return True
                else:
                    return False

            if 'gru' in dir(model):
                exit_flag = False
                if model.gru.weight_ih.grad is not None:
                    if weight_gradient_error(model.gru.weight_ih):
                        _max = model.gru.weight_ih.grad.data.max()
                        _min = model.gru.weight_ih.grad.data.min()
                        print('******** Weight Gradient Error *********')
                        print('GRU gradient', _max, _min, _max < _min, abs(_max) > 1000)
                        exit_flag = True
                if weight_gradient_error(model.lstm.weight_ih_l0):
                    print('******** Weight Gradient Error *********')
                    _max = model.lstm.weight_ih_l0.grad.data.max()
                    _min = model.lstm.weight_ih_l0.grad.data.min()
                    print('LSTM gradient', _max, _min, _max < _min, abs(_max) > 1000)
                    exit_flag = True
                if exit_flag:
                    exit()

            optimizer.step()

            if 'gru' in dir(model):
                q_max = model.attn.weight.data.max()
                lstm_max = model.lstm.weight_ih_l0.data.max()
                embed_max = model.embed.weight.data.max()
                mlp1_max = model.mlp1.weight.data.max()
                P_max = model.P.weight.data.max()
                gru_max = model.gru.weight_ih.data.max()
                #gru_max = model.gru.weight.data.max()
            '''
            print('--------- max_grad', max_grad, ' ----------')
            print('GRU max', gru_max, math.isnan(gru_max))
            print('q max', q_max, math.isnan(q_max))
            print('lstm max', lstm_max, math.isnan(lstm_max))
            print('embed max', embed_max, math.isnan(embed_max))
            print('mlp1 max', mlp1_max, math.isnan(mlp1_max))
            print('P max', P_max, math.isnan(P_max))
            '''


            '''
            for p in model.parameters():
                if p.grad is None:
                    continue
                p.data.add_(lr, p.grad.data)
            '''
            '''
            if steps % (division/2) == 0 and 'epsilon' in dir(model) and model.epsilon:
                print('loss: %.3f\t attn: %.3f\t lstm: %.3f\t embed: %.3f\t mlp1: %.3f' % (loss2.data.cpu().sum(), sum(old_q - model.attn.weight.data[0].cpu()), sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()), sum(old_embed - model.embed.weight.data[0].cpu()), sum(old_mlp1 - model.mlp1.weight.data[0].cpu())))
            '''

            if steps % division == 0:
                if 'gru' in dir(model):
                    if 'nlayers' in dir(model):
                        print('nlayers: %d\t rl_batch: %d' % (model.nlayers, model.rl_batch))
                    if 'epsilon' in dir(model):
                        print('epsilon: %.2f\t p_lambda: %.4f\t p_lambda2: %.4f' % (model.epsilon, model.p_lambda, model.p_lambda2))
                    print('params:', model.params)
                    print('attn: %.3f\t lstm: %.3f\t embed: %.3f' % (sum(old_q - model.attn.weight.data[0].cpu()), sum(old_lstm - model.lstm.weight_ih_l0.data[0].cpu()), sum(old_embed - model.embed.weight.data[0].cpu())))
                    print('gru: %.3f\t P: %.3f\t' % (sum(old_gru - model.gru.weight_ih.data[0].cpu()), sum(old_P - model.P.weight.data[0].cpu())))
                    '''
                    print('gru: %.2f \t attn: %.2f\t lstm: %.2f\t embed: %.2f' % (sum(model.gru.weight_ih.data[0].cpu()), sum(model.attn.weight.data[0].cpu()), sum(model.lstm.weight_ih_l0.data[0].cpu()), sum(model.embed.weight.data[0].cpu())))
                    '''

                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                ans = torch.max(logit, 1)[1].view(target.size()).data
                ans_dict = list2dict(ans.cpu())
                print(ans_dict)

                accuracy = 1.0 * corrects / target.size(0) * 100.0
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, loss.data[0], accuracy, corrects, target.size(0)))
                print(' ')
                _run.info['train_loss'].append(loss.data[0])
                _run.info['train_accuracy'].append(accuracy)

                if 'test_rl' in dir(model) and model.test_rl:
                    model.rl = True
                acc = evaluate(model, test_iter, use_cuda, _run, max_acc, stop=stop, dtype='test')
                if 'test_rl' in dir(model) and model.test_rl:
                    model.rl = False
                #evaluate(model, dev_iter, use_cuda, _run, max_acc, data_set=data_set, stop=stop, dtype='dev')
                if save_model and (max_acc < acc) and acc > 40.0:
                    f_name = '%.2f.torch' % (acc)
                    if 'search_n' in dir(model) and model.search_n > 1:
                        f_name = '%.d_' % (model.search_n) + f_name
                    if 'epsilon' in dir(model) and model.epsilon:
                        f_name = '%.2f_%.2f_' % (model.epsilon, model.p_lambda) + f_name
                    if 'rl_batch' in dir(model):
                        f_name = '%d_' % (model.rl_batch) + f_name
                    if 'nlayers' in dir(model):
                        f_name = '%d_' % (model.nlayers) + f_name
                    if 'rl' in dir(model) and model.rl:
                        f_name = 'rl_' + f_name
                    if 'pause' in dir(model) and model.pause:
                        f_name = 'pause_' + f_name
                    if lock_params:
                        f_name = 'lock' + '_' + f_name
                    for d in data_set:
                        d = d.split('/')
                        if len(d) > 1:
                            d = d[1]
                        else:
                            d = d[0]
                        f_name = d + '_' + f_name
                    f_name = str(model.__class__)[8:-2] + '_' + f_name

                    dont_save = False
                    files = os.listdir('../model/')
                    for f in files:
                        if f.startswith(f_name[:-12]):
                            if float(f[-11:-6]) < acc:
                                os.remove('../model/' + f)
                            else:
                                dont_save = True
                    if dont_save is False:
                        torch.save(model.cpu(), '../model/' + f_name)
                        model.cuda(model.device)
                    print('save model', f_name)
                if max_acc < acc:
                    max_acc = acc


def evaluate_tagger(model, sentences, dictionaries, lower):
    """
    Evaluate current model using CoNLL script.
    """
    output_path = '../tmp/evaluate.txt'
    scores_path = '../tmp/score.txt'
    eval_script = '../tmp/conlleval'
    with codecs.open(output_path, 'w', 'utf8') as f:
       for index in range(len(sentences)):
            #input sentence
            input_words = autograd.Variable(torch.LongTensor(sentences[index]['words']))
            
            #calculate the tag score
            if lower == 1:
            # We first convert it to one-hot, then input
                input_caps = torch.LongTensor(sentences[index]['caps'])
                tags = model.get_tags(input_words = input_words, input_caps = input_caps)
            else:
                tags = model.get_tags(input_words = input_words)

            #tags = model.get_tags(sentence_in)
            # get predict tags
            predict_tags = [dictionaries['id_to_tag'][tag] if (tag in dictionaries['id_to_tag']) else 'START_STOP' for tag in tags]

            # get true tags
            true_tags = [dictionaries['id_to_tag'][tag] for tag in sentences[index]['tags']]

            # write words pos true_tag predict_tag into a file
            
            for word, pos, true_tag, predict_tag in zip(sentences[index]['str_words'], sentences[index]['pos'], true_tags, predict_tags):
            #for word, pos, true_tag, predict_tag in zip(sentences[index]['str_words'], 'pos', true_tags, predict_tags):
                f.write('%s %s %s %s\n' % (word, pos ,true_tag, predict_tag))
            f.write('\n')
            

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    result={
       'accuracy' : float(eval_lines[1].strip().split()[1][:-2]),
       'precision': float(eval_lines[1].strip().split()[3][:-2]),
       'recall': float(eval_lines[1].strip().split()[5][:-2]),
       'FB1': float(eval_lines[1].strip().split()[7])
    }
    print(eval_lines[1])
    return result

def plot_result(accuracys, precisions, recalls, FB1s):
    plt.figure()
    plt.plot(accuracys,"g-",label="accuracy")
    plt.plot(precisions,"r-.",label="precision")
    plt.plot(recalls,"m-.",label="recalls")
    plt.plot(FB1s,"k-.",label="FB1s")

    plt.xlabel("epoches")
    plt.ylabel("%")
    plt.title("CONLL2000 dataset")

    plt.grid(True)
    plt.legend()
    plt.show()