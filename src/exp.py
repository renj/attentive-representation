from __future__ import unicode_literals, print_function, division
#import optparse
import os
from collections import OrderedDict
from loader import prepare_dictionaries, prepare_dataset, get_word_embedding_matrix 
import LstmModel
#import LstmCrfModel
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
from loader import CAP_DIM
from sacred import Experiment
#from sacred.observers import MongoObserver
import time
import logging


def init():
    log_id = str(int(time.time()*10)%(60*60*24*365*10))+str(os.getpid())
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
    #ex.observers.append(MongoObserver.create(url='10.60.43.110:27017', db_name='DNLP'))
    #ex.observers.append(MongoObserver.create(url='127.0.0.1:27017', db_name='nTrajMap'))
    return ex, logger

#init()

#@ex.config
def cfg():
    ex_name = 'test'
    dataset = 'sst2'
    lower = 1 #Lowercase words (this will not affect character inputs)
    zeros = 1 #Replace digits with 0
    pre_emb = '../embedding/glove.6B.100d.txt'
    vocab_size = 8000
    embedding_dim = 100
    hidden_dim = 100
    decode_method = 'viterbi'
    loss_function = 'likelihood'
    clip = 5.0
    n_epochs = 100 # number of epochs over the training set


#@ex.automain
#def main(ex_name, dataset, lower, zeros, pre_emb, vocab_size, embedding_dim, hidden_dim, decode_method, loss_function, clip, n_epochs, _log, _run):
#    _run.info['ex_name'] = ex_name
def main():
    dataset = 'sst2'
    ex_name = 'test'
    lower = 1 #Lowercase words (this will not affect character inputs)
    zeros = 1 #Replace digits with 0
    pre_emb = '../embedding/glove.6B.100d.txt'
    vocab_size = 8000
    embedding_dim = 100
    hidden_dim = 100
    decode_method = 'viterbi'
    loss_function = 'likelihood'
    clip = 5.0
    n_epochs = 100 # number of epochs over the training set

    if dataset == 'conll':
        train_path = '../data/conll2000.train.txt'
        dev_path = '../data/conll2000.test.txt'
    elif dataset == 'sst1':
        train_path = 'SST1_TRAIN'
        dev_path = 'SST1_DEV'
    elif dataset == 'sst2':
        train_path = 'SST2_TRAIN'
        dev_path = 'SST2_DEV'

    Parse_parameters = OrderedDict()
    Parse_parameters['lower'] = lower == 1
    Parse_parameters['zeros'] = zeros == 1
    Parse_parameters['pre_emb'] = pre_emb
    Parse_parameters['train'] = train_path
    Parse_parameters['development'] = dev_path
    Parse_parameters['vocab_size'] = vocab_size


    #assert os.path.isfile(train_path)
    #assert os.path.isfile(dev_path)
    if pre_emb:
        assert embedding_dim in [50, 100, 200, 300]
        assert lower == 1

    dictionaries = prepare_dictionaries(Parse_parameters)
    tagset_size = len(dictionaries['tag_to_id'])


    # Model parameters
    Model_parameters = OrderedDict()
    Model_parameters['vocab_size'] = vocab_size
    Model_parameters['embedding_dim'] = embedding_dim
    Model_parameters['hidden_dim'] = hidden_dim
    Model_parameters['tagset_size'] = tagset_size
    Model_parameters['lower'] = lower == 1
    Model_parameters['decode_method'] = decode_method
    Model_parameters['loss_function'] = loss_function


    if train_path.startswith('SST'):
        model = LstmModel.StackLSTMClassifier(Model_parameters)
        #model = LstmModel.LSTMClassifier(Model_parameters)
    else:
        model = LstmModel.LSTMTagger(Model_parameters)
        #model = LstmCrfModel.BiLSTM_CRF(Model_parameters)
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    train_data = prepare_dataset(Parse_parameters, train_path, dictionaries)
    dev_data = prepare_dataset(Parse_parameters, dev_path, dictionaries)


    # If using pre-train, we need to initialize word-embedding layer
    if pre_emb:
          print("Initialize the word-embedding layer")
          initial_matrix = get_word_embedding_matrix(dictionaries['word_to_id'], pre_emb, embedding_dim)
          model.init_word_embedding(initial_matrix)
    
    Division = 10
    accuracys = []
    precisions = []
    recalls = []
    FB1s =[]

    Division = 10
    for epoch in range(n_epochs): # again, normally you would NOT do 300 epochs, it is toy data
        epoch_costs = []
        print("Starting epoch %i..." % (epoch))
        #_run.info['result'] = {}
        for i, index in enumerate(np.random.permutation(len(train_data))):
            #print(i, index, len(train_data))
            if i % int(len(train_data)/Division) == 0:
                # evaluate
                if train_path.startswith('SST'):
                    eval_result = utils.evaluate(model, dev_data, dictionaries, lower)
                    print('%d, %.4f'%(i, eval_result*100))
                else:
                    utils.evaluate_tagger(model, dev_data, dictionaries, lower)
                    torch.save(model, '../data/conll_model.pytorch')
                #_run.info['result'][str(i)] = eval_result

            model.zero_grad()

            input_words = autograd.Variable(torch.LongTensor(train_data[index]['words']))
            targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

            if lower == 1:
                input_caps = torch.LongTensor(train_data[index]['caps'])
                loss = model.get_loss(targets, input_words = input_words, input_caps = input_caps)
            else:
                loss = model.get_loss(targets, input_words = input_words)
            
            epoch_costs.append(loss.data.numpy())
            loss.backward()
            '''
            if i % int(len(train_data)/Division) == 0:
                for p in model.parameters():
                    print(p.data.mean()),
                print(',')
            '''
            for p in model.parameters():
                p.data.add_(-0.01, p.grad.data)
            #nn.utils.clip_grad_norm(model.parameters(), clip)
            #optimizer.step()

        print("Epoch %i, cost average: %f" % (epoch, np.mean(epoch_costs)))


    '''
    for epoch in range(n_epochs): # again, normally you would NOT do 300 epochs, it is toy data
        epoch_costs = []


        print("Starting epoch %i..." % (epoch))
        #_run.info['result'] = {}
        for i, index in enumerate(np.random.permutation(len(train_data))):
            if i %(len(train_data)/Division) == 0:
                # evaluate
                eval_result = evaluate(model, dev_data, dictionaries, lower)
                accuracys.append(eval_result['accuracy'])
                precisions.append(eval_result['precision'])
                recalls.append(eval_result['recall'])
                FB1s.append(eval_result['FB1'])
                #_run.info['result'][str(i)] = eval_result
                #break


            # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
            # before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into Variables
            # of word indices.
            input_words = autograd.Variable(torch.LongTensor(train_data[index]['words']))
            targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

            # Step 3. Run our forward pass. We combine this step with get_loss function
            #tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by calling
            # first check whether we use lower this parameter
            if lower == 1:
                # We first convert it to one-hot, then input
                input_caps = torch.LongTensor(train_data[index]['caps'])
                loss = model.get_loss(targets, input_words = input_words, input_caps = input_caps)
            else:
                loss = model.get_loss(targets, input_words = input_words)

            epoch_costs.append(loss.data.numpy())
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
                
        print("Epoch %i, cost average: %f" % (epoch, np.mean(epoch_costs)))


    # Final Evaluation after training
    eval_result = evaluate(model, dev_data, dictionaries, lower)
    accuracys.append(eval_result['accuracy'])
    precisions.append(eval_result['precision'])
    recalls.append(eval_result['recall'])
    FB1s.append(eval_result['FB1'])
    '''

    '''
    _run.info['accuracy'] = accuracys
    _run.info['precision'] = precisions
    _run.info['recall'] = recalls
    _run.info['FB1'] = FB1s
    '''

    #print("Plot final result")
    #utils.plot_result(accuracys, precisions, recalls, FB1s)

main()