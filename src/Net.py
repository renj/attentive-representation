__author__ = 'chen'
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio


def getData():
    labelName = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    trainX = np.ones([11314, 2000], dtype = np.float32)
    trainY = np.ones([11314], dtype = np.int64)
    testX = np.ones([7532, 2000], dtype = np.float32)
    testY = np.ones([7532], dtype = np.int64)

    infile = open('20news-bydate_commonest_train_count2000.txt')
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        trainY[count] = int(line[0])
        trainX[count,:] = [(float(x)) for x in line[1:]]
        #trainX[count,:] /= trainX[count,:].sum() #normalization, bad
        count += 1
    print(count)
    infile.close()

    infile = open('20news-bydate_commonest_test_count2000.txt')
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        testY[count] = int(line[0])
        testX[count,:] = [(float(x)) for x in line[1:]]
        #testX[count,:] /= testX[count,:].sum() ##normalization, bad
        count += 1
    print(count)
    infile.close()

    trainX = trainX.reshape([11314,1,2000,1])
    testX = testX.reshape([7532,1,2000,1])

    return labelName, trainX, trainY, testX, testY


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2000, 100, bias=True)
        self.fc2 = nn.Linear(100, 20)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def applyMask(self, mask):
        weight = list(self.parameters())[0].data
        weight.mul_(mask)



labelName, trainX, trainY, testX, testY = getData()
batchsize = 100
batchnum = trainX.shape[0] / batchsize
randgen = np.random.RandomState(11)
randomTestIdx = randgen.choice(7532, 1000, replace=False)
randomTestX = torch.from_numpy(testX[randomTestIdx])
randomTestY = torch.from_numpy(testY[randomTestIdx])
randomTrainIdx = randgen.choice(11314, 1000, replace=False)
randomTrainX = torch.from_numpy(trainX[randomTrainIdx])
randomTrainY = torch.from_numpy(trainY[randomTrainIdx])


#mask = torch.from_numpy( sio.loadmat('mask_commonest2000_soft_cond_PEM_565hid_400.mat')['mask'].astype(np.float32).T )

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)#0.001

best = 0
for epoch in range(100):
    running_loss = 0.0
    for batch in range(batchnum):
        batchX = torch.from_numpy( trainX[ batch * batchsize: batch * batchsize + batchsize] )
        batchX = Variable(batchX)

        batchY = torch.from_numpy( trainY[ batch * batchsize: batch * batchsize + batchsize] )
        batchY = Variable(batchY)

        optimizer.zero_grad()
        #input = Variable(torch.randn(3,1,10,1))
        #target = Variable(torch.from_numpy(np.array([1,2,3])))

        #forward, backward, update
        output = net(batchX)
        loss = criterion(output, batchY)
        loss.backward()
        optimizer.step()
        #net.applyMask(mask)############################################

        # print statistics
        running_loss += loss.data[0]
        if batch % 10 == 9:
            print('[%d, %5d] training batch loss: %.3f' % (epoch+1, batch+1, running_loss / 10))
            running_loss = 0.0

            output = net(Variable(randomTrainX))
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == randomTrainY).sum()
            print('Accuracy of the network on the random training docs: %.1f %%' % (
                100.0 * correct / predicted.size()[0]))


            output = net(Variable(randomTestX))
            loss = criterion(output, Variable(randomTestY))
            print('random test loss: %.3f' % (loss.data[0]))

            _, predicted = torch.max(output.data, 1)
            correct = (predicted == randomTestY).sum()

            print('Accuracy of the network on the random test docs: %.1f %%' % (
                100.0 * correct / predicted.size()[0]))

            if 100.0 * correct / predicted.size()[0] > best:
                best = 100.0 * correct / predicted.size()[0]
            print('Best test Accuracy: %.1f %%' % (best))
