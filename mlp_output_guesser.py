import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import time 
import utils 
import numpy as np

import copy as cp
import json
import random as rd


class Net(nn.Module):
    def __init__(self, sizes):
        super(Net, self).__init__()

        self.layers = []
        for i, s in enumerate(sizes):
            if i > 0:
                self.layers.append(nn.Linear(sizes[i - 1], s, bias=True))
        self.param = nn.ParameterList(self.layers)
    
    # forward pass 
    def forward(self, x):
        y = cp.deepcopy(x)
        for l in self.layers:
            y = l(F.relu(y))
        return y


class Data:

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)

        train_test_ratio=0.7
        self.train = []
        self.test = []
        for d in self.data:
            x, y = self.raw_to_onehot(d)
            if rd.random() < train_test_ratio:
                self.train.append((x, y))
            else:
                self.test.append((x, y))

    @staticmethod
    def raw_to_onehot(d):
        onehot_in = []  # p1 rewards ; p2 rewards ; amounts
        for part in range(2):
            for item in range(3):
                for i in range(11):
                    onehot_in.append(1 if i == d["input"][str(part)][item] else 0)
        for item in range(3):
            for i in range(11):
                onehot_in.append(1 if i == d["input"]["amount"][item] else 0)

        onehot_out = []
        for item in range(3):
            for i in range(11):
                onehot_out.append(1 if i == d["output"][0][item] else 0)

        return torch.as_tensor(np.array(onehot_in), dtype=torch.float32), torch.as_tensor(np.array(onehot_out), dtype=torch.float32)


def error_on_test_set():
    current_error = 0
    num_batches = 0

    # evaluate error at every minibatch 
    for batch in range(0, 1000, bs):
        
        # extract test minibatch 
        test_mini_batch = test_set[batch:batch + bs]
        input_test = test_mini_batch.view(bs, 99)
        # extract corresponding labels 
        label_minibatch = test_targets[batch:batch+bs]
        # one forward pass over the network 
        y_hat = net(input_test)
        # compute error 
        error = utils.get_error(y_hat, label_minibatch)
        
        # update stats. 
        num_batches += 1
        current_error += error.item()
    
    avg_error = current_error / num_batches
    print('The error on test set = ' + str(avg_error * 100) + '%')


if __name__ == "__main__":
    # INPUT: reward => [0;10] * 3 obj * 2 participants ; amount => [0;10] * 3 obj : total_input = 99
    # OUTPUT: [0;10] * 3 obj : total_output = 33
    data = Data("./src/data/parsed.json")
    net = Net([99, 128, 64, 33])
    cross_entropy = nn.CrossEntropyLoss()
    print(len(list(net.parameters())))
    optimizer=torch.optim.SGD(net.parameters(), lr=0.01 )
    bs = 20
    print(net)

    start = time.time()

    input_tmp = []
    targets_tmp = []
    for i in range(len(data.test)):
        input_tmp.append(data.test[i][0])
        targets_tmp.append(data.test[i][1])
    test_set = np.array(input_tmp)
    test_targets = np.array(targets_tmp)

    input_tmp = []
    targets_tmp = []
    for i in range(len(data.train)):
        input_tmp.append(data.train[i][0])
        targets_tmp.append(data.train[i][1])
    train_set = np.array(input_tmp)
    train_targets = np.array(targets_tmp)

    for epoch in range(60):
    
        # keep track of the loss, error and the number of batches 
        current_loss  = 0
        current_error = 0
        num_batches = 0 

        # shuffle the data indices 
        shuffled_indices = torch.randperm(2055)
        for batch in range(0, 2055, bs):
            # extract mini-batches 
            shuffled_batch = shuffled_indices[batch:batch + bs]
            train_minibatch = train_set[shuffled_batch]
            label_minibatch = train_targets[shuffled_batch]
            
            # ---> forward pass 
            # reset the gradient 
            optimizer.zero_grad() 
            # view as a column vector
            inputs = train_minibatch.view(bs, 99)
            # start recording gradient
            inputs.requires_grad_()
            # a forward pass on the network 
            y_hat = net(inputs)
            # compute the cross entropy loss 
            loss = cross_entropy(y_hat, label_minibatch)
            # <--- back propagation
            loss.backward()
            # update the net parameters 
            optimizer.step() 
            # get the current error 
            error = utils.get_error( y_hat.detach() , label_minibatch)
            
            # update the running stats. 
            num_batches += 1 
            current_loss += loss.detach().item()
            current_error += error.item()
        
        # average loss/error over minibatches for the current epoch 
        avg_loss = current_loss / num_batches
        avg_error = current_error / num_batches
        elapsed_time = time.time() - start 

        # every 10 epochs display stats.
        if epoch % 10 == 0: 
            print('The loss for epoch number ' + str(epoch) + ' = ' + str(avg_loss))
            print('The error for epoch number ' + str(epoch) + ' = ' + str(avg_error))

            # evaluate error on test set
            error_on_test_set()