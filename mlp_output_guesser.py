#%%
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
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#%%
class Data:

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)

        train_test_ratio=0.7
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        for d in self.data:
            x, y, rx, ry = self.raw_to_float(d)  # self.raw_to_float_substract(d)  # self.raw_to_float(d)  # self.raw_to_onehot(d)
            if rd.random() < train_test_ratio:
                self.train_x.append(x)
                self.train_y.append(y)
                self.train_x.append(rx)
                self.train_y.append(ry)
            else:
                self.test_x.append(x)
                self.test_y.append(y)
                self.test_x.append(rx)
                self.test_y.append(ry)


    @staticmethod
    def raw_to_float(d):
        x = []
        y = []

        # input
        for part in range(2):
            for item in range(3):
                x.append(d["input"][str(part)][item] / 10)
        for item in range(3):
            x.append(d["input"]["amount"][item] / 10)

        # output
        for item in range(3):
            y.append(d["output"][0][item] / 10)

        # reversed situation
        rx = []
        ry = []
        # input
        for part in range(2):
            for item in range(3):
                rx.append(d["input"][str(1 - part)][item] / 10)
        for item in range(3):
            rx.append(d["input"]["amount"][item] / 10)

        # output
        for item in range(3):
            ry.append((d["input"]["amount"][item] - d["output"][0][item]) / 10)
        
        return x, y, rx, ry
    
    @staticmethod
    def raw_to_float_substract(d):
        x = []
        y = []

        # input
        for item in range(3):
            x.append((d["input"]["0"][item] - d["input"]["1"][item]) / 10)
        for item in range(3):
            x.append(d["input"]["amount"][item] / 10)

        # output
        for item in range(3):
            y.append(d["output"][0][item] / 10)

        # reversed situation
        rx = []
        ry = []
        # input
        for item in range(3):
            rx.append((d["input"]["1"][item] - d["input"]["0"][item]) / 10)
        for item in range(3):
            rx.append(d["input"]["amount"][item] / 10)

        # output
        for item in range(3):
            ry.append((d["input"]["amount"][item] - d["output"][0][item]) / 10)

        return x, y, rx, ry

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

        # return torch.as_tensor(onehot_in, dtype=torch.float32), torch.as_tensor(onehot_out, dtype=torch.float32)
        return onehot_in , onehot_out
    
    def tensor_geter_x(self):
        return self.train_x
    def tensor_geter_y(self):
        return self.train_y
#%%
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        # super(Net, self).__init__()

        # self.layers = []
        # for i, s in enumerate(sizes):
        #     if i > 0:
        #         self.layers.append(nn.Linear(sizes[i - 1], s, bias=True))
        # self.param = nn.ParameterList(self.layers)
        self.input_size  = input_size
        self.output_size = output_size
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    # forward pass 
    def forward(self, x):
        # y = cp.deepcopy(x)
        # for l in self.layers:
        #     y = l(F.relu(y))
        
        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=self.output_size)
        
        return x

#%%



#%%
data = Data("./src/data/parsed.json")
#%%
list_train_x = data.tensor_geter_x()
list_train_y = data.tensor_geter_y()
input_x = torch.from_numpy(np.array(list_train_x, dtype=np.float32))
input_y = torch.from_numpy(np.array(list_train_y, dtype=np.float32))
# input_x
#%%
input_x.shape
#%%

train_data = TensorDataset(input_x, input_y)
# train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,  batch_size=64)


#%%    
net = Net(9, 3)
cross_entropy = nn.MSELoss()# nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(), lr=0.01 )
bs = 20
#%%
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test_loss(ins, labels):
    ins = torch.from_numpy(np.array(ins))
    labels = torch.from_numpy(np.array(labels))
    test_data = TensorDataset(ins, labels)
    test_dataloader = DataLoader(test_data,  batch_size=len(test_data))
    for dl in test_dataloader:
        test_x, test_y = dl
        optimizer.zero_grad() 
        # view as a column vector
        # batch = tuple(t.cuda() for t in batch)
        b_input = test_x
        b_labels = test_y
        b_input = b_input.to(torch.float32)
        b_labels = b_labels.to(torch.float32)
        # print(b_labels)
        # start recording gradient
        # inputs.requires_grad_()
        # a forward pass on the network 
        y_pred = net(b_input)
        preds.append
        loss = cross_entropy(y_pred, b_labels)
        display_index = rd.randint(0, 200)

        total = 0
        cpt = 0
        for i, e in enumerate(y_pred):
            for j in range(3):
                total += abs(y_pred[i][j] - b_labels[i][j])
                cpt += 1
        total = total / cpt

        difference = []
        for i in range(3):
            difference.append(abs(y_pred[display_index].tolist()[i] - b_labels[display_index][i]))
        pred = y_pred[display_index].tolist()
        for i in range(len(pred)):
            pred[i] = round(pred[i], 1)
        print("========== Preview ==========")
        print("Input:", b_input[display_index])
        print("Output:", b_labels[display_index])
        print("Pred:", pred)
        print("Total:", total)
        # print(loss)
        return loss


#%%
for epoch in range(50000):

    # keep track of the loss, error and the number of batches 
    current_loss  = 0
    current_error = 0
    num_batches = 0 
    acc_flag = 0
    preds = []
    
    avg_loss = 0
    
    for batch in train_dataloader:
        
        # ---> forward pass 
        # reset the gradient 
        optimizer.zero_grad() 
        # view as a column vector
        # batch = tuple(t.cuda() for t in batch)
        b_input,  b_labels = batch
        b_input = b_input.to(torch.float32)
        b_labels = b_labels.to(torch.float32)
        # print(b_labels)
        # start recording gradient
        # inputs.requires_grad_()
        # a forward pass on the network 
        y_pred = net(b_input)
        preds.append
        loss = cross_entropy(y_pred, b_labels)
        # print(loss)
        avg_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
        
        logits = y_pred.detach().cpu().numpy()
        b_labels = b_labels.detach().cpu().numpy()
        accuracy = flat_accuracy(logits, b_labels)
        acc_flag+=accuracy
    
    if (epoch + 1) % 500 == 0:
        avg_loss_epoch = avg_loss / len(train_dataloader)
        loss_test = test_loss(data.test_x, data.test_y)
        print(epoch + 1, avg_loss_epoch, loss_test)
        
    # print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # print("Train acc: {}".format(acc_flag))
        # acc = (torch.argmax(y_pred, 1) == torch.argmax(b_labels, 1)).float().mean()
        # print(f"Epoch {epoch} validation: Cross-entropy={float(loss)}, Accuracy={float(acc)}")
        # compute the cross entropy loss 
        # loss = cross_entropy(y_hat, label_minibatch)
        # # <--- back propagation
        # loss.backward()
        # # update the net parameters 
        # optimizer.step() 
        # # get the current error 
        # error = utils.get_error( y_hat.detach() , label_minibatch)
        
        # # update the running stats. 
        # num_batches += 1 
        # current_loss += loss.detach().item()
        # current_error += error.item()
    
    # average loss/error over minibatches for the current epoch 
    # avg_loss = current_loss / num_batches
    # avg_error = current_error / num_batches
    # elapsed_time = time.time() - start 

    # every 10 epochs display stats.
    # if epoch % 10 == 0: 
    #     print('The loss for epoch number ' + str(epoch) + ' = ' + str(avg_loss))
    #     print('The error for epoch number ' + str(epoch) + ' = ' + str(avg_error))

    #     # evaluate error on test set
    #     error_on_test_set()
# %%
