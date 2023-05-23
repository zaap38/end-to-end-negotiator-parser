
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
    
    
    def tensor_geter_x(self):
            return self.train_x
    def tensor_geter_y(self):
        return self.train_y


    @staticmethod
    def raw_to_float(d):
        x = []
        y = []

        for i in range(3):
            x.append(d["input"]["amount"][i] / 10)
        for i in range(6):
            if i < len(d["conv"]):
                agent_id = 0 if d["conv"][i]["author"] == "YOU" else 1
                for j in range(3):
                    # print(d["conv"][i]["state"])
                    x.append(d["conv"][i]["state"][j] / 10)
            else:
                for _ in range(3):
                    x.append(0)

        for j in range(3):
            y.append(d["input"]['0'][j] / 10)

        """
        # reversed situation -------------------------------------
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
            # ry.append((d["input"]["amount"][item] - d["output"][0][item]) / 10)
            ry.append((d["input"]["amount"][item] - d["output"][0][item]) / d["input"]["amount"][item])
        """
        rx = x
        ry = y

        return x, y, rx, ry


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size  = input_size
        self.output_size = output_size
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    # forward pass 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def reshape_labels(x, y):
    for i in range(len(y)):
        y[i] = reshape_label(x[i], y[i])
    return y

def reshape_label(x, y):
    amounts = x[6:]
    for i in range(len(y)):
        y[i] = y[i] * amounts[i]
    return y

def reshape_pred(x, pred):
    amounts = x[6:]
    for i in range(len(pred)):
        pred[i] = pred[i] * amounts[i]
    return pred

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def test_loss(ins, labels):
    ins = torch.from_numpy(np.array(ins))
    labels = torch.from_numpy(np.array(labels))
    test_data = TensorDataset(ins, labels)
    test_dataloader = DataLoader(test_data,  batch_size=len(test_data))
    cpt = 0
    loss = 0
    
    rand_display = rd.randint(0, 200)

    for dl in test_dataloader:
        test_x, test_y = dl
        optimizer.zero_grad() 

        b_input = test_x
        b_labels = test_y
        b_input = b_input.to(torch.float32)
        b_labels = b_labels.to(torch.float32)

        y_pred = net(b_input)
        y_pred = torch.nn.ReLU()(y_pred)
        y_pred = torch.clamp(y_pred, min=0, max=1)
        loss += loss_function(y_pred, b_labels)


        for j in range(len(b_input)):

            cpt += 1
            # label = cp.deepcopy(reshape_label(b_input[j], b_labels[j]))
            # print(b_labels, y_pred)
            if cpt == rand_display:
                """states = split(b_input[j].tolist(), 6)
                you_bool = True
                print(states)
                for k in range(len(states)):
                    prefix = "YOU:" if you_bool else "THEM:"
                    you_bool = not you_bool
                    print(prefix, states[k])"""
                print("States:", b_input[j])
                print("Pred:", y_pred[j])
                print("Label:", b_labels[j])

        """print("Rwd:", round(reward / cpt, 2),
              "Avg_rwd:", round(avg_reward / cpt, 2),
              "Label_rwd:", round(label_reward / cpt, 2),
              "Dist:", round(manhattan / cpt, 2))"""

        return loss


if __name__ == "__main__":

    net = Net(21, 3)
    loss_function = nn.MSELoss() # custom_loss_no_negative  # nn.MSELoss()
    optimizer=torch.optim.SGD(net.parameters(), lr=0.01)
    bs = 20

    data = Data("./src/data/parsed.json")
    list_train_x = data.tensor_geter_x()
    list_train_y = data.tensor_geter_y()
    input_x = torch.from_numpy(np.array(list_train_x, dtype=np.float32))
    input_y = torch.from_numpy(np.array(list_train_y, dtype=np.float32))
    # input_y = reshape_labels(input_x, input_y)

    train_data = TensorDataset(input_x, input_y)
    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,  batch_size=64)

    for epoch in range(10000):

        # keep track of the loss, error and the number of batches 
        current_loss  = 0
        current_error = 0
        num_batches = 0 
        acc_flag = 0
        preds = []
        
        avg_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            b_input,  b_labels = batch
            b_input = b_input.to(torch.float32)
            b_labels = b_labels.to(torch.float32)
            y_pred = net(b_input)
            y_pred = torch.nn.ReLU()(y_pred)
            y_pred = torch.clamp(y_pred, min=0, max=1)
            loss = loss_function(y_pred, b_labels)
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
        
        if (epoch + 1) % 100 == 0:
            avg_loss_epoch = avg_loss / len(train_dataloader)
            loss_test = test_loss(data.test_x, data.test_y)
            print(epoch + 1, avg_loss_epoch, loss_test)
