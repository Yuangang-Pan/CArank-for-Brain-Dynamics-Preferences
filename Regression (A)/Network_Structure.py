#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:20:19 2019

@author: yuapan
"""
import torch
import torch.nn as nn
import numpy as np
import random
import copy
random.seed(2019)
theta = 0.65
## construct the neural network structure
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        x = self.dout(x)
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    ## the ACC
def MV_Baseline(Num_Channel, Z_score, Pair):
    delta_Z = Z_score[Pair[:,0]] - Z_score[Pair[:,1]]
    label = 1 / (1 + np.exp(-delta_Z))
    ord_pred = label.reshape(Num_Channel, -1)
    flag = ord_pred.sum(0) / ord_pred.shape[0]
    Idx_flag = copy.deepcopy(flag)
    flag[Idx_flag > theta] = 1
    flag[Idx_flag < theta] = 0
    flag[Idx_flag < 1 - theta] = -1

    Acc_all = (np.sum(flag == 1) + 0.5 * np.sum(flag == 0)) / flag.shape[0]
    return Acc_all

    ## the main training block
def Rank_Score_Calculate(Num_Channel, RT_GT, Z_score, Pair, Pair_ord):
    delta_Z = Z_score[Pair[:, 0]] - Z_score[Pair[:, 1]]
    label = 1 / (1 + np.exp(-delta_Z))
    ord_pred = label.reshape(Num_Channel, -1)
    flag = ord_pred.sum(0) / ord_pred.shape[0]
    Idx_flag = copy.deepcopy(flag)
    flag[Idx_flag > theta] = 1
    flag[Idx_flag < theta] = 0
    flag[Idx_flag < 1 - theta] = -1

    Score = np.zeros([RT_GT.shape[0], 2])
    RMSE = 0
    for n in range(flag.shape[0]):
        temp_score = np.zeros([1, 2])
        if RT_GT[Pair_ord[n,0]] > RT_GT[Pair_ord[n,1]]:
            temp_score[0,0] = 1
        else:
            temp_score[0,0] = -1

        if flag[n] == 1:
            temp_score[0,1] = 1
        else:
            if flag[n] == -1:
                temp_score[0,1] = -1
            else:
                temp_score[0,1] = 0

        Score[Pair_ord[n,0],:] = Score[Pair_ord[n,0],:] + (temp_score == 1) + 0.5*(temp_score == 0)
        Score[Pair_ord[n,1],:] = Score[Pair_ord[n,1],:] + (temp_score ==-1) + 0.5*(temp_score == 0)

    RMSE = np.sqrt(((Score[:,0] - Score[:,1])**2).mean())
    return RMSE
    ## the main training block


def train_epoch(Train_Feat, Train_RT, model, optimizer, criterion, flag_cuda, batch_size, epoch):
    losses = []
    i = 1

    ## Shuffle the training data
    Id = np.arange(Train_Feat.shape[0])
    random.shuffle(Id)
    X_tr = Train_Feat[Id,]
    Y_tr = Train_RT[Id]
    ## Transform the numpy data into Torch like tensor data
    Inputs = torch.FloatTensor(X_tr)
    Outputs = torch.FloatTensor(Y_tr)

    
    iteration = int(Inputs.size(0)/batch_size)
    for idx in range(0, Inputs.size(0), batch_size):       
        # (1) clear the gradient
        optimizer.zero_grad()        
        # (2) select the mini-batch
        x_batch = Inputs[idx : idx + batch_size, :]
        y_batch = Outputs[idx : idx + batch_size, :]    
        # (3) Forward
        if flag_cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        else:
            x_batch = x_batch
            y_batch = y_batch

        y_hat = model(x_batch)       
        # (4) Compute diff
        loss = criterion(y_hat, y_batch)                
        # (5) Compute gradients
        loss.backward()        
        # (6) update weights
        optimizer.step()         
        # (7) store the gradient
        if flag_cuda:
            loss = loss.cpu().data.numpy()
        else:
            loss = loss.data.numpy()
        losses.append(loss)
        #if (i+1) % 100 == 0:
        #    print ('Epoch {} [{}/{}], Loss: {:.4f}'.format(epoch, i+1, iteration, loss.item()))
            
        i = i + 1
    return losses