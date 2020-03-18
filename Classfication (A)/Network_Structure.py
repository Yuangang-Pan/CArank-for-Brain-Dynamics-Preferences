#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:20:19 2019

@author: yuapan
"""
import torch
import torch.nn as nn
import numpy as np
import copy

theta = 0.65
np.random.seed(2019)
## construct the neural network structure
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.dout(x)
        out = self.fc1(out)
        out = self.relu(out)
        #out = self.dout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def criterion(Z_score, Pair, Idx):
    delta_Z = Z_score[Pair[:,0]] - Z_score[Pair[:,1]]
    delta_Z = torch.clamp(delta_Z, -30, 30)
    delta_large = torch.log(1 + torch.exp(-delta_Z))
    delta_small = torch.log(1 + torch.exp(delta_Z))
    delta_ord = delta_large
    delta_eql = 0.5 * (delta_large + delta_small)
    loss = torch.reshape(Idx[:, 0], shape=[-1, 1]) * delta_ord + torch.reshape(Idx[:, 1], shape=[-1, 1]) * delta_eql
    losses = torch.mean(loss)
    return losses

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

def train_epoch(feat, ord_pair, eql_pair, model, optimizer, criterion, flag_cuda, batch_size, epoch):
    model.train()
    losses = []
    i = 1
    if eql_pair.shape[0] > ord_pair.shape[0]:
        ratio = int(eql_pair.shape[0] / ord_pair.shape[0])
    else:
        ratio = 1

    ## concatenate the feat_ord and feat_eql
    Extend_ord_pair = np.tile(ord_pair, (ratio, 1))
    ord_idx = np.ones((np.shape(Extend_ord_pair)[0], 2))
    ord_idx[:, 1] = 0
    eql_idx = np.ones((np.shape(eql_pair)[0], 2))
    eql_idx[:, 0] = 0
    Pair_train = np.concatenate((Extend_ord_pair, eql_pair), axis=0)
    ## generate the indx vector
    Idx_train = np.concatenate((ord_idx, eql_idx), axis=0)
    ## Shuffle the training data
    Id = np.arange(Pair_train.shape[0])
    np.random.shuffle(Id)
    ## Transform the numpy data into Torch like tensor data
    Pair_inputs = torch.LongTensor(Pair_train[Id, :])
    Idx_inputs = torch.FloatTensor(Idx_train[Id, :])
    Feat_Inputs = torch.FloatTensor(feat)

    iteration = int(Idx_inputs.shape[0] / batch_size)
    for idx in range(0, Idx_inputs.shape[0], batch_size):
        # (1) clear the gradient
        optimizer.zero_grad()
        # (2) select the mini-batch
        Pair_batch = Pair_inputs[idx: idx + batch_size, :]
        Idx_batch = Idx_inputs[idx: idx + batch_size, :]

        # store the mini-batch data in GPU
        if flag_cuda:
            Pair_batch = Pair_batch.cuda()
            Idx_batch = Idx_batch.cuda()
            Feat_Inputs = Feat_Inputs.cuda()
        else:
            Pair_batch = Pair_batch
            Idx_batch = Idx_batch
            Feat_Inputs = Feat_Inputs
        # (3) Forward
        Z_score = model(Feat_Inputs)
        # (4) Compute diff
        loss = criterion(Z_score, Pair_batch, Idx_batch)
        # (5) Compute gradients
        loss.backward()
        # (6) update weights
        optimizer.step()
        # fetch loss from GPU
        if flag_cuda:
            loss = loss.cpu().data.numpy()
        else:
            loss = loss.data.numpy()
        # (7) store the gradient
        losses.append(loss)
        #if (i + 1) % 100 == 0:
        #    print('Epoch {} [{}/{}], Loss: {:.4f}'.format(epoch, i + 1, iteration, loss.item()))

        i = i + 1
    return losses
