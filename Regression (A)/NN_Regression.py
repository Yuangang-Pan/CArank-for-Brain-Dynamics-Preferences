#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:49:41 2019

@author: yuapan
"""
## load the training and test dataset
from Document_Load import *
from Network_Structure import *

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
import pickle
## Initialize the parameter
Hidden_size = 100
Num_epoch = 50
Learning_rate = 0.005
Batch_size = 50
SEED = 9159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
Training_FFT, Training_RT, Training_pair, Testing_FFT, Testing_RT, Testing_pair = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Test_FFT', 'Test_RT', 'Test_ord')

Acc= []
RMSE = []
Train_score_pres = []
Test_score_pres = []
for i in range(Training_FFT.size):
    ## for subject i
    X_train = Training_FFT[0, i]
    Y_train = Training_RT[0, i]
    Pair_train = (Training_pair[0, i] - 1).astype(np.int64)
    X_test = Testing_FFT[0, i]
    Y_test = Testing_RT[0, i]
    Pair_test = (Testing_pair[0, i] - 1).astype(np.int64)
    ## stretch matrix
    Num_channel = X_train.shape[0]
    Dim = X_train.shape[1]
    Output_size = Y_train.shape[1]
    Train_Feat = X_train.transpose(2,0,1).reshape(-1, Dim)
    Train_RT = np.repeat(Y_train, Num_channel, axis = 0)
    Train_ord = np.hstack(((Pair_train[:,1] * Num_channel + Pair_train[:,0]).reshape(-1, 1), (Pair_train[:,2] * Num_channel + Pair_train[:,0]).reshape(-1, 1)))

    Test_Feat = X_test.transpose(2,0,1).reshape(-1, Dim)
    Test_RT = np.repeat(Y_test, Num_channel, axis = 0)
    Test_ord = np.hstack(((Pair_test[:, 1] * Num_channel + Pair_test[:, 0]).reshape(-1, 1), (Pair_test[:, 2] * Num_channel + Pair_test[:, 0]).reshape(-1, 1)))

    # pairwise comparison
    ID_tr_ord = np.where(Pair_train[:, 0] == 0)
    Train_ord_id = Pair_train[ID_tr_ord, 1:].reshape(-1, 2)

    ID_te_ord = np.where(Pair_test[:, 0] == 0)
    Test_ord_id = Pair_test[ID_te_ord, 1:].reshape(-1, 2)

    ## Initialize the network and define the Loss & optimizer
    model = NeuralNet(Dim, Hidden_size, Output_size)
    #optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

    criterion = nn.MSELoss()
    
    e_losses = []
    for e in range(Num_epoch):
        e_losses += train_epoch(Train_Feat, Train_RT, model, optimizer, criterion, Batch_size, e)
    plt.figure
    plt.plot(e_losses)
    
    
    ## Evaluation the model performance on the test dataset
    model.eval()

    # transform the training and test data into tensor
    Train_inputs = torch.FloatTensor(Train_Feat)
    Test_inputs = torch.FloatTensor(Test_Feat)
    # predication
    Train_RT_pre = model(Train_inputs).detach().numpy()
    Test_RT_pre = model(Test_inputs).detach().numpy()
    ## multi-channel ranking accuracy using majority voting
    Train_Acc = MV_Baseline(Num_channel, Train_RT_pre, Train_ord)
    Test_Acc = MV_Baseline(Num_channel, Test_RT_pre, Test_ord)
    print ('Subject: {}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'.format(i, Train_Acc, Test_Acc))
    temp_acc = [Train_Acc, Test_Acc]


    Train_RMSE = Rank_Score_Calculate(Num_channel, Y_train, Train_RT_pre, Train_ord, Train_ord_id)
    Test_RMSE= Rank_Score_Calculate(Num_channel, Y_test, Test_RT_pre, Test_ord, Test_ord_id)
    print ('Subject: {}, Train_RMSE: {:.4f}, Test_RMSE: {:.4f}'.format(i, Train_RMSE, Test_RMSE))
    temp_RMSE = [Train_RMSE, Test_RMSE]


    Acc.append(temp_acc)
    RMSE.append(temp_RMSE)
    Train_score_pres.append(Train_RT_pre)
    Test_score_pres.append(Test_RT_pre)

    scipy.io.savemat('/Regression (A)/Train_score_pres.mat', mdict={'Train_score_pres': Train_score_pres})
    scipy.io.savemat('/Regression (A)/Test_score_pres.mat', mdict={'Test_score_pres': Test_score_pres})
    scipy.io.savemat('/Regression (A)/Acc.mat', mdict={'Acc': Acc})
    scipy.io.savemat('/Regression (A)/RMSE.mat', mdict={'RMSE': RMSE})


