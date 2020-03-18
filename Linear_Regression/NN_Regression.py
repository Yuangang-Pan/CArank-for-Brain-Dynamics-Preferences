#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:49:41 2019

@author: yuapan
"""
import random
random.seed(2019)
## load the training and test dataset
from Document_Load import *
from Network_Structure import *

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
import pickle
## Initialize the parameter
Hidden_size = 100
Num_epoch = 50
Learning_rate = 0.005
Batch_size = 5
SEED = 9159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
flag_cuda = torch.cuda.is_available()
Training_FFT, Training_RT, Training_pair, Testing_FFT, Testing_RT, Testing_pair = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Test_FFT', 'Test_RT', 'Test_ord')
Acc= []
RMSE = []
Train_score_pres = []
Test_score_pres = []
for i in range(Training_FFT.size):
    ## for subject i
    X_train = Training_FFT[0, i]
    Y_train = Training_RT[0, i]

    X_test = Testing_FFT[0, i]
    Y_test = Testing_RT[0, i]

    ## stretch matrix
    Num_channel = X_train.shape[0]
    Output_size = Y_train.shape[1]
    Num_item_tr = X_train.shape[2]
    Train_Feat = X_train.reshape(-1, Num_item_tr).transpose(1, 0)
    Train_RT = Y_train
    Num_item_te = X_test.shape[2]
    Test_Feat = X_test.reshape(-1, Num_item_te).transpose(1,0)
    Test_RT = Y_test

    ## Pairwise comparison
    Pair_train = (Training_pair[0, i] - 1).astype(np.int64)
    Pair_test = (Testing_pair[0, i] - 1).astype(np.int64)
    ID_tr_ord = np.where(Pair_train[:, 0] == 0)
    Train_ord_id = Pair_train[ID_tr_ord, 1:].reshape(-1, 2)

    ID_te_ord = np.where(Pair_test[:, 0] == 0)
    Test_ord_id = Pair_test[ID_te_ord, 1:].reshape(-1, 2)

    Dim = Train_Feat.shape[1]
    ## Initialize the network and define the Loss & optimizer
    model = NeuralNet(Dim, Hidden_size, Output_size)
    # store the model in GPU
    if flag_cuda:
        model.cuda()

    #optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

    criterion = nn.MSELoss()
    
    e_losses = []
    for e in range(Num_epoch):
        e_losses += train_epoch(Train_Feat, Train_RT, model, optimizer, criterion, flag_cuda, Batch_size, e)

    ## Evaluation the model performance on the test dataset
    model.eval()

    # transform the training and test data into tensor
    Train_inputs = torch.FloatTensor(Train_Feat)
    Test_inputs = torch.FloatTensor(Test_Feat)
    # store the Inputs feat data in GPU
    if flag_cuda:
        Train_inputs = Train_inputs.cuda()
        Test_inputs = Test_inputs.cuda()
    else:
        Train_inputs = Train_inputs
        Test_inputs = Test_inputs
    # predication
    # fetch Outputs from GPU
    if flag_cuda:
        Train_RT_pre = model(Train_inputs).cpu().detach().numpy()
        Test_RT_pre = model(Test_inputs).cpu().detach().numpy()
    else:
        Train_RT_pre = model(Train_inputs).detach().numpy()
        Test_RT_pre = model(Test_inputs).detach().numpy()

    Train_loss = (np.square(Train_RT - Train_RT_pre)).mean()
    Test_loss = (np.square(Test_RT - Test_RT_pre)).mean()

    ## multi-channel ranking accuracy using majority voting
    Train_Acc = MV_Baseline(Train_RT_pre, Train_ord_id)
    Test_Acc = MV_Baseline(Test_RT_pre, Test_ord_id)
    print ('Subject: {}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'.format(i, Train_Acc, Test_Acc))
    temp_acc = [Train_Acc, Test_Acc]

    Train_RMSE = Rank_Score_Calculate(Y_train, Train_RT_pre, Train_ord_id)
    Test_RMSE = Rank_Score_Calculate(Y_test, Test_RT_pre, Test_ord_id)
    print ('Subject: {}, Train_RMSE: {:.4f}, Test_RMSE: {:.4f}'.format(i, Train_RMSE, Test_RMSE))
    temp_RMSE = [Train_RMSE, Test_RMSE]



    Acc.append(temp_acc)
    RMSE.append(temp_RMSE)
    Train_score_pres.append(Train_RT_pre)
    Test_score_pres.append(Test_RT_pre)

scipy.io.savemat('/LR/Train_score_pres.mat', mdict={'Train_score_pres': Train_score_pres})
scipy.io.savemat('/LR/Test_score_pres.mat', mdict={'Test_score_pres': Test_score_pres})
scipy.io.savemat('/LR/Acc.mat', mdict={'Acc': Acc})
scipy.io.savemat('/LR/RMSE.mat', mdict={'RMSE': RMSE})