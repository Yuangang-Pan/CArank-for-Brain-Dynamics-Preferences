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
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
import os
torch.backends.cudnn.deterministic = True
os. environ['CUDA_VISIBLE_DEVICES'] = '1'
flag_cuda = torch.cuda.is_available()
## Initialize the parameter
Hidden_size = 100
Num_epoch = 15
Learning_rate = 0.001
Batch_size = 300
Inner_EM = 7

Training_FFT, Training_RT, Training_ord, Training_eql, Testing_FFT, Testing_RT, Testing_ord, Testing_eql = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Training_eql', 'Test_FFT', 'Test_RT', 'Test_ord', 'Test_eql')
Acc= [0, 0]
RMSE = [0, 0]
Pis = []
Train_score_pres = []
Test_score_pres = []

#ID = np.array([9, 21, 25, 27, 29, 30, 32, 35]) - 1
for i in range(Training_FFT.size):
#for i in ID:
    ## for subject i
    Train_feat = Training_FFT[0, i]
    Train_RT = Training_RT[0, i]
    Pair_Train_ord = (Training_ord[0, i] -1).astype(np.int64)
    Pair_Train_eql = (Training_eql[0, i] -1).astype(np.int64)
    Test_feat = Testing_FFT[0, i]
    Test_RT = Testing_RT[0, i]
    Pair_Test_ord = (Testing_ord[0, i] -1).astype(np.int64)
    Pair_Test_eql = (Testing_eql[0, i] -1).astype(np.int64)

    ##channel & dimension
    Channel, Dim  = Train_feat.shape[0:2]
    Output_size = Train_RT.shape[1]
    Train_BCI_feat = Train_feat.transpose(2, 0 ,1).reshape(-1, Dim)
    Test_BCI_feat = Test_feat.transpose(2, 0 ,1).reshape(-1, Dim)
    ## data transform
    Train_ord = np.hstack(((Pair_Train_ord[:,1]* Channel +Pair_Train_ord[:,0]).reshape(-1,1),  (Pair_Train_ord[:,2]* Channel +Pair_Train_ord[:,0]).reshape(-1,1)))
    Train_eql = np.hstack(((Pair_Train_eql[:,1]* Channel +Pair_Train_eql[:,0]).reshape(-1,1),  (Pair_Train_eql[:,2]* Channel +Pair_Train_eql[:,0]).reshape(-1,1)))

    Test_ord = np.hstack(((Pair_Test_ord[:,1]* Channel +Pair_Test_ord[:,0]).reshape(-1,1),  (Pair_Test_ord[:,2]* Channel +Pair_Test_ord[:,0]).reshape(-1,1)))
    Test_eql = np.hstack(((Pair_Test_eql[:,1]* Channel +Pair_Test_eql[:,0]).reshape(-1,1),  (Pair_Test_eql[:,2]* Channel +Pair_Test_eql[:,0]).reshape(-1,1)))

    # pairwise comparison
    ID_tr_ord = np.where(Pair_Train_ord[:, 0] == 0)
    Train_ord_id = Pair_Train_ord[ID_tr_ord, 1:].reshape(-1, 2)

    ID_te_ord = np.where(Pair_Test_ord[:, 0] == 0)
    Test_ord_id = Pair_Test_ord[ID_te_ord, 1:].reshape(-1, 2)

    ## Initialize the network and define the Loss & optimizer
    model = NeuralNet(Dim, Hidden_size, Output_size)
    # store the model in GPU
    if flag_cuda:
        model.cuda()

    #optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999), weight_decay=0.01)

    ## initialize the model parameter
    Inputs = torch.FloatTensor(Train_BCI_feat)
    Gamma = 0.7 * np.ones([Train_ord.shape[0],1])
    Pi = 0.5 * np.ones([Channel, 1])
    e_losses = []
    for iter in range(Inner_EM):
        model.train()
        for e in range(Num_epoch):
            e_losses += train_epoch(Train_BCI_feat, Train_ord, Train_eql, model, optimizer, criterion, flag_cuda, Batch_size, Gamma, e)

        # Evaluation the model performance for expectation
        model.eval()
        # predication
        # store the Inputs feat data in GPU
        if flag_cuda:
            Inputs = Inputs.cuda()
        else:
            Inputs = Inputs

        Outputs = model(Inputs)
        # fetch Outputs from GPU
        if flag_cuda:
            Predict_score = Outputs.cpu().data.numpy()
        else:
            Predict_score = Outputs.data.numpy()
        Gamma, Pi = Posterior_expectation(Predict_score, Pair_Train_ord, Train_ord, Pi)

    plt.plot(e_losses)
    ## Evaluation the model performance on the test dataset
    # load the test data
    model.eval()

    # transform the training and test data into tensor
    Train_BCI_inputs = torch.FloatTensor(Train_BCI_feat)
    Test_BCI_inputs = torch.FloatTensor(Test_BCI_feat)
    # predication
    # store the Inputs feat data in GPU
    if flag_cuda:
        Train_BCI_inputs = Train_BCI_inputs.cuda()
        Test_BCI_inputs = Test_BCI_inputs.cuda()
    else:
        Train_BCI_inputs = Train_BCI_inputs
        Test_BCI_inputs = Test_BCI_inputs

    # fetch Outputs from GPU
    if flag_cuda:
        Train_score_pre = model(Train_BCI_inputs).cpu().detach().numpy()
        Test_score_pre = model(Test_BCI_inputs).cpu().detach().numpy()
    else:
        Train_score_pre = model(Train_BCI_inputs).detach().numpy()
        Test_score_pre = model(Test_BCI_inputs).detach().numpy()

    ## multi-channel ranking accuracy using majority voting
    Train_Acc = MV_Baseline(Pi, Train_score_pre, Train_ord)
    Test_Acc = MV_Baseline(Pi, Test_score_pre, Test_ord)
    print ('Subject: {}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'.format(i, Train_Acc, Test_Acc))
    temp_acc = [Train_Acc, Test_Acc]
    Acc = np.vstack((Acc, temp_acc))

    Train_RMSE = Rank_Score_Calculate(Pi, Train_RT, Train_score_pre, Train_ord, Train_ord_id)
    Test_RMSE = Rank_Score_Calculate(Pi, Test_RT, Test_score_pre, Test_ord, Test_ord_id)
    print ('Subject: {}, Train_RMSE: {:.4f}, Test_RMSE: {:.4f}'.format(i, Train_RMSE, Test_RMSE))
    temp_RMSE = [Train_RMSE, Test_RMSE]
    RMSE = np.vstack((RMSE, temp_RMSE))

    ## store the results
    Pis.append(Pi)
    Train_score_pres.append(Train_score_pre)
    Test_score_pres.append(Test_score_pre)

    scipy.io.savemat('/CArank/Pis.mat', mdict = {'Pis': Pis})
    scipy.io.savemat('/CArank/Train_score_pres.mat', mdict = {'Train_score_pres': Train_score_pres})
    scipy.io.savemat('/CArank/Test_score_pres.mat', mdict = {'Test_score_pres': Test_score_pres})
    scipy.io.savemat('/CArank/Acc.mat', mdict = {'Acc': Acc})
    scipy.io.savemat('/CArank/RMSE.mat', mdict = {'RMSE': RMSE})


