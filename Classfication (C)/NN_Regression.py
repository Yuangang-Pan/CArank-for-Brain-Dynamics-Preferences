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
import scipy.io
import matplotlib.pyplot as plt
import pickle
## Initialize the parameter

Hidden_size =100
Num_epoch = 15
Learning_rate = 0.001
Batch_size = 256

SEED = 9159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
os. environ['CUDA_VISIBLE_DEVICES'] = '1'
flag_cuda = torch.cuda.is_available()

Training_FFT, Training_RT, Training_ord, Training_eql, Testing_FFT, Testing_RT, Testing_ord, Testing_eql = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Training_eql', 'Test_FFT', 'Test_RT', 'Test_ord', 'Test_eql')
Acc= []
RMSE = []
Train_score_pres = []
Test_score_pres = []
for i in range(Training_FFT.size):
    ## for subject i
    Train_feat = Training_FFT[0, i]
    Train_RT = Training_RT[0, i]
    Train_ord = (Training_ord[0, i] -1).astype(np.int64)
    Train_eql = (Training_eql[0, i] -1).astype(np.int64)
    Test_feat = Testing_FFT[0, i] 
    Test_RT = Testing_RT[0, i] 
    Test_ord = (Testing_ord[0, i] -1).astype(np.int64)
    Test_eql = (Testing_eql[0, i] -1).astype(np.int64)
    
    ##channel and dimension
    Num_channel = Train_feat.shape[0]
    Output_size = Train_RT.shape[1]
    ## data transform
    Num_item_tr = Train_feat.shape[2]
    Num_item_te = Test_feat.shape[2]
    Train_Feat = Train_feat.reshape(-1, Num_item_tr).transpose(1,0)
    Test_Feat = Test_feat.reshape(-1, Num_item_te).transpose(1,0)
    ID_tr_ord = np.where(Train_ord[:, 0] == 0)
    Train_ord_id = Train_ord[ID_tr_ord,1:].reshape(-1,2)
    ID_tr_eql = np.where(Train_eql[:, 0] == 0)
    Train_eql_id = Train_eql[ID_tr_eql,1:].reshape(-1,2)

    ID_te_ord = np.where(Test_ord[:, 0] == 0)
    Test_ord_id = Test_ord[ID_te_ord,1:].reshape(-1,2)
    ID_te_eql = np.where(Test_eql[:, 0] == 0)
    Test_eql_id = Test_eql[ID_te_eql,1:].reshape(-1,2)
    Dim = Train_Feat.shape[1]
    ## Initialize the network and define the Loss & optimizer
    model = NeuralNet(Dim, Hidden_size, Output_size)
    # store the model in GPU
    if flag_cuda:
        model.cuda()
    #optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999), weight_decay=0.01)


    e_losses = []
    for e in range(Num_epoch):
        e_losses += train_epoch(Train_Feat, Train_ord_id, Train_eql_id, model, optimizer, criterion, flag_cuda, Batch_size, e)
    
    
    ## Evaluation the model performance on the test dataset
    # load the test data
    model.eval()

    # transform the training and test data into tensor
    Train_BCI_inputs = torch.FloatTensor(Train_Feat)
    Test_BCI_inputs = torch.FloatTensor(Test_Feat)
    # store the data in GPU
    if flag_cuda:
        Train_BCI_inputs = Train_BCI_inputs.cuda()
        Test_BCI_inputs = Test_BCI_inputs.cuda()
    else:
        Train_BCI_inputs = Train_BCI_inputs
        Test_BCI_inputs = Test_BCI_inputs
     # predication
    Train_out = model(Train_BCI_inputs)
    Test_out = model(Test_BCI_inputs)
    if flag_cuda:
        Train_score_pre = Train_out.cpu().data.numpy()
        Test_score_pre = Test_out.cpu().data.numpy()
    else:
        Train_score_pre = Train_out.data.numpy()
        Test_score_pre = Test_out.data.numpy()

    ## multi-channel ranking accuracy using majority voting
    Train_Acc = MV_Baseline(Train_score_pre, Train_ord_id)
    Test_Acc = MV_Baseline(Test_score_pre, Test_ord_id)
    print ('Subject: {}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'.format(i, Train_Acc, Test_Acc))
    temp_acc = [Train_Acc, Test_Acc]
    Acc.append(temp_acc)

    Train_RMSE = Rank_Score_Calculate(Train_RT, Train_score_pre, Train_ord_id)
    Test_RMSE = Rank_Score_Calculate(Test_RT, Test_score_pre, Test_ord_id)
    print ('Subject: {}, Train_RMSE: {:.4f}, Test_RMSE: {:.4f}'.format(i, Train_RMSE, Test_RMSE))
    temp_RMSE = [Train_RMSE, Test_RMSE]
    RMSE.append(temp_RMSE)

    Train_score_pres.append(Train_score_pre)
    Test_score_pres.append(Test_score_pre)

    scipy.io.savemat('/Classfication (C)/Train_score_pres.mat', mdict = {'Train_score_pres': Train_score_pres})
    scipy.io.savemat('/Classfication (C)/Test_score_pres.mat', mdict = {'Test_score_pres': Test_score_pres})
    scipy.io.savemat('/Classfication (C)/Acc.mat', mdict = {'Acc': Acc})
    scipy.io.savemat('/Classfication (C)/RMSE.mat', mdict = {'RMSE': RMSE})

