## Calculate the computational cost of naive regression
import random
import time
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
SEED = 194910
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
os. environ['CUDA_VISIBLE_DEVICES'] = '0'
flag_cuda = torch.cuda.is_available()
## Initialize the parameter
Hidden_size = 100
Learning_rate = 0.001
Batch_size = 256
Num_epoch = 50

Training_FFT, Training_RT, Training_pair, Testing_FFT, Testing_RT, Testing_pair = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Test_FFT', 'Test_RT', 'Test_ord')

computation_cost = []
Iteration = 1
for iter in range(Iteration):
    start = time.time()
    for i in range(1): ## for subject i
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
        Test_Feat = X_test.reshape(-1, Num_item_te).transpose(1, 0)
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

        # optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999))
        criterion = nn.MSELoss()

        e_losses = []

        model.train()
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

        # fetch Outputs from GPU
        if flag_cuda:
            Train_RT_pre = model(Train_inputs).cpu().detach().numpy()
            Test_RT_pre = model(Test_inputs).cpu().detach().numpy()
        else:
            Train_RT_pre = model(Train_inputs).detach().numpy()
            Test_RT_pre = model(Test_inputs).detach().numpy()


        Train_loss = (np.square(Train_RT - Train_RT_pre)).mean()
        Test_loss = (np.square(Test_RT - Test_RT_pre)).mean()

    end = time.time()
    print ('time_cost1 {:.4f}'.format(end-start))
    computation_cost.append(end - start)

#print ('computation_cost_mean: {:.4f}'.format(np.array(computation_cost).mean()))
#print ('computation_cost_std: {:.4f}'.format(np.array(computation_cost).std()))
#scipy.io.savemat('/home/yuangang/Downloads/BCI_Experiment/Naive_Regression/computation_cost.mat',
#                mdict={'computation_cost': computation_cost})