## load the training and test dataset
from Document_Load import *
from Network_Structure import *
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io
import pickle

## Initialize the parameter
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
Training_FFT, Training_RT, Training_pair, Testing_FFT, Testing_RT, Testing_pair = doc_load('Training_FFT', 'Training_RT', 'Training_ord', 'Test_FFT','Test_RT', 'Test_ord')

computation_cost = []
Iteration = 1
for iter in range(Iteration):

    start = time.time()
    for i in range(1):
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
        Train_Feat = X_train.transpose(2, 0, 1).reshape(-1, Dim)
        Train_RT = np.repeat(Y_train, Num_channel, axis=0)
        Train_ord = np.hstack(((Pair_train[:, 1] * Num_channel + Pair_train[:, 0]).reshape(-1, 1),
                               (Pair_train[:, 2] * Num_channel + Pair_train[:, 0]).reshape(-1, 1)))

        Test_Feat = X_test.transpose(2, 0, 1).reshape(-1, Dim)
        Test_RT = np.repeat(Y_test, Num_channel, axis=0)
        Test_ord = np.hstack(((Pair_test[:, 1] * Num_channel + Pair_test[:, 0]).reshape(-1, 1),
                              (Pair_test[:, 2] * Num_channel + Pair_test[:, 0]).reshape(-1, 1)))

        # pairwise comparison
        ID_tr_ord = np.where(Pair_train[:, 0] == 0)
        Train_ord_id = Pair_train[ID_tr_ord, 1:].reshape(-1, 2)

        ID_te_ord = np.where(Pair_test[:, 0] == 0)
        Test_ord_id = Pair_test[ID_te_ord, 1:].reshape(-1, 2)
        ## Initialize the network and define the Loss & optimizer
        model = NeuralNet(Dim, Hidden_size, Output_size)
        # store the model in GPU
        if flag_cuda:
            model.cuda()

        # optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999))
        criterion = nn.MSELoss()

        model.train()
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
        ## multi-channel ranking accuracy using majority voting
        Train_Acc = MV_Baseline(Num_channel, Train_RT_pre, Train_ord)
        Test_Acc = MV_Baseline(Num_channel, Test_RT_pre, Test_ord)

        Train_RMSE = Rank_Score_Calculate(Num_channel, Y_train, Train_RT_pre, Train_ord, Train_ord_id)
        Test_RMSE = Rank_Score_Calculate(Num_channel, Y_test, Test_RT_pre, Test_ord, Test_ord_id)

    end = time.time()
    print ('time_cost1 {:.4f}'.format(end - start))
    computation_cost.append(end - start)
#print ('computation_cost_mean: {:.4f}'.format(np.array(computation_cost).mean()))
#print ('computation_cost_std: {:.4f}'.format(np.array(computation_cost).std()))
#scipy.io.savemat('/home/yuangang/Downloads/BCI_Experiment/NN_Regression/computation_cost.mat',
#                         mdict={'computation_cost': computation_cost})
