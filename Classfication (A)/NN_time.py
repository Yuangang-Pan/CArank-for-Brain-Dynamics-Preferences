## load the training and test dataset
from Document_Load import *
from Network_Structure import *
import time
import torch
import numpy as np
import torch.optim as optim
import scipy.io
import matplotlib.pyplot as plt
import pickle

## Initialize the parameter
Hidden_size = 100
Num_epoch = 15
Learning_rate = 0.001
Batch_size = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
flag_cuda = torch.cuda.is_available()
SEED = 9159
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
Training_FFT, Training_RT, Training_ord, Training_eql, Testing_FFT, Testing_RT, Testing_ord, Testing_eql = doc_load(
    'Training_FFT', 'Training_RT', 'Training_ord', 'Training_eql', 'Test_FFT', 'Test_RT', 'Test_ord', 'Test_eql')
computation_cost = []
Iteration = 1
for iter in range(Iteration):
    start = time.time()
    for i in range(1):
        ## for subject i
        Train_feat = Training_FFT[0, i]
        Train_RT = Training_RT[0, i]
        Pair_Train_ord = (Training_ord[0, i] - 1).astype(np.int64)
        Pair_Train_eql = (Training_eql[0, i] - 1).astype(np.int64)
        Test_feat = Testing_FFT[0, i]
        Test_RT = Testing_RT[0, i]
        Pair_Test_ord = (Testing_ord[0, i] - 1).astype(np.int64)
        Pair_Test_eql = (Testing_eql[0, i] - 1).astype(np.int64)

        ##channel and dimension
        Num_Channel = Train_feat.shape[0]
        Dim = Train_feat.shape[1]
        Output_size = Train_RT.shape[1]
        ## data transform
        Train_BCI_feat = Train_feat.transpose(2, 0, 1).reshape(-1, Dim)
        Test_BCI_feat = Test_feat.transpose(2, 0, 1).reshape(-1, Dim)
        Train_ord = np.hstack(((Pair_Train_ord[:, 1] * Num_Channel + Pair_Train_ord[:, 0]).reshape(-1, 1),
                               (Pair_Train_ord[:, 2] * Num_Channel + Pair_Train_ord[:, 0]).reshape(-1, 1)))
        Train_eql = np.hstack(((Pair_Train_eql[:, 1] * Num_Channel + Pair_Train_eql[:, 0]).reshape(-1, 1),
                               (Pair_Train_eql[:, 2] * Num_Channel + Pair_Train_eql[:, 0]).reshape(-1, 1)))

        Test_ord = np.hstack(((Pair_Test_ord[:, 1] * Num_Channel + Pair_Test_ord[:, 0]).reshape(-1, 1),
                              (Pair_Test_ord[:, 2] * Num_Channel + Pair_Test_ord[:, 0]).reshape(-1, 1)))
        Test_eql = np.hstack(((Pair_Test_eql[:, 1] * Num_Channel + Pair_Test_eql[:, 0]).reshape(-1, 1),
                              (Pair_Test_eql[:, 2] * Num_Channel + Pair_Test_eql[:, 0]).reshape(-1, 1)))

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
        # optimizer = optim.SGD(model.parameters(), lr=Learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=Learning_rate, betas=(0.9, 0.999))

        e_losses = []
        for e in range(Num_epoch):
            e_losses += train_epoch(Train_BCI_feat, Train_ord, Train_eql, model, optimizer, criterion, flag_cuda,
                                    Batch_size, e)

        ## Evaluation the model performance on the test dataset
        # load the test data
        model.eval()

        # transform the training and test data into tensor
        Train_BCI_inputs = torch.FloatTensor(Train_BCI_feat)
        Test_BCI_inputs = torch.FloatTensor(Test_BCI_feat)

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
        Train_Acc = MV_Baseline(Num_Channel, Train_score_pre, Train_ord)
        Test_Acc = MV_Baseline(Num_Channel, Test_score_pre, Test_ord)
        Train_RMSE = Rank_Score_Calculate(Num_Channel, Train_RT, Train_score_pre, Train_ord, Train_ord_id)
        Test_RMSE = Rank_Score_Calculate(Num_Channel, Test_RT, Test_score_pre, Test_ord, Test_ord_id)

    end = time.time()
    print ('time_cost1 {:.4f}'.format(end - start))
    computation_cost.append(end - start)

#print ('computation_cost_mean: {:.4f}'.format(np.array(computation_cost).mean()))
#print ('computation_cost_std: {:.4f}'.format(np.array(computation_cost).std()))
#scipy.io.savemat('/home/yuangang/Downloads/BCI_Experiment/NN_Classfication/computation_cost.mat',
 #                        mdict={'computation_cost': computation_cost})
