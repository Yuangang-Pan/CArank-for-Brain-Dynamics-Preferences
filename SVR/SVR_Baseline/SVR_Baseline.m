clc;
clear;
path('..//liblinear_matlab', path);
path('..//EEG_Dataset', path);
%% load training data
load('Training_FFT_data.mat')
load('Training_RT.mat')
load('Training_Compairison_ord.mat')

%% load test data
load('Test_FFT_data.mat')
load('Test_RT.mat')
load('Test_Compairison_ord.mat')

N_subject = length(Training_RT);
ACC = [];
cmd = '-s 11';
for n = 1 : N_subject
    fprintf('iter%3d\n', n)
    %% prepare training dataset
    TR_X = Training_FFT_data{n};
    TR_y = Training_RT{n};
    TR_x = reshape(TR_X, size(TR_X,1) * size(TR_X,2), size(TR_X,3))';
    train_data = Training_Compairison_ord{n};
    Idx = train_data(:,1) == 1;
    train_data = train_data(Idx, 2:3);
    %% prepare test dataset
    Te_X = Test_FFT_data{n};
    Te_y = Test_RT{n};
    Te_x = reshape(Te_X, size(Te_X,1) * size(Te_X,2), size(Te_X,3))';
    test_data = Test_Compairison_ord{n};
    Idx = test_data(:,1) == 1;
    test_data = test_data(Idx, 2:3);
    
    model = train(double(TR_y), sparse(TR_x), cmd);    
    RT_train = predict(double(TR_y), sparse(TR_x), model);
    RT_test = predict(double(Te_y), sparse(Te_x), model);
    
    %%calculate the model performance
    [TR1,TR2] = Score_acc_Calculate(RT_train, train_data);
    [Te1,Te2] = Score_acc_Calculate(RT_test, test_data);
    ACC(:,n) = [TR1;Te1;TR2;Te2];
end
save ACC.mat ACC
