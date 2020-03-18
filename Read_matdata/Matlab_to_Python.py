#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:11:20 2019

@author: yuapan
"""
import pickle

import scipy.io

Training_FFT = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Training_FFT_data.mat')
Training_RT = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Training_RT.mat')
Training_ord = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Training_Compairison_ord.mat')
Training_eql = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Training_Compairison_eql.mat')

Testing_FFT = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Test_FFT_data.mat')
Testing_RT = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Test_RT.mat')
Testing_ord = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Test_Compairison_ord.mat')
Testing_eql = scipy.io.loadmat('/data/yuapan/Desktop/EEG_Code/ICLR_Submission/EEG_data44/Test_Compairison_eql.mat')

Train_FFT = Training_FFT['Training_FFT_data']
Train_RT = Training_RT['Training_RT']
Train_ord = Training_ord['Training_Compairison_ord']
Train_eql = Training_eql['Training_Compairison_eql']

Test_FFT = Testing_FFT['Test_FFT_data']
Test_RT = Testing_RT['Test_RT']
Test_ord = Testing_ord['Test_Compairison_ord']
Test_eql = Testing_eql['Test_Compairison_eql']

dbfile = open('Training_FFT', 'wb')
pickle.dump(Train_FFT, dbfile)
dbfile.close()

dbfile = open('Training_RT', 'wb')
pickle.dump(Train_RT, dbfile)
dbfile.close()

dbfile = open('Training_ord', 'wb')
pickle.dump(Train_ord, dbfile)
dbfile.close()

dbfile = open('Training_eql', 'wb')
pickle.dump(Train_eql, dbfile)
dbfile.close()

dbfile = open('Test_FFT', 'wb')
pickle.dump(Test_FFT, dbfile)
dbfile.close()

dbfile = open('Test_RT', 'wb')
pickle.dump(Test_RT, dbfile)
dbfile.close()

dbfile = open('Test_ord', 'wb')
pickle.dump(Test_ord, dbfile)
dbfile.close()

dbfile = open('Test_eql', 'wb')
pickle.dump(Test_eql, dbfile)
dbfile.close()



