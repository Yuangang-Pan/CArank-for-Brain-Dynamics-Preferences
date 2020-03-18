#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:34:13 2019

@author: yuapan
"""

import pickle

def doc_load(Training_FFT, Training_RT, Training_ord, Training_eql, Test_FFT, Test_RT, Test_ord, Test_eql):
    
    dbfile = open(Training_FFT, 'rb')
    Train_FFT = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Training_RT, 'rb')
    Train_RT = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Training_ord, 'rb')
    Train_ord = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Training_eql, 'rb')
    Train_eql = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Test_FFT, 'rb')
    Test_FFT = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Test_RT, 'rb')
    Test_RT = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Test_ord, 'rb')
    Test_ord = pickle.load(dbfile)
    dbfile.close()
    
    dbfile = open(Test_eql, 'rb')
    Test_eql = pickle.load(dbfile)
    dbfile.close()
    return Train_FFT, Train_RT, Train_ord, Train_eql, Test_FFT, Test_RT, Test_ord, Test_eql