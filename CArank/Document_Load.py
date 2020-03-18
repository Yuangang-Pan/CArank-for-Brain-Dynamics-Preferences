#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:34:13 2019

@author: yuapan
"""

import sys
import os
import pickle

root_path = '/Read_matdata/'

def doc_load(Training_FFT, Training_RT, Training_ord, Training_eql, Testing_FFT, Testing_RT, Testing_ord, Testing_eql):
    Training_FFT = os.path.join(root_path, Training_FFT)
    Training_RT = os.path.join(root_path, Training_RT)
    Training_ord = os.path.join(root_path, Training_ord)
    Training_eql = os.path.join(root_path, Training_eql)
    Testing_FFT = os.path.join(root_path, Testing_FFT)
    Testing_RT = os.path.join(root_path, Testing_RT)
    Testing_ord = os.path.join(root_path, Testing_ord)
    Testing_eql = os.path.join(root_path, Testing_eql)

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

    dbfile = open(Testing_FFT, 'rb')
    Test_FFT = pickle.load(dbfile)
    dbfile.close()

    dbfile = open(Testing_RT, 'rb')
    Test_RT = pickle.load(dbfile)
    dbfile.close()

    dbfile = open(Testing_ord, 'rb')
    Test_ord = pickle.load(dbfile)
    dbfile.close()

    dbfile = open(Testing_eql, 'rb')
    Test_eql = pickle.load(dbfile)
    dbfile.close()

    return Train_FFT, Train_RT, Train_ord, Train_eql, Test_FFT, Test_RT, Test_ord, Test_eql