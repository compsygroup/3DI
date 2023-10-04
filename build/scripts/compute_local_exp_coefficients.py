#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import DictionaryLearning

# expressions_dir = sys.argv[1] # '/media/v/SSD1TB/dataset/videos/treecam/ML/output/BFMmm-19830.cfg7.global4.curt'
# morphable_model = sys.argv[2] # 'BFMmm-19830'
# canonical_lmks_file = '/media/v/SSD1TB/dataset/videos/treecam/ML/output/BFMmm-19830.cfg7.global4.curt'
canonical_lmks_file = sys.argv[1] # '/home/v/code/3DI/build/output/BFMmm-19830.cfg1.global4/elaine.canonical_lmks'
local_exp_coeffs_file = sys.argv[2] # '/home/v/code/3DI/build/output/BFMmm-19830.cfg1.global4/elaine.local_bcoeffs.'
morphable_model = sys.argv[3] #  'BFMmm-19830'
basis_version = sys.argv[4] # '0.0.1.4'



sdir = f'models/MMs/{morphable_model}/'
localized_basis_file = f'models/MMs/{morphable_model}/E/localized_basis/v.{basis_version}.npy'
basis_set = np.load(localized_basis_file, allow_pickle=True).item()

P = np.loadtxt(canonical_lmks_file)

P0 = np.loadtxt(f'{sdir}/p0L_mat.dat')
X0 = P0[:,0]
Y0 = P0[:,1]
Z0 = P0[:,2]

tex0 = np.loadtxt(f'{sdir}/tex_mu.dat')


rel_ids   = {'lb': np.array(list(range(0, 5))),
             'rb': np.array(list(range(5, 10))),
             'no': np.array(list(range(10, 19))),
             'le': np.array(list(range(19, 25))),
             're': np.array(list(range(25, 31))),
             'ul': np.array(list(range(31, 37))+list(range(43, 47))),
             'll': np.array(list(range(37, 43))+list(range(47, 51)))}

facial_feats = list(rel_ids.keys())

T = P.shape[0]

C = []
for t in range(T):
    cur = []
    for feat in facial_feats:
        rel_id = rel_ids[feat]
    
        p = P[t,:]
        x = p[::3]
        y = p[1::3]
        z = p[2::3]
        
        dx = x-X0
        dy = y-X0
        dz = z-X0
        
        # @TODO the following code does not work for differential expression computation
        # but only for absolute expressions. It needs to be adapted to the case where
        # basis_set['use_abs'] is set to False!
        dp = np.concatenate((dx[rel_ids[feat]], dy[rel_ids[feat]], dz[rel_ids[feat]]))
        coeffs = basis_set[feat].transform(dp.reshape(1,-1)).T
        cur.append(coeffs)
    
    C.append(np.concatenate(cur).reshape(-1,))
C = np.array(C)

np.savetxt(local_exp_coeffs_file, C)
