#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:57:52 2023

@author: v
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from common import rel_ids, create_expression_sequence

from sklearn.exceptions import ConvergenceWarning

def robust_normalize(X, stats):
    eps = 1e-12
    N, T = X.shape

    def _vec(key):
        if key not in stats:
            raise ValueError(f"Required stat '{key}' is missing.")
        v = np.asarray(stats[key], dtype=float)
        if v.ndim != 1 or v.shape[0] != N:
            raise ValueError(f"Stat '{key}' must be shape (N,), got {v.shape}.")
        return v

    # 'min_0.5pctl', 'max_99.5pctl', 'min_2.5pctl', 'max_97.5pctl', 'Q1', 'Q3', 'median', 'mean', 'std'
    median = _vec("median")
    q1 = _vec("Q1")
    q3 = _vec("Q3")
    std = _vec("std")

    # Base robust scale: IQR -> sigma-like
    NORMAL_CONST_IQR = 1.3489795003921634
    base_scale = (q3 - q1)
    scale = base_scale / NORMAL_CONST_IQR

    # Identify invalid/too-small scales
    invalid_or_small = (~np.isfinite(scale)) | (scale <= eps)

    # Fallback to std if available, else 1.0
    if std is not None:
        std_safe = np.where(np.isfinite(std) & (std > eps), std, 1.0)
        scale = np.where(invalid_or_small, std_safe, scale)
    else:
        scale = np.where(invalid_or_small, 1.0, scale)

    # Final guard
    scale = np.maximum(scale, eps)

    # Normalize (broadcast across time axis)
    X = np.asarray(X, dtype=float, order="C")
    X_norm = (X - median[:, None]) / scale[:, None]
    
    return X_norm


#warnings.filterwarnings("ignore", category=ConvergenceWarning)

exp_coeffs_file = sys.argv[1] 
local_exp_coeffs_file = sys.argv[2]
morphable_model = sys.argv[3] #  'BFMmm-19830'
basis_version = sys.argv[4] # '0.0.1.4'
normalize = False
if len(sys.argv) >= 6:
    normalize = bool(int(sys.argv[5]))

sdir = f'models/MMs/{morphable_model}/'
localized_basis_file = f'models/MMs/{morphable_model}/E/localized_basis/v.{basis_version}.npy'

with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    basis_set = np.load(localized_basis_file, allow_pickle=True).item()

# @TODO the code does not work for differential expression computation
# but only for absolute expressions. It needs to be adapted to the case where
# basis_set['use_abs'] is set to False!
assert basis_set['use_abs']

li = np.loadtxt(f'{sdir}/li.dat').astype(int)

facial_feats = list(rel_ids.keys())

epsilons = np.loadtxt(exp_coeffs_file)

T = epsilons.shape[0]

Es = {}

for feat in rel_ids:
    rel_id = rel_ids[feat]
    EX  = np.loadtxt('%s/E/EX_79.dat' % sdir)[li[rel_id],:]
    EY  = np.loadtxt('%s/E/EY_79.dat' % sdir)[li[rel_id],:]
    EZ  = np.loadtxt('%s/E/EZ_79.dat' % sdir)[li[rel_id],:]
    Es[feat] = np.concatenate((EX, EY, EZ), axis=0)
    

ConvergenceWarning('ignore')

C = []
for feat in facial_feats:
    rel_id = rel_ids[feat]
    dp = create_expression_sequence(epsilons, Es[feat])
    dictionary = basis_set[feat]
    coeffs = dictionary.transform(dp).T

    # robust normalization using (x - median) / IQR
    if normalize:
        if hasattr(dictionary, 'stats'):
            coeffs = robust_normalize(coeffs, dictionary.stats)
        else:
            print("Skipping normalization because stats on localized expressions is not available.")

    C.append(coeffs)
    
C = np.concatenate(C).T

np.savetxt(local_exp_coeffs_file, C)
