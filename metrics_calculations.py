#!/usr/bin/env python
# coding: utf-8

# # Shared calculations for histograms and perfect fairness, for all metrics
# 
# In this file, a slightly different naming convention is used: `i` for m _i_ nority and `j` for ma _j_ ority group.

# ## Setup

# In[ ]:


import gc
import os
import pickle
from os import path

import numpy as np
import pandas as pd

from utils import *  # metrics functions


# In[ ]:


data_cols = [
    'i_tp',     # minority true positive
    'i_fp',     # minority false positive
    'i_tn',     # minority true negative
    'i_fn',     # minority false negative
    'j_tp',     # majority true positive
    'j_fp',     # majority false positive
    'j_tn',     # majority true negative
    'j_fn',     # majority false negative
]

sample_size = 28

calculations_dir = path.join('out', 'calculations', f'n{sample_size}')
os.makedirs(calculations_dir, exist_ok=True)
dataset_path = path.join('out', f'Set(08,{sample_size}).bin')


# In[ ]:


# Get half of the data
with open(dataset_path, "rb") as f:
    df = pd.DataFrame(pickle.load(f), columns=data_cols)

df.head()


# # Part 1: ratios and basic metrics
# 
# For each row, its group and imbalance ratio is calculated. Then, metrics that derive directly from these ratios are calculated.

# In[ ]:


# Calculate group ratios
with open(path.join(calculations_dir, "gr.bin"), "wb+") as f:
    get_group_ratios(df).to_numpy().tofile(f)

# Calculate imbalance ratios
with open(path.join(calculations_dir, "ir.bin"), "wb+") as f:
    get_imbalance_ratios(df).to_numpy().tofile(f)

# calculate metrics
with open(path.join(calculations_dir, "i_tpr.bin"), "wb+") as f:
    getTPR_i(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "j_tpr.bin"), "wb+") as f:
    getTPR_j(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "i_fpr.bin"), "wb+") as f:
    getFPR_i(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "j_fpr.bin"), "wb+") as f:
    getFPR_j(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "i_ppv.bin"), "wb+") as f:
    get_positive_predictive_value_i(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "j_ppv.bin"), "wb+") as f:
    get_positive_predictive_value_j(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "i_npv.bin"), "wb+") as f:
    get_negative_predictive_value_i(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "j_npv.bin"), "wb+") as f:
    get_negative_predictive_value_j(df).to_numpy().tofile(f)
    
with open(path.join(calculations_dir, "stat_parity.bin"), "wb+") as f:
    get_statistical_parity(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "disp_impact.bin"), "wb+") as f:
    get_disparate_impact(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "acc_equality_ratio.bin"), "wb+") as f:
    get_acc_equality_ratio(df).to_numpy().tofile(f)

with open(path.join(calculations_dir, "acc_equality_diff.bin"), "wb+") as f:
    get_acc_equality_diff(df).to_numpy().tofile(f)
    
# Free the memory
del df
gc.collect()
gc.get_stats()


# # Part 2: Get additional calculations
# 
# Calculations that are based on the previous ones. Some files from the previous part are used here, and new ones are created.

# In[ ]:


with open(path.join(calculations_dir, "i_tpr.bin"), "rb") as f:
    i_tpr = pd.DataFrame(np.fromfile(f), columns=["i_tpr"])

with open(path.join(calculations_dir, "j_tpr.bin"), "rb") as f:
    j_tpr = pd.DataFrame(np.fromfile(f), columns=["j_tpr"])

with open(path.join(calculations_dir, "equal_opp_ratio.bin"), "wb+") as f:
    get_equal_opp_ratio(j_tpr['j_tpr'], i_tpr['i_tpr']).to_numpy().tofile(f)
    
with open(path.join(calculations_dir, "equal_opp_diff.bin"), "wb+") as f:
    get_equal_opp_diff(j_tpr['j_tpr'], i_tpr['i_tpr']).to_numpy().tofile(f)

del j_tpr
del i_tpr
gc.collect()


# In[ ]:


with open(path.join(calculations_dir, "i_fpr.bin"), "rb") as f:
    i_fpr = pd.DataFrame(np.fromfile(f), columns=["i_fpr"])

with open(path.join(calculations_dir, "j_fpr.bin"), "rb") as f:
    j_fpr = pd.DataFrame(np.fromfile(f), columns=["j_fpr"])

with open(path.join(calculations_dir, "pred_equality_ratio.bin"), "wb+") as f:
    get_pred_equality_ratio(j_fpr['j_fpr'], i_fpr['i_fpr']).to_numpy().tofile(f)

with open(path.join(calculations_dir, "pred_equality_diff.bin"), "wb+") as f:
    get_pred_equality_diff(j_fpr['j_fpr'], i_fpr['i_fpr']).to_numpy().tofile(f)

del j_fpr
del i_fpr
gc.collect()


# In[ ]:


with open(path.join(calculations_dir, "i_ppv.bin"), "rb") as f:
    i_ppv = pd.DataFrame(np.fromfile(f), columns=["i_ppv"])

with open(path.join(calculations_dir, "j_ppv.bin"), "rb") as f:
    j_ppv = pd.DataFrame(np.fromfile(f), columns=["j_ppv"])

with open(path.join(calculations_dir, "pos_pred_parity_ratio.bin"), "wb+") as f:
    get_pos_pred_parity_ratio(j_ppv['j_ppv'], i_ppv['i_ppv']).to_numpy().tofile(f)

with open(path.join(calculations_dir, "pos_pred_parity_diff.bin"), "wb+") as f:
    get_pos_pred_parity_diff(j_ppv['j_ppv'], i_ppv['i_ppv']).to_numpy().tofile(f)

del j_ppv
del i_ppv
gc.collect()


# In[ ]:


with open(path.join(calculations_dir, "i_npv.bin"), "rb") as f:
    i_npv = pd.DataFrame(np.fromfile(f), columns=["i_npv"])

with open(path.join(calculations_dir, "j_npv.bin"), "rb") as f:
    j_npv = pd.DataFrame(np.fromfile(f), columns=["j_npv"])

with open(path.join(calculations_dir, "neg_pred_parity_ratio.bin"), "wb+") as f:
    get_neg_pred_parity_ratio(j_npv['j_npv'], i_npv['i_npv']).to_numpy().tofile(f)

with open(path.join(calculations_dir, "neg_pred_parity_diff.bin"), "wb+") as f:
    get_neg_pred_parity_diff(j_npv['j_npv'], i_npv['i_npv']).to_numpy().tofile(f)

del j_npv
del i_npv
gc.collect()

