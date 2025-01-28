# -*- coding: utf-8 -*-
#step 2
"""
Created on Fri Feb 16 15:23:37 2024

@author: mahmo
"""
# import pickle
import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj,save_object,get_train_test_Mids
import numpy as np
np.random.seed(8)

def agument_data(data_path,arr):
    if len(arr)==0:
        return []
    # M_ids = []
    X_list = []
    # y_list = []
    for counter,file_i in enumerate(arr):
        file = os.path.join(data_path,file_i)
        df = loadDatasetObj(file)
        X_list.append(df["XY"])

    XY = pd.concat(X_list)
    print('Done')
    return XY
#%%
from args_google import get_paths
base_path,processed_path,feat_google_step1,feat_google_step2,feat_google_step3,sav_path,sav_path_plots = get_paths()


data_path = feat_google_step1
sav_path = feat_google_step2
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
arr = os.listdir(data_path)
# google_ids = [fil.split('id_')[1].split('.obj')[0] for fil in arr]
#%%

XY_google = agument_data(data_path,arr)


data_set = {'XY_google':XY_google}
filename = os.path.join(sav_path,'X_Y_google_before_removing_features.obj')
save_object(data_set, filename)
