# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:37 2024

@author: mahmo
"""
import pickle
import os
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj,save_object,get_train_test_Mids
import numpy as np
np.random.seed(8)

def agument_data(data_path,arr):
    if len(arr)==0:
        return []
    M_ids = []
    X_list = []
    # y_list = []
    for counter,file_i in enumerate(arr):
        file = os.path.join(data_path,file_i)
        df = loadDatasetObj(file)
        # M_ids.append(df["X"]['M_id'])
        X_list.append(df["XY"])
        # y_list.append(df["y"])
        # if counter == 0:
        #     X = df["X"]
        #     # y = df["y"]
            
        # else:
        #     # X  = pd.concat([X, df["X"]], ignore_index=True)
        #     y = y + df["y"]
    XY = pd.concat(X_list)
    print('Done')
    return XY




#%%
base_path = "data/"
data_path = base_path+"feature_statistical"
sav_path = base_path+"feature_statistical_proccessed"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
arr = np.array(os.listdir(data_path))

train_val_per = 0.8
val_per = 0

M_ids_train, M_ids_val, M_ids_test = get_train_test_Mids(base_path,train_val_per,val_per)

arr_train = ['X_Y_M_id_'+str(M_id)+'.obj' for M_id in M_ids_train]

arr_val = ['X_Y_M_id_'+str(M_id)+'.obj' for M_id in M_ids_val]

arr_test = ['X_Y_M_id_'+str(M_id)+'.obj' for M_id in M_ids_test]



XY_train = agument_data(data_path,arr_train)

XY_val = agument_data(data_path,arr_val)

XY_test = agument_data(data_path,arr_test)



data_set = {'XY_train':XY_train,
            'XY_val':XY_val,
            'XY_test':XY_test}
filename = os.path.join(sav_path,'X_Y_alibaba_train_val_test_before_removing_features.obj')
save_object(data_set, filename)

#%%
XY_all = agument_data(data_path,arr)
data_set = {'XY_all':XY_all}
filename = os.path.join(sav_path,'X_Y_alibaba_all.obj')
save_object(data_set, filename)