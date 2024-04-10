# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:37 2024

@author: mahmo
"""
import pickle
import os
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj,save_object
# sav_path = 'C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/feature_obj'
sav_path = "data/feature_obj"
arr = os.listdir(sav_path)
#%%

test_per = 0.2
for counter,file_i in enumerate(arr):
    print(counter)
    file = os.path.join(sav_path,file_i)
    df = loadDatasetObj(file)
    train_len = int(len(df["X"])*(1-test_per))
    if counter == 0:
        X_train = df["X"].iloc[:train_len,:]
        y_train = df["y"][:train_len]
        
        X_test = df["X"].iloc[train_len:,:]
        y_test = df["y"][train_len:]
    else:
        X_train  = pd.concat([X_train, df["X"].iloc[:train_len,:]], ignore_index=True)
        y_train = y_train + df["y"][:train_len]
        
        X_test  = pd.concat([X_test, df["X"].iloc[train_len:,:]], ignore_index=True)
        y_test = y_train + df["y"][train_len:]

    
data_set = {'X_train':X_train,'y_train':y_train,
            'X_test':X_test,'y_test':y_test}
filename = os.path.join(sav_path,'X_Y_alibaba.obj')


save_object(data_set, filename)