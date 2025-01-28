# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np

from models_lib import reg_all
import time
from Alibaba_helper_functions import loadDatasetObj,save_object,get_data_stat
import os
from args import get_paths
base_path,processed_path,_,_,feat_stats_step3,sav_path = get_paths()
#%%
models = ["linear_reg","svr_reg","GBT_reg"]
data_path = os.path.join(feat_stats_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
sav_path = os.path.join(base_path,'base_proposed')
if not os.path.exists(sav_path):
    os.makedirs(sav_path)

# RMSE_opt_all = []
# test_set_len = []
train_time = []
#%%
for model_counter , model in enumerate(models):
    reg_name =  os.path.join(sav_path ,model+'.obj')
    if not os.path.exists(reg_name):
        print(model)
        X_train,y_train,X_test,y_test,scaler,_ = get_data_stat(data_path)
        start_train = time.time()
        reg_trained = reg_all(X_train,y_train,X_test,model)
        end_train = time.time()
        train_time = (end_train - start_train)/60
        save_object({'reg_trained':reg_trained,'train_time':train_time},reg_name)
    

    
    












