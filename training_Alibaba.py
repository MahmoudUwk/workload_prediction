# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all

from Alibaba_helper_functions import loadDatasetObj,save_object
import os
#%%
models = ["linear_reg","svr_reg","GBT_reg"]#,"GPR_reg"]
data_path = 'data/Datasets/Proccessed_Alibaba'
sav_path = 'data/models'
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
file_read  = os.listdir(data_path)
reg_trained_all = []
class_trained_all = []
RMSE_opt_all = []
for M_id,file_M_id in enumerate(file_read):
    reg_trained_M_id = []
    print(M_id,file_M_id)
    filename = os.path.join(data_path,file_M_id)
    df = loadDatasetObj(filename)
    
    
    X_train = np.array(df['X_train'] .drop(['M_id'],axis=1))
    y_train = df['y_train']
    
    X_test = np.array(df['X_test'].drop(['M_id'],axis=1))
    y_test = df['y_test']
    
    print(X_train.shape,X_test.shape)
    #%%
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #%%
    
    reg_path = os.path.join(sav_path,str(M_id))
    if not os.path.exists(reg_path):
        os.makedirs(reg_path)
    pred_rr_train = np.zeros((len(X_train),len(models)))
    pred_rr_test = np.zeros((len(X_test),len(models)))
    for model_counter , model in enumerate(models):
        reg_name =  os.path.join(reg_path ,model+'_m_id_'+str(M_id)+'.obj')
        if not os.path.exists(reg_name):
            print(model)
            reg_trained = reg_all(X_train,y_train,X_test,model)
            save_object(reg_trained,reg_name)
        

    
    












