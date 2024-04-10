# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all

from Alibaba_helper_functions import loadDatasetObj,save_object,diff,RMSE,expand_dims,expand_dims_st
import os
#%%
# base_path = "data/"
models = ["linear_reg","svr_reg","GBT_reg"]#,"GPR_reg"]
class_models_names = ["KNN","GNB","RDF","GBT","MLP"]
data_path = 'data/Datasets/Proccessed_Alibaba'
sav_path = 'data/models'
file_read  = os.listdir(data_path)

RMSE_opt_all = []
for M_id,file_M_id in enumerate(file_read):
    reg_trained_M_id = []
    # class_trained_M_id  = []
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
    pred_rr_train = np.zeros((len(X_train),len(models)))
    pred_rr_test = np.zeros((len(X_test),len(models)))
    for model_counter , model in enumerate(models):
        print(model)
        reg_name = os.path.join(sav_path,str(M_id))
        reg_name =  os.path.join(reg_name ,model+'_m_id_'+str(M_id)+'.obj')
        reg_trained = loadDatasetObj(reg_name)
        y_pred_train = reg_trained.predict(X_train)
        y_pred_test = reg_trained.predict(X_test)
        pred_rr_train[:,model_counter] = np.abs(y_pred_train-y_train)
        pred_rr_test[:,model_counter] = np.abs(y_pred_test-y_test)
        reg_trained_M_id.append(reg_trained)
        
        
     
    y_train_c = np.argmin(pred_rr_train,axis=1)
    y_test_c = np.argmin(pred_rr_test,axis=1)

    # class_XY = {'X_train':X_train,'y_train_c':y_train_c,'X_test':X_test,'y_test_c':y_test_c}
    # save_object(class_XY,'data\\data.obj')
    #%%
    
    acc_c = np.zeros((len(class_models_names)))
    y_pred_all = []
    for model_counter , model in enumerate(class_models_names):
        print(model)
        classifier_trained,y_pred = class_all(X_train,y_train_c,X_test,model)
        y_pred_all.append(y_pred)
        # print(y_pred.round(2))
        acc_c[model_counter]= np.mean(y_pred==y_test_c)
        # class_trained_M_id.append(classifier_trained)
    
    
    print(acc_c)
    print(class_models_names[np.argmax(acc_c)])
    #%%
    
    model_best = class_models_names[np.argmax(acc_c)]
    
    y_pred = y_pred_all[np.argmax(acc_c)]
    y_pred_opt = np.zeros((len(X_test)))
    for c_i , test_instance in enumerate(X_test):
        y_pred_opt[c_i] = reg_trained_M_id[y_pred[c_i]].predict(expand_dims_st(test_instance))[0]
    
    
    
    RMSE_opt_all.append(RMSE(y_pred_opt,y_test))
    
#%%
    
clusters= [459, 13, 205, 633]

RMSE_all =np.sum( np.array(RMSE_opt_all)*np.array(clusters)/sum(clusters))

print(RMSE_all)











