# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all


import pickle
import os


def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def RMSE(test,pred):
    return np.abs(test-pred)

def expand_dims(X):
    return np.expand_dims(X, axis = len(X.shape))


data_path = 'C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/Proccessed_Alibaba'
file_read  = os.listdir(data_path)
filename = os.path.join(data_path,file_read[0])
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

# y_train = expand_dims(y_train)
# y_test = expand_dims(y_test)
# scaler = MinMaxScaler()
# scaler.fit(y_train)
# y_train = np.squeeze(scaler.transform(y_train))
# y_test = np.squeeze(scaler.transform(y_test))


#%%
models = ["linear_reg","svr_reg","GPR_reg","GBT_reg"]
# models_dict = dict((el,c) for c,el in enumerate(models))

reg_trained_all = []
pred_rr_train = np.zeros((len(X_train),len(models)))
pred_rr_test = np.zeros((len(X_test),len(models)))
for model_counter , model in enumerate(models):
    print(model)
    reg_trained,y_pred_train,y_pred_test = reg_all(X_train,y_train,X_test,model)
    # print(y_pred.round(2))
    pred_rr_train[:,model_counter] = np.abs(y_pred_train-y_train)
    pred_rr_test[:,model_counter] = np.abs(y_pred_test-y_test)
    reg_trained_all.append(reg_trained)
    
    
 
y_train_c = np.argmin(pred_rr_train,axis=1)
y_test_c = np.argmin(pred_rr_test,axis=1)
#%%
class_models_names = ["KNN","GNB","RDF","GBT"]
# class_models_names = ["KNN","MLP","GNB","RDF","GBT"]
class_trained_all = []
acc_c = np.zeros((len(class_models_names)))
for model_counter , model in enumerate(class_models_names):
    print(model)
    classifier_trained,y_pred = class_all(X_train,y_train_c,X_test,model)
    # print(y_pred.round(2))
    acc_c[model_counter]= np.mean(y_pred==y_test_c)
    class_trained_all.append(classifier_trained)


print(acc_c)
print(class_models_names[np.argmax(acc_c)])























