# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from feature_selector import FeatureSelector
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

data_path = 'C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/feature_obj'
filename = os.path.join(data_path,'X_Y_alibaba.obj')
df = loadDatasetObj(filename)
df['X_train'] = df['X_train'].set_index("M_id")

#%%

X, y = make_regression(random_state=0,n_samples=4000, n_features=15,bias=10,noise=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = expand_dims(y_train)
y_test = expand_dims(y_test)
scaler = MinMaxScaler()
scaler.fit(y_train)
y_train = np.squeeze(scaler.transform(y_train))
y_test = np.squeeze(scaler.transform(y_test))


#%%
models = ["linear_reg","svr_reg","GPR_reg","GBT_reg"]
models_dict = dict((el,c) for c,el in enumerate(models))

reg_trained_all = []
pred_rr_train = np.zeros((len(X_train),len(models)))
pred_rr_test = np.zeros((len(X_test),len(models)))
for model_counter , model in enumerate(models):
    reg_trained,y_pred_train,y_pred_test = reg_all(X_train,y_train,X_test,model)
    # print(y_pred.round(2))
    pred_rr_train[:,model_counter] = np.abs(y_pred_train-y_train)
    pred_rr_test[:,model_counter] = np.abs(y_pred_test-y_test)
    reg_trained_all.append(reg_trained)
    
    
 
y_train_c = np.argmin(pred_rr_train,axis=1)
y_test_c = np.argmin(pred_rr_test,axis=1)
#%%

class_models_names = ["KNN","MLP","GNB","RDF","GBT"]
class_trained_all = []
acc_c = np.zeros((len(class_models_names)))
for model_counter , model in enumerate(class_models_names):
    classifier_trained,y_pred = class_all(X_train,y_train_c,X_test,model)
    # print(y_pred.round(2))
    acc_c[model_counter]= np.mean(y_pred==y_test_c)
    class_trained_all.append(classifier_trained)


print(acc_c)
print(class_models_names[np.argmax(acc_c)])























