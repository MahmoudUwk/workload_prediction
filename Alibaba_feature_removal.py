# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:31:26 2024

@author: msallam
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all
from feature_selector import FeatureSelector
import pickle
import os
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict


data_path = 'C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/feature_obj'
filename = os.path.join(data_path,'X_Y_alibaba.obj')
df = loadDatasetObj(filename)
df['X_train'] = df['X_train'].set_index("M_id")

#%%
fs = FeatureSelector(data = df['X_train'], labels = df['y_train'])
#%%
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
fs.plot_unique()
#%%
fs.identify_collinear(correlation_threshold=0.98)
correlated_features = fs.ops['collinear']
print(correlated_features[:5])
fs.plot_collinear()
#%%
fs.identify_zero_importance(task = 'regression', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
zero_importance_f = fs.ops['zero_importance']
print('LGBM finished-------------------')
#%%
all_to_remove = fs.check_removal()
print(all_to_remove)
train_removed = fs.remove(methods = 'all')

















