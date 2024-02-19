# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:23:37 2024

@author: mahmo
"""
import pickle
import os
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/feature_obj"
arr = os.listdir(sav_path)
#%%
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

for file_i in arr:
    file = os.path.join(sav_path,file_i)
    df = loadDatasetObj(file)
    print(df.shape)