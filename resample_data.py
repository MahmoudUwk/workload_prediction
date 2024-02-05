# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:16:39 2024

@author: mahmo
"""
import pandas as pd
import pickle

import os
from os import listdir
from os.path import isfile, join

txt = 'fastStorage' #fastStorage #rnd

path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/proccessed"


#%%
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def resample(df,sav_path,txt_resample):
    df_downsampled = df.resample(txt_resample).mean()

    df_downsampled = df_downsampled.dropna()
    sav_obj(df_downsampled,sav_path,txt_resample)
    print(df_downsampled.shape)
    
def sav_obj(df,sav_path,txt_resample):
    filehandler = open(os.path.join(sav_path, txt_resample+'.obj'), 'wb') #save the dataset using pickle as an object 
    pickle.dump(df, filehandler)#saving the dataset
    filehandler.close() #closing the object that saved the dataset


#%%
txt = "rnd"
data_path = os.path.join(path,txt+'.obj')
# sav_path = os.path.join(path,txt)

df = loadDatasetObj(data_path) #loading the dataset

df.set_index(df.date, inplace=True)
df.drop(columns=["Timestamp [ms]"], inplace=True)
df.drop(columns=["date"], inplace=True)


#%%
resample(df,path,'10T')

resample(df,path,'15T')

resample(df,path,'30T')

