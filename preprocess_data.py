# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:09:37 2024

@author: mahmo
"""

import pandas as pd
import pickle
import numpy as np
import os
from os import listdir
from os.path import isfile, join

txt = 'fastStorage' #fastStorage #rnd

data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/rnd"

sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/proccessed"

files_names = [] #file names including the complete path of each file.

rample_rate = "30T"

def save_data(sav_path,txt,df):
    filehandler = open(os.path.join(sav_path, txt+'.obj'), 'wb') #save the dataset using pickle as an object 
    pickle.dump(df, filehandler)#saving the dataset
    filehandler.close() #closing the object that saved the dataset



for subdir, dirs, files in os.walk(data_path): #loop over files
  for file in files:
      if file.endswith(".csv"):
          full_file = os.path.join(subdir,file).lower()
          files_names.append(full_file) #append file names, this is redundant, not needed but stored with the dataset


#%%
def get_date(df):
    df.set_index(df.date, inplace=True)
    df.drop(columns=["Timestamp [ms]"], inplace=True)
    df.drop(columns=["date"], inplace=True)
    return df
def resample(df,txt):
    df_downsampled = df.resample(txt).mean()
    df_downsampled = df_downsampled.dropna()
    # df_downsampled.to_csv(os.path.join(sav_path,txt+'.csv'))
    print(df_downsampled.shape)
    return df_downsampled

dataset = []
for counter , full_path in enumerate(files_names):
    df_temp =  pd.read_csv(full_path,sep=';\t',engine='python')
    df_temp['date'] = pd.to_datetime(df_temp['Timestamp [ms]'],unit='s')#.apply(lambda x: x.date())
    df_temp = get_date(df_temp)
    dataset.append(np.array(resample(df_temp,rample_rate)))
    # print(df_temp.shape)





