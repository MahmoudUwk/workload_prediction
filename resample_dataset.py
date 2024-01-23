# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:04:26 2023

@author: mahmo
"""
import pandas as pd


import os
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/1Hz"
# data_path = "C:/Users/msallam/Desktop/Kuljeet/1Hz"
# data_path = 'C:/Users/msallam/Desktop/Energy Prediction/1Hz'
# sav_path = "C:/Users/msallam/Desktop/Energy Prediction/resampled data"
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/pwr data paper 2/resampled data"


onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and '.csv' in f]

def resample(df,txt):
    df_downsampled = df.resample(txt).mean()
    del df
    df_downsampled = df_downsampled.dropna()
    df_downsampled.to_csv(os.path.join(sav_path,txt+'.csv'))
    print(df_downsampled.shape)

for counter , file in enumerate(onlyfiles):
    full_path = os.path.join(data_path,file)
    if counter == 0:
        df = pd.read_csv(full_path)
    else:
        df_temp = pd.read_csv(full_path)
        df = pd.concat([df, df_temp])#.sort_values('timestamp').reset_index(drop=True)
        print(df_temp,full_path)
print(df.shape)
#%%
df.set_index(pd.to_datetime(df.timestamp), inplace=True)
df.drop(columns=["timestamp"], inplace=True)
# df = df.dropna()
# df.to_csv(os.path.join(sav_path,'1Hz.csv'))
# df = pd.read_csv(data_path)

resample(df,'1T')

resample(df,'10T')

resample(df,'15T')

resample(df,'30T')

