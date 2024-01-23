# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:04:26 2023

@author: mahmo
"""
import pandas as pd


import os
from os import listdir
from os.path import isfile, join

data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/rnd"

sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/proccessed"

files_names = [] #file names including the complete path of each file.

for subdir, dirs, files in os.walk(data_path): #loop over files
  for file in files:
      if file.endswith(".csv"):
          full_file = os.path.join(subdir,file).lower()
          files_names.append(full_file) #append file names, this is redundant, not needed but stored with the dataset

txt = 'rnd'
#%%
def resample(df,txt):
    df_downsampled = df.resample(txt).mean()
    del df
    df_downsampled = df_downsampled.dropna()
    df_downsampled.to_csv(os.path.join(sav_path,txt+'.csv'))
    print(df_downsampled.shape)

for counter , full_path in enumerate(files_names):
    print(counter)
    # df_temp = pd.read_csv(full_path,sep=';/t',engine='python')
    if counter == 0:
        df =  pd.read_csv(full_path,sep=';\t',engine='python')
        # print(df.shape)
    else:
        df_temp =  pd.read_csv(full_path,sep=';\t',engine='python')
        df = pd.concat([df, df_temp])
        # print(df_temp.shape)
    # if counter == 0:
    #     dict_dataset = {col: list(df[col]) for col in list(df.columns)}

    # else:
    #     [dict_dataset[col].append(df[col]) for col in list(df.columns)]

# df = pd.DataFrame.from_dict(dict_dataset)
#%%
print(df.shape)

df.to_csv(os.path.join(sav_path,txt+'.csv'), index=False) 
#%%
# df.set_index(pd.to_datetime(df.timestamp), inplace=True)
# df.drop(columns=["timestamp"], inplace=True)
# # df = df.dropna()
# # df.to_csv(os.path.join(sav_path,'1Hz.csv'))
# # df = pd.read_csv(data_path)

# resample(df,'1T')

# resample(df,'10T')

# resample(df,'15T')

# resample(df,'30T')

