# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:04:26 2023

@author: mahmo
"""
import pandas as pd
import pickle

import os
from os import listdir
from os.path import isfile, join
from args_BB import get_paths
base_path,processed_path,feat_google_step1,feat_google_step2,feat_google_step3,sav_path,sav_path_plots = get_paths()

txt = 'rnd' #fastStorage #rnd

data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/BB/rnd/2013-9"

sav_path = base_path

files_names = [] #file names including the complete path of each file.
machineid = []
for subdir, dirs, files in os.walk(data_path): #loop over files
  for file in files:
      if file.endswith(".csv"):
          full_file = os.path.join(subdir,file).lower()
          files_names.append(full_file) #append file names, this is redundant, not needed but stored with the dataset
          machineid.append(int(file.split('.csv')[0]))
#%%
for counter , full_path in enumerate(files_names):
    print(counter)
    if counter == 0:
        df =  pd.read_csv(full_path,sep=';\t',engine='python')
        df['date'] = pd.to_datetime(df['Timestamp [ms]'],unit='s')#.apply(lambda x: x.date())
        df['machine_id'] = [machineid[counter]]*len(df)
    else:
        df_temp =  pd.read_csv(full_path,sep=';\t',engine='python')
        df_temp['date'] = pd.to_datetime(df_temp['Timestamp [ms]'],unit='s')#.apply(lambda x: x.date())
        df_temp['machine_id'] = [machineid[counter]]*len(df_temp)
        df = pd.concat([df, df_temp])

#%%
print(df.shape)
filehandler = open(os.path.join(sav_path, txt+'.obj'), 'wb') #save the dataset using pickle as an object 
pickle.dump(df, filehandler)#saving the dataset
filehandler.close() #closing the object that saved the dataset


