# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:49:43 2024

@author: mahmo
"""

import pandas as pd
import os
base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
# base_path = "data/"

sav_path = base_path+"features_lstm"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
    
    
script = "server_usage.csv"
target = " used percent of cpus(%)"
id_str = " machine id"
# cols = [' used percent of cpus(%)']
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

info_path = base_path+"schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']

full_path = base_path+script
nrows = None
df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))
df = df.dropna()

#%%
# df[[id_str,target]].boxplot(by=id_str)