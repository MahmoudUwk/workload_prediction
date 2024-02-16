# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:55:58 2024

@author: mahmo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:09:55 2024

@author: mahmo
"""
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
import pickle
import os
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


data_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba"
# data_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"
sav_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/feature_obj"


if not os.path.exists(sav_path):
    os.makedirs(sav_path)


script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

info_path = data_path+"/schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']



full_path = os.path.join(data_path,script)
df =  pd.read_csv(full_path,header=None,names=list(df_info))

df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
# df = df[df.notna()]
df = df.dropna()
seq = 12
#%%

Ids = np.sort(np.unique(df[" machine id"]))
num_splits = 5
batch_sizes = np.split(Ids,num_splits)

for batch_num, batch_indeces in enumerate(batch_sizes):
    print(batch_num)
    df_batch = df.loc[df[" machine id"].isin(batch_indeces)]
    # print(M_id,M_id_val[target][:142].shape)
    df_rolled = roll_time_series(df_batch, column_id=' machine id', column_sort=" timestamp",n_jobs = 1,max_timeshift=seq)
    df_features = extract_features(df_rolled, column_id="id", column_sort=" timestamp",n_jobs = 1).dropna(axis=1, how='all')

    save_object(df_features, os.path.join(sav_path,'df_features'+batch_num+'.obj'))













