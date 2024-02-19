# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:09:55 2024

@author: mahmo
"""
import pandas as pd
import numpy as np
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import roll_time_series
script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

info_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']
#%%
def sliding_windows(data, seq_length):
    x = np.zeros((len(data)-seq_length,seq_length))
    y = np.zeros((len(data)-seq_length))
    #print(x.shape,y.shape)
    for ind in range(len(x)):
        #print((i,(i+seq_length)))
        x[ind,:] = data[ind:ind+seq_length]
        #print(data[ind+seq_length:ind+seq_length+k_step])
        y[ind] = data[ind+seq_length:ind+seq_length+1][0]
    return x,y

def list_to_array(lst,seq_length):
    shapes = 0
    for sub_list in lst:
        shapes += len(sub_list)
    ind = 0
    if seq_length != 0:
        X = np.zeros((shapes,seq_length))
        for sub_list in lst:
            X[ind:ind+len(sub_list),:] = sub_list
            ind = ind + len(sub_list)
    else:
        X = np.zeros((shapes,))
        for sub_list in lst:
            X[ind:ind+len(sub_list)] = sub_list
            ind = ind + len(sub_list)
  
    return X

full_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/"+script
nrows = None
df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))

df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
# df = df[df.notna()]
df = df.dropna()
seq = 12

#%%
grouped = df.groupby([" machine id"])
dataset_widnows = []
M_ids = []
label_pred = []
for M_id, M_id_val in grouped:
    # print(M_id,M_id_val[target][:142].shape)
    x,y = sliding_windows(np.array(M_id_val[target]), seq)
    # if x.shape!=(130,12):
        # print(M_id)
        # print(M_id,M_id_val[target][:142].shape)
    label_pred.append(y)
    dataset_widnows.append(x)
    M_ids.append([M_id[0]]*len(y))

M_ids = [ x for xs in M_ids for x in xs]
dataset_widnows = list_to_array(dataset_widnows,seq)

label_pred = list_to_array(label_pred,0)









