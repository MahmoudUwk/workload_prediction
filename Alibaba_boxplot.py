# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:23:49 2024

@author: mahmo
"""

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
        y[ind] = data[ind+seq_length:ind+seq_length+1]
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

#%%


df = df[df[target]<63]
# print(df.head())

# print(df.mean())


df_CPU = df[target]
# df.groupby([" machine id"]).std()[target].hist()
# df.groupby([" machine id"]).boxplot(column=[target])
df_grouped_id = df.groupby([" machine id"])
std_therhold = [2,7]

M4 = np.where(df_grouped_id.std()[target]<std_therhold[0])[0]
M12 = np.where(np.array(df_grouped_id.std()[target]<std_therhold[1]) * np.array(df_grouped_id.std()[target]>std_therhold[0]))[0]
M3 = np.where(df_grouped_id.std()[target]>std_therhold[1])[0]

print(len(M4),len(M12),len(M3))

df_M4 = df.loc[df[" machine id"].isin(M4)][target]
df_M12 = df.loc[df[" machine id"].isin(M12)][target]
df_M3= df.loc[df[" machine id"].isin(M3)][target]

df_M4.boxplot(column=[' used percent of cpus(%)'])
df_M12.boxplot(column=[' used percent of cpus(%)'])
df_M3.boxplot(column=[' used percent of cpus(%)'])

import matplotlib.pyplot as plt

plt.boxplot([df_M4,df_M12,df_M3], labels=["M4","M12","M3"])













