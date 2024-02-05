# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:26:24 2024

@author: mahmo
"""
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pandas as pd
import pickle
path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/proccessed"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None
sav_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/results"
def feature_creation(data):
    df = data.copy()
    df['Minute'] = data.index.minute
    # df['Second'] = data.index.second
    df['DOW'] = data.index.dayofweek
    df['H'] = data.index.hour
    # df['W'] = data.index.week
    return df

def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

#rnd fastStorage
txt = "rnd"
data_path = os.path.join(path,txt+'.obj')
sav_path = os.path.join(sav_path,txt)

df = loadDatasetObj(data_path) #loading the dataset

df.set_index(df.date, inplace=True)
df.drop(columns=["Timestamp [ms]"], inplace=True)
df.drop(columns=["date"], inplace=True)

# df = feature_creation(df)
df_corr = df.corr()
#%%

plt.figure(figsize=(10,7),dpi=180)
xticks_font_size = 5
# sns.set(font_scale=1.2)
plt.rc('xtick', labelsize=xticks_font_size)
plt.rc('ytick', labelsize=xticks_font_size)
sns.heatmap(df_corr.round(2), annot=True, annot_kws={"size": 10})
plt.savefig(os.path.join(sav_path,"corr.png"))








