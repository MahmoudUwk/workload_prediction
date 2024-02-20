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

def df_from_M_id(df,M):
    return df.loc[df[" machine id"].isin(M)][" used percent of cpus(%)"]

script = "server_usage.csv"
target = " used percent of cpus(%)"
pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_columns = None

base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
info_path = base_path+"schema.csv"

df_info =  pd.read_csv(info_path)
df_info = df_info[df_info["file name"] == script]['content']



full_path = base_path+script
nrows = None
df =  pd.read_csv(full_path,nrows=nrows,header=None,names=list(df_info))

df = df[[" machine id", " timestamp"," used percent of cpus(%)"]]
# df = df[df.notna()]
df = df.dropna()



df_grouped_id = df.groupby([" machine id"])

mean_Mid= np.expand_dims(np.array(df_grouped_id.mean()[target]), axis=1)
std_Mid= np.expand_dims(np.array(df_grouped_id.std()[target]), axis=1)
X_Mid = np.concatenate((mean_Mid,std_Mid),axis=1)

#%%

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X_Mid)





#%%


df_grouped_id.mean()[target].hist()

#%%
mean_therhold = [5,30]
values_thershold = 63

M1 = np.where(df_grouped_id.mean()[target]>mean_therhold[1])[0]
M2 = np.where(df_grouped_id.mean()[target]<mean_therhold[0])[0]

df_M1 = df_from_M_id(df,M1)
df_M2 = df_from_M_id(df,M2)



M34 = np.array(list(set(list(df[" machine id"])) - set(list(M1)+list(M2))))


df_34 = df.loc[df[" machine id"].isin(M34)]

df_grouped_id = df_34.groupby([" machine id"])

df_grouped_id.std()[target].hist()
std_therhold = 1

M4 = np.where(df_grouped_id.std()[target]<std_therhold)[0]
M3 = np.where(df_grouped_id.std()[target]>std_therhold)[0]


df_M4 = df_from_M_id(df_34,M4)
df_M3 = df_from_M_id(df_34,M3)



print(len(df_M1),len(df_M2),len(df_M3),len(df_M4))
import matplotlib.pyplot as plt
plt.figure()
ax = plt.boxplot([df_M1,df_M2,df_M3,df_M4], labels=["M1","M2","M3","M4"])













