# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:10:02 2024

@author: msallam
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def get_Mid():
    script = "server_usage.csv"
    target = " used percent of cpus(%)"
    
    
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
    
    
    
    
    
    kmeans = KMeans(n_clusters=4, n_init="auto",algorithm='elkan',max_iter=5000,random_state=5).fit(X_Mid)
    
    M_id_labels = kmeans.labels_
    
    return [np.where(M_id_labels==class_Mid)[0] for  class_Mid in np.unique(kmeans.labels_)]
