# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:59:14 2024

@author: mahmo
"""

import os
from args import get_paths
import pickle
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3 = get_paths()

files = os.listdir(feat_stats_step1)
#%%
for file in files:
    df = loadDatasetObj(os.path.join(feat_stats_step1,file))
    df_features = df['XY'].copy()
    ID = df_features['M_id'].iloc[0][0]
    df_features['M_id'] = [ID]*len(df_features)
    save_object({"XY":df_features}, os.path.join(feat_stats_step1,'X_Y_M_id_'+str(ID)+'.obj'))
    