# -*- coding: utf-8 -*-
#step 3
"""
Created on Tue Feb 20 11:31:26 2024
df roll
load feat
feature removal
@author: msallam
"""

import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
from Alibaba_helper_functions import loadDatasetObj,save_object
from args_google import get_paths
base_path,processed_path,feat_google_step1,feat_google_step2,feat_google_step3,sav_path,sav_path_plots = get_paths()

data_path = feat_google_step2
sav_path = feat_google_step3
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
    
from args import get_paths
_,_,_,_,feat_stats_step3,_,_ = get_paths()
df_ali_path = os.path.join(feat_stats_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
col_feat_ali = list(loadDatasetObj(df_ali_path)['XY_train'].columns)
col_feat_google = ['cpu_utilization'+ feat.split(' used percent of cpus(%)')[1] for feat in col_feat_ali if ' used percent of cpus(%)' in feat]
col_feat_google = col_feat_google + ['M_id','y']
#%%
filename = os.path.join(feat_google_step2,'X_Y_google_before_removing_features.obj')
df = loadDatasetObj(filename)
# feat_all = set(list(df.columns))

#%%
data_set = {'XY_test_ready':df['XY_google'][col_feat_google].dropna(axis=0)}
filename = os.path.join(sav_path,'XY_test_ready.obj')
save_object(data_set, filename)















