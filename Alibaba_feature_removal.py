# -*- coding: utf-8 -*-
#step 3
"""
Created on Tue Feb 20 11:31:26 2024
df roll
load feat
feature removal
@author: msallam
"""
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from models_lib import reg_all,class_all
from feature_selector import FeatureSelector
# import pickle
# from Alibaba_helper_functions import get_Mid
import os
from Alibaba_helper_functions import loadDatasetObj,save_object,drop_col_nan_inf
from args import get_paths

#%%
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3 = get_paths()

data_path = feat_stats_step2
sav_path = feat_stats_step3
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
filename_features_remove = os.path.join(sav_path,'all_to_remove.obj')
run_removal = 1

filename = os.path.join(data_path,'X_Y_alibaba_all.obj')
df_original = loadDatasetObj(filename)

df = df_original['XY_all'].drop(['y'], axis=1)
feat_all = set(list(df.columns))
df = drop_col_nan_inf(df)
feat_nan = list(feat_all-set(list(df.columns)))
y = np.array(df_original['XY_all']['y'])
#%%
if run_removal == 1:
    fs = FeatureSelector(data = df, labels = y)

    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    # fs.plot_unique()
    #%%
    fs.identify_collinear(correlation_threshold=0.98)
    correlated_features = fs.ops['collinear']
    # fs.plot_collinear()
    #%%
    fs.identify_zero_importance(task = 'regression', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
    zero_importance_f = fs.ops['zero_importance']
    # fs.identify_low_importance(0.95)
    # low_importance_f = fs.ops['low_importance']
    
    print('LGBM finished-------------------')
    #%%
    all_to_remove = fs.check_removal()
    print(all_to_remove)
    train_removed = fs.remove(methods = 'all')
    
    save_object(all_to_remove, filename_features_remove)
    
else:
    #%%
    all_to_remove = loadDatasetObj(filename_features_remove)
    

all_to_remove = list(set(all_to_remove + feat_nan))
#%%
filename = os.path.join(data_path,'X_Y_alibaba_train_val_test_before_removing_features.obj')

df_tvt = loadDatasetObj(filename)
XY_train = df_tvt['XY_train'].drop(all_to_remove, axis=1)
if len(df_tvt['XY_val'])!=0:
    XY_val = df_tvt['XY_val'].drop(all_to_remove, axis=1)
else:
    XY_val=[]
XY_test = df_tvt['XY_test'].drop(all_to_remove, axis=1)

#%%
data_set = {'XY_train':XY_train,
            'XY_val':XY_val,
            'XY_test':XY_test}
filename = os.path.join(sav_path,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
save_object(data_set, filename)















