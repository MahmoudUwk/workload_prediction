# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:31:26 2024

@author: msallam
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models_lib import reg_all,class_all
from feature_selector import FeatureSelector
import pickle
from Alibaba_helper_functions import get_Mid
import os
def loadDatasetObj(fname):
    file_id = open(fname, 'rb') 
    data_dict = pickle.load(file_id)
    file_id.close()
    return data_dict

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

base_path = "C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/"
sav_path = base_path+"Proccessed_Alibaba"
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
    
M_ids = get_Mid()
data_path = 'C:/Users/msallam/Desktop/Cloud project/Datasets/Alidbaba/feature_obj'
filename = os.path.join(data_path,'X_Y_alibaba.obj')
df_original = loadDatasetObj(filename)
# df['X_train'] = df['X_train'].set_index("M_id")

def drop_col_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)
#%%
for counter,M_id in enumerate(M_ids):
    
    ind_Mid = np.where(df_original['X_train']["M_id"].isin(M_id))[0]
    
    df_train = df_original['X_train'].loc[ind_Mid]
    df_train = drop_col_nan_inf(df_train)
    y_M_id = np.array(df_original['y_train'])[ind_Mid]
    
    ind_Mid_test = np.where(df_original['X_test']["M_id"].isin(M_id))[0]
    df_test_M_id = df_original['X_test'].loc[ind_Mid_test]
    df_test_M_id = drop_col_nan_inf(df_test_M_id)
    
    y_M_id_test = np.array(df_original['y_test'])[ind_Mid_test]

    fs = FeatureSelector(data = df_train, labels = y_M_id)
    #%%
    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    # fs.plot_unique()
    #%%
    fs.identify_collinear(correlation_threshold=0.98)
    correlated_features = fs.ops['collinear']
    # print(correlated_features[:5])
    # fs.plot_collinear()
    #%%
    fs.identify_zero_importance(task = 'regression', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
    zero_importance_f = fs.ops['zero_importance']
    print('LGBM finished-------------------')
    #%%
    all_to_remove = fs.check_removal()
    # print(all_to_remove)
    train_removed = fs.remove(methods = 'all')
    
    df_test_M_id = df_test_M_id.drop(all_to_remove, axis=1)

    data_set = {'X_train':train_removed,'y_train':y_M_id,
                'X_test':df_test_M_id,'y_test':y_M_id_test}
    filename = os.path.join(sav_path,'X_Y_alibaba_M_id'+str(counter)+'.obj')


    save_object(data_set, filename)














