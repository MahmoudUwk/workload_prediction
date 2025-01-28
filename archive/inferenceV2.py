import os
import keras
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from Alibaba_helper_functions import loadDatasetObj,save_object,flatten,MAPE,MAE,RMSE,expand_dims,list_to_array
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
import warnings
# from keras.models import load_model 
import time
from args import get_paths
from tensorflow.saved_model import load
_,_,_,_,_,sav_path,_ = get_paths()

def split_3d_array(array_3d, batch_size):
    num_samples = array_3d.shape[0]
    sub_arrays = []
    for i in range(0, num_samples, batch_size):
        sub_array = array_3d[i:i + batch_size]
        sub_arrays.append(sub_array)
    return sub_arrays

if not os.path.exists(sav_path):
    os.makedirs(sav_path)
    
opt = 1 #0 for endelstmatt and 1 for regular LSTM
folder_name = ['saved_models','LSTM']
sig = ['EnDeAtt','LSTM']
sav_path_global = sav_path
models_path = os.path.join(sav_path,folder_name[opt])
best_params = loadDatasetObj(os.path.join(models_path,'Best_paramCuckooSearch_val_population_5_itr_15.obj'))
seq_len = best_params['best_para_save']['seq']
#%% Alib
model = load(os.path.join(models_path,'Alibaba'))
filename = os.path.join(sav_path_global,'cuckoosearch_Ali'+sig[opt]+'.obj')
X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(seq_len,2)

y_test_pred_list = []
rmse_list = []
start_test = time.time()
for c,test_sample in enumerate(X_test_list):
    pred_i = model.serve(test_sample)
    # pred_i = model.predict(test_sample)
    y_test_pred_list.append(pred_i*scaler)
    rmse_i_list = RMSE(y_test_list[c]*scaler,pred_i*scaler)
    y_test_list[c] = y_test_list[c]*scaler
    rmse_list.append(rmse_i_list)
end_test = time.time()
test_time = end_test - start_test
obj = {'y_test':y_test_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list),'best_params':best_params,'Mids_test':Mids_test}
save_object(obj, filename)

print(np.mean(rmse_list))
#%% google
# model = load(os.path.join(models_path,'google'))
from get_data_infere_py import get_data_inf
scaler = 100
feat_names = ['cpu_utilization', 'memory_utilization']
target = ['cpu_utilization']
X_list,Y_list,M_ids = get_data_inf(seq_len,feat_names,target)
batch_size = 2**9
y_test_pred_list = []
rmse_list_google = []
from args_google import get_paths
_,_,_,_,_,sav_path,_ = get_paths()

for c,test_sample_all in enumerate(X_list):
    if len(test_sample_all)>batch_size:
        test_sample_all = split_3d_array(test_sample_all, batch_size)
    else:
        test_sample_all = [test_sample_all]
    pred_i = []
    for test_sample in test_sample_all:
        pred_ii = list(np.squeeze(np.array(model.serve(test_sample/scaler))) *scaler)
        pred_i.append(pred_ii)
    pred_i = flatten(pred_i)
    y_test_pred_list.append(pred_i)
    rmse_i_list = RMSE(Y_list[c],pred_i)
    rmse_list_google.append(rmse_i_list)

obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_google),'Mids_test':M_ids}
filename = os.path.join(sav_path,'cuckoosearch_google'+sig[opt]+'.obj')
save_object(obj, filename)
# [len(i) for i in  y_test_pred_list]
print(np.mean(rmse_list_google))
#%% bitbrain
# model = load(os.path.join(models_path,'BB'))
from get_data_infere_py_BB import get_data_inf_BB
feat_names = ['CPU usage [%]', 'memory_utilization']
target = 'CPU usage [%]'
id_m = "machine_id"
sort_by = 'Timestamp [ms]'
X_list,Y_list,M_ids = get_data_inf_BB(seq_len,feat_names,target)
y_test_pred_list = []
rmse_list_BB = []
from args_BB import get_paths
_,_,_,_,_,sav_path,_ = get_paths()
for c,test_sample_all in enumerate(X_list):
    if len(test_sample_all)>batch_size:
        test_sample_all = split_3d_array(test_sample_all, batch_size)
    else:
        test_sample_all = [test_sample_all]
    pred_i = []
    for test_sample in test_sample_all:
        pred_ii = list(np.squeeze(np.array(model.serve(test_sample/scaler))) *scaler)
        pred_i.append(pred_ii)
    # pred_i = np.clip(pred_i,0,100)
    pred_i = flatten(pred_i)
    #print(len(pred_i))
    y_test_pred_list.append(pred_i)
    rmse_i_list = RMSE(Y_list[c],pred_i)
    rmse_list_BB.append(rmse_i_list)

obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_BB),'Mids_test':M_ids}
filename = os.path.join(sav_path,'cuckoosearch_BB'+sig[opt]+'.obj')
save_object(obj, filename)

print(np.mean(rmse_list_BB))
