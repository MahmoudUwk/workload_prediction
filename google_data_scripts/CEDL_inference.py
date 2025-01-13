import os
import numpy as np
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_path)
import keras
# from args import get_paths
from Alibaba_helper_functions import list_to_array,loadDatasetObj,save_object,MAPE,MAE,RMSE,expand_dims,list_to_array,get_EN_DE_LSTM_model
from get_data_infere_py import get_data_inf
base_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/'
save_path_dat = base_path+'/pred_results_all'
loaded_model = keras.models.load_model(os.path.join(save_path_dat,"CEDL"))

scaler = 100
#%%
alg_name = 'CuckooSearch'
param_path = 'C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/pred_results_all\\CuckooSearch'
best_params = loadDatasetObj(os.path.join(param_path,'Best_param'+alg_name+'.obj'))['best_para_save']
feat_names = ['cpu_utilization', 'memory_utilization']
target = ['cpu_utilization']

X_list,Y_list,M_ids = get_data_inf(best_params['seq'],feat_names,target)


#%%
# X = list_to_array(X_list,best_params['seq'],2)
# Y = list_to_array(Y_list,0,2)
# print(X.shape,Y.shape)
# Y = expand_dims(expand_dims(Y))
# input_dim=(X.shape[1],X.shape[2])
# output_dim = Y.shape[-1]


# y_pred = (loaded_model.predict(X/scaler))*scaler
# rmse = RMSE(Y,y_pred)
# mae = MAE(Y,y_pred)
# mape = MAPE(Y,y_pred)
# print(rmse,mae,mape)
#%%
y_test_pred_list = []
rmse_list = []
from args_google import get_paths
base_path,processed_path,feat_google_step1,feat_google_step2,feat_google_step3,sav_path,sav_path_plots = get_paths()
# Y_list_scaled = Y_list.copy()
for c,test_sample in enumerate(X_list):
    pred_i = (loaded_model.predict(np.clip(test_sample,0,100)/scaler)) *scaler
    pred_i = np.clip(pred_i,0,100)
    y_test_pred_list.append(pred_i)
    rmse_i_list = RMSE(Y_list[c],pred_i)
    # Y_list_scaled[c] = Y_list[c]*scaler
    rmse_list.append(rmse_i_list)

obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list),'Mids_test':M_ids}
filename = os.path.join(sav_path,'CDEL_google_inference.obj')
save_object(obj, filename)

print(np.mean(rmse_list))