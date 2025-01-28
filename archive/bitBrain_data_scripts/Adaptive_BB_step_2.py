# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:05:49 2024

@author: mahmo
"""
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from models_lib import class_all
import time
from Alibaba_helper_functions import loadDatasetObj,flatten,save_object,RMSE,expand_dims_st,get_data_stat
import os
from args_BB import get_paths
_,_,_,_,feat_BB_step3,sav_path,_ = get_paths()
if not os.path.exists(sav_path):
    os.makedirs(sav_path)
#%%
from args import get_paths
_,_,_,_,feat_stats_step3,_,_ = get_paths()
dat_path_obi = os.path.join(feat_stats_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
_,_,_,_,scaler,_ = get_data_stat(dat_path_obi)


models = ["linear_reg","svr_reg","GBT_reg"]
model_path = "C:/Users/mahmo/OneDrive/Desktop/kuljeet/Cloud project/Datasets/Alidbaba/base_proposed"

reg_trained_all = []
for model_counter , model in enumerate(models):
    reg_name =  os.path.join(model_path ,model+'.obj')
    print(model)
    data_reg = loadDatasetObj(reg_name)
    reg_trained = data_reg['reg_trained']
    reg_trained_all.append(reg_trained)

BC_path = os.path.join(model_path,'best_classifier.obj')
class_trained_best = loadDatasetObj(BC_path)

#%%
dat_BB_obi = os.path.join(feat_BB_step3,'XY_test_ready.obj')

df_test_xy = loadDatasetObj(dat_BB_obi)['XY_test_ready']
id_m = 'M_id'
X_test = scaler.transform(np.array(df_test_xy.drop(['M_id','y'],axis=1)))
y_test  = np.array(df_test_xy['y'])
#%%.
save_pred_save = sav_path#os.path.join(sav_path,'Adaptive_Bitbrain_results')
if not os.path.exists(save_pred_save):
    os.makedirs(save_pred_save)
Mids_test = []
y_test_list = []
y_test_pred_list = []
rmse_list = []
start_test = time.time()
for m_id, group_val in  df_test_xy.groupby(["M_id"]):
    Mids_test.append(m_id[0])
    y_test_list.append(np.array(group_val['y']))
    X_test_Mid = scaler.transform(np.array(group_val.drop(['y','M_id'],axis=1)))
    y_pred_reg_best = []
    ind_regs = class_trained_best.predict(X_test_Mid)
    for counter,test_instance in enumerate(X_test_Mid):
        ind_reg = ind_regs[counter]
        # ind_reg = np.argmax(class_trained_best.predict(expand_dims_st(test_instance)))
        y_i = reg_trained_all[ind_reg].predict(expand_dims_st(test_instance))[0]
        y_pred_reg_best.append(y_i)
    
    y_test_pred_list.append(y_pred_reg_best)
    rmse_i_list = RMSE(np.array(group_val['y']),y_pred_reg_best)
    rmse_list.append(rmse_i_list)
end_test = time.time()
test_time = end_test - start_test
obj = {'y_test':y_test_list,'y_test_pred':y_test_pred_list,'rmse_list':np.array(rmse_list),'Mids_test':Mids_test}
filename = os.path.join(save_pred_save,'Adaptive_predictor.obj')
save_object(obj, filename)


print('RMSE:',np.mean(rmse_list))
print("RMSE:",RMSE(flatten(y_test_list),flatten(y_test_pred_list)))








