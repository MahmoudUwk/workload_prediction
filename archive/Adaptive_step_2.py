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
from args import get_paths
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

#%%
class_models_names = ["KNN","GNB","RDF","GBT","MLP"]
models = ["linear_reg","svr_reg","GBT_reg"]#,"GPR_reg"]

data_path = os.path.join(feat_stats_step3,'X_Y_alibaba_train_val_test_after_feature_removal.obj')
sav_path = os.path.join(base_path,'base_proposed')
if not os.path.exists(sav_path):
    os.makedirs(sav_path)


test_set_len = []
df = loadDatasetObj(data_path)

X_train,y_train,X_test,y_test,scaler,df_test_xy = get_data_stat(data_path)
#%%
pred_rr_train = np.zeros((len(X_train),len(models)))
pred_rr_test = np.zeros((len(X_test),len(models)))
reg_trained_all = []
train_time_all = []
for model_counter , model in enumerate(models):
    
    reg_name =  os.path.join(sav_path ,model+'.obj')
    print(model)
    data_reg = loadDatasetObj(reg_name)
    train_time_all.append(data_reg['train_time'])
    reg_trained = data_reg['reg_trained']
    y_pred_train = reg_trained.predict(X_train)
    y_pred_test = reg_trained.predict(X_test)
    pred_rr_train[:,model_counter] = np.abs(y_pred_train-y_train)
    pred_rr_test[:,model_counter] = np.abs(y_pred_test-y_test)
    reg_trained_all.append(reg_trained)
    
y_train_c = np.argmin(pred_rr_train,axis=1)
y_test_c = np.argmin(pred_rr_test,axis=1)

#%%

acc_c = np.zeros((len(class_models_names)))
y_pred_all = []
class_trained_all = []
for model_counter , model in enumerate(class_models_names):
    print(model)
    start_train = time.time()
    classifier_trained,y_pred = class_all(X_train,y_train_c,X_test,model)
    end_train = time.time()
    train_time_all.append((end_train - start_train)/60)
    y_pred_all.append(y_pred)
    # print(y_pred.round(2))
    acc_c[model_counter]= np.mean(y_pred==y_test_c)
    class_trained_all.append(classifier_trained)

ind_best_classifier = np.argmax(acc_c)
print(acc_c)
print('Best classifier:',class_models_names[ind_best_classifier])
#%%
# model_best = class_models_names[np.argmax(acc_c)]
class_trained_best = class_trained_all[ind_best_classifier]
y_pred_regressor = y_pred_all[ind_best_classifier]
y_reg_adaptive = np.zeros((len(X_test)))
RMSE_opt_all = []
for c_i , test_instance in enumerate(X_test):
    y_reg_adaptive[c_i] = reg_trained_all[y_pred_regressor[c_i]].predict(expand_dims_st(test_instance))[0]

RMSE_opt_all = RMSE(y_reg_adaptive,y_test)
    

print(RMSE_opt_all)
BC = os.path.join(sav_path,'best_classifier.obj')
save_object(class_trained_best, BC)

#%%
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
        y_i = reg_trained_all[ind_reg].predict(expand_dims_st(test_instance))[0]
        y_pred_reg_best.append(y_i)
    y_test_pred_list.append(y_pred_reg_best)
    rmse_i_list = RMSE(np.array(group_val['y']),y_pred_reg_best)
    rmse_list.append(rmse_i_list)
end_test = time.time()
test_time = end_test - start_test
obj = {'acc_c':acc_c,'test_time':test_time,'train_time':train_time_all,'y_test':y_test_list,'y_test_pred':y_test_pred_list,'rmse_list':np.array(rmse_list),'Mids_test':Mids_test,'Best classifier':class_models_names[ind_best_classifier]}
filename = os.path.join(sav_path,'Adaptive_predictor.obj')
save_object(obj, filename)
print(RMSE_opt_all)
print('RMSE:',np.mean(rmse_list))
print("RMSE:",RMSE(flatten(y_test_list),flatten(y_test_pred_list)))








