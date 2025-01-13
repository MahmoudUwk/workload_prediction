import os
from keras.callbacks import EarlyStopping
import numpy as np
from Alibaba_helper_functions import loadDatasetObj,save_object,MAPE,MAE,RMSE
from Alibaba_helper_functions import expand_dims,list_to_array,get_en_de_lstm_model_attention
from Alibaba_fet_features_LSTM_no_cluster import get_dataset_alibaba_lstm_no_cluster
import warnings
import time
warnings.filterwarnings('ignore')


def get_data(num_feat,seq_len):
    X_train,y_train,X_val,y_val,X_test_list ,y_test_list,scaler,Mids_test = get_dataset_alibaba_lstm_no_cluster(seq_len,num_feat)
    X_test =list_to_array(X_test_list,seq_len,num_feat)
    y_test = list_to_array(y_test_list,0,num_feat)
    return X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test


from args import get_paths
base_path,processed_path,feat_stats_step1,feat_stats_step2,feat_stats_step3,sav_path,sav_path_plots = get_paths()

if not os.path.exists(sav_path):
    os.makedirs(sav_path)
num_feat = 2

num_epoc = 700
folder_name = 'CuckooSearch_population_5_itr_15'
file_para_name = 'Best_paramCuckooSearch_val_population_5_itr_15.obj'
batch_size = 2**8
best_params = loadDatasetObj(os.path.join(os.path.join(sav_path,folder_name),file_para_name))['best_para_save']

X_train,y_train,X_val,y_val,X_test ,y_test,scaler,X_test_list,y_test_list,Mids_test = get_data(num_feat,best_params['seq'])

output_dim = 1
input_dim=(X_train.shape[1],X_train.shape[2])
y_train = expand_dims(expand_dims(y_train))
y_val = expand_dims(expand_dims(y_val))



model= get_en_de_lstm_model_attention(input_dim,output_dim,**best_params)

callbacks_list = [EarlyStopping(monitor='val_loss', 
                                patience=15, restore_best_weights=True)]#,lr_scheduler]

start_train = time.time()
history = model.fit(X_train, y_train, epochs=num_epoc , 
          batch_size=batch_size, verbose=2, shuffle=True, 
          validation_data=(X_val,y_val),callbacks=callbacks_list)

end_train = time.time()
train_time = (end_train - start_train)/60

y_test_pred = (model.predict(X_test))*scaler

rmse = RMSE(y_test*scaler,y_test_pred)
mae = MAE(y_test*scaler,y_test_pred)
mape = MAPE(y_test*scaler,y_test_pred)
print(rmse,mae,mape)

filename = os.path.join(sav_path,'cuckoosearch_EnDeAtt.obj')
y_test_pred_list = []
rmse_list = []
start_test = time.time()
for c,test_sample in enumerate(X_test_list):
    pred_i = model.predict(test_sample)
    y_test_pred_list.append(pred_i*scaler)
    rmse_i_list = RMSE(y_test_list[c]*scaler,pred_i*scaler)
    y_test_list[c] = y_test_list[c]*scaler
    rmse_list.append(rmse_i_list)
end_test = time.time()
test_time = end_test - start_test
val_loss = history.history['val_loss']
train_loss = history.history['loss']
obj = {'test_time':test_time,'train_time':train_time,'y_test':y_test_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list),'best_params':best_params,'Mids_test':Mids_test,'val_loss':val_loss,'train_loss':train_loss}
save_object(obj, filename)
#%%
from get_data_infere_py import get_data_inf
scaler = 100
feat_names = ['cpu_utilization', 'memory_utilization']
target = ['cpu_utilization']
X_list,Y_list,M_ids = get_data_inf(best_params['seq'],feat_names,target)

y_test_pred_list = []
rmse_list_google = []
from args_google import get_paths
_,_,_,_,_,sav_path,_ = get_paths()
Y_list_scaled = Y_list.copy()
for c,test_sample in enumerate(X_list):
    pred_i = (model.predict(test_sample/scaler)) *scaler
    # pred_i = np.clip(pred_i,0,100)
    y_test_pred_list.append(pred_i)
    rmse_i_list = RMSE(Y_list[c],pred_i)
    # Y_list_scaled[c] = Y_list[c]*scaler
    rmse_list_google.append(rmse_i_list)

obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_google),'Mids_test':M_ids}
filename = os.path.join(sav_path,'CDEL_google_inference.obj')
save_object(obj, filename)

print(np.mean(rmse_list_google))
#%%
from get_data_infere_py_BB import get_data_inf_BB
feat_names = ['CPU usage [%]', 'memory_utilization']
target = 'CPU usage [%]'
id_m = "machine_id"
sort_by = 'Timestamp [ms]'
X_list,Y_list,M_ids = get_data_inf_BB(best_params['seq'],feat_names,target)
y_test_pred_list = []
rmse_list_BB = []
from args_BB import get_paths
_,_,_,_,_,sav_path,_ = get_paths()
Y_list_scaled = Y_list.copy()
for c,test_sample in enumerate(X_list):
    pred_i = (model.predict(test_sample/scaler)) *scaler
    # pred_i = np.clip(pred_i,0,100)
    y_test_pred_list.append(pred_i)
    rmse_i_list = RMSE(Y_list[c],pred_i)
    # Y_list_scaled[c] = Y_list[c]*scaler
    rmse_list_BB.append(rmse_i_list)

obj = {'y_test':Y_list,'y_test_pred':y_test_pred_list,'scaler':scaler,'rmse_list':np.array(rmse_list_BB),'Mids_test':M_ids}
filename = os.path.join(sav_path,'CDEL_BB_inference.obj')
save_object(obj, filename)

print(np.mean(rmse_list_BB))
